from __future__ import annotations

import io
import sys
import threading
import wave
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORDING_SECONDS = 120  # cap at 2 minutes to prevent runaway RAM usage


@dataclass
class RecorderState:
    recording: bool = False
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    stream: sd.InputStream | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    enabled: bool = True
    _callback_count: int = 0


def get_input_devices() -> list[tuple[int, str]]:
    """Return list of (device_index, device_name) for all input devices."""
    result = []
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                result.append((i, dev["name"]))
    except Exception:
        pass
    return result


def _resolve_device(device_index: int | None) -> int:
    """Resolve a device index, falling back to system default if invalid."""
    if device_index is not None:
        try:
            dev = sd.query_devices(device_index)
            if dev["max_input_channels"] > 0:
                return device_index
        except Exception:
            print(f"  Device {device_index} unavailable, falling back to default")

    chosen = sd.default.device[0]
    if chosen is None or chosen < 0:
        chosen = sd.query_devices(kind="input")["index"]
    return chosen


def start_recording(
    state: RecorderState,
    play_sound_fn: Callable[[str], None],
    device_index: int | None = None,
) -> None:
    """Start capturing audio from the specified (or default) microphone."""
    if not state.enabled:
        return

    with state.lock:
        if state.recording:
            return
        state.recording = True
        state.audio_chunks = []
        state._callback_count = 0

    play_sound_fn("start")
    print("Recording...")

    max_chunks = int(MAX_RECORDING_SECONDS * SAMPLE_RATE / 1024)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  Audio status: {status}", file=sys.stderr)
        with state.lock:
            if state.recording:
                if state._callback_count >= max_chunks:
                    return  # silently stop capturing to cap memory
                state.audio_chunks.append(indata.copy())
                state._callback_count += 1

    try:
        chosen = _resolve_device(device_index)
        dev_info = sd.query_devices(chosen)
        print(f"  Mic: {dev_info['name']} (device {chosen})")

        stream = sd.InputStream(
            device=chosen,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=audio_callback,
            blocksize=1024,
        )
        stream.start()
        with state.lock:
            state.stream = stream
        print(f"  Stream active: {stream.active}")
    except Exception as e:
        print(f"  ERROR starting audio stream: {e}")
        # Clean up stream if it was created but failed to start
        if state.stream is not None:
            try:
                state.stream.close()
            except Exception:
                pass
            state.stream = None
        with state.lock:
            state.recording = False


def stop_recording(state: RecorderState, play_sound_fn: Callable[[str], None]) -> io.BytesIO | None:
    """Stop recording and return the audio as a WAV bytes buffer."""
    with state.lock:
        if not state.recording:
            return None
        state.recording = False
        stream = state.stream
        state.stream = None

    play_sound_fn("stop")

    if stream:
        try:
            print(f"  Stream active before stop: {stream.active}, callbacks fired: {state._callback_count}")
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"  ERROR stopping stream: {e}")

    with state.lock:
        chunks = state.audio_chunks
        state.audio_chunks = []

    if not chunks:
        if state._callback_count == 0:
            print("  No audio captured - mic may be in use by another app (call, meeting, etc.)")
        else:
            print("  No audio captured")
        return None

    audio_data = np.concatenate(chunks, axis=0)
    del chunks  # free chunk list immediately
    duration = len(audio_data) / SAMPLE_RATE
    print(f"  Captured {duration:.1f}s of audio ({len(audio_data)} samples)")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    del audio_data  # free raw numpy array, WAV is in buf now
    buf.seek(0)
    return buf
