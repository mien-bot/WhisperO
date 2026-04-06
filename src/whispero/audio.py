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
    lock: threading.Lock = field(default_factory=threading.Lock)
    enabled: bool = True
    _callback_count: int = 0


class SharedStream:
    """Single mic stream shared between push-to-talk and meeting mode.

    Only one sd.InputStream is open per device.  Consumers register
    callbacks via add_consumer(); the stream auto-starts on the first
    consumer and auto-stops when the last consumer is removed.
    """

    def __init__(self, device_index: int | None = None):
        self._device_index = device_index
        self._stream: sd.InputStream | None = None
        self._consumers: dict[str, Callable] = {}
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active

    def add_consumer(self, name: str, callback: Callable) -> None:
        need_start = False
        with self._lock:
            self._consumers[name] = callback
            if self._stream is None or not self._stream.active:
                need_start = True
        if need_start:
            self._start_stream()

    def remove_consumer(self, name: str) -> None:
        stream_to_stop = None
        with self._lock:
            self._consumers.pop(name, None)
            if not self._consumers and self._stream:
                stream_to_stop = self._stream
                self._stream = None
        # Stop outside lock — Pa_StopStream waits for callback to finish,
        # and the callback acquires self._lock → would deadlock if held here.
        if stream_to_stop:
            try:
                stream_to_stop.stop()
                stream_to_stop.close()
            except Exception:
                pass

    def _start_stream(self) -> None:
        """Start the underlying sd.InputStream."""
        chosen = _resolve_device(self._device_index)
        dev_info = sd.query_devices(chosen)
        print(f"  Mic: {dev_info['name']} (device {chosen})")

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"  Audio status: {status}", file=sys.stderr)
            data = indata.copy()
            with self._lock:
                for cb in list(self._consumers.values()):
                    try:
                        cb(data)
                    except Exception:
                        pass

        stream = sd.InputStream(
            device=chosen,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=audio_callback,
            blocksize=1024,
        )
        stream.start()
        with self._lock:
            self._stream = stream

    def stop(self) -> None:
        """Force-stop: clear all consumers and close the stream."""
        stream_to_stop = None
        with self._lock:
            self._consumers.clear()
            if self._stream:
                stream_to_stop = self._stream
                self._stream = None
        if stream_to_stop:
            try:
                stream_to_stop.stop()
                stream_to_stop.close()
            except Exception:
                pass


# ── Module-level shared stream ────────���─────────────────────────────────

_shared_stream: SharedStream | None = None
_shared_stream_lock = threading.Lock()


def get_shared_stream(device_index: int | None = None) -> SharedStream:
    """Get or create the module-level SharedStream."""
    global _shared_stream
    with _shared_stream_lock:
        if _shared_stream is None:
            _shared_stream = SharedStream(device_index)
        elif device_index is not None:
            _shared_stream._device_index = device_index
        return _shared_stream


# ── Helpers ────────���─────────────────────────────────���──────────────────

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


# ── Push-to-talk recording (uses SharedStream) ─────────────────────────

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

    def on_audio(indata: np.ndarray) -> None:
        with state.lock:
            if state.recording:
                if state._callback_count >= max_chunks:
                    return  # silently stop capturing to cap memory
                state.audio_chunks.append(indata.copy())
                state._callback_count += 1

    try:
        stream = get_shared_stream(device_index)
        stream.add_consumer("push_to_talk", on_audio)
        print(f"  Stream active: {stream.active}")
    except Exception as e:
        print(f"  ERROR starting audio stream: {e}")
        with state.lock:
            state.recording = False


def stop_recording(state: RecorderState, play_sound_fn: Callable[[str], None]) -> io.BytesIO | None:
    """Stop recording and return the audio as a WAV bytes buffer."""
    with state.lock:
        if not state.recording:
            return None
        state.recording = False

    play_sound_fn("stop")

    stream = get_shared_stream()
    print(f"  Stream active before stop: {stream.active}, callbacks fired: {state._callback_count}")
    stream.remove_consumer("push_to_talk")

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
