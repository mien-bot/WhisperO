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

class LoopbackStream:
    """Captures system/desktop audio via WASAPI loopback (Windows only).

    Similar to SharedStream but uses pyaudiowpatch to record whatever
    is playing through the speakers.  Audio is resampled to 16 kHz mono
    to match the mic stream format.
    """

    def __init__(self):
        self._stream = None
        self._pa = None
        self._consumers: dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._device_info: dict | None = None

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.is_active()

    @staticmethod
    def is_available() -> bool:
        """Check if WASAPI loopback is supported on this system."""
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            try:
                p.get_host_api_info_by_type(pyaudio.paWASAPI)
                loopback = p.get_default_wasapi_loopback()
                return loopback is not None
            except (OSError, StopIteration):
                return False
            finally:
                p.terminate()
        except ImportError:
            return False

    def add_consumer(self, name: str, callback: Callable) -> None:
        need_start = False
        with self._lock:
            self._consumers[name] = callback
            if self._stream is None or not self._stream.is_active():
                need_start = True
        if need_start:
            self._start_stream()

    def remove_consumer(self, name: str) -> None:
        should_stop = False
        with self._lock:
            self._consumers.pop(name, None)
            if not self._consumers and self._stream:
                should_stop = True
        if should_stop:
            self._stop_stream()

    def _start_stream(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        try:
            loopback = self._pa.get_default_wasapi_loopback()
        except (OSError, StopIteration):
            print("  No WASAPI loopback device found")
            self._pa.terminate()
            self._pa = None
            return

        self._device_info = loopback
        src_rate = int(loopback["defaultSampleRate"])
        src_channels = loopback["maxInputChannels"]
        print(f"  System audio: {loopback['name']} ({src_rate}Hz, {src_channels}ch)")

        def audio_callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.int16)
            # Convert to mono if stereo
            if src_channels > 1:
                audio = audio.reshape(-1, src_channels).mean(axis=1).astype(np.int16)
            # Resample to 16 kHz if needed
            if src_rate != SAMPLE_RATE:
                num_samples = int(len(audio) * SAMPLE_RATE / src_rate)
                indices = np.linspace(0, len(audio) - 1, num_samples).astype(int)
                audio = audio[indices]
            data = audio.reshape(-1, 1)
            with self._lock:
                for cb in list(self._consumers.values()):
                    try:
                        cb(data)
                    except Exception:
                        pass
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=src_channels,
            rate=src_rate,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=1024,
            stream_callback=audio_callback,
        )
        self._stream.start_stream()

    def _stop_stream(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def stop(self) -> None:
        with self._lock:
            self._consumers.clear()
        self._stop_stream()


# ── Module-level loopback stream ─────────────────────────────────────

_loopback_stream: LoopbackStream | None = None
_loopback_stream_lock = threading.Lock()


def get_loopback_stream() -> LoopbackStream:
    """Get or create the module-level LoopbackStream."""
    global _loopback_stream
    with _loopback_stream_lock:
        if _loopback_stream is None:
            _loopback_stream = LoopbackStream()
        return _loopback_stream


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
