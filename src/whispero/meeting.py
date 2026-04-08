from __future__ import annotations

import collections
import gc
import io
import json
import os
import platform
import queue
import subprocess
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import CHANNELS, SAMPLE_RATE, get_loopback_stream, get_shared_stream, LoopbackStream
from .config import MEETINGS_DIR, load_config
from .transcribe import transcribe_meeting_segment, transcription_lock

# Max chunks the ring buffer will hold before silently dropping old audio.
# ~20s at 16 kHz / 1024 blocksize ≈ 312 chunks.  Generous enough for a
# max-length segment + timing jitter, but bounded so a stalled segment
# thread can never balloon RAM.
_MAX_RING_CHUNKS = 320

# Silence-aware segmentation tuning
_POLL_INTERVAL_SECS = 0.25        # how often the segment loop wakes up
_MIN_SEGMENT_SECS = 0.8           # don't emit segments shorter than this
_SILENCE_FLUSH_MS = 500           # trailing silence that triggers a flush


def _find_cut_point(
    total_samples: int,
    speech_ts: list[dict] | None,
    silence_samples: int,
    force: bool,
) -> int | None:
    """Decide where to split an accumulated audio buffer.

    Returns a sample index up to which we should flush, or None to wait
    for more audio.  Preferred cut: just after the last speech region
    when it's followed by ≥ silence_samples of silence.  If the buffer
    is too long (force=True), cut at the largest internal silence gap
    or, failing that, flush the whole buffer.
    """
    if not speech_ts:
        return total_samples if force else None

    last_end = speech_ts[-1]["end"]
    tail_silence = total_samples - last_end
    if tail_silence >= silence_samples:
        # Cut a little past the end of speech so we don't clip the final word
        return min(total_samples, last_end + silence_samples // 2)

    if not force:
        return None

    best_cut = None
    best_gap = 0
    for i in range(len(speech_ts) - 1):
        gap = speech_ts[i + 1]["start"] - speech_ts[i]["end"]
        if gap >= silence_samples and gap > best_gap:
            best_gap = gap
            best_cut = speech_ts[i]["end"] + gap // 2
    if best_cut is not None:
        return best_cut

    return total_samples


class MeetingRecorder:
    """Continuously records audio, segments it, runs VAD, and transcribes speech."""

    def __init__(
        self,
        session: MeetingSession,
        device_index: int | None = None,
        max_segment_duration: int = 15,
        audio_source: str = "mic",
    ):
        self._session = session
        self._device_index = device_index
        self._max_segment_duration = max_segment_duration
        self._audio_source = audio_source  # "mic", "system", or "both"

        # Per-source ring buffers — kept separate so mic + system audio
        # can be sample-aligned and mixed rather than arbitrarily
        # interleaved.
        self._mic_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=_MAX_RING_CHUNKS,
        )
        self._sys_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=_MAX_RING_CHUNKS,
        )
        self._segment_queue: queue.Queue = queue.Queue(maxsize=20)

        # Lifecycle
        self._running = False
        self._stop_event = threading.Event()
        self._segment_thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._stop_event.clear()

        source = self._audio_source
        if source in ("mic", "both"):
            stream = get_shared_stream(self._device_index)
            stream.add_consumer("meeting", self._on_mic_audio)
            print(f"  Meeting: mic capture active")

        if source in ("system", "both"):
            if LoopbackStream.is_available():
                loopback = get_loopback_stream()
                loopback.add_consumer("meeting", self._on_sys_audio)
                print(f"  Meeting: system audio capture active")
            else:
                print("  System audio capture unavailable (WASAPI loopback not found)")
                if source == "system":
                    # Fall back to mic if system-only was requested but unavailable
                    stream = get_shared_stream(self._device_index)
                    stream.add_consumer("meeting", self._on_mic_audio)
                    print(f"  Meeting: falling back to mic capture")

        self._segment_thread = threading.Thread(
            target=self._segment_loop, daemon=True, name="meeting-segment",
        )
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop, daemon=True, name="meeting-transcribe",
        )
        self._segment_thread.start()
        self._transcribe_thread.start()
        print("  Meeting recorder started")

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

        get_shared_stream().remove_consumer("meeting")
        try:
            get_loopback_stream().remove_consumer("meeting")
        except Exception:
            pass

        # Sentinel to unblock the transcribe thread
        self._segment_queue.put(None)

        if self._segment_thread:
            self._segment_thread.join(timeout=5)
        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=30)

        # Explicit cleanup
        self._mic_buffer.clear()
        self._sys_buffer.clear()
        gc.collect()
        print("  Meeting recorder stopped")

    # ── Stream consumer callbacks ───────────────────────────────────────

    def _on_mic_audio(self, indata: np.ndarray) -> None:
        if self._running:
            self._mic_buffer.append(indata)

    def _on_sys_audio(self, indata: np.ndarray) -> None:
        if self._running:
            self._sys_buffer.append(indata)

    def _drain_and_mix(self) -> np.ndarray | None:
        """Drain per-source buffers and return a single int16 waveform.

        With only one source, passes the audio through.  With both mic
        and system audio active, trims to the shorter of the two, sums
        them sample-aligned (clipping to int16), and pushes any
        leftover samples back so the next drain stays aligned.
        """
        mic_chunks: list[np.ndarray] = []
        while self._mic_buffer:
            try:
                mic_chunks.append(self._mic_buffer.popleft())
            except IndexError:
                break

        sys_chunks: list[np.ndarray] = []
        while self._sys_buffer:
            try:
                sys_chunks.append(self._sys_buffer.popleft())
            except IndexError:
                break

        mic = np.concatenate(mic_chunks, axis=0) if mic_chunks else None
        sys = np.concatenate(sys_chunks, axis=0) if sys_chunks else None

        if mic is None and sys is None:
            return None
        if mic is None:
            return sys
        if sys is None:
            return mic

        n = min(len(mic), len(sys))
        if n == 0:
            if len(mic) > 0:
                self._mic_buffer.appendleft(mic)
            if len(sys) > 0:
                self._sys_buffer.appendleft(sys)
            return None

        if len(mic) > n:
            self._mic_buffer.appendleft(mic[n:])
        if len(sys) > n:
            self._sys_buffer.appendleft(sys[n:])

        mixed = mic[:n].astype(np.int32) + sys[:n].astype(np.int32)
        return np.clip(mixed, -32768, 32767).astype(np.int16)

    # ── Segment extraction thread ───────────────────────────────────────

    def _segment_loop(self) -> None:
        """Accumulate audio and flush utterance-sized segments on silence.

        Instead of fixed 10s windows, we poll every _POLL_INTERVAL_SECS,
        append new audio to an accumulator, and run Silero VAD.  When
        the accumulator ends with ≥ _SILENCE_FLUSH_MS of silence after
        speech (or hits the max-length safety cap), we flush that span
        as one segment — a natural utterance.  Each flushed segment is
        typically one speaker's turn, which gives the diarizer clean
        embeddings and produces transcript output much more frequently.
        """
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        accum_parts: list[np.ndarray] = []
        accum_len = 0
        accum_start_time = 0.0

        max_samples = int(self._max_segment_duration * SAMPLE_RATE)
        min_samples = int(_MIN_SEGMENT_SECS * SAMPLE_RATE)
        silence_samples = int((_SILENCE_FLUSH_MS / 1000.0) * SAMPLE_RATE)

        def flush(segment_audio: np.ndarray, segment_start: float) -> None:
            if len(segment_audio) < min_samples:
                return

            seg_float = segment_audio.flatten().astype(np.float32) / 32768.0

            # Skip if the flushed window has no actual speech
            try:
                seg_speech = get_speech_timestamps(
                    seg_float,
                    vad_options=VadOptions(min_silence_duration_ms=500),
                )
                if not seg_speech:
                    return
            except Exception:
                pass

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(segment_audio.tobytes())
            buf.seek(0)

            try:
                self._segment_queue.put_nowait((segment_start, buf, seg_float))
            except queue.Full:
                print("  Meeting segment queue full, dropping segment")

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_POLL_INTERVAL_SECS)

            new_audio = self._drain_and_mix()
            if new_audio is not None and len(new_audio) > 0:
                accum_parts.append(new_audio)
                accum_len += len(new_audio)

            if accum_len == 0:
                continue

            audio = (
                np.concatenate(accum_parts, axis=0)
                if len(accum_parts) > 1
                else accum_parts[0]
            )
            audio_float = audio.flatten().astype(np.float32) / 32768.0

            try:
                speech_ts = get_speech_timestamps(
                    audio_float,
                    vad_options=VadOptions(min_silence_duration_ms=_SILENCE_FLUSH_MS),
                )
            except Exception as e:
                print(f"  VAD error: {e}")
                speech_ts = None

            force = accum_len >= max_samples
            cut = _find_cut_point(len(audio_float), speech_ts, silence_samples, force)

            if cut is None:
                # Wait for more audio / a natural pause
                accum_parts = [audio]
                continue

            cut = max(0, min(cut, len(audio)))
            segment_audio = audio[:cut]
            remainder = audio[cut:]
            segment_start = accum_start_time
            accum_start_time += cut / SAMPLE_RATE

            if len(remainder) > 0:
                accum_parts = [remainder]
                accum_len = len(remainder)
            else:
                accum_parts = []
                accum_len = 0

            flush(segment_audio, segment_start)

        # Final flush on stop — drain anything still pending
        tail = self._drain_and_mix()
        if tail is not None and len(tail) > 0:
            accum_parts.append(tail)
            accum_len += len(tail)
        if accum_len > 0:
            audio = (
                np.concatenate(accum_parts, axis=0)
                if len(accum_parts) > 1
                else accum_parts[0]
            )
            flush(audio, accum_start_time)

    # ── Transcription thread ────────────────────────────────────────────

    def _transcribe_loop(self) -> None:
        cfg = load_config()
        model_size = cfg.get("model", "large-v3")
        languages = cfg.get("languages", ["en"])
        diarization_enabled = cfg.get("meeting_diarization", False)

        diarizer = None
        if diarization_enabled:
            try:
                from .diarize import SpeakerDiarizer

                threshold = cfg.get("meeting_diarization_threshold", 0.75)
                max_spk = cfg.get("meeting_max_speakers", 10)
                device = "cuda" if cfg.get("device", "gpu") == "gpu" else "cpu"
                diarizer = SpeakerDiarizer(device=device, threshold=threshold, max_speakers=max_spk)
                print("  Speaker diarization enabled")
            except Exception as e:
                print(f"  Diarization unavailable: {e}")

        while True:
            item = self._segment_queue.get()
            if item is None:
                break

            segment_start, audio_buf, audio_float = item
            del item  # drop tuple reference

            # Wait up to 10s for the lock (yields to push-to-talk)
            acquired = transcription_lock.acquire(timeout=10)
            if not acquired:
                print("  Meeting segment skipped (transcription busy)")
                del audio_buf, audio_float
                continue

            try:
                segments = transcribe_meeting_segment(
                    audio_buf=audio_buf,
                    model_size=model_size,
                    languages=languages,
                    word_timestamps=bool(diarizer),
                )
                del audio_buf  # free WAV buffer

                if not segments:
                    del audio_float
                    continue

                if diarizer:
                    self._write_with_diarization(
                        segments, segment_start, audio_float, diarizer,
                    )
                else:
                    for start, end, text in segments:
                        self._session.write_segment(
                            segment_start + start, segment_start + end, text,
                        )
            except Exception as e:
                print(f"  Meeting transcription error: {e}")
            finally:
                transcription_lock.release()
                del audio_float  # ensure float buffer is freed
                gc.collect()

    def _write_with_diarization(self, segments, segment_start, audio_float, diarizer):
        """Assign speaker labels and write segments."""
        try:
            speaker_ids = diarizer.identify_speakers(audio_float, segments)
            names = self._session.speaker_names
            for (start, end, text), spk_id in zip(segments, speaker_ids):
                key = str(spk_id + 1)
                label = names.get(key, f"Speaker {spk_id + 1}")
                self._session.write_segment(
                    segment_start + start, segment_start + end, text, label,
                )
        except Exception as e:
            print(f"  Diarization error: {e}")
            for start, end, text in segments:
                self._session.write_segment(
                    segment_start + start, segment_start + end, text,
                )


class MeetingSession:
    """Manages a single meeting recording session (lifecycle + output files)."""

    def __init__(
        self,
        device_index: int | None = None,
        config: dict | None = None,
    ):
        self._config = config or load_config()
        self._device_index = device_index
        self._start_time = datetime.now()
        self._recorder: MeetingRecorder | None = None
        self._running = False
        self._write_lock = threading.Lock()
        self._speakers_seen: set[str] = set()

        # Output paths
        MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = self._start_time.strftime("%Y-%m-%d_%H-%M")
        self._txt_path = MEETINGS_DIR / f"Meeting_{ts}.txt"
        self._jsonl_path = MEETINGS_DIR / f"Meeting_{ts}.jsonl"

        # Write header
        header = (
            f"Meeting Transcript - {self._start_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"{'=' * 40}\n\n"
        )
        self._txt_path.write_text(header, encoding="utf-8")

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def start_time(self) -> datetime:
        return self._start_time

    @property
    def running(self) -> bool:
        return self._running

    @property
    def txt_path(self) -> Path:
        return self._txt_path

    @property
    def speaker_names(self) -> dict[str, str]:
        """Speaker ID → display name, e.g. {"1": "Ian", "2": "Parker"}."""
        return self._config.get("meeting_speaker_names", {})

    @property
    def speakers_seen(self) -> set[str]:
        return set(self._speakers_seen)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Used as the safety cap for silence-aware flushing — if no pause
        # is detected within this many seconds, the buffer is force-flushed.
        max_seg_dur = self._config.get("meeting_segment_duration", 15)

        audio_source = self._config.get("meeting_audio_source", "mic")
        self._recorder = MeetingRecorder(
            session=self,
            device_index=self._device_index,
            max_segment_duration=max_seg_dur,
            audio_source=audio_source,
        )
        self._recorder.start()
        print(f"  Meeting started: {self._txt_path.name}")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._recorder:
            self._recorder.stop()
            self._recorder = None

        # Update header with speakers if diarization was used
        if self._speakers_seen:
            try:
                content = self._txt_path.read_text(encoding="utf-8")
                marker = "=" * 40
                idx = content.index(marker) + len(marker)
                speaker_line = f"\nSpeakers: {', '.join(sorted(self._speakers_seen))}"
                content = content[:idx] + speaker_line + content[idx:]
                self._txt_path.write_text(content, encoding="utf-8")
            except Exception:
                pass

        if self._config.get("meeting_auto_open", True):
            self._open_transcript()

        print(f"  Meeting stopped: {self._txt_path.name}")

    # ── Output ──────────────────────────────────────────────────────────

    def write_segment(
        self,
        start: float,
        end: float,
        text: str,
        speaker: str | None = None,
    ) -> None:
        """Append a transcribed segment to .txt and .jsonl output files."""
        text = text.strip()
        if not text:
            return

        h, remainder = divmod(int(start), 3600)
        m, s = divmod(remainder, 60)
        ts_str = f"[{h:02d}:{m:02d}:{s:02d}]"

        with self._write_lock:
            if speaker:
                line = f"{ts_str} {speaker}: {text}\n"
                self._speakers_seen.add(speaker)
            else:
                line = f"{ts_str} {text}\n"

            with open(self._txt_path, "a", encoding="utf-8") as f:
                f.write(line)

            record = {
                "ts": round(start, 1),
                "end": round(end, 1),
                "text": text,
                "speaker": speaker,
            }
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if speaker:
            print(f"  {ts_str} {speaker}: {text}")
        else:
            print(f"  {ts_str} {text}")

    def elapsed_str(self) -> str:
        """Return elapsed time as M:SS string."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        m, s = divmod(int(elapsed), 60)
        return f"{m}:{s:02d}"

    # ── Speaker renaming ──────────────────────────────────────────────

    @staticmethod
    def rename_speakers_in_files(
        txt_path: Path,
        jsonl_path: Path,
        rename_map: dict[str, str],
    ) -> None:
        """Find-and-replace speaker labels in existing transcript files.

        rename_map: {"Speaker 1": "Ian", "Speaker 2": "Parker"}
        """
        # Rewrite .txt
        if txt_path.exists():
            content = txt_path.read_text(encoding="utf-8")
            for old_label, new_name in rename_map.items():
                content = content.replace(f"] {old_label}:", f"] {new_name}:")
            # Update Speakers header line if present
            for old_label, new_name in rename_map.items():
                content = content.replace(old_label, new_name)
            txt_path.write_text(content, encoding="utf-8")

        # Rewrite .jsonl
        if jsonl_path.exists():
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            new_lines = []
            for line in lines:
                try:
                    record = json.loads(line)
                    spk = record.get("speaker")
                    if spk and spk in rename_map:
                        record["speaker"] = rename_map[spk]
                    new_lines.append(json.dumps(record, ensure_ascii=False))
                except Exception:
                    new_lines.append(line)
            jsonl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # ── Internal ────────────────────────────────────────────────────────

    def _open_transcript(self) -> None:
        try:
            if platform.system() == "Windows":
                os.startfile(str(self._txt_path))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(self._txt_path)])
            else:
                subprocess.Popen(["xdg-open", str(self._txt_path)])
        except Exception as e:
            print(f"  Could not open transcript: {e}")
