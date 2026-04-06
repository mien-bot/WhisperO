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

from .audio import CHANNELS, SAMPLE_RATE, get_shared_stream
from .config import MEETINGS_DIR, load_config
from .transcribe import transcribe_meeting_segment, transcription_lock

# Max chunks the ring buffer will hold before silently dropping old audio.
# ~20s at 16 kHz / 1024 blocksize ≈ 312 chunks.  Generous enough for a
# 10s segment + overlap + timing jitter, but bounded so a stalled segment
# thread can never balloon RAM.
_MAX_RING_CHUNKS = 320


class MeetingRecorder:
    """Continuously records audio, segments it, runs VAD, and transcribes speech."""

    def __init__(
        self,
        session: MeetingSession,
        device_index: int | None = None,
        segment_duration: int = 10,
        overlap: float = 0.5,
    ):
        self._session = session
        self._device_index = device_index
        self._segment_duration = segment_duration
        self._overlap = overlap

        # Audio buffering — bounded ring buffer
        self._chunk_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=_MAX_RING_CHUNKS,
        )
        self._overlap_chunks: list[np.ndarray] = []
        self._segment_queue: queue.Queue = queue.Queue(maxsize=20)

        # Lifecycle
        self._running = False
        self._stop_event = threading.Event()
        self._segment_thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None

        # Timestamp tracking — cumulative new-audio time (excludes overlap)
        self._cumulative_new_seconds = 0.0

    def start(self) -> None:
        self._running = True
        self._stop_event.clear()

        stream = get_shared_stream(self._device_index)
        stream.add_consumer("meeting", self._on_audio)

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

        # Sentinel to unblock the transcribe thread
        self._segment_queue.put(None)

        if self._segment_thread:
            self._segment_thread.join(timeout=5)
        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=30)

        # Explicit cleanup
        self._chunk_buffer.clear()
        self._overlap_chunks.clear()
        gc.collect()
        print("  Meeting recorder stopped")

    # ── SharedStream consumer callback ──────────────────────────────────

    def _on_audio(self, indata: np.ndarray) -> None:
        if self._running:
            self._chunk_buffer.append(indata)

    # ── Segment extraction thread ───────────────────────────────────────

    def _segment_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._segment_duration)
            if self._stop_event.is_set():
                break

            # Drain chunk buffer
            chunks: list[np.ndarray] = []
            while self._chunk_buffer:
                try:
                    chunks.append(self._chunk_buffer.popleft())
                except IndexError:
                    break

            if not chunks:
                continue

            # Prepend overlap from previous segment
            prepended_overlap = self._overlap_chunks
            all_chunks = prepended_overlap + chunks
            audio = np.concatenate(all_chunks, axis=0)

            # Save tail of new chunks for next segment's overlap (O(1) slice)
            overlap_samples = int(self._overlap * SAMPLE_RATE)
            total = 0
            split_idx = len(chunks)
            for i in range(len(chunks) - 1, -1, -1):
                total += len(chunks[i])
                split_idx = i
                if total >= overlap_samples:
                    break
            self._overlap_chunks = chunks[split_idx:]

            # Calculate timing: prepended overlap duration comes from the
            # difference between total audio and new-only audio.
            new_samples = sum(len(c) for c in chunks)
            prepended_secs = (len(audio) - new_samples) / SAMPLE_RATE
            segment_audio_start = max(0.0, self._cumulative_new_seconds - prepended_secs)
            self._cumulative_new_seconds += new_samples / SAMPLE_RATE

            # Free list references (numpy arrays still alive via `audio`)
            del prepended_overlap, all_chunks, chunks

            # Create WAV buffer from int16 audio
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            buf.seek(0)

            # Convert to float32 for VAD, then free int16
            audio_float = audio.flatten().astype(np.float32) / 32768.0
            del audio

            # Run Silero VAD — skip silence
            try:
                from faster_whisper.vad import VadOptions, get_speech_timestamps

                vad_opts = VadOptions(min_silence_duration_ms=500)
                speech_ts = get_speech_timestamps(audio_float, vad_options=vad_opts)
                if not speech_ts:
                    del audio_float, buf
                    continue  # no speech in this segment
            except Exception as e:
                print(f"  VAD error (transcribing anyway): {e}")

            try:
                self._segment_queue.put_nowait(
                    (segment_audio_start, buf, audio_float)
                )
            except queue.Full:
                print("  Meeting segment queue full, dropping segment")
                del buf, audio_float

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
        seg_dur = self._config.get("meeting_segment_duration", 10)
        overlap = self._config.get("meeting_overlap", 0.5)

        self._recorder = MeetingRecorder(
            session=self,
            device_index=self._device_index,
            segment_duration=seg_dur,
            overlap=overlap,
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
