from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400          # 25 ms window at 16 kHz
HOP_LENGTH = 160     # 10 ms hop
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0       # Nyquist for 16 kHz

MODEL_DIR = Path.home() / ".whispero" / "models" / "ecapa-tdnn"
ONNX_PATH = MODEL_DIR / "ecapa_tdnn_voxceleb.onnx"
STATS_PATH = MODEL_DIR / "ecapa_stats.npz"


class DiarizationModelNotFound(Exception):
    """Raised when the ONNX diarization model has not been downloaded."""


def is_model_downloaded() -> bool:
    return ONNX_PATH.exists() and ONNX_PATH.with_suffix(".onnx.data").exists()


# ── Numpy Mel filterbank ─────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Create a (n_mels, n_fft//2 + 1) Mel filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


# Module-level cache
_mel_filterbank: np.ndarray | None = None


def _compute_fbank(audio: np.ndarray) -> np.ndarray:
    """Compute 80-bin log-Mel filterbank features from raw float32 audio.

    Returns (time_frames, 80) float32 array matching SpeechBrain's Fbank output.
    """
    global _mel_filterbank
    if _mel_filterbank is None:
        _mel_filterbank = _create_mel_filterbank()

    # Pad audio to fill last frame
    pad_amount = N_FFT - (len(audio) % HOP_LENGTH)
    if pad_amount > 0:
        audio = np.pad(audio, (0, pad_amount), mode="constant")

    # STFT using numpy
    window = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
    )
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=N_FFT)
    power = np.abs(spectrum) ** 2

    # Apply Mel filterbank
    mel_spec = power @ _mel_filterbank.T  # (time, n_mels)

    # Log with floor to avoid log(0)
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    return log_mel.astype(np.float32)


def _normalize_features(features: np.ndarray, stats_path: Path | None = None) -> np.ndarray:
    """Per-utterance mean normalization (matches SpeechBrain sentence-level norm)."""
    if stats_path and stats_path.exists():
        try:
            stats = np.load(stats_path)
            mean = stats["mean"].astype(np.float32)
            std = stats["std"].astype(np.float32)
            if mean.size > 0 and std.size > 0:
                std = np.maximum(std, 1e-10)
                return (features - mean) / std
        except Exception:
            pass

    # Sentence-level mean subtraction (default for ECAPA-TDNN)
    mean = features.mean(axis=0, keepdims=True)
    return features - mean


# ── Cosine similarity ────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Segment merging ─────────────────────────────────────────────────────

_MIN_EMBED_SECS = 1.5   # minimum audio for a reliable voice embedding
_TURN_GAP_SECS = 0.8    # merge segments within this gap into one turn
_CENTROID_MAX_WEIGHT = 20  # cap running-mean weight so centroids stay adaptive


def _merge_into_turns(
    segments: list[tuple[float, float, str]],
    gap: float = _TURN_GAP_SECS,
) -> list[tuple[float, float, list[int]]]:
    """Group adjacent segments (within *gap* seconds) into speaking turns.

    Returns list of (turn_start, turn_end, [original_segment_indices]).
    """
    if not segments:
        return []
    turns: list[tuple[float, float, list[int]]] = []
    start, end = segments[0][0], segments[0][1]
    indices = [0]
    for i in range(1, len(segments)):
        s, e, _ = segments[i]
        if s - end <= gap:
            end = max(end, e)
            indices.append(i)
        else:
            turns.append((start, end, indices))
            start, end = s, e
            indices = [i]
    turns.append((start, end, indices))
    return turns


# ── Speaker diarizer ────────────────────────────────────────────────────

class SpeakerDiarizer:
    """Identify speakers using ECAPA-TDNN ONNX model + online clustering."""

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.55,
        max_speakers: int = 10,
    ):
        self._threshold = threshold
        self._max_speakers = max_speakers

        # Running centroids for the entire meeting
        self._centroids: list[np.ndarray] = []
        self._centroid_counts: list[int] = []
        # Persist last-assigned speaker across calls so a short utterance
        # at the start of a new segment inherits the most recent speaker
        # instead of defaulting back to speaker 0.
        self._last_assigned: int = 0

        if not ONNX_PATH.exists():
            raise DiarizationModelNotFound(
                "Speaker diarization model not downloaded. "
                "Use the tray menu to download it."
            )

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(ONNX_PATH), providers=providers)

        # Load normalization stats if available
        self._stats_path = STATS_PATH if STATS_PATH.exists() else None

        provider_used = self._session.get_providers()[0]
        device_label = "GPU" if "CUDA" in provider_used else "CPU"
        print(f"  Speaker diarization model loaded (ONNX, {device_label})")

    # ── Public API ──────────────────────────────────────────────────────

    def identify_speakers(
        self,
        audio: np.ndarray,
        segments: list[tuple[float, float, str]],
    ) -> list[int]:
        """Full diarization pipeline for one chunk of meeting audio.

        1. Merge adjacent Whisper segments into speaking turns (longer audio
           = much more reliable voice embeddings).
        2. Extract one ECAPA-TDNN embedding per turn (skip turns < 1.5 s).
        3. Online-cluster turns against running centroids.
        4. Map turn-level speaker IDs back to the original segments.

        Returns one speaker-ID (0-based) per input segment.
        """
        if not segments:
            return []

        # 1. Merge adjacent segments into speaking turns
        turns = _merge_into_turns(segments)

        # 2. Extract one embedding per turn (None for short turns)
        turn_embeddings: list[np.ndarray | None] = []
        for t_start, t_end, _ in turns:
            duration = t_end - t_start
            s = int(t_start * SAMPLE_RATE)
            e = min(int(t_end * SAMPLE_RATE), len(audio))
            if duration < _MIN_EMBED_SECS or e <= s:
                turn_embeddings.append(None)
            else:
                turn_embeddings.append(self._extract_embedding(audio[s:e]))

        # 3. Assign speaker IDs to turns
        turn_ids = self._cluster_turns(turn_embeddings)

        # 4. Map back to original segments
        result = [0] * len(segments)
        for turn_idx, (_, _, seg_indices) in enumerate(turns):
            for si in seg_indices:
                result[si] = turn_ids[turn_idx]
        return result

    # ── Internals ───────────────────────────────────────────────────────

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract a 192-dim speaker embedding from a raw float32 audio array."""
        fbank = _compute_fbank(audio)
        fbank = _normalize_features(fbank, self._stats_path)
        fbank_input = fbank[np.newaxis, :, :]  # (1, time, 80)
        result = self._session.run(None, {"features": fbank_input})
        return result[0].flatten()

    def _cluster_turns(
        self, embeddings: list[np.ndarray | None],
    ) -> list[int]:
        """Assign speakers to turns, handling short turns by temporal fallback."""
        speaker_ids: list[int] = []

        for emb in embeddings:
            if emb is None or np.allclose(emb, 0):
                # Short turn — inherit the most recent speaker (persistent
                # across calls so a brief utterance at the start of a new
                # segment doesn't snap back to speaker 0).
                speaker_ids.append(self._last_assigned)
                continue

            if not self._centroids:
                self._centroids.append(emb.copy())
                self._centroid_counts.append(1)
                speaker_ids.append(0)
                self._last_assigned = 0
                continue

            similarities = [_cosine_similarity(emb, c) for c in self._centroids]
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]

            if best_sim >= self._threshold:
                # Match — update centroid (capped running mean)
                n = min(self._centroid_counts[best_idx], _CENTROID_MAX_WEIGHT)
                self._centroids[best_idx] = (
                    self._centroids[best_idx] * n + emb
                ) / (n + 1)
                self._centroid_counts[best_idx] += 1
                speaker_ids.append(best_idx)
                self._last_assigned = best_idx
            elif len(self._centroids) < self._max_speakers:
                # New speaker
                new_idx = len(self._centroids)
                self._centroids.append(emb.copy())
                self._centroid_counts.append(1)
                speaker_ids.append(new_idx)
                self._last_assigned = new_idx
            else:
                # At speaker limit — assign to closest
                speaker_ids.append(best_idx)
                self._last_assigned = best_idx

        return speaker_ids
