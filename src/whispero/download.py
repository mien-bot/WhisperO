"""Download manager for optional diarization model files."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable

import requests

from .diarize import MODEL_DIR, ONNX_PATH, STATS_PATH, is_model_downloaded

# ── Model file URLs and checksums ────────────────────────────────────────
_BASE_URL = "https://github.com/mien-bot/WhisperO/releases/download/models-v1"
ONNX_URL = f"{_BASE_URL}/ecapa_tdnn_voxceleb.onnx"
ONNX_DATA_URL = f"{_BASE_URL}/ecapa_tdnn_voxceleb.onnx.data"
STATS_URL = f"{_BASE_URL}/ecapa_stats.npz"

ONNX_SHA256 = "3783724564edbe7818ffe9f0bde7e6f59fc3197630957f081b5f9f68f7887c7c"
ONNX_DATA_SHA256 = "069be348f6c01a15dbd1c6ea4d77a141c444a0ed0ae895af4d2e61123cb9e27d"
STATS_SHA256 = "df08b272fad05be5512712c584aea8e23f5d4fbe1c325408e3c402abfd89d596"

# Total download: .onnx (~0.6 MB) + .onnx.data (~79 MB) + stats (~0.5 KB)
TOTAL_DOWNLOAD_BYTES = 586_914 + 83_230_720 + 500  # ~80 MB


ProgressCallback = Callable[[int, int], None]  # (downloaded, total)


def get_model_size() -> int:
    """Return expected total download size in bytes."""
    return TOTAL_DOWNLOAD_BYTES


def download_diarization_model(
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Download the ONNX diarization model files.

    Downloads three files: .onnx (graph), .onnx.data (weights), .npz (stats).
    Streams with resume support. Verifies SHA256 checksums.
    Returns the path to the ONNX model file.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Track cumulative progress across all files
    cumulative = [0]
    total = TOTAL_DOWNLOAD_BYTES

    def _wrap_progress(downloaded: int, _file_total: int) -> None:
        if progress_callback:
            progress_callback(cumulative[0] + downloaded, total)

    # 1. Download .onnx graph file (~0.6 MB)
    _download_file(
        url=ONNX_URL,
        dest=ONNX_PATH,
        expected_sha256=ONNX_SHA256,
        progress_callback=_wrap_progress,
    )
    cumulative[0] += ONNX_PATH.stat().st_size

    # 2. Download .onnx.data weights file (~79 MB — the big one)
    data_path = ONNX_PATH.with_suffix(".onnx.data")
    _download_file(
        url=ONNX_DATA_URL,
        dest=data_path,
        expected_sha256=ONNX_DATA_SHA256,
        progress_callback=_wrap_progress,
    )
    cumulative[0] += data_path.stat().st_size

    # 3. Stats file is small, download without progress
    try:
        _download_file(
            url=STATS_URL,
            dest=STATS_PATH,
            expected_sha256=STATS_SHA256,
        )
    except Exception as e:
        # Stats are optional — per-utterance norm works as fallback
        print(f"  Stats download skipped: {e}")

    return ONNX_PATH


def remove_diarization_model() -> None:
    """Remove downloaded model files."""
    for path in [ONNX_PATH, ONNX_PATH.with_suffix(".onnx.data"), STATS_PATH]:
        if path.exists():
            path.unlink()
    # Remove .part files too
    for path in MODEL_DIR.glob("*.part"):
        path.unlink()


def _download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Download a file with streaming, resume support, and optional SHA256 check."""
    if dest.exists():
        if expected_sha256 and _sha256(dest) == expected_sha256:
            return  # already valid
        elif not expected_sha256:
            return  # assume good if no hash configured

    part_path = dest.with_suffix(dest.suffix + ".part")
    downloaded = part_path.stat().st_size if part_path.exists() else 0

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    resp = requests.get(url, headers=headers, stream=True, timeout=(10, 60))

    if resp.status_code == 416:
        # Range not satisfiable — file may be complete or server doesn't support resume
        part_path.unlink(missing_ok=True)
        downloaded = 0
        resp = requests.get(url, stream=True, timeout=(10, 60))

    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + downloaded

    mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
    if mode == "wb":
        downloaded = 0

    with open(part_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total)

    # Verify hash
    if expected_sha256:
        actual = _sha256(part_path)
        if actual != expected_sha256:
            part_path.unlink()
            raise ValueError(
                f"SHA256 mismatch for {dest.name}: "
                f"expected {expected_sha256[:16]}..., got {actual[:16]}..."
            )

    shutil.move(str(part_path), str(dest))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
