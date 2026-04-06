"""Download manager for optional diarization model files."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable

import requests

from .diarize import MODEL_DIR, ONNX_PATH, STATS_PATH, is_model_downloaded

# ── Update these after running scripts/export_ecapa_onnx.py ─────────────
# Host the exported files on GitHub Releases and paste the URLs + hashes here.
ONNX_URL = "https://github.com/parkercai/whispero/releases/download/models-v1/ecapa_tdnn_voxceleb.onnx"
STATS_URL = "https://github.com/parkercai/whispero/releases/download/models-v1/ecapa_stats.npz"
ONNX_SHA256 = ""   # fill after export
STATS_SHA256 = ""  # fill after export
ONNX_SIZE_BYTES = 85_000_000  # approximate, updated after export


ProgressCallback = Callable[[int, int], None]  # (downloaded, total)


def get_model_size() -> int:
    """Return expected download size in bytes."""
    return ONNX_SIZE_BYTES


def download_diarization_model(
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Download the ONNX diarization model and stats file.

    Streams with resume support (.part files). Verifies SHA256 if configured.
    Returns the path to the downloaded ONNX model.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _download_file(
        url=ONNX_URL,
        dest=ONNX_PATH,
        expected_sha256=ONNX_SHA256 or None,
        progress_callback=progress_callback,
    )

    # Stats file is small, download without progress
    try:
        _download_file(
            url=STATS_URL,
            dest=STATS_PATH,
            expected_sha256=STATS_SHA256 or None,
        )
    except Exception as e:
        # Stats are optional — per-utterance norm works as fallback
        print(f"  Stats download skipped: {e}")

    return ONNX_PATH


def remove_diarization_model() -> None:
    """Remove downloaded model files."""
    for path in [ONNX_PATH, STATS_PATH]:
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
