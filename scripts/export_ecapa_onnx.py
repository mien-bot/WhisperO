#!/usr/bin/env python3
"""Export the SpeechBrain ECAPA-TDNN speaker embedding model to ONNX.

One-time script — requires speechbrain + torch installed:
    pip install speechbrain torch

Produces:
    models/ecapa_tdnn_voxceleb.onnx   (~80-90 MB)
    models/ecapa_stats.npz             (norm stats, few KB)

After export, speechbrain/torch can be uninstalled.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("This script requires speechbrain and torch:")
        print("  pip install speechbrain torch")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load the pretrained model ────────────────────────────────────────
    print("Loading speechbrain ECAPA-TDNN model...")
    save_dir = Path.home() / ".whispero" / "models" / "spkrec-ecapa-voxceleb"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(save_dir),
        run_opts={"device": "cpu"},
    )

    # ── Extract the embedding model ──────────────────────────────────────
    embedding_model = classifier.mods.embedding_model
    embedding_model.eval()

    # The embedding model expects (batch, time, 80) fbank features + lengths
    # We need a wrapper that takes fbank features and returns embeddings
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """features: (batch, time, 80) -> embeddings: (batch, 192)"""
            # embedding_model expects features and a lengths tensor
            lengths = torch.ones(features.shape[0], device=features.device)
            embeddings = self.model(features, lengths)
            return embeddings.squeeze(1)

    wrapper = EmbeddingWrapper(embedding_model)
    wrapper.eval()

    # ── Export mean/var norm statistics ───────────────────────────────────
    print("Extracting normalization statistics...")
    try:
        mean_var_norm = classifier.mods.mean_var_norm
        # SpeechBrain InputNormalization stores running stats
        glob_mean = mean_var_norm.glob_mean.cpu().numpy().flatten()
        glob_std = mean_var_norm.glob_std.cpu().numpy().flatten()
        stats_path = OUTPUT_DIR / "ecapa_stats.npz"
        np.savez(stats_path, mean=glob_mean, std=glob_std)
        print(f"  Saved: {stats_path} ({stats_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"  Warning: Could not extract norm stats ({e})")
        print("  The ONNX model will use per-utterance normalization instead")
        stats_path = None

    # ── Export to ONNX ───────────────────────────────────────────────────
    print("Exporting to ONNX...")
    # Create dummy input: (batch=1, time_frames=100, mel_bins=80)
    dummy_features = torch.randn(1, 100, 80)

    onnx_path = OUTPUT_DIR / "ecapa_tdnn_voxceleb.onnx"
    torch.onnx.export(
        wrapper,
        dummy_features,
        str(onnx_path),
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "embedding": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  Saved: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── Verify ONNX output matches PyTorch ───────────────────────────────
    print("Verifying ONNX output...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))

    # Test with a few random inputs
    for i in range(3):
        test_input = torch.randn(1, 50 + i * 30, 80)
        with torch.no_grad():
            torch_out = wrapper(test_input).numpy()
        onnx_out = session.run(None, {"features": test_input.numpy()})[0]

        cos_sim = np.dot(torch_out.flatten(), onnx_out.flatten()) / (
            np.linalg.norm(torch_out) * np.linalg.norm(onnx_out)
        )
        print(f"  Test {i + 1}: cosine similarity = {cos_sim:.6f}")
        if cos_sim < 0.999:
            print("  WARNING: Low similarity — ONNX export may be inaccurate")

    # ── Compute SHA256 ───────────────────────────────────────────────────
    sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    print(f"\n  SHA256: {sha256}")
    print(f"  Size:   {onnx_path.stat().st_size} bytes")
    print(f"\nUpdate ONNX_SHA256 in src/whispero/download.py with this hash.")

    if stats_path:
        stats_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        print(f"  Stats SHA256: {stats_sha}")

    print("\nDone! Host these files on GitHub Releases and update the URL in download.py.")


if __name__ == "__main__":
    main()
