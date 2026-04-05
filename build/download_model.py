"""Post-install helper: download the default whisper model with progress."""
from __future__ import annotations

import sys
import os

def main():
    model_size = "large-v3"
    cache_dir = os.path.join(os.path.expanduser("~"), ".whispero", "models")
    os.makedirs(cache_dir, exist_ok=True)

    # Check if already cached
    repo_dir = os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}")
    if os.path.exists(repo_dir):
        for ext in ("*.bin", "*.safetensors"):
            import glob
            if glob.glob(os.path.join(repo_dir, "**", ext), recursive=True):
                print(f"Model {model_size} already downloaded.")
                return 0

    print(f"Downloading {model_size} model to {cache_dir}...")
    print("This may take several minutes depending on your connection.")

    try:
        from faster_whisper import WhisperModel
        WhisperModel(model_size, device="cpu", compute_type="auto", download_root=cache_dir)
        print(f"Model {model_size} downloaded successfully!")
        return 0
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("The model will be downloaded when you first launch WhisperO.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
