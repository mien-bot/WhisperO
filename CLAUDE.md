# CLAUDE.md - WhisperO

## What is this?

Push-to-talk desktop dictation app. Hold a hotkey, speak, release — text appears at cursor. Also has a meeting mode for continuous transcription with optional speaker diarization.

## Quick commands

```bash
# Run from source
python -m whispero

# Build standalone .exe
PYTHONIOENCODING=utf-8 python build/build.py

# Build Windows installer (requires Inno Setup 6)
PYTHONIOENCODING=utf-8 python build/build.py --installer

# Export ONNX diarization model (one-time, requires speechbrain+torch)
pip install speechbrain torch
PYTHONIOENCODING=utf-8 python scripts/export_ecapa_onnx.py
```

## Project structure

```
src/whispero/
  app.py          # Main: tray icon, hotkey listener, UI dialogs
  audio.py        # SharedStream (single mic stream shared between modes)
  transcribe.py   # Whisper model loading & transcription (local + server)
  meeting.py      # Meeting mode: continuous segmented recording + transcription
  diarize.py      # Speaker diarization via ONNX (ECAPA-TDNN, no torch needed)
  download.py     # On-demand model download with progress + SHA256 verification
  config.py       # Config loading (JSON file + env vars + defaults)
  clipboard.py    # Paste text at cursor (platform-specific)
  dictionary.py   # Custom word dictionary
  sounds.py       # Sound effect playback

build/
  build.py        # PyInstaller build script
  installer.iss   # Inno Setup config for Windows installer
  icons/          # App icons (.ico, .png, .icns)

scripts/
  export_ecapa_onnx.py  # One-time: convert SpeechBrain model to ONNX
```

## Architecture

- **SharedStream**: One `sounddevice.InputStream` per device, shared between push-to-talk and meeting mode via consumer callbacks. Auto-starts on first consumer, auto-stops on last removal.
- **Transcription lock**: `transcription_lock` in `transcribe.py` prevents push-to-talk and meeting mode from running Whisper simultaneously. Push-to-talk has priority (meeting segments yield).
- **ONNX diarization**: Speaker identification uses ECAPA-TDNN exported to ONNX format. Runs on onnxruntime (already bundled via faster-whisper). Model (~80 MB) is downloaded on-demand from GitHub Releases — not bundled in the installer.
- **Meeting mode**: Records in overlapping segments (10s default, 0.5s overlap), runs VAD to skip silence, transcribes with timestamps, writes `.txt` + `.jsonl` output.
- **Model loading**: faster-whisper with CTranslate2. GPU (CUDA) preferred, falls back to CPU. Models cached in `~/.whispero/models/`. Uses `local_files_only=True` when model is already cached to avoid connecting to HuggingFace.
- **First-launch model download**: On startup, if the Whisper model isn't cached, a tkinter dialog prompts the user to download it with a progress bar. A "Download Model..." tray menu item appears if the model is missing.
- **HuggingFace mirror fallback**: If `huggingface.co` is blocked (SSL/connection errors), the download automatically retries via `hf-mirror.com`. A working mirror is saved to config for future use. Users can also set `hf_mirror` in config or `WHISPERO_HF_MIRROR` env var.
- **CUDA detection**: On startup with GPU mode, checks if CUDA libraries (`cublas64_12.dll`) are loadable. If missing, shows a dialog with options to continue on CPU or download CUDA Toolkit 12.2. Users without admin can copy the 3 CUDA DLLs directly into the WhisperO install folder.
- **DLL search path**: `__main__.py` adds the exe's directory to the DLL search path (`os.add_dll_directory` + `PATH`), so CUDA DLLs placed next to `WhisperO.exe` are found automatically.

## Config

**Location**: `~/.whispero/config.json`

**Priority**: env vars > config file > defaults (in `config.py:DEFAULTS`)

**Key settings**: `backend` (local/server), `model` (large-v3/medium/small/base/tiny), `device` (gpu/cpu), `languages` (list of codes), `mic_device` (int index or null), `hotkey`, `meeting_diarization` (bool), `hf_mirror` (HuggingFace mirror URL).

**Env vars**: `WHISPERO_BACKEND`, `WHISPERO_MODEL`, `WHISPERO_SERVER`, `WHISPERO_HF_MIRROR`

## Build notes

- Standalone app: `dist/WhisperO/WhisperO.exe` (~278 MB folder)
- Windows installer: `dist/WhisperO-Setup.exe` (~80 MB) — CPU-only, no CUDA DLLs bundled
- Windows installer with CUDA: `dist/WhisperO-Setup-CUDA.exe` (~332 MB) — includes CUDA 12 DLLs for GPU support out of the box
- To build CUDA variant: run normal `--installer` build, then copy `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll` from CUDA toolkit into `dist/WhisperO/_internal/`, and re-run Inno Setup with `/FWhisperO-Setup-CUDA`
- Must use `PYTHONIOENCODING=utf-8` on Windows (build script prints Unicode)
- Build script auto-strips CUDA DLLs (~608 MB) from the dist folder to keep the default installer small
- Diarization model is NOT bundled — downloaded at runtime from GitHub Releases (`models-v1` tag)
- The `models/` directory is gitignored (exported files are on GitHub Releases)
- Entry script `.whispero_entry.py` is auto-created and cleaned up during build

## Diarization model workflow

The ONNX model files are hosted at: `https://github.com/mien-bot/WhisperO/releases/tag/models-v1`

To re-export (only needed if model changes):
1. `pip install speechbrain torch`
2. `python scripts/export_ecapa_onnx.py`
3. Update SHA256 hashes in `src/whispero/download.py`
4. `gh release upload models-v1 models/*.onnx* models/*.npz --clobber`
5. Uninstall speechbrain+torch

## CUDA / GPU setup

GPU transcription requires CUDA Toolkit 12 libraries. Three options:

1. **Use the CUDA installer** (`WhisperO-Setup-CUDA.exe`) — includes DLLs, works out of the box
2. **Install CUDA Toolkit 12.2** — download from NVIDIA, custom install with only Runtime > Libraries
3. **Copy DLLs manually** (no admin needed) — copy these 3 files from a machine with CUDA into the WhisperO install folder:
   - `cublas64_12.dll` (94 MB)
   - `cublasLt64_12.dll` (514 MB)
   - `cudart64_12.dll` (522 KB)

Source path on a machine with CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\`

## No tests

No test suite exists. No CI/CD pipeline.
