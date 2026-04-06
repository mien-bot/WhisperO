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
- **Model loading**: faster-whisper with CTranslate2. GPU (CUDA) preferred, falls back to CPU. Models cached in `~/.whispero/models/`.

## Config

**Location**: `~/.whispero/config.json`

**Priority**: env vars > config file > defaults (in `config.py:DEFAULTS`)

**Key settings**: `backend` (local/server), `model` (large-v3/medium/small/base/tiny), `device` (gpu/cpu), `languages` (list of codes), `mic_device` (int index or null), `hotkey`, `meeting_diarization` (bool).

**Env vars**: `WHISPERO_BACKEND`, `WHISPERO_MODEL`, `WHISPERO_SERVER`

## Build notes

- Build output: `dist/WhisperO/WhisperO.exe` (~278 MB)
- Must use `PYTHONIOENCODING=utf-8` on Windows (build script prints Unicode)
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

## No tests

No test suite exists. No CI/CD pipeline.
