from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULTS = {
    "backend": "local",
    "server": "http://localhost:8080",
    "fallback_servers": [],
    "model": "large-v3",
    "device": "gpu",          # GPU by default
    "languages": ["en"],      # checked languages; 1 = fixed, 2+ = auto-detect with hint
    "mic_device": None,       # None = system default
    "hotkey": {"windows": ["win", "ctrl"], "mac": ["cmd", "ctrl"]},
    "sounds": True,
    "start_sound": "start",
    "stop_sound": "stop",
    "meeting_segment_duration": 10,    # seconds per segment
    "meeting_overlap": 0.5,            # overlap between segments (avoid cut words)
    "meeting_auto_open": True,         # open transcript when meeting stops
    "meeting_audio_source": "mic",     # "mic", "system", or "both"
    "meeting_diarization": False,      # speaker identification (requires speechbrain)
    "meeting_diarization_threshold": 0.55,  # speaker matching sensitivity (lower = more lenient)
    "meeting_max_speakers": 10,
    "meeting_speaker_names": {},        # e.g. {"1": "Ian", "2": "Parker"}
    "hf_mirror": "",                    # HuggingFace mirror URL (e.g. "https://hf-mirror.com")
}

SOUND_OPTIONS = [
    "start", "stop", "beep_high", "beep_low", "pop", "chime", "click", "ding",
    "duck", "fart", "seven_eleven", "boing", "laser", "coin", "horn",
    "whoosh", "bruh", "sparkle", "vine_boom",
    "cha_ching", "boom", "scratch", "sad_trombone", "suspense", "wrong_buzzer",
    "correct", "bonk", "emotional_damage", "rizz", "slide_whistle", "bell",
    "metal_pipe", "yeet", "siren", "windows_error",
    "none",
]

VALID_BACKENDS = {"local", "server"}
VALID_MODELS = {"large-v3", "medium", "small", "base", "tiny"}

# (code, display label) — ordered for the tray menu
# Check one language for fastest speed; check multiple to auto-detect between them
# "auto" catches anything not in the preset list
LANGUAGES = [
    ("en", "English"),
    ("zh", "Chinese Simplified (简体)"),
    ("zh-Hant", "Chinese Traditional (繁體)"),
    ("ja", "Japanese (日本語)"),
    ("ko", "Korean (한국어)"),
    ("de", "German (Deutsch)"),
    ("auto", "Auto-Detect (Other)"),
]
LANG_LABELS = dict(LANGUAGES)

CONFIG_DIR = Path.home() / ".whispero"
CONFIG_PATH = CONFIG_DIR / "config.json"
MEETINGS_DIR = CONFIG_DIR / "meetings"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)

    backend = str(normalized.get("backend", DEFAULTS["backend"])).lower()
    normalized["backend"] = backend if backend in VALID_BACKENDS else DEFAULTS["backend"]

    model = str(normalized.get("model", DEFAULTS["model"])).lower()
    normalized["model"] = model if model in VALID_MODELS else DEFAULTS["model"]

    normalized["server"] = str(normalized.get("server", DEFAULTS["server"]))
    return normalized


def _apply_env(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)

    env_backend = os.environ.get("WHISPERO_BACKEND")
    if env_backend:
        backend = env_backend.strip().lower()
        if backend in VALID_BACKENDS:
            updated["backend"] = backend

    env_server = os.environ.get("WHISPERO_SERVER")
    if env_server:
        updated["server"] = env_server

    env_model = os.environ.get("WHISPERO_MODEL")
    if env_model:
        model = env_model.strip().lower()
        if model in VALID_MODELS:
            updated["model"] = model

    env_mirror = os.environ.get("WHISPERO_HF_MIRROR")
    if env_mirror:
        updated["hf_mirror"] = env_mirror.strip()

    return updated


def save_config_value(key: str, value: Any) -> None:
    """Update a single key in the user config file, preserving other settings."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    file_config = _load_config_file()
    file_config[key] = value
    CONFIG_PATH.write_text(json.dumps(file_config, indent=2) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    file_config = _load_config_file()
    merged = _deep_merge(DEFAULTS, file_config)
    env_applied = _apply_env(merged)
    return _normalize(env_applied)
