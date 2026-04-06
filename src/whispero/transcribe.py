from __future__ import annotations

import gc
import io
import sys
import threading
from pathlib import Path
from typing import Any

import requests

_model = None
_model_size: str | None = None
_model_lock = threading.RLock()  # protects _model and _model_size across threads
transcription_lock = threading.Lock()  # shared between push-to-talk and meeting mode
_last_working_server: str | None = None


def unload_model() -> None:
    """Unload the current model and flush RAM (+ VRAM if applicable)."""
    global _model, _model_size
    with _model_lock:
        if _model is None:
            return
        name = _model_size or "unknown"
        _model = None
        _model_size = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"  Model {name} unloaded, RAM freed")


def get_model_cache_dir() -> Path:
    cache_dir = Path.home() / ".whispero" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_model_cached(model_size: str = "large-v3") -> bool:
    repo_dir = get_model_cache_dir() / f"models--Systran--faster-whisper-{model_size}"
    if not repo_dir.exists():
        return False

    for extension in ("*.bin", "*.safetensors"):
        if any(repo_dir.rglob(extension)):
            return True

    snapshots_dir = repo_dir / "snapshots"
    return snapshots_dir.exists() and any(path.is_file() for path in snapshots_dir.rglob("*"))


def download_model(model_size: str = "large-v3") -> None:
    """Download the model files without loading into memory."""
    from faster_whisper.utils import download_model as _dl
    cache_dir = str(get_model_cache_dir())
    _dl(model_size, cache_dir=cache_dir)


def get_model(model_size: str = "large-v3", device_pref: str | None = None):
    global _model, _model_size

    with _model_lock:
        if _model is not None and _model_size == model_size and device_pref is None:
            return _model

        # Free the old model before loading a new one
        if _model is not None:
            _model = None
            _model_size = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            from faster_whisper import WhisperModel
        except ImportError as err:
            raise RuntimeError(
                "faster-whisper is not installed. Run `pip install faster-whisper`."
            ) from err

        # Determine device order based on preference
        if device_pref == "cpu":
            devices = ("cpu",)
        else:
            # "gpu" or unset: try CUDA first, fall back to CPU
            devices = ("cuda", "cpu")

        device = "cpu"
        for device in devices:
            try:
                _model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type="auto",
                    download_root=str(get_model_cache_dir()),
                )
                break
            except Exception:
                if device == devices[-1]:
                    raise
                print(f"  GPU init failed, falling back to CPU")
        _model_size = model_size

    device_label = "GPU" if device == "cuda" else "CPU"
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                device_label = f"GPU ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass

    print(f"  Model: {model_size} | Device: {device_label}")
    return _model


def reload_model(model_size: str = "large-v3", device_pref: str = "gpu"):
    """Force reload the model on a specific device."""
    unload_model()
    return get_model(model_size, device_pref=device_pref)


def transcribe_server(audio_buf: io.BytesIO, server: str, prompt: str = "") -> str | None:
    """Send audio to whisper.cpp server and return transcribed text."""
    print("  Sending to server...")
    try:
        audio_buf.seek(0)
        post_data = {"response_format": "text"}
        if prompt:
            post_data["prompt"] = prompt

        resp = requests.post(
            f"{server}/inference",
            files={"file": ("audio.wav", audio_buf, "audio/wav")},
            data=post_data,
            timeout=(3, 30),  # 3s connect, 30s read
        )
        resp.raise_for_status()

        # Validate response looks like text, not an HTML error page
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            print("  Server returned HTML instead of text, ignoring", file=sys.stderr)
            return None

        text = resp.text.strip()
        if not text:
            return None
        return text
    except requests.ConnectionError:
        print("  Cannot reach server.", file=sys.stderr)
        return None
    except requests.Timeout:
        print("  Server timeout", file=sys.stderr)
        return None
    except requests.HTTPError as err:
        print(f"  Server error: {err}", file=sys.stderr)
        return None


def transcribe_local(
    audio_buf: io.BytesIO,
    model_size: str = "large-v3",
    prompt: str = "",
    languages: list[str] | None = None,
) -> str | None:
    """Transcribe audio using local faster-whisper model."""
    if languages is None:
        languages = ["en"]
    try:
        model = get_model(model_size)
        audio_buf.seek(0)

        kwargs: dict = {
            "initial_prompt": prompt,
            "beam_size": 1,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }

        # Map custom codes to Whisper codes and build prompt hints
        _WHISPER_CODE = {"zh-Hant": "zh"}  # Traditional Chinese → zh
        _SCRIPT_HINTS = {"zh-Hant": "以下是繁體中文的語音轉錄。"}  # bias toward Traditional

        real_langs = [c for c in languages if c != "auto"]
        has_auto = "auto" in languages
        whisper_langs = list(dict.fromkeys(_WHISPER_CODE.get(c, c) for c in real_langs))

        # Prepend script hints (e.g. Traditional Chinese) to the prompt
        script_hint = " ".join(_SCRIPT_HINTS[c] for c in real_langs if c in _SCRIPT_HINTS)
        if script_hint:
            prompt = f"{script_hint} {prompt}".strip()
            kwargs["initial_prompt"] = prompt

        if len(whisper_langs) == 1 and not has_auto:
            # Single language — skip detection pass (fastest)
            kwargs["language"] = whisper_langs[0]
        elif whisper_langs and not has_auto:
            # Multiple specific languages — auto-detect, hint via prompt
            from .config import LANG_LABELS
            lang_names = [LANG_LABELS.get(c, c) for c in real_langs]
            hint = f"This audio may contain: {', '.join(lang_names)}."
            kwargs["initial_prompt"] = f"{hint} {prompt}".strip()
        # else: full auto-detect, no language hint

        segments, _info = model.transcribe(audio_buf, **kwargs)
        return " ".join(seg.text for seg in segments).strip()
    except Exception as err:
        print(f"  Local transcription error: {err}", file=sys.stderr)
        return None


def transcribe_meeting_segment(
    audio_buf: io.BytesIO,
    model_size: str = "large-v3",
    prompt: str = "",
    languages: list[str] | None = None,
    word_timestamps: bool = False,
) -> list[tuple[float, float, str]]:
    """Transcribe audio segment, returning list of (start_sec, end_sec, text) tuples."""
    if languages is None:
        languages = ["en"]
    try:
        model = get_model(model_size)
        audio_buf.seek(0)

        kwargs: dict = {
            "initial_prompt": prompt,
            "beam_size": 1,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "word_timestamps": word_timestamps,
        }

        _WHISPER_CODE = {"zh-Hant": "zh"}
        _SCRIPT_HINTS = {"zh-Hant": "以下是繁體中文的語音轉錄。"}

        real_langs = [c for c in languages if c != "auto"]
        has_auto = "auto" in languages
        whisper_langs = list(dict.fromkeys(_WHISPER_CODE.get(c, c) for c in real_langs))

        script_hint = " ".join(_SCRIPT_HINTS[c] for c in real_langs if c in _SCRIPT_HINTS)
        if script_hint:
            prompt = f"{script_hint} {prompt}".strip()
            kwargs["initial_prompt"] = prompt

        if len(whisper_langs) == 1 and not has_auto:
            kwargs["language"] = whisper_langs[0]
        elif whisper_langs and not has_auto:
            from .config import LANG_LABELS
            lang_names = [LANG_LABELS.get(c, c) for c in real_langs]
            hint = f"This audio may contain: {', '.join(lang_names)}."
            kwargs["initial_prompt"] = f"{hint} {prompt}".strip()

        segments, _info = model.transcribe(audio_buf, **kwargs)

        result = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                result.append((seg.start, seg.end, text))
        return result
    except Exception as err:
        print(f"  Meeting segment transcription error: {err}", file=sys.stderr)
        return []


def transcribe(
    audio_buf: io.BytesIO,
    config: dict[str, Any] | None = None,
    server: str | None = None,
    prompt: str = "",
    backend: str | None = None,
    model_size: str | None = None,
) -> str | None:
    """Transcribe using configured backend (local faster-whisper or remote server)."""
    global _last_working_server
    cfg = config or {}
    backend_name = (backend or cfg.get("backend") or ("server" if server else "local")).lower()
    languages = cfg.get("languages", ["en"])

    if backend_name == "server":
        server_url = server or cfg.get("server", "http://localhost:8080")
        all_servers = [server_url] + cfg.get("fallback_servers", [])

        # Try last working server first, then the rest
        if _last_working_server and _last_working_server in all_servers:
            servers = [_last_working_server] + [s for s in all_servers if s != _last_working_server]
        else:
            servers = all_servers

        for url in servers:
            audio_buf.seek(0)
            result = transcribe_server(audio_buf=audio_buf, server=url, prompt=prompt)
            if result is not None:
                if _last_working_server != url:
                    _last_working_server = url
                    # Persist last working server to config
                    try:
                        from .config import save_config_value
                        save_config_value("last_working_server", url)
                    except Exception:
                        pass
                return result

        # All servers unreachable — fall back to local
        _last_working_server = None
        print("  All servers unavailable, falling back to local...")
        audio_buf.seek(0)
        resolved_model = model_size or cfg.get("model", "large-v3")
        return transcribe_local(audio_buf=audio_buf, model_size=resolved_model, prompt=prompt, languages=languages)

    resolved_model = model_size or cfg.get("model", "large-v3")
    return transcribe_local(audio_buf=audio_buf, model_size=resolved_model, prompt=prompt, languages=languages)
