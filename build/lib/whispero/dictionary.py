from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import CONFIG_DIR

DICT_PATH = CONFIG_DIR / "dictionary.txt"


def load_dictionary(seed_path: Path | None = None) -> str:
    """Load custom words from dictionary.txt, return as prompt string."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not DICT_PATH.exists():
        if seed_path and seed_path.exists():
            shutil.copy2(seed_path, DICT_PATH)
        else:
            return ""

    try:
        words = []
        for line in DICT_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
        return ", ".join(words) if words else ""
    except Exception:
        return ""


def open_dictionary() -> None:
    """Open dictionary.txt in the default text editor."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not DICT_PATH.exists():
        DICT_PATH.write_text("# Add one word or phrase per line\n", encoding="utf-8")

    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", str(DICT_PATH)])
    elif system == "Windows":
        os.startfile(str(DICT_PATH))
    else:
        subprocess.Popen(["xdg-open", str(DICT_PATH)])
