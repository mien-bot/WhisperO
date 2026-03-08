from __future__ import annotations

import platform
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd


def play_sound(name: str, sounds_enabled: bool, sounds_dir: Path) -> None:
    """Play a notification sound in a background thread."""
    if not sounds_enabled:
        return

    sound_file = sounds_dir / f"{name}.wav"
    if not sound_file.exists():
        return

    def _play() -> None:
        try:
            if platform.system() == "Windows":
                # Use winsound on Windows, no conflict with sounddevice recording.
                import winsound

                winsound.PlaySound(
                    str(sound_file), winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            else:
                import wave as _wave

                with _wave.open(str(sound_file), "rb") as wf:
                    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    sd.play(data, samplerate=wf.getframerate(), blocking=True)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()
