from __future__ import annotations

import os
import platform
import signal
import sys
import threading
from pathlib import Path

import requests
from pynput import keyboard

from .audio import RecorderState, start_recording, stop_recording
from .clipboard import paste_text
from .config import load_config, save_config_value
from .dictionary import load_dictionary, open_dictionary
from .sounds import play_sound
from .transcribe import transcribe

signal.signal(signal.SIGINT, lambda *_: (print("\n  😮 Stopping WhisperO..."), os._exit(0)))

config = load_config()
state = RecorderState()


KEY_MAP = {
    "win": keyboard.Key.cmd,
    "cmd": keyboard.Key.cmd,
    "cmd_r": keyboard.Key.cmd_r,
    "ctrl": keyboard.Key.ctrl_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "shift": keyboard.Key.shift,
    "shift_r": keyboard.Key.shift_r,
    "alt": keyboard.Key.alt,
    "alt_r": keyboard.Key.alt_r,
}


class _HotkeyListener:
    """Manages the global keyboard listener for hotkey detection."""

    def __init__(self):
        self.listener = None
        self.trigger_keys: set = set()
        self._keys_held: set = set()
        self._recording_active = False

    def start(self):
        self.trigger_keys = get_trigger_keys()
        self._keys_held = set()
        self._recording_active = False

        def on_press(key):
            self._keys_held.add(key)
            if self.trigger_keys.issubset(self._keys_held) and not self._recording_active:
                self._recording_active = True
                on_hotkey_press()

        def on_release(key):
            if key in self.trigger_keys and self._recording_active:
                self._recording_active = False
                on_hotkey_release()
            self._keys_held.discard(key)

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()

    def restart(self):
        if self.listener:
            self.listener.stop()
        self.start()


_hotkey_listener = _HotkeyListener()

_KEY_ORDER = {"win": 0, "cmd": 0, "cmd_r": 1, "ctrl": 2, "ctrl_r": 3,
              "shift": 4, "shift_r": 5, "alt": 6, "alt_r": 7}


def _hotkey_display() -> str:
    """Return a human-readable string for the current hotkey."""
    is_mac = platform.system() == "Darwin"
    key_names = config["hotkey"].get("mac" if is_mac else "windows", ["cmd", "ctrl"])
    if is_mac:
        sym = {"cmd": "⌘", "cmd_r": "⌘", "ctrl": "⌃", "ctrl_r": "⌃",
               "shift": "⇧", "shift_r": "⇧", "alt": "⌥", "alt_r": "⌥", "win": "⌘"}
        return "".join(sym.get(k, k.title()) for k in key_names)
    sym = {"win": "Win", "cmd": "Win", "ctrl": "Ctrl", "ctrl_r": "Ctrl",
           "shift": "Shift", "shift_r": "Shift", "alt": "Alt", "alt_r": "Alt"}
    return "+".join(sym.get(k, k.title()) for k in key_names)


def _open_hotkey_dialog(tray_icon=None):
    """Open a tkinter dialog to capture a new hotkey combination."""

    def _run():
        import tkinter as tk

        is_mac = platform.system() == "Darwin"
        reverse = {
            keyboard.Key.cmd: "cmd" if is_mac else "win",
            keyboard.Key.cmd_r: "cmd_r" if is_mac else "win",
            keyboard.Key.ctrl_l: "ctrl",
            keyboard.Key.ctrl_r: "ctrl_r",
            keyboard.Key.shift: "shift",
            keyboard.Key.shift_r: "shift_r",
            keyboard.Key.alt: "alt",
            keyboard.Key.alt_r: "alt_r",
        }

        root = tk.Tk()
        root.title("WhisperO \u2013 Change Hotkey")
        root.geometry("380x200")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.update_idletasks()
        x = (root.winfo_screenwidth() - 380) // 2
        y = (root.winfo_screenheight() - 200) // 2
        root.geometry(f"+{x}+{y}")

        current = config["hotkey"].get("mac" if is_mac else "windows", ["cmd", "ctrl"])
        tk.Label(root, text=f"Current: {' + '.join(n.title() for n in current)}",
                 font=("Segoe UI", 10), fg="#666666").pack(pady=(15, 2))
        tk.Label(root, text="Hold 2+ modifier keys to set new hotkey",
                 font=("Segoe UI", 10)).pack(pady=(2, 8))

        key_var = tk.StringVar(value="Waiting...")
        tk.Label(root, textvariable=key_var, font=("Segoe UI", 16, "bold"),
                 fg="#0066cc").pack(pady=8)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=12)

        held: set = set()
        best: list = []

        def on_press(key):
            nonlocal best
            if key in reverse:
                held.add(key)
                names = sorted([reverse[k] for k in held if k in reverse],
                               key=lambda n: _KEY_ORDER.get(n, 99))
                if len(names) >= 2:
                    best = names
                    try:
                        key_var.set(" + ".join(n.title() for n in names))
                    except Exception:
                        pass

        def on_release(key):
            held.discard(key)

        tmp = keyboard.Listener(on_press=on_press, on_release=on_release)
        tmp.daemon = True
        tmp.start()

        def save():
            if best:
                plat = "mac" if is_mac else "windows"
                config["hotkey"][plat] = best
                save_config_value("hotkey", config["hotkey"])
                _hotkey_listener.restart()
                print(f"  🎹 Hotkey changed to: {' + '.join(n.title() for n in best)}")
                if tray_icon:
                    tray_icon.update_menu()
            tmp.stop()
            root.destroy()

        def cancel():
            tmp.stop()
            root.destroy()

        tk.Button(btn_frame, text="Save", command=save, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=cancel, width=10).pack(side=tk.LEFT, padx=5)
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.mainloop()

    threading.Thread(target=_run, daemon=True).start()


def _bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def _sounds_dir() -> Path:
    if getattr(sys, "frozen", False):
        return _bundle_dir() / "sounds"
    # Check package assets first, then project root
    pkg_sounds = Path(__file__).resolve().parent / "assets" / "sounds"
    if pkg_sounds.exists():
        return pkg_sounds
    return _bundle_dir() / "assets" / "sounds"


def _dictionary_seed_path() -> Path:
    if getattr(sys, "frozen", False):
        return _bundle_dir() / "dictionary.txt"
    pkg_dict = Path(__file__).resolve().parent / "assets" / "dictionary.txt"
    if pkg_dict.exists():
        return pkg_dict
    return _bundle_dir() / "dictionary.txt"


def _play_sound(name: str) -> None:
    play_sound(name=name, sounds_enabled=bool(config.get("sounds", True)), sounds_dir=_sounds_dir())


def get_trigger_keys() -> set:
    """Get trigger keys from config based on platform."""
    is_mac = platform.system() == "Darwin"
    key_names = config["hotkey"].get("mac" if is_mac else "windows", ["cmd", "ctrl"])
    keys = set()
    for name in key_names:
        if name.lower() in KEY_MAP:
            keys.add(KEY_MAP[name.lower()])
        else:
            print(f"  ⚠️  Unknown key: {name}")
    return keys


def on_hotkey_press() -> None:
    start_recording(state, _play_sound)


def on_hotkey_release() -> None:
    audio_buf = stop_recording(state, _play_sound)
    if audio_buf is None:
        return

    def do_transcribe() -> None:
        prompt = load_dictionary(seed_path=_dictionary_seed_path())
        text = transcribe(audio_buf=audio_buf, config=config, prompt=prompt)
        if text:
            print(f"  📝 \"{text}\"")
            paste_text(text)
            print("  ✅ Pasted!")
        else:
            print("  ⚠️  No transcription returned")

    threading.Thread(target=do_transcribe, daemon=True).start()


def create_tray_icon():
    """Create and run the system tray icon."""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        print("  ⚠️  pystray/Pillow not installed, running without tray icon")
        return None

    def make_icon():
        # Try to load the 😮 icon from bundled or project icons
        icon_paths = [
            _bundle_dir() / "icons" / "icon_128.png",
            _bundle_dir() / "icons" / "icon.png",
            Path(__file__).resolve().parents[2] / "icons" / "icon_128.png",
            Path(__file__).resolve().parents[2] / "icons" / "icon.png",
        ]
        for p in icon_paths:
            if p.exists():
                return Image.open(p).resize((64, 64), Image.LANCZOS)
        # Fallback: 😮 face (yellow circle, two eyes, open mouth)
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, 60, 60], fill="#FFD93D", outline="#2D2D2D", width=3)
        draw.ellipse([20, 22, 28, 30], fill="#2D2D2D")  # left eye
        draw.ellipse([36, 22, 44, 30], fill="#2D2D2D")  # right eye
        draw.ellipse([26, 38, 38, 50], fill="#2D2D2D")  # open mouth
        return img

    MODELS = ["large-v3", "medium", "small", "base", "tiny"]

    def on_toggle(icon, item):
        state.enabled = not state.enabled
        status = "Enabled" if state.enabled else "Disabled"
        print(f"  🔄 Dictation {status}")

    def on_quit(icon, item):
        icon.stop()
        os._exit(0)

    def on_edit_dict(icon, item):
        open_dictionary()

    def make_model_callback(model_name):
        def callback(icon, item):
            if config.get("model") == model_name:
                return
            config["model"] = model_name
            save_config_value("model", model_name)
            print(f"  🔄 Switching to {model_name}...")
            if config.get("backend", "local") == "local":
                try:
                    from .transcribe import get_model, is_model_cached
                    if not is_model_cached(model_name):
                        print(f"  ⏳ Downloading {model_name}...")
                    get_model(model_name)
                    print(f"  ✓ {model_name} ready")
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {e}")
        return callback

    def is_current_model(model_name):
        return lambda item: config.get("model", "large-v3") == model_name

    def on_change_hotkey(icon, item):
        _open_hotkey_dialog(icon)

    model_menu = pystray.Menu(
        *[pystray.MenuItem(
            m, make_model_callback(m), checked=is_current_model(m), radio=True
        ) for m in MODELS]
    )

    # Build server list from config (deduplicated, stable order)
    _all_servers = list(dict.fromkeys(
        [config.get("server", "http://localhost:8080")] + config.get("fallback_servers", [])
    ))

    def make_server_callback(url):
        def callback(icon, item):
            config["backend"] = "server"
            config["server"] = url
            save_config_value("backend", "server")
            save_config_value("server", url)
            print(f"  🔄 Server: {url}")
            icon.update_menu()
        return callback

    def is_current_server(url):
        return lambda item: config.get("backend", "local") == "server" and config.get("server") == url

    def make_backend_callback(backend_name):
        def callback(icon, item):
            config["backend"] = backend_name
            save_config_value("backend", backend_name)
            print(f"  🔄 Backend: {backend_name}")
            icon.update_menu()
        return callback

    def is_current_backend(backend_name):
        return lambda item: config.get("backend", "local") == backend_name

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: f"Hold {_hotkey_display()} to dictate", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(lambda item: "✓ Enabled" if state.enabled else "  Disabled", on_toggle),
        pystray.MenuItem("Select Backend", pystray.Menu(
            pystray.MenuItem(
                "Local", make_backend_callback("local"),
                checked=is_current_backend("local"), radio=True
            ),
            *[pystray.MenuItem(
                f"Server ({s})", make_server_callback(s),
                checked=is_current_server(s), radio=True
            ) for s in _all_servers],
        )),
        pystray.MenuItem(
            "Select Model", model_menu,
            enabled=lambda item: config.get("backend", "local") == "local"
        ),
        pystray.MenuItem("Edit Dictionary", on_edit_dict),
        pystray.MenuItem("Change Hotkey...", on_change_hotkey),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon("WhisperO", make_icon(), "WhisperO", menu)
    return icon


def main() -> None:
    backend = config.get("backend", "local")

    if backend == "local":
        try:
            from .transcribe import get_model, is_model_cached
            model_name = config.get("model", "large-v3")
            print(f"😮 WhisperO (local, model: {model_name})")
            if not is_model_cached(model_name):
                print("  ⏳ Downloading model (this may take a few minutes)...")
            else:
                print("  ⏳ Loading model...")
            get_model(model_name)
            print("  ✓ Model ready")
        except (ImportError, RuntimeError):
            print("  ⚠️  faster-whisper not available, falling back to server mode")
            backend = "server"
            config["backend"] = "server"
    if backend == "server":
        print(f"🎤 WhisperO (server: {config['server']})")
        try:
            response = requests.get(f"{config['server']}/health", timeout=5)
            if response.json().get("status") == "ok":
                print("  ✓ Server is healthy")
            else:
                print("  ⚠️  Unexpected server response")
        except Exception:
            print("  ❌ Cannot reach server, will retry on each recording")

    print(f"🎹 Hotkey: hold [{_hotkey_display()}] to record")
    print("🔇 Press Ctrl+C to quit\n")

    _hotkey_listener.start()

    tray = create_tray_icon()
    if tray:
        if platform.system() == "Windows":
            # Windows: run tray in background thread so Ctrl+C works
            tray_thread = threading.Thread(target=tray.run, daemon=True)
            tray_thread.start()
            try:
                tray_thread.join()
            except KeyboardInterrupt:
                tray.stop()
                print("\n👋 Bye!")
        else:
            # macOS/Linux: tray must run on main thread (AppKit requirement)
            tray.run()
    else:
        try:
            _hotkey_listener.listener.join()
        except KeyboardInterrupt:
            print("\n👋 Bye!")


if __name__ == "__main__":
    main()
