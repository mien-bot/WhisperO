from __future__ import annotations

import os
import platform
import signal
import sys
import threading
from pathlib import Path

import requests
from pynput import keyboard

from .audio import RecorderState, get_input_devices, start_recording, stop_recording
from .clipboard import paste_text
from .config import SOUND_OPTIONS, load_config, save_config_value
from .dictionary import load_dictionary, open_dictionary
from .sounds import play_sound
from .transcribe import transcribe

config = load_config()
state = RecorderState()
_transcription_lock = threading.Lock()  # prevent overlapping transcriptions


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
    # Function keys
    "f1": keyboard.Key.f1, "f2": keyboard.Key.f2, "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4, "f5": keyboard.Key.f5, "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7, "f8": keyboard.Key.f8, "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10, "f11": keyboard.Key.f11, "f12": keyboard.Key.f12,
    # Navigation keys
    "insert": keyboard.Key.insert, "delete": keyboard.Key.delete,
    "home": keyboard.Key.home, "end": keyboard.Key.end,
    "page_up": keyboard.Key.page_up, "page_down": keyboard.Key.page_down,
    # Other special keys
    "menu": keyboard.Key.menu, "scroll_lock": keyboard.Key.scroll_lock,
    "pause": keyboard.Key.pause, "caps_lock": keyboard.Key.caps_lock,
    "num_lock": keyboard.Key.num_lock, "print_screen": keyboard.Key.print_screen,
    "esc": keyboard.Key.esc, "space": keyboard.Key.space,
    "tab": keyboard.Key.tab, "enter": keyboard.Key.enter,
    "backspace": keyboard.Key.backspace,
}

# Reverse lookup: pynput key object -> normalized config name
# Built from KEY_MAP so there's a single source of truth.
_NORMALIZE_ALIASES = {"cmd": "win", "cmd_r": "win", "ctrl_r": "ctrl", "shift_r": "shift", "alt_r": "alt"}
_SPECIAL_KEY_NAMES: dict = {}
for _name, _key_obj in KEY_MAP.items():
    _resolved = _NORMALIZE_ALIASES.get(_name, _name)
    # First mapping wins (e.g. "win" before "cmd" for keyboard.Key.cmd)
    if _key_obj not in _SPECIAL_KEY_NAMES:
        _SPECIAL_KEY_NAMES[_key_obj] = _resolved


# Windows OEM virtual key codes → config names
_VK_OEM = {
    0xC0: "`", 0xBD: "-", 0xBB: "=",
    0xDB: "[", 0xDD: "]", 0xDC: "\\",
    0xBA: ";", 0xDE: "'", 0xBC: ",",
    0xBE: ".", 0xBF: "/",
}


def _key_to_name(key) -> str | None:
    """Convert a pynput key event to a config-friendly name string."""
    if key in _SPECIAL_KEY_NAMES:
        return _SPECIAL_KEY_NAMES[key]
    # Check vk (virtual key code) first — it's reliable even when modifiers
    # are held (e.g. Ctrl+A gives char='\x01' but vk=0x41).
    if hasattr(key, "vk") and key.vk is not None:
        vk = key.vk
        if 0x30 <= vk <= 0x39:  # 0-9
            return chr(vk).lower()
        if 0x41 <= vk <= 0x5A:  # A-Z
            return chr(vk).lower()
        if vk in _VK_OEM:
            return _VK_OEM[vk]
    # Fallback to char for keys without a standard vk (rare)
    if hasattr(key, "char") and key.char is not None and key.char.isprintable():
        return key.char.lower()
    return None


def _name_to_key(name: str):
    """Convert a config name to a pynput key for comparison."""
    if name in KEY_MAP:
        return KEY_MAP[name]
    # Single character -> KeyCode
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return None


class _HotkeyListener:
    """Manages the global keyboard listener for hotkey detection."""

    def __init__(self):
        self.listener = None
        self.trigger_names: set = set()  # config names like {"ctrl", "f1", "a"}
        self._held_names: set = set()
        self._recording_active = False

    def start(self):
        self.trigger_names = get_trigger_key_names()
        self._held_names = set()
        self._recording_active = False

        def on_press(key):
            try:
                name = _key_to_name(key)
                if name:
                    self._held_names.add(name)
                if self.trigger_names.issubset(self._held_names) and not self._recording_active:
                    self._recording_active = True
                    on_hotkey_press()
            except Exception as e:
                print(f"  ❌ Hotkey press error: {e}")

        def on_release(key):
            try:
                name = _key_to_name(key)
                if name and name in self.trigger_names and self._recording_active:
                    self._recording_active = False
                    on_hotkey_release()
                if name:
                    self._held_names.discard(name)
            except Exception as e:
                print(f"  ❌ Hotkey release error: {e}")

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

    def restart(self):
        self.stop()
        self.start()


_hotkey_listener = _HotkeyListener()


def _signal_exit(*_):
    print("\n  Stopping WhisperO...")
    _hotkey_listener.stop()
    try:
        from .transcribe import unload_model
        unload_model()
    except Exception:
        pass
    os._exit(0)


signal.signal(signal.SIGINT, _signal_exit)

_KEY_ORDER = {"win": 0, "cmd": 0, "cmd_r": 1, "ctrl": 2, "ctrl_r": 3,
              "shift": 4, "shift_r": 5, "alt": 6, "alt_r": 7}
# Non-modifiers sort after modifiers (order 10+)


_DISPLAY_NAMES = {
    "win": "Win", "cmd": "Win", "ctrl": "Ctrl", "ctrl_r": "Ctrl",
    "shift": "Shift", "shift_r": "Shift", "alt": "Alt", "alt_r": "Alt",
    "page_up": "PgUp", "page_down": "PgDn", "print_screen": "PrtSc",
    "scroll_lock": "ScrLk", "caps_lock": "CapsLk", "num_lock": "NumLk",
    "insert": "Ins", "delete": "Del", "backspace": "Bksp",
    "esc": "Esc", "space": "Space", "tab": "Tab", "enter": "Enter",
    "menu": "Menu", "pause": "Pause", "home": "Home", "end": "End",
    "`": "`", "-": "-", "=": "=", "[": "[", "]": "]", "\\": "\\",
    ";": ";", "'": "'", ",": ",", ".": ".", "/": "/",
}
_MAC_DISPLAY = {"cmd": "\u2318", "cmd_r": "\u2318", "ctrl": "\u2303", "ctrl_r": "\u2303",
                "shift": "\u21e7", "shift_r": "\u21e7", "alt": "\u2325", "alt_r": "\u2325", "win": "\u2318"}


def _hotkey_display() -> str:
    """Return a human-readable string for the current hotkey."""
    is_mac = platform.system() == "Darwin"
    hotkey = config.get("hotkey", {})
    key_names = hotkey.get("mac" if is_mac else "windows", ["cmd", "ctrl"])
    if is_mac:
        return "".join(_MAC_DISPLAY.get(k, k.upper() if len(k) == 1 else k.title()) for k in key_names)
    return "+".join(_DISPLAY_NAMES.get(k, k.upper() if len(k) == 1 else k.title().replace("_", " ")) for k in key_names)


def _open_hotkey_dialog(tray_icon=None):
    """Open a tkinter dialog to capture a new hotkey combination."""

    def _run():
        import tkinter as tk

        # Pause the global hotkey listener so keypresses go to the dialog only
        if _hotkey_listener.listener:
            _hotkey_listener.listener.stop()

        is_mac = platform.system() == "Darwin"

        # -- Colors --
        BG = "#1a1a2e"
        BG_CARD = "#16213e"
        ACCENT = "#e94560"
        ACCENT_HOVER = "#ff6b81"
        TEXT = "#eaeaea"
        TEXT_DIM = "#8892a0"
        TEXT_KEY = "#00d2ff"
        BTN_SAVE_BG = "#e94560"
        BTN_SAVE_FG = "#ffffff"
        BTN_CANCEL_BG = "#2a2a4a"
        BTN_CANCEL_FG = "#aaaaaa"
        BORDER = "#2a2a4a"

        W, H = 440, 280
        root = tk.Tk()
        root.title("WhisperO")
        root.geometry(f"{W}x{H}")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.configure(bg=BG)
        root.overrideredirect(True)  # borderless window
        root.update_idletasks()
        x = (root.winfo_screenwidth() - W) // 2
        y = (root.winfo_screenheight() - H) // 2
        root.geometry(f"+{x}+{y}")

        # Allow dragging the borderless window
        _drag = {"x": 0, "y": 0}

        def _start_drag(e):
            _drag["x"], _drag["y"] = e.x, e.y

        def _do_drag(e):
            root.geometry(f"+{root.winfo_x() + e.x - _drag['x']}+{root.winfo_y() + e.y - _drag['y']}")

        root.bind("<Button-1>", _start_drag)
        root.bind("<B1-Motion>", _do_drag)

        # Outer border effect
        outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
        outer.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer, bg=BG)
        inner.pack(fill=tk.BOTH, expand=True)

        # Title bar
        title_bar = tk.Frame(inner, bg=BG, height=36)
        title_bar.pack(fill=tk.X, padx=16, pady=(12, 0))
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="Change Hotkey", font=("Segoe UI Semibold", 13),
                 bg=BG, fg=TEXT).pack(side=tk.LEFT)
        close_btn = tk.Label(title_bar, text="\u2715", font=("Segoe UI", 12),
                             bg=BG, fg=TEXT_DIM, cursor="hand2")
        close_btn.pack(side=tk.RIGHT)

        # Divider
        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=(4, 0))

        # Current hotkey
        current = config.get("hotkey", {}).get("mac" if is_mac else "windows", ["cmd", "ctrl"])
        current_display = " + ".join(
            _DISPLAY_NAMES.get(n, n.upper() if len(n) == 1 else n.title()) for n in current
        )
        tk.Label(inner, text=f"Current:  {current_display}", font=("Segoe UI", 10),
                 bg=BG, fg=TEXT_DIM).pack(pady=(14, 4))

        # Key display card
        card = tk.Frame(inner, bg=BG_CARD, highlightbackground=BORDER,
                        highlightthickness=1, padx=20, pady=14)
        card.pack(padx=24, pady=(4, 6))

        key_label = tk.Label(card, text="Press any key...", font=("Segoe UI", 20, "bold"),
                             bg=BG_CARD, fg=TEXT_KEY)
        key_label.pack()

        hint_label = tk.Label(inner, text="Hold one or more keys, then click Save",
                              font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM)
        hint_label.pack(pady=(2, 8))

        # Buttons
        btn_frame = tk.Frame(inner, bg=BG)
        btn_frame.pack(pady=(4, 16))

        held_names: set = set()
        best: list = []

        def _display_name(n: str) -> str:
            return _DISPLAY_NAMES.get(n, n.upper() if len(n) == 1 else n.title().replace("_", " "))

        def on_press(key):
            nonlocal best
            name = _key_to_name(key)
            if name is None:
                return
            held_names.add(name)
            names = sorted(list(held_names),
                           key=lambda n: _KEY_ORDER.get(n, 99))
            if len(names) >= 1:
                best = names
                label = "  +  ".join(_display_name(n) for n in names)
                try:
                    root.after(0, key_label.configure, {"text": label})
                except Exception:
                    pass

        def on_release(key):
            name = _key_to_name(key)
            if name:
                held_names.discard(name)

        tmp = keyboard.Listener(on_press=on_press, on_release=on_release)
        tmp.daemon = True
        tmp.start()

        def _cleanup_and_close():
            tmp.stop()
            root.destroy()
            _hotkey_listener.start()

        def save():
            if best:
                plat = "mac" if is_mac else "windows"
                config["hotkey"][plat] = best
                save_config_value("hotkey", config["hotkey"])
                print(f"  🎹 Hotkey changed to: {' + '.join(n.title() for n in best)}")
                if tray_icon:
                    tray_icon.update_menu()
            _cleanup_and_close()

        def cancel():
            _cleanup_and_close()

        def _make_btn(parent, text, command, bg, fg, hover_bg):
            btn = tk.Label(parent, text=text, font=("Segoe UI Semibold", 10),
                           bg=bg, fg=fg, padx=24, pady=6, cursor="hand2")
            btn.bind("<Button-1>", lambda e: command())
            btn.bind("<Enter>", lambda e: btn.configure(bg=hover_bg))
            btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
            return btn

        save_btn = _make_btn(btn_frame, "Save", save, BTN_SAVE_BG, BTN_SAVE_FG, ACCENT_HOVER)
        save_btn.pack(side=tk.LEFT, padx=8)
        cancel_btn = _make_btn(btn_frame, "Cancel", cancel, BTN_CANCEL_BG, BTN_CANCEL_FG, "#3a3a5a")
        cancel_btn.pack(side=tk.LEFT, padx=8)

        close_btn.bind("<Button-1>", lambda e: cancel())
        close_btn.bind("<Enter>", lambda e: close_btn.configure(fg=ACCENT))
        close_btn.bind("<Leave>", lambda e: close_btn.configure(fg=TEXT_DIM))
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.mainloop()

    threading.Thread(target=_run, daemon=True).start()


_SOUND_LABELS = {
    "start": "Start", "stop": "Stop", "beep_high": "Beep High", "beep_low": "Beep Low",
    "pop": "Pop", "chime": "Chime", "click": "Click", "ding": "Ding",
    "duck": "Duck", "fart": "Fart", "seven_eleven": "7-Eleven", "boing": "Boing",
    "laser": "Laser", "coin": "Coin", "horn": "Horn", "whoosh": "Whoosh",
    "bruh": "Bruh", "sparkle": "Sparkle", "vine_boom": "Vine Boom",
    "cha_ching": "Cha-Ching", "boom": "Boom", "scratch": "Scratch",
    "sad_trombone": "Sad Trombone", "suspense": "Suspense", "wrong_buzzer": "Wrong Buzzer",
    "correct": "Correct", "bonk": "Bonk", "emotional_damage": "Emotional Damage",
    "rizz": "Rizz", "slide_whistle": "Slide Whistle", "bell": "Bell",
    "metal_pipe": "Metal Pipe", "yeet": "Yeet", "siren": "Siren",
    "windows_error": "Windows Error", "none": "None (Silent)",
}


def _open_sound_dialog():
    """Open a dark-themed dialog to select start and stop sounds."""

    def _run():
        import tkinter as tk

        BG = "#1a1a2e"
        BG_CARD = "#16213e"
        ACCENT = "#e94560"
        ACCENT_HOVER = "#ff6b81"
        TEXT = "#eaeaea"
        TEXT_DIM = "#8892a0"
        TEXT_KEY = "#00d2ff"
        BTN_SAVE_BG = "#e94560"
        BTN_SAVE_FG = "#ffffff"
        BTN_CANCEL_BG = "#2a2a4a"
        BTN_CANCEL_FG = "#aaaaaa"
        BORDER = "#2a2a4a"
        HOVER_BG = "#1e2a4a"

        W, H = 540, 520
        root = tk.Tk()
        root.title("WhisperO")
        root.geometry(f"{W}x{H}")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.configure(bg=BG)
        root.overrideredirect(True)
        root.update_idletasks()
        x = (root.winfo_screenwidth() - W) // 2
        y = (root.winfo_screenheight() - H) // 2
        root.geometry(f"+{x}+{y}")

        # Drag via title bar only (so scrolling inside lists works)
        _drag = {"x": 0, "y": 0}

        def _start_drag(e):
            _drag["x"], _drag["y"] = e.x, e.y

        def _do_drag(e):
            root.geometry(f"+{root.winfo_x() + e.x - _drag['x']}+"
                          f"{root.winfo_y() + e.y - _drag['y']}")

        # Border
        outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
        outer.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer, bg=BG)
        inner.pack(fill=tk.BOTH, expand=True)

        # Title bar
        title_bar = tk.Frame(inner, bg=BG, height=36)
        title_bar.pack(fill=tk.X, padx=16, pady=(12, 0))
        title_bar.pack_propagate(False)
        title_lbl = tk.Label(title_bar, text="Change Sounds",
                             font=("Segoe UI Semibold", 13), bg=BG, fg=TEXT)
        title_lbl.pack(side=tk.LEFT)
        close_btn = tk.Label(title_bar, text="\u2715", font=("Segoe UI", 12),
                             bg=BG, fg=TEXT_DIM, cursor="hand2")
        close_btn.pack(side=tk.RIGHT)

        # Bind drag on title bar and its label (not close button)
        for w in (title_bar, title_lbl):
            w.bind("<Button-1>", _start_drag)
            w.bind("<B1-Motion>", _do_drag)

        # Divider
        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=(4, 0))

        # Hint
        tk.Label(inner, text="Click a sound to preview  \u2022  Select one for each event",
                 font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM).pack(pady=(10, 6))

        # Two-column container
        columns = tk.Frame(inner, bg=BG)
        columns.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        def _make_column(parent, title, current_selection):
            col = tk.Frame(parent, bg=BG)

            tk.Label(col, text=title, font=("Segoe UI Semibold", 11),
                     bg=BG, fg=TEXT_KEY).pack(pady=(0, 6))

            card = tk.Frame(col, bg=BG_CARD, highlightbackground=BORDER,
                            highlightthickness=1)
            card.pack(fill=tk.BOTH, expand=True)

            canvas = tk.Canvas(card, bg=BG_CARD, highlightthickness=0, bd=0)
            scrollbar = tk.Scrollbar(card, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas, bg=BG_CARD)

            scroll_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            def _on_canvas_resize(e):
                canvas.itemconfigure(win_id, width=e.width)
            canvas.bind("<Configure>", _on_canvas_resize)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            selection = {"current": current_selection}
            item_widgets = {}

            def _scroll(e):
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

            canvas.bind("<MouseWheel>", _scroll)
            scroll_frame.bind("<MouseWheel>", _scroll)

            for snd in SOUND_OPTIONS:
                label_text = _SOUND_LABELS.get(snd, snd.replace("_", " ").title())
                is_sel = snd == current_selection
                bg = ACCENT if is_sel else BG_CARD
                fg = BTN_SAVE_FG if is_sel else TEXT

                row = tk.Frame(scroll_frame, bg=bg, cursor="hand2")
                row.pack(fill=tk.X, padx=3, pady=1)
                lbl = tk.Label(row, text=f"  {label_text}", font=("Segoe UI", 10),
                               bg=bg, fg=fg, anchor="w", pady=4)
                lbl.pack(fill=tk.X)
                item_widgets[snd] = (row, lbl)

                def _on_click(e, s=snd):
                    old = selection["current"]
                    if old in item_widgets:
                        r, l = item_widgets[old]
                        r.configure(bg=BG_CARD)
                        l.configure(bg=BG_CARD, fg=TEXT)
                    selection["current"] = s
                    r, l = item_widgets[s]
                    r.configure(bg=ACCENT)
                    l.configure(bg=ACCENT, fg=BTN_SAVE_FG)
                    if s != "none":
                        play_sound(name=s, sounds_enabled=True, sounds_dir=_sounds_dir())

                def _on_enter(e, s=snd):
                    if selection["current"] != s:
                        r, l = item_widgets[s]
                        r.configure(bg=HOVER_BG)
                        l.configure(bg=HOVER_BG)

                def _on_leave(e, s=snd):
                    if selection["current"] != s:
                        r, l = item_widgets[s]
                        r.configure(bg=BG_CARD)
                        l.configure(bg=BG_CARD)

                for w in (row, lbl):
                    w.bind("<Button-1>", _on_click)
                    w.bind("<Enter>", _on_enter)
                    w.bind("<Leave>", _on_leave)
                    w.bind("<MouseWheel>", _scroll)

            return col, selection

        left_col, start_sel = _make_column(
            columns, "Start Sound", config.get("start_sound", "start"))
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        right_col, stop_sel = _make_column(
            columns, "Stop Sound", config.get("stop_sound", "stop"))
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # Buttons
        btn_frame = tk.Frame(inner, bg=BG)
        btn_frame.pack(pady=(4, 16))

        def save():
            try:
                config["start_sound"] = start_sel["current"]
                config["stop_sound"] = stop_sel["current"]
                save_config_value("start_sound", start_sel["current"])
                save_config_value("stop_sound", stop_sel["current"])
                print(f"  Sounds updated: start={start_sel['current']}, stop={stop_sel['current']}")
            except Exception as e:
                print(f"  Error saving sounds: {e}")
            root.destroy()

        def cancel():
            root.destroy()

        def _make_btn(parent, text, command, bg, fg, hover_bg):
            btn = tk.Label(parent, text=text, font=("Segoe UI Semibold", 10),
                           bg=bg, fg=fg, padx=24, pady=6, cursor="hand2")
            btn.bind("<Button-1>", lambda e: command())
            btn.bind("<Enter>", lambda e: btn.configure(bg=hover_bg))
            btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
            return btn

        save_btn = _make_btn(btn_frame, "Save", save, BTN_SAVE_BG, BTN_SAVE_FG, ACCENT_HOVER)
        save_btn.pack(side=tk.LEFT, padx=8)
        cancel_btn = _make_btn(btn_frame, "Cancel", cancel, BTN_CANCEL_BG, BTN_CANCEL_FG, "#3a3a5a")
        cancel_btn.pack(side=tk.LEFT, padx=8)

        close_btn.bind("<Button-1>", lambda e: cancel())
        close_btn.bind("<Enter>", lambda e: close_btn.configure(fg=ACCENT))
        close_btn.bind("<Leave>", lambda e: close_btn.configure(fg=TEXT_DIM))
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
    # Map "start"/"stop" events to the user-configured sound file
    if name == "start":
        sound_file = config.get("start_sound", "start")
    elif name == "stop":
        sound_file = config.get("stop_sound", "stop")
    else:
        sound_file = name
    if sound_file == "none":
        return
    play_sound(name=sound_file, sounds_enabled=bool(config.get("sounds", True)), sounds_dir=_sounds_dir())


def get_trigger_key_names() -> set:
    """Get trigger key names from config based on platform."""
    is_mac = platform.system() == "Darwin"
    hotkey = config.get("hotkey", {})
    key_names = hotkey.get("mac" if is_mac else "windows", ["cmd", "ctrl"])
    return set(_NORMALIZE_ALIASES.get(n.lower(), n.lower()) for n in key_names)


def on_hotkey_press() -> None:
    try:
        mic = config.get("mic_device")  # None = system default
        start_recording(state, _play_sound, device_index=mic)
    except Exception as e:
        print(f"  Recording start error: {e}")


def on_hotkey_release() -> None:
    try:
        audio_buf = stop_recording(state, _play_sound)
    except Exception as e:
        print(f"  ❌ Recording stop error: {e}")
        return
    if audio_buf is None:
        return

    def do_transcribe() -> None:
        if not _transcription_lock.acquire(blocking=False):
            print("  ⏳ Transcription already in progress, skipping")
            return
        try:
            prompt = load_dictionary(seed_path=_dictionary_seed_path())
            text = transcribe(audio_buf=audio_buf, config=config, prompt=prompt)
            if text:
                print(f"  📝 \"{text}\"")
                paste_text(text)
                print("  ✅ Pasted!")
            else:
                print("  ⚠️  No transcription returned")
        except Exception as e:
            print(f"  ❌ Transcription error: {e}")
        finally:
            _transcription_lock.release()

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
        _hotkey_listener.stop()
        try:
            from .transcribe import unload_model
            unload_model()
        except Exception:
            pass
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
            print(f"  Switching to {model_name}...")

            def _switch():
                if config.get("backend", "local") == "local":
                    try:
                        from .transcribe import get_model, is_model_cached
                        if not is_model_cached(model_name):
                            print(f"  Downloading {model_name}...")
                        get_model(model_name)
                        print(f"  {model_name} ready")
                    except Exception as e:
                        print(f"  Failed to load {model_name}: {e}")
                    icon.update_menu()

            threading.Thread(target=_switch, daemon=True).start()
        return callback

    def is_current_model(model_name):
        return lambda item: config.get("model", "large-v3") == model_name

    def on_change_hotkey(icon, item):
        _open_hotkey_dialog(icon)

    def on_change_sounds(icon, item):
        _open_sound_dialog()

    DEVICES = ["GPU", "CPU"]

    def make_device_callback(device_name):
        def callback(icon, item):
            target = device_name.lower()
            if config.get("device") == target:
                return
            config["device"] = target
            save_config_value("device", target)
            print(f"  Switching to {device_name}...")

            def _switch():
                if config.get("backend", "local") == "local":
                    try:
                        from .transcribe import reload_model
                        reload_model(config.get("model", "large-v3"), target)
                        print(f"  Model reloaded on {device_name}")
                    except Exception as e:
                        print(f"  Failed to switch to {device_name}: {e}")
                icon.update_menu()

            threading.Thread(target=_switch, daemon=True).start()
        return callback

    def is_current_device(device_name):
        return lambda item: config.get("device", "gpu") == device_name.lower()

    # --- Microphone selection ---
    _input_devices = get_input_devices()

    def make_mic_callback(dev_idx, dev_name):
        def callback(icon, item):
            config["mic_device"] = dev_idx
            save_config_value("mic_device", dev_idx)
            print(f"  Microphone: {dev_name} (device {dev_idx})")
            icon.update_menu()
        return callback

    def on_mic_default(icon, item):
        config.pop("mic_device", None)
        save_config_value("mic_device", None)
        print("  Microphone: System Default")
        icon.update_menu()

    def is_mic_default():
        return lambda item: config.get("mic_device") is None

    def is_current_mic(dev_idx):
        return lambda item: config.get("mic_device") == dev_idx

    mic_menu = pystray.Menu(
        pystray.MenuItem(
            "System Default", on_mic_default,
            checked=is_mic_default(), radio=True
        ),
        *[pystray.MenuItem(
            name, make_mic_callback(idx, name),
            checked=is_current_mic(idx), radio=True
        ) for idx, name in _input_devices]
    )

    # --- Model status ---
    def _model_status_label(item):
        try:
            from .transcribe import _model, _model_lock, _model_size, is_model_cached
            model_name = config.get("model", "large-v3")
            with _model_lock:
                loaded = _model is not None
                loaded_size = _model_size
            if loaded:
                dev = (config.get("device") or "gpu").upper()
                return f"Model: {loaded_size or model_name} ({dev}) - Loaded"
            elif is_model_cached(model_name):
                return f"Model: {model_name} - Cached (not loaded)"
            else:
                return f"Model: {model_name} - Not downloaded"
        except Exception:
            return "Model: unknown"

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
            from .transcribe import unload_model
            unload_model()
            icon.update_menu()
        return callback

    def is_current_server(url):
        return lambda item: config.get("backend", "local") == "server" and config.get("server") == url

    def make_backend_callback(backend_name):
        def callback(icon, item):
            config["backend"] = backend_name
            save_config_value("backend", backend_name)
            print(f"  🔄 Backend: {backend_name}")
            if backend_name == "server":
                from .transcribe import unload_model
                unload_model()
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
        pystray.MenuItem("Device (GPU/CPU)", pystray.Menu(
            *[pystray.MenuItem(
                d, make_device_callback(d), checked=is_current_device(d), radio=True
            ) for d in DEVICES]
        ), enabled=lambda item: config.get("backend", "local") == "local"),
        pystray.MenuItem("Select Microphone", mic_menu),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_model_status_label, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Change Sounds...", on_change_sounds),
        pystray.MenuItem("Edit Dictionary", on_edit_dict),
        pystray.MenuItem("Change Hotkey...", on_change_hotkey),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon("WhisperO", make_icon(), "WhisperO", menu)
    return icon


def _download_model_and_exit() -> None:
    """Download the default model and exit. Used by the installer post-install step."""
    from .transcribe import download_model, is_model_cached
    model_name = config.get("model", "large-v3")
    if is_model_cached(model_name):
        print(f"Model {model_name} already downloaded.")
        sys.exit(0)
    print(f"Downloading {model_name} model (~3 GB)...")
    try:
        download_model(model_name)
        print(f"Model {model_name} downloaded successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("The model will be downloaded when you first launch WhisperO.")
        sys.exit(1)


def main() -> None:
    if "--download-model" in sys.argv:
        _download_model_and_exit()

    backend = config.get("backend", "local")

    # Restore persisted last working server
    saved_server = config.get("last_working_server")
    if saved_server:
        from . import transcribe as _transcribe_mod
        _transcribe_mod._last_working_server = saved_server

    if backend == "local":
        model_name = config.get("model", "large-v3")
        print(f"WhisperO (local, model: {model_name})")

        def _load_model_bg():
            try:
                from .transcribe import get_model, is_model_cached
                if not is_model_cached(model_name):
                    print("  Downloading model (this may take a few minutes)...")
                else:
                    print("  Loading model...")
                device_pref = config.get("device")  # None = auto (GPU first)
                get_model(model_name, device_pref=device_pref)
                print("  Model ready (hotkey active)")
            except (ImportError, RuntimeError) as e:
                print(f"  faster-whisper not available ({e}), falling back to server mode")
                config["backend"] = "server"

        threading.Thread(target=_load_model_bg, daemon=True).start()
    if backend == "server":
        print(f"WhisperO (server: {config['server']})")
        try:
            response = requests.get(f"{config['server']}/health", timeout=5)
            if response.json().get("status") == "ok":
                print("  Server is healthy")
            else:
                print("  Unexpected server response")
        except Exception:
            print("  Cannot reach server, will retry on each recording")

    print(f"Hotkey: hold [{_hotkey_display()}] to record")
    print("Press Ctrl+C to quit\n")

    _hotkey_listener.start()

    tray = create_tray_icon()
    if tray:
        def _run_tray():
            try:
                tray.run()
            except Exception as e:
                print(f"  Tray icon crashed: {e}")

        if platform.system() == "Windows":
            # Windows: run tray in background thread so Ctrl+C works
            tray_thread = threading.Thread(target=_run_tray, daemon=True)
            tray_thread.start()
            try:
                # Keep alive even if tray crashes — hotkey still works
                while tray_thread.is_alive():
                    tray_thread.join(timeout=1)
                # Tray exited — keep running on hotkey listener alone
                print("  Tray exited, hotkey still active")
                if _hotkey_listener.listener:
                    _hotkey_listener.listener.join()
            except KeyboardInterrupt:
                _hotkey_listener.stop()
                tray.stop()
                print("\nBye!")
        else:
            # macOS/Linux: tray must run on main thread (AppKit requirement)
            tray.run()
    else:
        try:
            if _hotkey_listener.listener:
                _hotkey_listener.listener.join()
        except KeyboardInterrupt:
            _hotkey_listener.stop()
            print("\nBye!")


if __name__ == "__main__":
    main()
