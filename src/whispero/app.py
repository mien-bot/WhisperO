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
from .config import (
    LANG_LABELS, LANGUAGES, MEETINGS_DIR, SOUND_OPTIONS,
    load_config, save_config_value,
)
from .diarize import is_model_downloaded
from .dictionary import load_dictionary, open_dictionary
from .download import download_diarization_model, get_model_size, remove_diarization_model
from .meeting import MeetingSession
from .sounds import play_sound
from .transcribe import transcribe, transcription_lock

config = load_config()
state = RecorderState()
_meeting_session: MeetingSession | None = None


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
        # Stop any lingering listener so we never have two Windows keyboard
        # hooks installed simultaneously (that causes the hook to misbehave
        # and, on Windows, can hard-crash the process when a key is pressed).
        self.stop()

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
                    # Run in thread — blocking the Windows keyboard hook callback
                    # for >300ms causes the OS to silently kill the hook.
                    threading.Thread(target=on_hotkey_press, daemon=True).start()
            except Exception as e:
                print(f"  ❌ Hotkey press error: {e}")

        def on_release(key):
            try:
                name = _key_to_name(key)
                if name and name in self.trigger_names and self._recording_active:
                    self._recording_active = False
                    threading.Thread(target=on_hotkey_release, daemon=True).start()
                if name:
                    self._held_names.discard(name)
            except Exception as e:
                print(f"  ❌ Hotkey release error: {e}")

        try:
            self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.daemon = True
            self.listener.start()
        except Exception as e:
            print(f"  ❌ Failed to start hotkey listener: {e}")
            self.listener = None

    def stop(self):
        listener = self.listener
        self.listener = None
        if listener:
            try:
                listener.stop()
            except Exception:
                pass
            # Wait for pynput's internal thread to finish uninstalling the
            # Windows keyboard hook before we install a new one.  Without
            # this join, start() can install the new hook while the old
            # hook is still live, which on Windows causes the process to
            # die on the next keypress.
            try:
                listener.join(timeout=2)
            except Exception:
                pass

    def restart(self):
        self.stop()
        self.start()


_hotkey_listener = _HotkeyListener()


def _signal_exit(*_):
    print("\n  Stopping WhisperO...")
    global _meeting_session
    if _meeting_session and _meeting_session.running:
        _meeting_session.stop()
        _meeting_session = None
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

        # Pause the global hotkey listener so keypresses go to the dialog
        # only.  Use stop() (not listener.stop()) so the reference is
        # cleared and the pynput thread is properly joined — otherwise a
        # stale hook can conflict with the new hook when we restart.
        _hotkey_listener.stop()

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
            try:
                tmp.stop()
                try:
                    tmp.join(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass
            # Restart the main listener from a fresh background thread —
            # not this tkinter dialog thread, which is about to exit.
            # Installing a Windows keyboard hook from a thread that's
            # immediately going away has caused crashes in the past.
            threading.Thread(
                target=_hotkey_listener.start, daemon=True, name="hotkey-restart",
            ).start()

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


def _open_rename_speakers_dialog():
    """Open a dialog to rename Speaker 1, Speaker 2, etc."""

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

        # Determine which speakers to show (from config or up to 5 slots)
        current_names = dict(config.get("meeting_speaker_names", {}))
        max_rows = max(5, len(current_names))

        W, H = 400, 60 + max_rows * 40 + 80
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

        _drag = {"x": 0, "y": 0}

        def _start_drag(e):
            _drag["x"], _drag["y"] = e.x, e.y

        def _do_drag(e):
            root.geometry(f"+{root.winfo_x() + e.x - _drag['x']}+"
                          f"{root.winfo_y() + e.y - _drag['y']}")

        outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
        outer.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer, bg=BG)
        inner.pack(fill=tk.BOTH, expand=True)

        title_bar = tk.Frame(inner, bg=BG, height=36)
        title_bar.pack(fill=tk.X, padx=16, pady=(12, 0))
        title_bar.pack_propagate(False)
        title_lbl = tk.Label(title_bar, text="Rename Speakers",
                             font=("Segoe UI Semibold", 13), bg=BG, fg=TEXT)
        title_lbl.pack(side=tk.LEFT)
        close_btn = tk.Label(title_bar, text="\u2715", font=("Segoe UI", 12),
                             bg=BG, fg=TEXT_DIM, cursor="hand2")
        close_btn.pack(side=tk.RIGHT)
        for w in (title_bar, title_lbl):
            w.bind("<Button-1>", _start_drag)
            w.bind("<B1-Motion>", _do_drag)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=(4, 0))

        tk.Label(inner, text="Type a name for each speaker (leave blank to skip)",
                 font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM).pack(pady=(10, 6))

        # Speaker name entries
        entries: dict[str, tk.Entry] = {}
        for i in range(1, max_rows + 1):
            key = str(i)
            row = tk.Frame(inner, bg=BG)
            row.pack(fill=tk.X, padx=24, pady=2)
            tk.Label(row, text=f"Speaker {i}:", font=("Segoe UI", 10),
                     bg=BG, fg=TEXT_KEY, width=10, anchor="e").pack(side=tk.LEFT)
            ent = tk.Entry(row, font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT,
                           insertbackground=TEXT, relief=tk.FLAT,
                           highlightbackground=BORDER, highlightthickness=1)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
            if key in current_names:
                ent.insert(0, current_names[key])
            entries[key] = ent

        # Also rename in last meeting's transcript
        apply_var = tk.BooleanVar(value=True)
        tk.Checkbutton(inner, text="Also rename in last meeting transcript",
                       variable=apply_var, font=("Segoe UI", 9),
                       bg=BG, fg=TEXT_DIM, selectcolor=BG_CARD,
                       activebackground=BG, activeforeground=TEXT
                       ).pack(pady=(8, 4))

        btn_frame = tk.Frame(inner, bg=BG)
        btn_frame.pack(pady=(4, 16))

        def save():
            names = {}
            for key, ent in entries.items():
                name = ent.get().strip()
                if name:
                    names[key] = name
            config["meeting_speaker_names"] = names
            save_config_value("meeting_speaker_names", names)
            print(f"  Speaker names: {names}")

            # Apply to last transcript
            if apply_var.get() and names:
                from .config import MEETINGS_DIR
                try:
                    txts = sorted(MEETINGS_DIR.glob("Meeting_*.txt"), reverse=True)
                    if txts:
                        txt = txts[0]
                        jsonl = txt.with_suffix(".jsonl")
                        rename_map = {f"Speaker {k}": v for k, v in names.items()}
                        MeetingSession.rename_speakers_in_files(txt, jsonl, rename_map)
                        print(f"  Renamed speakers in {txt.name}")
                except Exception as e:
                    print(f"  Rename failed: {e}")
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


def _open_diarization_download_dialog(tray_icon=None):
    """Show a dialog to download the speaker diarization model (~80 MB)."""

    def _run():
        import tkinter as tk
        from tkinter import ttk

        BG = "#1a1a2e"
        ACCENT = "#e94560"
        TEXT = "#eaeaea"
        TEXT_DIM = "#8892a0"

        root = tk.Tk()
        root.title("Speaker Identification")
        root.configure(bg=BG)
        root.resizable(False, False)

        w, h = 420, 200
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.attributes("-topmost", True)

        size_mb = get_model_size() / 1024 / 1024

        info = tk.Label(
            root, text=f"Speaker identification requires a one-time\n"
                       f"download (~{size_mb:.0f} MB). Download now?",
            font=("Segoe UI", 11), bg=BG, fg=TEXT, justify=tk.CENTER,
        )
        info.pack(pady=(20, 10))

        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=350)

        status_label = tk.Label(root, text="", font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM)

        _cancel_flag = [False]
        _downloading = [False]

        def on_download():
            info.pack_forget()
            btn_frame.pack_forget()
            progress_bar.pack(pady=(20, 5))
            status_label.pack()
            _downloading[0] = True

            def _do_download():
                try:
                    def progress(downloaded, total):
                        if _cancel_flag[0]:
                            raise InterruptedError("Cancelled")
                        if total > 0:
                            pct = downloaded / total * 100
                            dl_mb = downloaded / 1024 / 1024
                            tot_mb = total / 1024 / 1024
                            root.after(0, lambda: progress_var.set(pct))
                            root.after(0, lambda: status_label.configure(
                                text=f"{dl_mb:.1f} / {tot_mb:.1f} MB"
                            ))

                    download_diarization_model(progress_callback=progress)

                    # Enable diarization
                    config["meeting_diarization"] = True
                    save_config_value("meeting_diarization", True)
                    print("  Speaker diarization model downloaded")
                    if tray_icon:
                        tray_icon.update_menu()
                    root.after(0, root.destroy)
                except InterruptedError:
                    remove_diarization_model()
                    root.after(0, root.destroy)
                except Exception as e:
                    print(f"  Download error: {e}")
                    root.after(0, lambda: status_label.configure(
                        text=f"Download failed: {e}", fg="#ff6b6b",
                    ))
                    root.after(0, lambda: progress_bar.configure(style="red.Horizontal.TProgressbar"))

            threading.Thread(target=_do_download, daemon=True).start()

        def on_cancel():
            if _downloading[0]:
                _cancel_flag[0] = True
            else:
                root.destroy()

        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=(5, 15))

        dl_btn = tk.Label(
            btn_frame, text="Download", font=("Segoe UI Semibold", 10),
            bg=ACCENT, fg="#fff", padx=24, pady=6, cursor="hand2",
        )
        dl_btn.bind("<Button-1>", lambda e: on_download())
        dl_btn.pack(side=tk.LEFT, padx=8)

        cancel_btn = tk.Label(
            btn_frame, text="Cancel", font=("Segoe UI Semibold", 10),
            bg="#2a2a4a", fg=TEXT_DIM, padx=24, pady=6, cursor="hand2",
        )
        cancel_btn.bind("<Button-1>", lambda e: on_cancel())
        cancel_btn.pack(side=tk.LEFT, padx=8)

        root.protocol("WM_DELETE_WINDOW", on_cancel)
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
        # Blocking acquire — push-to-talk has priority over meeting transcription
        if not transcription_lock.acquire(timeout=10):
            print("  ⏳ Transcription lock busy, skipping")
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
            transcription_lock.release()

    threading.Thread(target=do_transcribe, daemon=True).start()


def _is_current_model_cached() -> bool:
    """Check if the currently selected model is already downloaded."""
    from .transcribe import is_model_cached
    return is_model_cached(config.get("model", "large-v3"))


def _on_download_model(tray_icon=None):
    """Handle 'Download Model' menu click — opens download dialog in a thread."""
    model_name = config.get("model", "large-v3")

    def _run():
        downloaded = _open_model_download_dialog(model_name)
        if downloaded:
            # Switch back to local backend and load the model
            config["backend"] = "local"
            save_config_value("backend", "local")
            if tray_icon:
                tray_icon.update_menu()

            def _load():
                try:
                    from .transcribe import get_model
                    print("  Loading model...")
                    device_pref = config.get("device", "gpu")
                    get_model(model_name, device_pref=device_pref)
                    print("  Model ready (hotkey active)")
                except Exception as e:
                    print(f"  Model loading failed ({e})")
                if tray_icon:
                    tray_icon.update_menu()

            threading.Thread(target=_load, daemon=True).start()
        if tray_icon:
            tray_icon.update_menu()

    threading.Thread(target=_run, daemon=True).start()


def _on_check_model_update(tray_icon=None):
    """Re-download the model with local_files_only=False to pick up any updates."""
    model_name = config.get("model", "large-v3")

    def _run():
        try:
            from .transcribe import download_model
            print(f"  Checking for {model_name} model updates...")
            download_model(model_name, status_callback=lambda msg: print(f"  {msg}"))
            print(f"  Model {model_name} is up to date")

            # Reload model if it was already loaded
            from .transcribe import _model, _model_lock
            with _model_lock:
                was_loaded = _model is not None
            if was_loaded:
                from .transcribe import reload_model
                device_pref = config.get("device", "gpu")
                reload_model(model_name, device_pref=device_pref)
                print("  Model reloaded")
        except Exception as e:
            print(f"  Update check failed: {e}")
        if tray_icon:
            tray_icon.update_menu()

    threading.Thread(target=_run, daemon=True).start()


def create_tray_icon():
    """Create and run the system tray icon."""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        print("  ⚠️  pystray/Pillow not installed, running without tray icon")
        return None

    def make_icon(meeting: bool = False):
        # Try to load the 😮 icon from bundled or project icons
        icon_paths = [
            _bundle_dir() / "icons" / "icon_128.png",
            _bundle_dir() / "icons" / "icon.png",
            Path(__file__).resolve().parents[2] / "icons" / "icon_128.png",
            Path(__file__).resolve().parents[2] / "icons" / "icon.png",
        ]
        img = None
        for p in icon_paths:
            if p.exists():
                img = Image.open(p).resize((64, 64), Image.LANCZOS)
                break
        if img is None:
            # Fallback: 😮 face (yellow circle, two eyes, open mouth)
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse([4, 4, 60, 60], fill="#FFD93D", outline="#2D2D2D", width=3)
            draw.ellipse([20, 22, 28, 30], fill="#2D2D2D")  # left eye
            draw.ellipse([36, 22, 44, 30], fill="#2D2D2D")  # right eye
            draw.ellipse([26, 38, 38, 50], fill="#2D2D2D")  # open mouth
        if meeting:
            # Red recording dot in bottom-right corner
            draw = ImageDraw.Draw(img)
            draw.ellipse([46, 46, 62, 62], fill="#FF0000", outline="#FFFFFF", width=1)
        return img

    MODELS = ["large-v3", "medium", "small", "base", "tiny"]

    def on_toggle(icon, item):
        state.enabled = not state.enabled
        status = "Enabled" if state.enabled else "Disabled"
        print(f"  🔄 Dictation {status}")

    # ── Meeting mode ────────────────────────────────────────────────

    # Live timer: refresh the tray menu while a meeting is running
    _menu_timer_stop = threading.Event()

    def _start_menu_timer(ic):
        _menu_timer_stop.clear()
        def _tick():
            while not _menu_timer_stop.wait(timeout=5):
                try:
                    ic.update_menu()
                except Exception:
                    pass
        threading.Thread(target=_tick, daemon=True).start()

    def _stop_menu_timer():
        _menu_timer_stop.set()

    def on_meeting_toggle(icon, item):
        global _meeting_session
        if _meeting_session and _meeting_session.running:
            _meeting_session.stop()
            _play_sound("stop")
            _meeting_session = None
            icon.icon = make_icon(meeting=False)
            _stop_menu_timer()
        else:
            mic = config.get("mic_device")
            _meeting_session = MeetingSession(device_index=mic, config=config)
            _meeting_session.start()
            _play_sound("start")
            icon.icon = make_icon(meeting=True)
            _start_menu_timer(icon)
        icon.update_menu()

    def _meeting_label(item):
        if _meeting_session and _meeting_session.running:
            return f"Stop Meeting ({_meeting_session.elapsed_str()})"
        return "Start Meeting"

    def on_open_meetings(icon, item):
        MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Windows":
            os.startfile(str(MEETINGS_DIR))
        elif platform.system() == "Darwin":
            import subprocess
            subprocess.Popen(["open", str(MEETINGS_DIR)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(MEETINGS_DIR)])

    def _diarization_available():
        return is_model_downloaded()

    def on_toggle_diarization(icon, item):
        if not _diarization_available():
            # Model not downloaded — prompt user
            _open_diarization_download_dialog(icon)
            return
        val = not config.get("meeting_diarization", False)
        config["meeting_diarization"] = val
        save_config_value("meeting_diarization", val)
        print(f"  Speaker identification: {'on' if val else 'off'}")
        icon.update_menu()

    def on_rename_speakers(icon, item):
        _open_rename_speakers_dialog()

    # ── Attendees selector ──────────────────────────────────────────

    def make_attendees_callback(n):
        def callback(icon, item):
            config["meeting_max_speakers"] = n
            save_config_value("meeting_max_speakers", n)
            print(f"  Meeting attendees set to {n}")
            icon.update_menu()
        return callback

    def is_current_attendees(n):
        return lambda item: config.get("meeting_max_speakers", 10) == n

    def _attendees_label(item):
        n = config.get("meeting_max_speakers", 10)
        return f"Attendees: {n}"

    attendees_menu = pystray.Menu(
        *[pystray.MenuItem(
            str(n), make_attendees_callback(n),
            checked=is_current_attendees(n), radio=True,
        ) for n in range(2, 11)]
    )

    def on_quit(icon, item):
        global _meeting_session
        _stop_menu_timer()
        if _meeting_session and _meeting_session.running:
            _meeting_session.stop()
            _meeting_session = None
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
                        from .transcribe import reload_model, is_model_cached
                        if not is_model_cached(model_name):
                            print(f"  Downloading {model_name}...")
                        device_pref = config.get("device", "gpu")
                        reload_model(model_name, device_pref=device_pref)
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
        return lambda item: (config.get("device") or "gpu") == device_name.lower()

    # --- Language selection (multi-select checkboxes) ---
    def make_lang_callback(lang_code):
        def callback(icon, item):
            langs = list(config.get("languages", ["en"]))
            if lang_code in langs:
                if len(langs) > 1:
                    langs.remove(lang_code)
                else:
                    return  # must keep at least one
            else:
                langs.append(lang_code)
            config["languages"] = langs
            save_config_value("languages", langs)
            names = [LANG_LABELS.get(c, c) for c in langs]
            print(f"  Languages: {', '.join(names)}")
            icon.update_menu()
        return callback

    def is_lang_checked(lang_code):
        return lambda item: lang_code in config.get("languages", ["en"])

    def _lang_menu_label(item):
        langs = config.get("languages", ["en"])
        if len(langs) == 1:
            return f"Language: {LANG_LABELS.get(langs[0], langs[0])}"
        return f"Language: {len(langs)} selected (auto-detect)"

    lang_menu = pystray.Menu(
        *[pystray.MenuItem(
            label, make_lang_callback(code),
            checked=is_lang_checked(code),
        ) for code, label in LANGUAGES]
    )

    # --- Microphone selection (refreshes on each menu open) ---
    def _on_mic_click(icon, item):
        label = str(item)
        if label == "System Default":
            config.pop("mic_device", None)
            save_config_value("mic_device", None)
            print("  Microphone: System Default")
        else:
            # Find device index by matching name from current device list
            for idx, name in get_input_devices():
                if name == label:
                    config["mic_device"] = idx
                    save_config_value("mic_device", idx)
                    print(f"  Microphone: {name} (device {idx})")
                    break
        icon.update_menu()

    def _build_mic_menu():
        """Build mic menu dynamically — queries available devices each time."""
        current = config.get("mic_device")
        devices = get_input_devices()
        items = [
            pystray.MenuItem(
                "System Default", _on_mic_click,
                checked=lambda item: config.get("mic_device") is None, radio=True,
            ),
        ]
        for idx, name in devices:
            items.append(pystray.MenuItem(
                name, _on_mic_click,
                checked=(lambda i: lambda item: config.get("mic_device") == i)(idx),
                radio=True,
            ))
        return items

    mic_menu = pystray.Menu(lambda: _build_mic_menu())

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

    def _diarization_label(item):
        if _diarization_available():
            on_off = "on" if config.get("meeting_diarization", False) else "off"
            return f"Speaker Identification ({on_off})"
        return "Speaker Identification (download)"

    # --- Meeting audio source ---
    _AUDIO_SOURCES = [("mic", "Microphone Only"), ("system", "System Audio Only"), ("both", "Mic + System Audio")]

    def _make_audio_source_callback(source_key):
        def callback(icon, item):
            config["meeting_audio_source"] = source_key
            save_config_value("meeting_audio_source", source_key)
            label = dict(_AUDIO_SOURCES).get(source_key, source_key)
            print(f"  Meeting audio: {label}")
            icon.update_menu()
        return callback

    def _is_current_audio_source(source_key):
        return lambda item: config.get("meeting_audio_source", "mic") == source_key

    def _audio_source_available(item):
        from .audio import LoopbackStream
        return LoopbackStream.is_available()

    audio_source_menu = pystray.Menu(
        *[pystray.MenuItem(
            label, _make_audio_source_callback(key),
            checked=_is_current_audio_source(key), radio=True,
        ) for key, label in _AUDIO_SOURCES],
    )

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: f"Hold {_hotkey_display()} to dictate", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(lambda item: "✓ Enabled" if state.enabled else "  Disabled", on_toggle),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_meeting_label, on_meeting_toggle),
        pystray.MenuItem("Open Meetings Folder", on_open_meetings),
        pystray.MenuItem("Meeting Audio Source", audio_source_menu),
        pystray.MenuItem(
            _diarization_label,
            on_toggle_diarization,
        ),
        pystray.MenuItem(_attendees_label, attendees_menu),
        pystray.MenuItem("Rename Speakers...", on_rename_speakers),
        pystray.Menu.SEPARATOR,
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
        pystray.MenuItem(_lang_menu_label, lang_menu),
        pystray.MenuItem("Select Microphone", mic_menu),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_model_status_label, None, enabled=False),
        pystray.MenuItem(
            "Download Model...",
            lambda icon, item: _on_download_model(icon),
            visible=lambda item: not _is_current_model_cached(),
        ),
        pystray.MenuItem(
            "Check for Model Updates",
            lambda icon, item: _on_check_model_update(icon),
            visible=lambda item: _is_current_model_cached(),
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Change Sounds...", on_change_sounds),
        pystray.MenuItem("Edit Dictionary", on_edit_dict),
        pystray.MenuItem("Change Hotkey...", on_change_hotkey),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon("WhisperO", make_icon(), "WhisperO", menu)
    return icon


def _open_model_download_dialog(model_name: str) -> bool:
    """Show a first-launch dialog to download the Whisper model. Returns True if downloaded."""

    result = [False]

    def _run():
        import tkinter as tk
        from tkinter import ttk

        BG = "#1a1a2e"
        ACCENT = "#e94560"
        TEXT = "#eaeaea"
        TEXT_DIM = "#8892a0"

        root = tk.Tk()
        root.title("WhisperO — First-Time Setup")
        root.configure(bg=BG)
        root.resizable(False, False)

        w, h = 480, 220
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.attributes("-topmost", True)

        # Model sizes (approximate download)
        _MODEL_SIZES = {
            "large-v3": "~3 GB", "medium": "~1.5 GB", "small": "~500 MB",
            "base": "~150 MB", "tiny": "~75 MB",
        }
        size_str = _MODEL_SIZES.get(model_name, "several GB")

        info = tk.Label(
            root,
            text=f"WhisperO needs to download the {model_name} model\n"
                 f"({size_str}) for speech recognition.\n\n"
                 f"This is a one-time download.",
            font=("Segoe UI", 11), bg=BG, fg=TEXT, justify=tk.CENTER,
        )
        info.pack(pady=(20, 10))

        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=400)
        status_label = tk.Label(root, text="", font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM)

        _cancel_flag = [False]

        def on_download():
            info.pack_forget()
            btn_frame.pack_forget()
            progress_bar.pack(pady=(30, 5))
            status_label.pack()
            status_label.configure(text="Connecting...")
            root.geometry(f"{w}x{180}+{x}+{y}")

            def _do_download():
                try:
                    from .transcribe import download_model

                    def progress(downloaded, total):
                        if _cancel_flag[0]:
                            raise InterruptedError("Cancelled")
                        if total > 0:
                            pct = downloaded / total * 100
                            dl_mb = downloaded / 1024 / 1024
                            tot_mb = total / 1024 / 1024
                            root.after(0, lambda: progress_var.set(pct))
                            root.after(0, lambda: status_label.configure(
                                text=f"{dl_mb:.0f} / {tot_mb:.0f} MB ({pct:.0f}%)"
                            ))

                    def status(msg):
                        root.after(0, lambda: status_label.configure(
                            text=msg, fg=TEXT_DIM,
                        ))

                    download_model(model_name, progress_callback=progress,
                                   status_callback=status)
                    result[0] = True
                    print(f"  Model {model_name} downloaded successfully")
                    root.after(0, root.destroy)
                except InterruptedError:
                    root.after(0, root.destroy)
                except Exception as e:
                    print(f"  Download error: {e}")
                    root.after(0, lambda: status_label.configure(
                        text=f"Failed: {e}", fg="#ff6b6b",
                    ))

            threading.Thread(target=_do_download, daemon=True).start()

        def on_skip():
            root.destroy()

        def on_close():
            _cancel_flag[0] = True
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)

        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Download", font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg="white", activebackground="#c73a52", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2",
            command=on_download,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btn_frame, text="Skip (use server)", font=("Segoe UI", 10),
            bg="#2a2a4a", fg=TEXT_DIM, activebackground="#3a3a5a", activeforeground=TEXT,
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2",
            command=on_skip,
        ).pack(side=tk.LEFT, padx=8)

        root.mainloop()

    _run()
    return result[0]


def _check_cuda_available() -> bool:
    """Check if CUDA libraries are loadable.

    Tries the executable's directory first (so users who manually drop
    CUDA DLLs into the WhisperO install folder are detected), then falls
    back to the default Windows DLL search path.
    """
    import ctypes
    import os
    import sys

    candidates = ["cublas64_12.dll", "cudart64_12.dll", "cublasLt64_12.dll"]

    # Search dirs: app dir first, then PyInstaller _internal, then frozen sys.path
    search_dirs = []
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
        search_dirs.append(app_dir)
        internal = os.path.join(app_dir, "_internal")
        if os.path.isdir(internal):
            search_dirs.append(internal)
    else:
        search_dirs.append(os.path.dirname(os.path.abspath(__file__)))

    for d in search_dirs:
        full_path = os.path.join(d, "cublas64_12.dll")
        if os.path.exists(full_path):
            try:
                # Add dir so dependent CUDA DLLs (cudart, cublasLt) can be found
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(d)
                    except (OSError, FileNotFoundError):
                        pass
                ctypes.CDLL(full_path)
                return True
            except OSError as e:
                print(f"  Found {full_path} but failed to load: {e}")

    # Fallback: default DLL search (PATH, system32, etc.)
    try:
        ctypes.cdll.LoadLibrary("cublas64_12.dll")
        return True
    except OSError:
        return False


def _open_cuda_missing_dialog() -> str:
    """Show dialog when CUDA is missing. Returns 'cpu', 'install', or 'quit'."""
    result = ["cpu"]

    def _run():
        import tkinter as tk
        import webbrowser

        BG = "#1a1a2e"
        ACCENT = "#e94560"
        TEXT = "#eaeaea"
        TEXT_DIM = "#8892a0"
        LINK = "#5dade2"

        root = tk.Tk()
        root.title("WhisperO — CUDA Not Found")
        root.configure(bg=BG)
        root.resizable(False, False)

        w, h = 520, 300
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.attributes("-topmost", True)

        tk.Label(
            root, text="CUDA libraries not found",
            font=("Segoe UI", 13, "bold"), bg=BG, fg=TEXT,
        ).pack(pady=(20, 5))

        tk.Label(
            root,
            text="An NVIDIA GPU was detected, but CUDA runtime\n"
                 "libraries are not installed. GPU transcription\n"
                 "requires CUDA Toolkit 12.",
            font=("Segoe UI", 10), bg=BG, fg=TEXT_DIM, justify=tk.CENTER,
        ).pack(pady=(0, 10))

        # Instructions
        tk.Label(
            root,
            text="To install: download CUDA Toolkit 12.2, run the installer,\n"
                 'select "Custom" and check only CUDA > Runtime > Libraries.\n'
                 "Then restart WhisperO.",
            font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM, justify=tk.CENTER,
        ).pack(pady=(0, 5))

        cuda_url = "https://developer.nvidia.com/cuda-12-2-0-download-archive"

        link = tk.Label(
            root, text=cuda_url,
            font=("Segoe UI", 9, "underline"), bg=BG, fg=LINK, cursor="hand2",
        )
        link.pack(pady=(0, 15))
        link.bind("<Button-1>", lambda e: webbrowser.open(cuda_url))

        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=5)

        def on_cpu():
            result[0] = "cpu"
            root.destroy()

        def on_install():
            webbrowser.open(cuda_url)
            result[0] = "quit"
            root.destroy()

        def on_close():
            result[0] = "cpu"
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)

        tk.Button(
            btn_frame, text="Continue on CPU", font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg="white", activebackground="#c73a52", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2",
            command=on_cpu,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btn_frame, text="Download CUDA & Quit", font=("Segoe UI", 10),
            bg="#2a2a4a", fg=TEXT_DIM, activebackground="#3a3a5a", activeforeground=TEXT,
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2",
            command=on_install,
        ).pack(side=tk.LEFT, padx=8)

        root.mainloop()

    _run()
    return result[0]


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

    # Installer post-install: reset device preference to GPU so a
    # fresh install always starts on GPU, even if the user previously
    # had device=cpu saved from a past CUDA-missing session.
    if "--reset-device-gpu" in sys.argv:
        try:
            save_config_value("device", "gpu")
            print("Device preference reset to GPU.")
        except Exception as e:
            print(f"Failed to reset device: {e}")
        sys.exit(0)

    backend = config.get("backend", "local")

    # Restore persisted last working server
    saved_server = config.get("last_working_server")
    if saved_server:
        from . import transcribe as _transcribe_mod
        _transcribe_mod._last_working_server = saved_server

    if backend == "local":
        model_name = config.get("model", "large-v3")
        print(f"WhisperO (local, model: {model_name})")

        # Check CUDA availability when GPU is selected
        device_pref = config.get("device", "gpu")
        if device_pref == "gpu" and platform.system() == "Windows":
            if not _check_cuda_available():
                choice = _open_cuda_missing_dialog()
                if choice == "cpu":
                    config["device"] = "cpu"
                    save_config_value("device", "cpu")
                    print("  CUDA not found, using CPU")
                elif choice == "quit":
                    print("  Please install CUDA and restart WhisperO")
                    sys.exit(0)

        # First launch: show download dialog if model isn't cached
        from .transcribe import is_model_cached
        if not is_model_cached(model_name):
            downloaded = _open_model_download_dialog(model_name)
            if not downloaded:
                # User skipped — fall back to server mode for this session
                print("  Model not downloaded, using server mode")
                config["backend"] = "server"
                backend = "server"

        if backend == "local":
            def _load_model_bg():
                try:
                    from .transcribe import get_model
                    print("  Loading model...")
                    device_pref = config.get("device", "gpu")
                    get_model(model_name, device_pref=device_pref)
                    print("  Model ready (hotkey active)")
                except Exception as e:
                    print(f"  Model loading failed ({e}), falling back to server mode")
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
