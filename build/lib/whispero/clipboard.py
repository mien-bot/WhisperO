from __future__ import annotations

import platform
import time

import pyperclip


def save_clipboard_win32():
    """Save all clipboard formats on Windows. Returns list of (format, data) tuples."""
    try:
        import win32clipboard

        saved = []
        win32clipboard.OpenClipboard()
        try:
            fmt = 0
            while True:
                fmt = win32clipboard.EnumClipboardFormats(fmt)
                if fmt == 0:
                    break
                try:
                    data = win32clipboard.GetClipboardData(fmt)
                    saved.append((fmt, data))
                except Exception:
                    pass  # Some formats can't be read, skip.
        finally:
            win32clipboard.CloseClipboard()
        return saved
    except Exception:
        return None


def restore_clipboard_win32(saved):
    """Restore previously saved clipboard contents on Windows."""
    if saved is None:
        return
    try:
        import win32clipboard

        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            for fmt, data in saved:
                try:
                    win32clipboard.SetClipboardData(fmt, data)
                except Exception:
                    pass  # Some formats can't be restored, skip.
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        pass


def save_clipboard_macos():
    """Save all clipboard types on macOS via AppKit as (uti, bytes) tuples."""
    try:
        from AppKit import NSPasteboard

        pb = NSPasteboard.generalPasteboard()
        types = pb.types() or []
        saved = []
        for uti in types:
            data = pb.dataForType_(uti)
            if data is not None:
                saved.append((str(uti), bytes(data)))
        return saved
    except Exception:
        # Fallback: preserve plain text only when AppKit is unavailable.
        try:
            return [("public.utf8-plain-text", pyperclip.paste().encode("utf-8"))]
        except Exception:
            return None


def restore_clipboard_macos(saved):
    """Restore clipboard contents on macOS from (uti, bytes) tuples."""
    if saved is None:
        return
    try:
        from AppKit import NSPasteboard
        from Foundation import NSData

        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        for uti, payload in saved:
            try:
                data = NSData.dataWithBytes_length_(payload, len(payload))
                pb.setData_forType_(data, uti)
            except Exception:
                pass
    except Exception:
        pass


def paste_text(text: str) -> None:
    """Copy text to clipboard, simulate paste, then restore original clipboard contents."""
    system = platform.system()

    # Save clipboard to avoid clobbering user data (images/files/rich text).
    saved_clipboard = None
    if system == "Windows":
        saved_clipboard = save_clipboard_win32()
    elif system == "Darwin":
        saved_clipboard = save_clipboard_macos()

    pyperclip.copy(text)
    time.sleep(0.05)

    from pynput.keyboard import Controller, Key

    kb = Controller()

    if system == "Darwin":
        kb.press(Key.cmd)
        kb.press("v")
        kb.release("v")
        kb.release(Key.cmd)
    else:
        kb.press(Key.ctrl)
        kb.press("v")
        kb.release("v")
        kb.release(Key.ctrl)

    # Restore original clipboard after paste completes
    if saved_clipboard is not None:
        time.sleep(0.15)  # Wait for paste to finish
        if system == "Windows":
            restore_clipboard_win32(saved_clipboard)
        elif system == "Darwin":
            restore_clipboard_macos(saved_clipboard)
