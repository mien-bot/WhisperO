import multiprocessing
multiprocessing.freeze_support()

import os
import sys

# Add the exe's directory to DLL search path so bundled CUDA DLLs are found
if getattr(sys, "frozen", False):
    _app_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = _app_dir + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_app_dir)

# In frozen windowed builds stdout/stderr are None — redirect to a log file.
if getattr(sys, "frozen", False) and sys.stdout is None:
    from pathlib import Path

    _log = Path.home() / ".whispero" / "whispero.log"
    _log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(_log, "w", encoding="utf-8")  # noqa: SIM115

from .app import main


if __name__ == "__main__":
    main()
