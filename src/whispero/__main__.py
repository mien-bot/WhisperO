import multiprocessing
multiprocessing.freeze_support()

import os
import sys

# Add the exe's directory to DLL search path so bundled or manually-placed
# CUDA DLLs (cublas64_12.dll, cudart64_12.dll, ...) are found.
if getattr(sys, "frozen", False):
    _app_dir = os.path.dirname(sys.executable)
    _dll_dirs = [_app_dir]
    _internal = os.path.join(_app_dir, "_internal")
    if os.path.isdir(_internal):
        _dll_dirs.append(_internal)
    os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        for _d in _dll_dirs:
            try:
                os.add_dll_directory(_d)
            except (OSError, FileNotFoundError):
                pass

# In frozen windowed builds stdout/stderr are None — redirect to a log file.
if getattr(sys, "frozen", False) and sys.stdout is None:
    from pathlib import Path

    _log = Path.home() / ".whispero" / "whispero.log"
    _log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(_log, "w", encoding="utf-8")  # noqa: SIM115

from .app import main


if __name__ == "__main__":
    main()
