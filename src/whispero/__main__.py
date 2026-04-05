import multiprocessing
multiprocessing.freeze_support()

import sys

# In frozen windowed builds stdout/stderr are None — redirect to a log file.
if getattr(sys, "frozen", False) and sys.stdout is None:
    from pathlib import Path

    _log = Path.home() / ".whispero" / "whispero.log"
    _log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(_log, "w", encoding="utf-8")  # noqa: SIM115

from .app import main


if __name__ == "__main__":
    main()
