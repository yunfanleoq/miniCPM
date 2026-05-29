"""Windows/PowerShell UTF-8 console helpers."""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Make Python print UTF-8 correctly in Windows terminals."""
    if sys.platform != "win32":
        return

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
