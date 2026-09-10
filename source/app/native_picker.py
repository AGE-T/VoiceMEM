"""Native Windows file picker for the LLM GGUF model selector (v0.4.16).

The web UI is served by the LOCAL Python backend on the user's machine, so
a browser ``<input type=file>`` would be the WRONG tool (it uploads CONTENT
— we only want the PATH). This module is the smallest safe bridge: the
backend opens the REAL Windows common file dialog and returns the selected
absolute path to the UI. Nothing is uploaded, nothing is copied.

Strategy (Windows):
  1. ``ctypes`` call to ``comdlg32.GetOpenFileNameW`` — the classic native
     dialog, zero non-stdlib dependencies, works from a worker thread with
     no owner window (the dialog pumps its own modal loop).
  2. ``tkinter.filedialog.askopenfilename`` fallback when comdlg32 is
     unavailable for any reason (stripped Windows, dialog subsystem off).

Non-Windows (dev sandbox / Linux): returns ``(None, reason)`` — the REST
layer reports ``supported: false`` and the UI falls back to the manual
path input (also useful for diagnostics, per the task contract).

The dialog is opened from ``asyncio.to_thread`` by the caller; this module
is fully synchronous and thread-agnostic.
"""

from __future__ import annotations

import platform
import threading
from typing import Optional

_lock = threading.Lock()

#: Picker filter: GGUF first (the task contract), then OLLAMA BLOBS
#: (content-addressed ``sha256-<hex>`` files carry no .gguf extension but
#: ARE complete GGUFs llama-server loads directly — v0.4.17), then all
#: files for diagnostics.
GGUF_FILTER = (
    "GGUF model files (*.gguf)\0*.gguf\0"
    "Ollama blobs (sha256-*)\0sha256-*\0"
    "All files (*.*)\0*.*\0\0"
)

# GetOpenFileNameW flags:
#   OFN_EXPLORER      0x00080000  modern Explorer-style dialog
#   OFN_FILEMUSTEXIST 0x00001000  only existing files selectable
#   OFN_PATHMUSTEXIST 0x00000800  typed paths must exist
#   OFN_HIDEREADONLY  0x00000004  hide the pointless "open read-only" box
#   OFN_NOCHANGEDIR   0x00000008  never chdir() the calling process
_OFN_FLAGS = 0x00080000 | 0x00001000 | 0x00000800 | 0x00000004 | 0x00000008


def is_supported() -> bool:
    """True when this platform can open a native file dialog at all."""
    return platform.system() == "Windows"


def pick_file(
    title: str = "Select the LLM model file (.gguf)",
    filter_text: str = GGUF_FILTER,
) -> tuple[Optional[str], str]:
    """Open the NATIVE file dialog; return ``(selected_path, status)``.

    ``status`` is one of ``"selected"``, ``"cancelled"`` (user closed the
    dialog), ``"unsupported"`` (non-Windows) or ``"failed:<detail>"``.
    Never raises.
    """
    if not is_supported():
        return None, "unsupported"
    # Serialise: only one native dialog at a time (a second open while the
    # first is modal would nest modal loops on the same thread pool).
    with _lock:
        try:
            path = _pick_comdlg32(title, filter_text)
            if path is not None:
                return path, "selected"
            last_error = _LAST_COMDLG_ERROR[0]
            if last_error:
                return None, f"failed:{last_error}"
            return None, "cancelled"
        except Exception as exc:  # noqa: BLE001 - fall through to tkinter
            comdlg_detail = str(exc)[:120]
        try:
            path = _pick_tkinter(title, filter_text)
            if path:
                return path, "selected"
            return None, "cancelled"
        except Exception as exc:  # noqa: BLE001
            return None, f"failed:comdlg32({comdlg_detail}) tkinter({exc})"


_LAST_COMDLG_ERROR = [""]


def _pick_comdlg32(title: str, filter_text: str) -> Optional[str]:
    """ctypes GetOpenFileNameW; ``None`` = user cancelled; raises on setup
    problems (caller falls back to tkinter)."""
    import ctypes
    from ctypes import wintypes

    class _OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", wintypes.LPVOID),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", wintypes.LPVOID),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    comdlg32 = ctypes.windll.comdlg32
    buf = ctypes.create_unicode_buffer(4096)  # MAX_PATH is 260; 4096 is safe
    ofn = _OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(_OPENFILENAMEW)
    ofn.hwndOwner = None
    ofn.lpstrFilter = filter_text
    ofn.nFilterIndex = 1  # preselect the *.gguf entry
    ofn.lpstrFile = buf
    ofn.nMaxFile = ctypes.sizeof(buf) // 2
    ofn.lpstrTitle = title
    ofn.Flags = _OFN_FLAGS
    _LAST_COMDLG_ERROR[0] = ""
    if not comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        # 0 = cancelled OR error; CommDlgExtendedError tells them apart.
        err = comdlg32.CommDlgExtendedError()
        if err:
            _LAST_COMDLG_ERROR[0] = f"CommDlgExtendedError={err}"
        return None
    return buf.value


def _pick_tkinter(title: str, filter_text: str) -> Optional[str]:
    """tkinter fallback (same semantics; raises when tkinter is absent)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)  # keep the dialog above the browser
        # "GGUF model files (*.gguf)\0*.gguf\0All files (*.*)\0*.*\0\0"
        # -> tkinter wants "{'GGUF model files' {*.gguf} {'All files' {*.*}}}"
        pairs: list[tuple[str, str]] = []
        parts = [p for p in filter_text.split("\0") if p]
        for i in range(0, len(parts) - 1, 2):
            pairs.append((parts[i], parts[i + 1]))
        filetypes = [(label, pattern.replace("*.", "*.")) for label, pattern in pairs]
        path = filedialog.askopenfilename(
            parent=root, title=title, filetypes=filetypes or [("All files", "*.*")]
        )
        return str(path) if path else None
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass
