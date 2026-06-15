from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_HANDLES = []
_DIR_HANDLES = []

threads = os.environ.get("BANKTRANSLATOR_TORCH_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", threads)
os.environ.setdefault("MKL_NUM_THREADS", threads)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if platform.system() == "Windows":
    paths = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        paths.append(Path(meipass) / "torch" / "lib")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        paths.extend([exe_dir / "_internal" / "torch" / "lib", exe_dir / "torch" / "lib"])

    for path in paths:
        if not path.exists():
            continue
        text = str(path)
        os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                _DIR_HANDLES.append(os.add_dll_directory(text))
            except OSError:
                pass
        c10 = path / "c10.dll"
        if c10.exists():
            try:
                _HANDLES.append(ctypes.WinDLL(str(c10)))
            except OSError:
                pass
