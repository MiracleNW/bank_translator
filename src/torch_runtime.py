from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_DLL_HANDLES = []
_DLL_DIR_HANDLES = []


def _add_path_to_env(path: Path) -> None:
    text = str(path)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if text not in parts:
        os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")


def _candidate_torch_lib_dirs() -> list[Path]:
    dirs: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass) / "torch" / "lib")

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        dirs.extend([
            exe_dir / "_internal" / "torch" / "lib",
            exe_dir / "torch" / "lib",
        ])

    for entry in map(Path, sys.path):
        dirs.append(entry / "torch" / "lib")

    unique: list[Path] = []
    for path in dirs:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.exists() and resolved not in unique:
            unique.append(resolved)
    return unique


def _desired_thread_count() -> int:
    raw = os.environ.get("BANKTRANSLATOR_TORCH_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count // 2 or 1))


def prepare_torch_runtime() -> None:
    threads = str(_desired_thread_count())
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


    if platform.system() != "Windows":
        return

    for dll_dir in _candidate_torch_lib_dirs():
        _add_path_to_env(dll_dir)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            except OSError:
                pass

        c10 = dll_dir / "c10.dll"
        if c10.exists():
            try:
                _DLL_HANDLES.append(ctypes.WinDLL(str(c10)))
            except OSError:
                pass


def import_torch_early():
    prepare_torch_runtime()
    import torch

    try:
        threads = _desired_thread_count()
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    return torch
