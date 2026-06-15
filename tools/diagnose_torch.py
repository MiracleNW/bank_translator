from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=== Torch diagnostics ===")
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("CWD:", os.getcwd())
print("BANKTRANSLATOR_TORCH_THREADS:", os.environ.get("BANKTRANSLATOR_TORCH_THREADS", "not set"))

try:
    from src.torch_runtime import import_torch_early, _candidate_torch_lib_dirs
    print("Torch lib dirs:")
    for p in _candidate_torch_lib_dirs():
        print(" -", p)
    torch = import_torch_early()
    print("Torch version:", torch.__version__)
    print("Torch file:", torch.__file__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA capability:", torch.cuda.get_device_capability(0))
    print("Torch threads:", torch.get_num_threads())
    print("Tensor test:", float(torch.tensor([1.0, 2.0, 3.0]).sum()))
    print("DIAG_OK")
except Exception as exc:
    print("DIAG_FAILED")
    print(type(exc).__name__ + ":", exc)
    raise
