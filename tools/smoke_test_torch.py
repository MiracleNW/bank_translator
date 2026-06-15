from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())

from src.torch_runtime import import_torch_early

torch = import_torch_early()
print("Torch version:", torch.__version__)
print("Torch file:", torch.__file__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("CUDA device name error:", exc)

if os.environ.get("BANKTRANSLATOR_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
    print("ERROR: CUDA build was selected, but torch.cuda.is_available() is False.")
    print("Use build_exe.bat and choose CPU, or update/install NVIDIA driver and try CUDA again.")
    raise SystemExit(2)

from PyQt5.QtWidgets import QApplication
print("PyQt import after torch: OK")

x = torch.tensor([1.0, 2.0, 3.0])
print("Torch tensor test:", float(x.sum()))
if torch.cuda.is_available():
    y = x.to("cuda")
    print("Torch CUDA tensor test:", float(y.sum().cpu()))
print("SMOKE_TEST_OK")
