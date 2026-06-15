from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = {"__pycache__", ".git", ".idea", ".vscode"}
SKIP_FILE_SUFFIXES = {".pyc", ".pyo"}


def should_skip_dir(path: Path) -> bool:
    return path.name in SKIP_DIR_NAMES


def should_skip_file(path: Path) -> bool:
    return path.suffix in SKIP_FILE_SUFFIXES


def copy_tree_filtered(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if any(should_skip_dir(parent) for parent in item.parents if parent != src.parent):
            continue
        target = dst / rel
        if item.is_dir():
            if not should_skip_dir(item):
                target.mkdir(parents=True, exist_ok=True)
            continue
        if should_skip_file(item):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: copy_runtime_resources.py <dist_app_folder>")
        return 2

    dist_app = Path(sys.argv[1]).resolve()
    dist_app.mkdir(parents=True, exist_ok=True)

    configs = ROOT / "configs"

    if not configs.exists():
        print(f"ERROR: missing configs folder: {configs}")
        return 1

    print(f"Copy configs only: {configs} -> {dist_app / 'configs'}")
    copy_tree_filtered(configs, dist_app / "configs")

    for folder_name in ("models", "input_docx", "output_docx"):
        old = dist_app / folder_name
        if old.exists():
            print(f"Remove stale folder from dist: {old}")
            shutil.rmtree(old)

    print("Resources copied. Models remain external in project_root/models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
