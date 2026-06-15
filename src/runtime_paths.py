from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def exe_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def source_project_root_from_exe() -> Path | None:
    if not is_frozen():
        return Path(__file__).resolve().parents[1]

    folder = exe_dir()
    if folder.parent.name.lower() == "dist":
        candidate = folder.parent.parent
        if candidate.exists():
            return candidate
    return None


def app_root() -> Path:
    env_root = os.environ.get("BANKTRANSLATOR_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    project_root = source_project_root_from_exe()
    if project_root is not None:
        return project_root

    return exe_dir()


def bundled_root() -> Path | None:
    value = getattr(sys, "_MEIPASS", None)
    return Path(value).resolve() if value else None


def candidate_roots() -> list[Path]:
    roots: list[Path] = []

    env_root = os.environ.get("BANKTRANSLATOR_PROJECT_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())

    project_root = source_project_root_from_exe()
    if project_root is not None:
        roots.append(project_root)

    roots.append(exe_dir())

    internal = bundled_root()
    if internal is not None:
        roots.append(internal)

    roots.append(Path.cwd().resolve())

    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved not in unique:
            unique.append(resolved)
    return unique


def resource_path(*parts: str | Path, must_exist: bool = False) -> Path:
    relative = Path(*parts)

    if relative.is_absolute():
        if relative.exists() or not must_exist:
            return relative
        raise FileNotFoundError(f"Не найден ресурс: {relative}")

    for root in candidate_roots():
        path = root / relative
        if path.exists():
            return path

    fallback = app_root() / relative
    if must_exist:
        searched = "\n".join(str(root / relative) for root in candidate_roots())
        raise FileNotFoundError(f"Не найден ресурс: {relative}\nПроверенные пути:\n{searched}")
    return fallback
