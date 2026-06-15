from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from src.runtime_paths import app_root, candidate_roots, resource_path

StatusCallback = Callable[[str], None]


_MODEL_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "*.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def is_full_transformers_model_dir(path: str | Path) -> bool:
    folder = Path(path)
    if not folder.exists() or not folder.is_dir():
        return False
    if not (folder / "config.json").exists():
        return False
    for pattern in _MODEL_WEIGHT_PATTERNS:
        if any(folder.glob(pattern)):
            return True
    return False


def looks_like_peft_adapter_dir(path: str | Path) -> bool:
    folder = Path(path)
    return (
        folder.exists()
        and folder.is_dir()
        and (folder / "adapter_config.json").exists()
        and any(folder.glob("adapter_model.*"))
    )


def _status(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _as_candidate_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    for root in candidate_roots():
        candidate = root / path
        if candidate.exists():
            return candidate

    return app_root() / path


def _generated_candidate_names(base_model_name_or_path: str) -> list[Path]:
    names: list[Path] = []
    raw = str(base_model_name_or_path).strip()
    if not raw:
        return names
    
    parts = raw.split("/")
    leaf = parts[-1]
    candidates = [
        Path("models") / leaf,
        Path("models") / raw.replace("/", "__"),
        Path("models") / raw.replace("/", "_"),
        Path("models") / "yandex_gpt_base",
        Path("models") / "yandex_gpt_full",
        Path("models") / "yandex_gpt",
    ]
    if len(parts) > 1:
        candidates.append(Path("models").joinpath(*parts))

    for item in candidates:
        if item not in names:
            names.append(item)
    return names


def base_model_local_files_only(settings: dict | None = None) -> bool:
    settings = settings or {}
    if os.environ.get("BANKTRANSLATOR_ALLOW_HF_DOWNLOADS") == "1":
        return False
    if os.environ.get("BANKTRANSLATOR_LOCAL_FILES_ONLY") == "1":
        return True
    return not bool(settings.get("allow_hf_download", False))


def candidate_base_model_paths(
    base_model_name_or_path: str,
    adapter_path: str | Path | None = None,
    settings: dict | None = None,
) -> list[Path]:
    settings = settings or {}
    candidates: list[Path] = []

    env_path = os.environ.get("BANKTRANSLATOR_BASE_MODEL_PATH", "").strip()
    if env_path:
        candidates.append(_as_candidate_path(env_path))

    explicit = str(settings.get("base_model_path", "")).strip()
    if explicit:
        candidates.append(_as_candidate_path(explicit))

    if adapter_path is not None:
        adapter = Path(adapter_path)
        candidates.extend([
            adapter / "base_model",
            adapter.parent / "base_model",
        ])

    for item in settings.get("base_model_candidates", []) or []:
        text = str(item).strip()
        if text:
            candidates.append(_as_candidate_path(text))

    for item in _generated_candidate_names(base_model_name_or_path):
        candidates.append(_as_candidate_path(item))

    raw = str(base_model_name_or_path).strip()
    if raw:
        leaf = raw.split("/")[-1]
        for parts in [
            ("models", leaf),
            ("models", raw.replace("/", "__")),
            ("models", raw.replace("/", "_")),
            ("models", "yandex_gpt_base"),
            ("models", "yandex_gpt_full"),
            ("models", "yandex_gpt"),
        ]:
            try:
                candidates.append(resource_path(*parts))
            except Exception:
                pass
        if "/" in raw:
            try:
                candidates.append(resource_path("models", *raw.split("/")))
            except Exception:
                pass

    unique: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved not in unique:
            unique.append(resolved)
    return unique


def resolve_base_model_source(
    base_model_name_or_path: str,
    adapter_path: str | Path | None = None,
    settings: dict | None = None,
    status_callback: StatusCallback | None = None,
) -> str:
    """
    Return local full model folder when available; otherwise return the original
    Hugging Face id/name so Transformers can use the local HF cache.

    This function intentionally ignores PEFT adapter folders as base models.
    """
    settings = settings or {}

    for candidate in candidate_base_model_paths(base_model_name_or_path, adapter_path, settings):
        if not candidate.exists():
            continue
        if is_full_transformers_model_dir(candidate):
            _status(status_callback, f"Локальная full base model найдена: {candidate}")
            return str(candidate)
        if looks_like_peft_adapter_dir(candidate):
            _status(
                status_callback,
                "Найдена папка, похожая на LoRA/PEFT adapter, но это НЕ full base model; "
                f"пропускаю как базу: {candidate}",
            )

    if base_model_local_files_only(settings):
        _status(
            status_callback,
            "Локальная full base model не найдена. Будет использован Hugging Face cache по имени модели; скачивание отключено.",
        )
    else:
        _status(
            status_callback,
            "Локальная full base model не найдена. Разрешена загрузка из Hugging Face.",
        )
    return str(base_model_name_or_path)
