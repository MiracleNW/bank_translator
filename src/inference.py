from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Callable, Iterable

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

from src.model_resolver import base_model_local_files_only, resolve_base_model_source
from src.runtime_paths import resource_path
from src.torch_runtime import import_torch_early

torch = import_torch_early()

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]

def _read_active_model_name() -> str:
    active_model_file = resource_path("configs", "active_model.txt", must_exist=True)
    model_name = active_model_file.read_text(encoding="utf-8").strip()
    if not model_name:
        raise ValueError(f"Файл {active_model_file} пустой. Укажите имя папки модели.")
    return model_name


MODEL_NAME = _read_active_model_name()
MODEL_PATH = resource_path("models", MODEL_NAME)
SETTINGS_PATH = resource_path("configs", "translation_settings.json")


_tokenizer = None
_model = None
_device = None
_model_lock = threading.RLock()
_translation_cache: dict[str, str] = {}
_base_model_source: str | None = None
_cuda_strategy_used: str | None = None

_DEFAULT_SETTINGS = {
    "max_input_tokens": 512,
    "max_new_tokens": 160,
    "batch_size_cpu": 1,
    "batch_size_cuda": 4,
    "repetition_penalty": 1.05,
    "use_cache": True,
    "skip_without_latin": True,
    "cuda_strategy": "full_gpu",
    "allow_hf_download": False,
    "base_model_path": "",
    "base_model_candidates": [
        "models/yandex_gpt_base",
        "models/yandex_gpt_full",
        "models/YandexGPT-5-Lite-8B-pretrain",
        "models/yandex_gpt",
        "models/yandex/YandexGPT-5-Lite-8B-pretrain",
    ],
}


def _status(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _load_settings() -> dict:
    settings = dict(_DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                settings.update(loaded)
        except Exception:
            pass
    return settings


SETTINGS = _load_settings()


def _local_files_only() -> bool:
    return base_model_local_files_only(SETTINGS)


def _load_tokenizer(tokenizer_source: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        use_fast=True,
        local_files_only=_local_files_only(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cuda_strategy() -> str:
    value = str(SETTINGS.get("cuda_strategy", "full_gpu")).strip().lower()
    if value in {"auto", "auto_offload", "balanced", "offload"}:
        return "auto_offload"
    return "full_gpu"


def _model_dtype_for_device(device):
    if device.type == "cuda":
        return torch.float16
    return "auto"


def _from_pretrained_kwargs(device) -> dict:
    kwargs = {
        "torch_dtype": _model_dtype_for_device(device),
        "low_cpu_mem_usage": True,
        "local_files_only": _local_files_only(),
    }
    if device.type == "cuda":
        strategy = _cuda_strategy()
        if strategy == "full_gpu":
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["device_map"] = "auto"
    return kwargs


def _peft_from_pretrained_kwargs(device) -> dict:
    if device.type != "cuda":
        return {"local_files_only": _local_files_only()}
    if _cuda_strategy() == "full_gpu":
        return {"device_map": {"": 0}, "local_files_only": _local_files_only()}
    return {"device_map": "auto", "local_files_only": _local_files_only()}


def _model_input_device():
    """Return the device where input_ids must live for this model."""
    if _model is None:
        return _device or torch.device("cpu")

    try:
        embeddings = _model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    except Exception:
        pass

    try:
        return next(_model.parameters()).device
    except Exception:
        return _device or torch.device("cpu")


def _raise_better_model_load_error(exc: Exception, base_source: str | None = None) -> None:
    text = str(exc)
    lower = text.lower()

    if "out of memory" in lower or "cuda error: out of memory" in lower:
        raise RuntimeError(
            "Недостаточно VRAM видеокарты для полной загрузки модели на GPU.\n\n"
            "Что можно сделать:\n"
            "1. Закройте другие программы, которые используют видеокарту.\n"
            "2. Если видеокарта слабая, в configs/translation_settings.json поменяйте "
            '"cuda_strategy" на "auto_offload" — будет медленнее и может использовать RAM.\n'
            "3. Для быстрой работы 8B-модели нужна NVIDIA GPU с достаточным объёмом VRAM.\n\n"
            f"Оригинальная ошибка: {exc}"
        ) from exc

    if "local_files_only" in lower or "couldn't connect" in lower or "not the path to a directory" in lower:
        raise RuntimeError(
            "Полная базовая YandexGPT-модель не найдена локально и скачивание отключено.\n\n"
            "Приложение использует активную дообученную модель из configs/active_model.txt, "
            "но LoRA-адаптеру нужен полный base model YandexGPT.\n\n"
            "Важно: папка с adapter_config.json и adapter_model.safetensors — это LoRA/PEFT adapter, "
            "а не полная базовая модель. Полная базовая модель обычно содержит config.json "
            "и большие файлы весов model*.safetensors или pytorch_model*.bin.\n\n"
            "Положите полную базовую модель в одну из папок:\n"
            "- models/yandex_gpt_base\n"
            "- models/YandexGPT-5-Lite-8B-pretrain\n"
            "- models/yandex/YandexGPT-5-Lite-8B-pretrain\n\n"
            "Либо укажите путь в configs/translation_settings.json в поле base_model_path.\n\n"
            f"Пробовал источник: {base_source}\n"
            f"Оригинальная ошибка: {exc}"
        ) from exc

    raise exc


def _load_model(status_callback: StatusCallback | None = None):
    global _tokenizer, _model, _device, _base_model_source, _cuda_strategy_used

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            _status(status_callback, "Модель уже загружена.")
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Папка активной модели не найдена.\n"
                f"Ожидалось: {MODEL_PATH}\n"
                f"Имя модели берётся из configs/active_model.txt: {MODEL_NAME}"
            )

        _device = _select_device()
        _cuda_strategy_used = _cuda_strategy() if _device.type == "cuda" else "cpu"
        _status(status_callback, f"Устройство для модели: {_device.type.upper()}")
        if _device.type == "cuda":
            _status(status_callback, f"Стратегия CUDA: {_cuda_strategy_used}")
        _status(
            status_callback,
            "Режим базовой модели: локальные файлы/HF cache без скачивания"
            if _local_files_only()
            else "Режим базовой модели: локальные файлы или загрузка из Hugging Face разрешена",
        )

        adapter_config_path = MODEL_PATH / "adapter_config.json"
        pretrained_kwargs = _from_pretrained_kwargs(_device)

        try:
            if adapter_config_path.exists():
                _status(status_callback, f"Активная дообученная модель: {MODEL_NAME} (LoRA/PEFT adapter).")
                _status(status_callback, "Важно: сначала загружается базовая YandexGPT, затем поверх неё подключается активный bank_translation adapter.")
                _status(status_callback, "Читаю конфигурацию LoRA adapter...")
                peft_config = PeftConfig.from_pretrained(str(MODEL_PATH))
                base_model_name = peft_config.base_model_name_or_path
                _base_model_source = resolve_base_model_source(
                    base_model_name,
                    adapter_path=MODEL_PATH,
                    settings=SETTINGS,
                    status_callback=status_callback,
                )

                tokenizer_source = MODEL_PATH
                if not (MODEL_PATH / "tokenizer_config.json").exists():
                    tokenizer_source = _base_model_source

                _status(status_callback, "Загружаю tokenizer...")
                _tokenizer = _load_tokenizer(tokenizer_source)

                _status(status_callback, f"Загружаю базовую YandexGPT-модель: {_base_model_source}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    _base_model_source,
                    **pretrained_kwargs,
                )

                _status(status_callback, f"Подключаю дообученный adapter: {MODEL_NAME}...")
                _model = PeftModel.from_pretrained(
                    base_model,
                    str(MODEL_PATH),
                    **_peft_from_pretrained_kwargs(_device),
                )
            else:
                _base_model_source = str(MODEL_PATH)
                _status(status_callback, "Загружаю tokenizer...")
                _tokenizer = _load_tokenizer(MODEL_PATH)

                _status(status_callback, f"Загружаю полную модель: {MODEL_PATH}")
                _model = AutoModelForCausalLM.from_pretrained(
                    str(MODEL_PATH),
                    **pretrained_kwargs,
                )
        except Exception as exc:
            _raise_better_model_load_error(exc, _base_model_source)

        if _device.type != "cuda":
            _model.to(_device)

        try:
            _model.config.use_cache = bool(SETTINGS.get("use_cache", True))
        except Exception:
            pass

        _model.eval()
        torch.set_grad_enabled(False)

        input_device = _model_input_device()
        if _device.type == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                _status(status_callback, f"Модель загружена. GPU: {name}. Input device: {input_device}")
            except Exception:
                _status(status_callback, f"Модель загружена на GPU. Input device: {input_device}")
        else:
            _status(status_callback, "Модель загружена на CPU. Для 8B модели это будет медленно.")


def preload_model(status_callback: StatusCallback | None = None) -> None:
    """Load tokenizer/model once. Intended to run at application startup."""
    _load_model(status_callback=status_callback)


def is_model_loaded() -> bool:
    return _model is not None and _tokenizer is not None


def get_runtime_info() -> str:
    if _device is None:
        return "Модель ещё не загружена."
    if _device.type == "cuda":
        try:
            return (
                f"Модель готова. Устройство: GPU — {torch.cuda.get_device_name(0)}. "
                f"Стратегия: {_cuda_strategy_used}."
            )
        except Exception:
            return f"Модель готова. Устройство: GPU. Стратегия: {_cuda_strategy_used}."
    return "Модель готова. Устройство: CPU"


def _contains_latin_letters(text: str) -> bool:
    return any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)


def _make_prompt(text: str) -> str:
    return f"""### Instruction:
Ты профессиональный переводчик банковских и юридических документов.
Переведи текст с английского на русский, сохраняя официальный стиль и юридическую терминологию.

### Input:
{text}

### Response:
"""


def _clean_translation_text(text: str) -> str:
    """Remove prompt/template spillover from a generated translation.

    The model was trained with an Alpaca-like template:
    ### Instruction / ### Input / ### Response.  During long generation it can
    continue the training template after the answer.  That text must never be
    written back into DOCX.
    """
    if text is None:
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    if "### Response:" in text:
        text = text.split("### Response:")[-1]

    stop_markers = [
        "\n### Instruction:", "\n### Input:", "\n### Response:",
        "### Instruction:", "### Input:", "### Response:",
        "\nInstruction:", "\nInput:", "\nResponse:",
    ]
    cut_at = len(text)
    for marker in stop_markers:
        pos = text.find(marker)
        if pos != -1:
            cut_at = min(cut_at, pos)
    text = text[:cut_at]

    text = re.sub(r"^\s*(Ответ|Перевод|Translation|Response)\s*[:：-]\s*", "", text, flags=re.IGNORECASE)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in {"###", "### Instruction:", "### Input:", "### Response:"}:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _extract_response(decoded: str) -> str:
    return _clean_translation_text(decoded)


def _batch_size() -> int:
    if _device is not None and _device.type == "cuda":
        return max(1, int(SETTINGS.get("batch_size_cuda", 4)))
    return max(1, int(SETTINGS.get("batch_size_cpu", 1)))


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def translate_many(
    texts: list[str],
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> list[str]:
    global _tokenizer, _model, _device

    def _raise_if_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise InterruptedError("Перевод отменён пользователем.")

    _raise_if_cancelled()

    if _model is None or _tokenizer is None:
        _load_model(status_callback=status_callback)

    _raise_if_cancelled()

    if not texts:
        return []

    results: list[str | None] = [None] * len(texts)
    pending_indexes: list[int] = []
    pending_texts: list[str] = []

    for idx, text in enumerate(texts):
        _raise_if_cancelled()
        key = text.strip()
        if not key:
            results[idx] = text
        elif bool(SETTINGS.get("skip_without_latin", True)) and not _contains_latin_letters(key):
            results[idx] = text
            _translation_cache[key] = text
        elif key in _translation_cache:
            results[idx] = _translation_cache[key]
        else:
            pending_indexes.append(idx)
            pending_texts.append(text)

    completed = len(texts) - len(pending_texts)
    if progress_callback is not None:
        progress_callback(completed, len(texts))

    settings = SETTINGS
    batch_size = _batch_size()
    max_input_tokens = int(settings.get("max_input_tokens", 512))
    max_new_tokens = int(settings.get("max_new_tokens", 160))
    repetition_penalty = float(settings.get("repetition_penalty", 1.05))

    pending_total = len(pending_texts)
    if status_callback is not None and completed > 0:
        status_callback(f"Уже готово без повторной генерации: {completed} из {len(texts)} фрагментов (кэш/пропуск).")

    visible_done = 0
    pointer = 0
    for batch in _chunks(pending_texts, batch_size):
        _raise_if_cancelled()
        batch_indexes = pending_indexes[pointer:pointer + len(batch)]
        pointer += len(batch)

        if status_callback is not None:
            start = visible_done + 1
            end = visible_done + len(batch)
            input_device = _model_input_device()
            if pending_total == len(texts):
                status_callback(f"Перевожу фрагменты {start}-{end} из {len(texts)}... batch={len(batch)}, device={input_device}")
            else:
                status_callback(f"Перевожу новые фрагменты {start}-{end} из {pending_total} (всего в файле: {len(texts)})... batch={len(batch)}, device={input_device}")

        prompts = [_make_prompt(text) for text in batch]
        inputs = _tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )

        input_device = _model_input_device()
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with _model_lock:
            with torch.inference_mode():
                out = _model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=_tokenizer.eos_token_id,
                    pad_token_id=_tokenizer.eos_token_id,
                    use_cache=bool(settings.get("use_cache", True)),
                )

        _raise_if_cancelled()

        prompt_token_count = int(inputs["input_ids"].shape[1])
        generated_only = out[:, prompt_token_count:]
        decoded_items = _tokenizer.batch_decode(generated_only.detach().cpu(), skip_special_tokens=True)

        for idx, source_text, decoded in zip(batch_indexes, batch, decoded_items):
            translation = _extract_response(decoded)
            results[idx] = translation
            _translation_cache[source_text.strip()] = translation

        visible_done += len(batch)
        completed += len(batch)
        if progress_callback is not None:
            progress_callback(completed, len(texts))

    _raise_if_cancelled()
    return [item if item is not None else "" for item in results]


def translate(text: str) -> str:
    return translate_many([text])[0]
