from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_resolver import base_model_local_files_only, resolve_base_model_source
from src.utils.utils import load_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_tokenize_fn(tokenizer, max_length: int):
    def tokenize(example):
        prompt = f"""### Instruction:
Ты профессиональный переводчик банковских и юридических документов.
Переведи текст с английского на русский, сохраняя официальный стиль и юридическую терминологию.

### Input:
{example['input']}

### Response:
"""

        answer = example["output"] + tokenizer.eos_token
        full_text = prompt + answer

        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
        )

        labels = tokens["input_ids"].copy()
        prompt_length = len(prompt_tokens["input_ids"])
        labels[:prompt_length] = [-100] * prompt_length

        tokens["labels"] = labels
        return tokens

    return tokenize


def load_data(path: Path, tokenize_fn):
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    df = pd.read_csv(path).dropna()
    if df.empty:
        raise ValueError(f"Файл датасета пустой после dropna(): {path}")

    required_columns = {"instruction", "input", "output"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В датасете {path} нет колонок: {', '.join(sorted(missing))}")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
    )
    return dataset


def _model_settings_from_train_config(config: dict) -> dict:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    return {
        "allow_hf_download": bool(model_cfg.get("allow_hf_download", False)),
        "base_model_path": str(model_cfg.get("base_model_path", "") or ""),
        "base_model_candidates": model_cfg.get("base_model_candidates", []) or [],
    }


def main():
    config_path = BASE_DIR / "configs" / "train_config.yaml"
    config = load_config(config_path)

    requested_model_name = config["model"]["name"]
    model_settings = _model_settings_from_train_config(config)
    model_name = resolve_base_model_source(
        requested_model_name,
        adapter_path=None,
        settings=model_settings,
        status_callback=print,
    )
    local_only = base_model_local_files_only(model_settings)
    max_length = int(config["data"]["max_length"])

    train_path = BASE_DIR / config["data"].get("train_path", "data/train.csv")
    val_path = BASE_DIR / config["data"].get("val_path", "data/val.csv")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = "bank_translation"
    model_dir = BASE_DIR / "models" / f"{experiment_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Модель будет сохранена в: {model_dir}")
    print(f"Base model source: {model_name}")
    print(f"Local files only:  {local_only}")
    print(f"Train CSV: {train_path}")
    print(f"Val CSV:   {val_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = build_tokenize_fn(tokenizer, max_length)
    train_dataset = load_data(train_path, tokenize_fn)
    val_dataset = load_data(val_path, tokenize_fn)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=local_only,
    )

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["grad_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["epochs"],
        warmup_steps=100,
        fp16=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(model_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    active_model_file = BASE_DIR / "configs" / "active_model.txt"
    active_model_file.write_text(model_dir.name, encoding="utf-8")

    print(f"\nTraining complete! Model saved to: {model_dir}")
    print(f"Active model updated: {active_model_file} -> {model_dir.name}")


if __name__ == "__main__":
    main()
