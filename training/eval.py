from __future__ import annotations

import json
import sys
from pathlib import Path

import evaluate
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_resolver import base_model_local_files_only, resolve_base_model_source


ACTIVE_MODEL_FILE = BASE_DIR / "configs" / "active_model.txt"
MODEL_NAME = ACTIVE_MODEL_FILE.read_text(encoding="utf-8").strip()
MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
TEST_PATH = BASE_DIR / "data" / "test.csv"
SETTINGS_PATH = BASE_DIR / "configs" / "translation_settings.json"
MAX_SAMPLES = 5

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Папка активной модели не найдена: {MODEL_PATH}")

settings = {}
if SETTINGS_PATH.exists():
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        settings = {}

adapter_config_path = MODEL_PATH / "adapter_config.json"


def _input_device(model):
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tokenizer + Model
local_only = base_model_local_files_only(settings)

if adapter_config_path.exists():
    peft_config = PeftConfig.from_pretrained(str(MODEL_PATH))
    BASE_MODEL = resolve_base_model_source(
        peft_config.base_model_name_or_path,
        adapter_path=MODEL_PATH,
        settings=settings,
        status_callback=print,
    )

    tokenizer_source = MODEL_PATH if (MODEL_PATH / "tokenizer_config.json").exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=local_only)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=local_only,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        )
        model.to("cpu")

    model = PeftModel.from_pretrained(model, str(MODEL_PATH), local_files_only=local_only)
else:
    BASE_MODEL = str(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )
    if not torch.cuda.is_available():
        model.to("cpu")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")


def generate(text):
    prompt = f"""### Instruction:
Ты профессиональный переводчик банковских и юридических документов.
Переведи текст с английского на русский, сохраняя официальный стиль и юридическую терминологию.

### Input:
{text}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(out[0].detach().cpu(), skip_special_tokens=True)
    result = result.split("### Response:")[-1].strip()
    return result


def main():
    df = pd.read_csv(TEST_PATH).dropna()
    df = df.head(MAX_SAMPLES)

    preds = []
    refs = []

    print("\n===== START EVALUATION =====\n")
    print(f"Active model: {MODEL_PATH}")
    print(f"Base model:   {BASE_MODEL}")
    print(f"Local only:   {local_only}")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        pred = generate(row["input"])
        ref = row["output"]

        preds.append(pred)
        refs.append(ref)

        print("\n" + "=" * 60)
        print(f"EXAMPLE {i + 1}")
        print("\nINPUT:")
        print(row["input"])
        print("\nPREDICT:")
        print(pred)
        print("\nREFERENCE:")
        print(ref)
        print("=" * 60)

    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_score = chrf.compute(predictions=preds, references=refs)

    print("\n\n===== FINAL RESULTS =====")
    print(f"BLEU: {bleu_score['score']:.2f}")
    print(f"chrF: {chrf_score['score']:.2f}")


if __name__ == "__main__":
    main()
