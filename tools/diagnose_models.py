from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_resolver import (
    candidate_base_model_paths,
    is_full_transformers_model_dir,
    looks_like_peft_adapter_dir,
    resolve_base_model_source,
)


def main() -> int:
    print("=== BankTranslator model diagnostics ===")
    active_file = ROOT / "configs" / "active_model.txt"
    settings_file = ROOT / "configs" / "translation_settings.json"
    if not active_file.exists():
        print(f"ERROR: missing {active_file}")
        return 1

    active = active_file.read_text(encoding="utf-8").strip()
    print(f"Active model: {active}")
    active_path = ROOT / "models" / active
    print(f"Active model path: {active_path}")
    print(f"Active exists: {active_path.exists()}")

    settings = {}
    if settings_file.exists():
        settings = json.loads(settings_file.read_text(encoding="utf-8"))

    if not active_path.exists():
        return 1

    if (active_path / "adapter_config.json").exists():
        adapter_config = json.loads((active_path / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = adapter_config.get("base_model_name_or_path", "")
        print(f"Adapter base_model_name_or_path: {base_name}")
        print("\nCandidate local base model folders:")
        for candidate in candidate_base_model_paths(base_name, active_path, settings):
            exists = candidate.exists()
            full = is_full_transformers_model_dir(candidate) if exists else False
            adapter = looks_like_peft_adapter_dir(candidate) if exists else False
            if exists:
                kind = "FULL BASE MODEL" if full else "PEFT/LoRA ADAPTER" if adapter else "not a full model"
            else:
                kind = "missing"
            print(f"- {candidate} -> {kind}")
        print("\nResolved base source:")
        print(resolve_base_model_source(base_name, active_path, settings, status_callback=print))
    else:
        print("Active model looks like a full model folder, not LoRA adapter.")
        print(f"Full model valid: {is_full_transformers_model_dir(active_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
