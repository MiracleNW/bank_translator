This uploaded yandex_gpt folder looks like a LoRA/PEFT adapter, not a full base model.
It has adapter_config.json and adapter_model.safetensors, but no config.json and no full model weight shards.

The app will not use this folder as the full YandexGPT base model unless it contains a valid full Transformers model layout.
For the base model, place the full YandexGPT files into models/yandex_gpt_base or set base_model_path in configs/translation_settings.json.
