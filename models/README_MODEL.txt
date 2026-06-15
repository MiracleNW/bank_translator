Active fine-tuned adapter is selected in configs/active_model.txt.

For bank translation the active adapter should be:
models/bank_translation_20260507_1747

A LoRA/PEFT adapter folder contains adapter_config.json and adapter_model.safetensors.
It is not a full base model.

The full YandexGPT base model should be placed in one of these folders:
models/yandex_gpt_base
models/yandex_gpt_full
models/YandexGPT-5-Lite-8B-pretrain
models/yandex/YandexGPT-5-Lite-8B-pretrain

A full Transformers base model folder must contain config.json and model*.safetensors or pytorch_model*.bin files.
