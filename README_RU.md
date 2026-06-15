# BankTranslator — CUDA EXE, модели остаются вне dist

## Главное

В этой версии оставлен только один bat-файл:

```text
build_cuda_exe.bat
```

Он создаёт локальный `.venv`, ставит CUDA PyTorch и собирает EXE.

## Где должны лежать модели

Модели остаются в обычной папке проекта:

```text
models\bank_translation_20260507_1747
```

EXE создаётся здесь:

```text
dist\BankTranslator\BankTranslator.exe
```

При запуске из `dist\BankTranslator` приложение сначала ищет ресурсы не в `dist`, а в корне проекта:

```text
<project_root>\models
<project_root>\configs
```

То есть тяжёлые модели больше не копируются в `dist\BankTranslator\models`.

## Как работает LoRA-модель

Активная модель указана в:

```text
configs\active_model.txt
```

Сейчас активная модель:

```text
bank_translation_20260507_1747
```

Это LoRA/PEFT adapter, а не самостоятельная full-модель. Поэтому порядок загрузки такой:

1. загрузить полную базовую YandexGPT-модель;
2. подключить поверх неё adapter `bank_translation_20260507_1747`;
3. переводить уже через эту дообученную связку.

Лог `Загружаю базовую YandexGPT-модель...` не означает, что перевод идёт чистой YandexGPT. Это обязательный первый этап перед подключением твоего adapter.

## Важное про папку yandex_gpt

Папка `models\yandex_gpt` будет использована как базовая модель только если внутри есть:

```text
config.json
model*.safetensors
```

или:

```text
config.json
pytorch_model*.bin
```

Если внутри только `adapter_config.json` и `adapter_model.safetensors`, это тоже LoRA adapter, а не полная базовая модель. Такая папка не может заменить base model.

## Сборка

Запустить:

```text
build_cuda_exe.bat
```

После сборки:

```text
dist\BankTranslator\BankTranslator.exe
```

## Сохранение результата

После перевода приложение спрашивает, куда сохранить переведённый DOCX. Папки `input_docx` и `output_docx` больше не используются и не создаются специально при сборке.
