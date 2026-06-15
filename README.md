# BankTranslator — CUDA EXE, модели остаются вне dist

## Главное


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

## Как работает LoRA-модель

Активная модель указана в:

```text
configs\active_model.txt
```

Это LoRA/PEFT adapter, а не самостоятельная full-модель. Поэтому порядок загрузки такой:

1. загрузить полную базовую YandexGPT-модель;
2. подключить поверх неё adapter `bank_translation_20260507_1747`;
3. переводить уже через эту дообученную связку.


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

После перевода приложение спрашивает, куда сохранить переведённый DOCX.
