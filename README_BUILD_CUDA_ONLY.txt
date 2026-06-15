CUDA build only

1. Keep models in the project folder:
   models/bank_translation_20260507_1747
   models/<full_base_yandexgpt_folder>

2. Run only:
   build_cuda_exe.bat

3. The EXE is created here:
   dist/BankTranslator/BankTranslator.exe

4. The EXE searches models first in the project root models folder, not in dist.

5. After translation, the app asks where to save the translated DOCX.
