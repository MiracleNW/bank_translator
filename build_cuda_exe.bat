@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_NAME=BankTranslator"
set "ENTRY=app\app.py"
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "BASE_PY=python"
set "TORCH_VERSION=2.8.0"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "BANKTRANSLATOR_REQUIRE_CUDA=1"

echo ==============================================
echo Build CUDA EXE: %APP_NAME%
echo ==============================================
echo Project folder: %CD%
echo Models folder used at runtime: %CD%\models
echo VENV folder: %CD%\%VENV_DIR%
echo PyTorch: %TORCH_VERSION% CUDA 12.8
echo.

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found. Install 64-bit Python 3.10, 3.11 or 3.12 and add it to PATH.
        pause
        exit /b 1
    )
    set "BASE_PY=py -3"
)

if not exist "%VENV_PY%" (
    echo [1/6] Creating local virtual environment...
    %BASE_PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo [1/6] Using existing local virtual environment.
)

if not exist "%VENV_PY%" (
    echo ERROR: venv Python was not found: %VENV_PY%
    pause
    exit /b 1
)

"%VENV_PY%" -c "import sys, struct; raise SystemExit(0 if ((3,10) <= sys.version_info[:2] <= (3,12) and struct.calcsize('P')*8 == 64) else 1)"
if errorlevel 1 (
    echo ERROR: Use 64-bit Python 3.10, 3.11 or 3.12.
    pause
    exit /b 1
)

if not exist "%ENTRY%" (
    echo ERROR: Entry file not found: %ENTRY%
    pause
    exit /b 1
)

if not exist "configs\active_model.txt" (
    echo ERROR: configs\active_model.txt not found.
    pause
    exit /b 1
)

set "ACTIVE_MODEL="
for /f "usebackq delims=" %%A in ("configs\active_model.txt") do (
    if not defined ACTIVE_MODEL set "ACTIVE_MODEL=%%A"
)

if not defined ACTIVE_MODEL (
    echo ERROR: configs\active_model.txt is empty.
    pause
    exit /b 1
)

echo Active adapter model: %ACTIVE_MODEL%
if exist "models\%ACTIVE_MODEL%\" (
    echo Active adapter folder found: models\%ACTIVE_MODEL%
) else (
    echo ERROR: Active adapter folder was not found: models\%ACTIVE_MODEL%
    pause
    exit /b 1
)

echo.
echo [2/6] Installing dependencies into local venv...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :pip_error

echo Installing CUDA PyTorch %TORCH_VERSION% from: %TORCH_INDEX%
"%VENV_PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
"%VENV_PY%" -m pip install --no-cache-dir --index-url "%TORCH_INDEX%" "torch==%TORCH_VERSION%"
if errorlevel 1 goto :pip_error

"%VENV_PY%" -m pip install -r requirements-build.txt
if errorlevel 1 goto :pip_error

echo.
echo [3/6] Running PyTorch smoke test before build...
"%VENV_PY%" tools\smoke_test_torch.py
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch CUDA failed inside .venv before PyInstaller.
    pause
    exit /b 1
)

echo.
echo [4/6] Cleaning old build folders...
if exist "build" rmdir /s /q "build"
if exist "build_temp" rmdir /s /q "build_temp"
if exist "dist\%APP_NAME%" rmdir /s /q "dist\%APP_NAME%"
if not exist "build_temp" mkdir "build_temp"

echo.
echo [5/6] Running PyInstaller from local venv...
"%VENV_PY%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --noupx ^
    --name "%APP_NAME%" ^
    --paths "%CD%" ^
    --specpath "build_temp" ^
    --workpath "build" ^
    --distpath "dist" ^
    --runtime-hook "tools\pyinstaller_rth_torch_dlls.py" ^
    --hidden-import "torch" ^
    --hidden-import "torch._C" ^
    --hidden-import "transformers" ^
    --hidden-import "peft" ^
    --hidden-import "safetensors" ^
    --hidden-import "tokenizers" ^
    --hidden-import "huggingface_hub" ^
    --hidden-import "accelerate" ^
    --hidden-import "sentencepiece" ^
    --hidden-import "protobuf" ^
    --hidden-import "docx" ^
    --hidden-import "src.model_resolver" ^
    --hidden-import "PyQt5.QtCore" ^
    --hidden-import "PyQt5.QtWidgets" ^
    --hidden-import "PyQt5.QtGui" ^
    --collect-all "torch" ^
    --collect-submodules "transformers" ^
    --collect-submodules "peft" ^
    --collect-submodules "tokenizers" ^
    --collect-submodules "huggingface_hub" ^
    --collect-submodules "accelerate" ^
    --collect-submodules "safetensors" ^
    --collect-data "transformers" ^
    --collect-data "huggingface_hub" ^
    --copy-metadata "torch" ^
    --copy-metadata "transformers" ^
    --copy-metadata "peft" ^
    --copy-metadata "tokenizers" ^
    --copy-metadata "huggingface_hub" ^
    --copy-metadata "accelerate" ^
    --copy-metadata "safetensors" ^
    "%ENTRY%"

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)

if not exist "dist\%APP_NAME%\%APP_NAME%.exe" (
    echo ERROR: EXE was not created: dist\%APP_NAME%\%APP_NAME%.exe
    pause
    exit /b 1
)

echo.
echo [6/6] Copying configs only. Models stay in project_root\models ...
"%VENV_PY%" tools\copy_runtime_resources.py "dist\%APP_NAME%"
if errorlevel 1 (
    echo ERROR: Failed to copy configs next to EXE.
    pause
    exit /b 1
)

if exist "build" rmdir /s /q "build"
if exist "build_temp" rmdir /s /q "build_temp"

echo.
echo ==============================================
echo DONE
echo EXE: dist\%APP_NAME%\%APP_NAME%.exe
echo Runtime models folder: %CD%\models
echo Active adapter: models\%ACTIVE_MODEL%
echo Local venv used: %VENV_DIR%
echo ==============================================
pause
exit /b 0

:pip_error
echo.
echo ERROR: Dependency installation failed inside .venv.
echo Check internet connection, Python version, and pip output above.
pause
exit /b 1
