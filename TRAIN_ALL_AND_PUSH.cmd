@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv-rl\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set "PARALLEL_TECHNIQUES=2"
set "DEVICE=auto"

echo Training all GalagAI brain families, assembling the selector, and pushing master + gh-pages.
echo Python: %PYTHON_EXE%
echo Device: %DEVICE%
echo Parallel technique jobs: %PARALLEL_TECHNIQUES%
echo.

"%PYTHON_EXE%" tools\train_all.py ^
  --device %DEVICE% ^
  --target-rounds 4 ^
  --pilot-warmup-generations 3 ^
  --enemy-warmup-generations 1 ^
  --curriculum-waves 3 ^
  --candidate-spawns 2 ^
  --train-workers 1 ^
  --eval-workers 4 ^
  --parallel-techniques %PARALLEL_TECHNIQUES% ^
  --deploy

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Training and push finished.
) else (
  echo Training or push failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
