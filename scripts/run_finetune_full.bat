@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001 >nul
echo [1/2] Generating training data...
python scripts\generate_training_data.py
echo.
echo [2/2] Starting LoRA fine-tuning...
python scripts\finetune_lora.py %*
