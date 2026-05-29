@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001 >nul
python scripts\finetune_lora.py %*