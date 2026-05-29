@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
if "%1"=="" (
    python scripts\test_inference.py
) else (
    python scripts\test_inference.py %*
)
