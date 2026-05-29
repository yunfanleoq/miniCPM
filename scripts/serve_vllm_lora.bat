@echo off
cd /d "%~dp0.."
docker compose -f docker-compose.vllm-lora.yml up -d
echo.
echo vLLM + LoRA server starting on http://localhost:8001
echo LoRA module name: xiaozhi
echo Test: python scripts\test_vllm_lora_api.py
