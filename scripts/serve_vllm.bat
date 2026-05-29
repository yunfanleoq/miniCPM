@echo off
cd /d "%~dp0.."
docker compose -f docker-compose.vllm.yml up -d
echo.
echo vLLM server starting on http://localhost:8000
echo Check logs: docker logs -f minicpm5-vllm
echo Test API:   python scripts\test_vllm_api.py --prompt "你好"
