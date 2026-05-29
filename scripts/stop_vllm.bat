@echo off
cd /d "%~dp0.."
docker compose -f docker-compose.vllm.yml down
echo vLLM server stopped.
