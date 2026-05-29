@echo off
curl.exe -s http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"MiniCPM5-1B\",\"messages\":[{\"role\":\"user\",\"content\":\"%*\"}],\"temperature\":0.7,\"top_p\":0.95,\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
