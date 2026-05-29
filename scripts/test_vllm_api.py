"""Test vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utf8_console import configure_utf8_stdio

DEFAULT_URL = "http://localhost:8000/v1/chat/completions"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Test vLLM chat completions API")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt", default="1+1=?")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    payload = {
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.9 if args.enable_thinking else 0.7,
        "top_p": 0.95,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }

    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== Answer ===")
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
