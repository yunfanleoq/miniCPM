"""Quick GPU inference smoke test for MiniCPM5-1B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utf8_console import configure_utf8_stdio

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "MiniCPM5-1B"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="MiniCPM5-1B inference smoke test")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Local model path or HuggingFace model id",
    )
    parser.add_argument(
        "--prompt",
        default="用一句话介绍你自己。",
        help="User prompt",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from: {args.model_path}")
    print(f"Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
    ).eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7 if not args.enable_thinking else 0.9,
            top_p=0.95,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
