"""Test LoRA adapter inference with transformers + PEFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utf8_console import configure_utf8_stdio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "models" / "MiniCPM5-1B"
DEFAULT_ADAPTER = PROJECT_ROOT / "runs" / "minicpm5_lora" / "adapter_final"


def main() -> None:
    configure_utf8_stdio()

    parser = argparse.ArgumentParser(description="Test LoRA adapter inference")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--prompt", default="你好，你是谁？")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Base   : {args.base_model}")
    print(f"Adapter: {args.adapter}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter).eval()

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
    print("\n=== LoRA Response ===")
    print(response)


if __name__ == "__main__":
    main()
