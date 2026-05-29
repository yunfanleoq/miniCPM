"""LoRA fine-tuning for MiniCPM5-1B using TRL + PEFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utf8_console import configure_utf8_stdio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "models" / "MiniCPM5-1B"
DEFAULT_DATA = PROJECT_ROOT / "data" / "xiaozhi_assistant_500.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "runs" / "minicpm5_lora"

TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def load_jsonl(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append({"messages": json.loads(line)["messages"]})
    return Dataset.from_list(rows)


def main() -> None:
    configure_utf8_stdio()

    parser = argparse.ArgumentParser(description="LoRA fine-tune MiniCPM5-1B")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model : {args.base_model}")
    print(f"Data       : {args.data}")
    print(f"Output     : {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = TRAIN_CHAT_TEMPLATE

    dataset = load_jsonl(Path(args.data))
    print(f"Samples    : {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to=["tensorboard"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
