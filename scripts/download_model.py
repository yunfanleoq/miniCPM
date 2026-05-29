"""Download MiniCPM5-1B model weights to local models/ directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parents[1] / "models" / "MiniCPM5-1B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MiniCPM5-1B model")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace model id")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help="Local save path")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"Downloading {args.model_id} -> {local_dir}")
    print(f"HF endpoint: {endpoint}")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(local_dir),
    )

    print(f"Done. Model saved to: {local_dir}")


if __name__ == "__main__":
    main()
