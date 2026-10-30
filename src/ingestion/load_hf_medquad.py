"""
src/ingestion/load_hf_medquad.py

Downloads the MedQuAD dataset from HuggingFace and verifies its structure.

Uses the hf-mirror.com endpoint to avoid connectivity issues with the
default huggingface.co endpoint.

Usage:
    python src/ingestion/load_hf_medquad.py
"""

import os

# Must be set BEFORE importing datasets/huggingface_hub so that all
# HTTP requests go through the mirror.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "mukulb/clustered_MEDQUAD_dataset_with_groups"


def main() -> None:
    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
    print(f"Loading dataset: {DATASET_ID}")
    print("(This may take a minute on first run while the dataset is downloaded and cached.)\n")

    dataset = load_dataset(DATASET_ID)

    # ── Structure ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("DATASET STRUCTURE")
    print("=" * 60)
    print(dataset)
    print()

    # Determine the split to inspect (prefer 'train', fall back to first)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    split = dataset[split_name]

    print(f"Split used for inspection : {split_name!r}")
    print(f"Number of rows            : {len(split):,}")
    print(f"Column names              : {split.column_names}")
    print()

    # ── First record ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("FIRST RECORD")
    print("=" * 60)
    first = split[0]
    for key, value in first.items():
        # Truncate long string values for readability
        if isinstance(value, str) and len(value) > 200:
            display = value[:200] + "  [... truncated]"
        else:
            display = value
        print(f"  {key}: {display!r}")

    print()
    print("Dataset loaded successfully.")


if __name__ == "__main__":
    main()
