#!/usr/bin/env python3
"""
Targeted Dataset Expansion Script for TypeScript & Python.
Expands dataset_train.jsonl and dataset_val.jsonl with targeted subtle mutations
while preserving 100% symbolic gate compliance and exact 50/50 class balance.
Updates data/code_oracle_dataset.zip.
"""

import json
import random
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from code_oracle.dataset import DatasetGenerator, DatasetRecord
from code_oracle.linearizer import estimate_tokens


def main():
    data_dir = REPO_ROOT / "data"
    train_file = data_dir / "dataset_train.jsonl"
    val_file = data_dir / "dataset_val.jsonl"
    heldout_file = data_dir / "dataset_heldout_eval.jsonl"
    zip_file = data_dir / "code_oracle_dataset.zip"

    print("[*] Loading existing datasets...")
    with open(train_file, "r", encoding="utf-8") as f:
        train_records = [DatasetRecord.from_dict(json.loads(line)) for line in f if line.strip()]
    with open(val_file, "r", encoding="utf-8") as f:
        val_records = [DatasetRecord.from_dict(json.loads(line)) for line in f if line.strip()]

    print(f"    - Existing train samples: {len(train_records)}")
    print(f"    - Existing val samples:   {len(val_records)}")

    # Initialize generator
    generator = DatasetGenerator(languages=["typescript", "python"], seed=42)

    print("[*] Generating targeted TypeScript & Python subtle mutation samples...")
    # 400 TS samples (200 pass, 200 reject across 4 types -> 50 per type)
    # 400 Py samples (200 pass, 200 reject across 4 types -> 50 per type)
    expanded_train, expanded_val = generator.expand_dataset(
        train_records=train_records,
        val_records=val_records,
        num_ts_samples=400,
        num_py_samples=400,
        val_ratio=0.2,
    )

    print(f"[+] Dataset expansion complete:")
    print(f"    - Expanded train samples: {len(expanded_train)}")
    print(f"    - Expanded val samples:   {len(expanded_val)}")

    # Verify all records conform to requirements
    for name, ds in [("train", expanded_train), ("val", expanded_val)]:
        pass_n = sum(1 for r in ds if r.label == 1)
        reject_n = sum(1 for r in ds if r.label == 0)
        assert pass_n == reject_n, f"Class imbalance in {name}: {pass_n} vs {reject_n}"
        for idx, r in enumerate(ds):
            assert r.symbolic_gate_passed is True, f"Gate failed in {name} #{idx}"
            tok = estimate_tokens(r.input_dsl)
            assert tok <= 400, f"Token limit exceeded in {name} #{idx}: {tok}"

    print("[*] Writing expanded datasets to disk...")
    with open(train_file, "w", encoding="utf-8") as f:
        for r in expanded_train:
            f.write(json.dumps(r.to_dict()) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in expanded_val:
            f.write(json.dumps(r.to_dict()) + "\n")

    print("[*] Updating data/code_oracle_dataset.zip...")
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(train_file, arcname="dataset_train.jsonl")
        zf.write(val_file, arcname="dataset_val.jsonl")
        if heldout_file.exists():
            zf.write(heldout_file, arcname="dataset_heldout_eval.jsonl")

    print(f"[✓] Zip archive successfully updated: {zip_file} ({zip_file.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
