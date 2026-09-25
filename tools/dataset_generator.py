#!/usr/bin/env python3
"""
CLI Tool for Multi-Language Dataset Mining & Synthetic Generation.
Generates balanced JSONL datasets for Laya ModernBERT decision head training.
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_oracle.dataset import DatasetGenerator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine and generate balanced multi-language training datasets for Code Oracle / Laya ModernBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Path to an existing code repository to mine for call graphs and mutations.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./dataset_output"),
        help="Directory to emit dataset_train.jsonl and dataset_val.jsonl.",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=100,
        help="Target total number of balanced positive/negative samples.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of generated dataset allocated to validation set.",
    )
    parser.add_argument(
        "--languages",
        "-l",
        type=str,
        default="python,typescript,go,rust",
        help="Comma-separated Tier 1 languages to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--positive-label",
        type=int,
        default=1,
        help="Integer label for positive (PASS) instances.",
    )
    parser.add_argument(
        "--negative-label",
        type=int,
        default=0,
        help="Integer label for negative (REJECT) instances.",
    )
    parser.add_argument(
        "--filter-symbolic-gate",
        action="store_true",
        default=False,
        help="Filter out mutations that fail symbolic gate (keep only gray-area hard negatives).",
    )
    parser.add_argument(
        "--include-subtle",
        action="store_true",
        default=True,
        help="Include subtle gray-area semantic mutations across risk taxonomy.",
    )
    parser.add_argument(
        "--no-subtle",
        dest="include_subtle",
        action="store_false",
        help="Disable subtle gray-area semantic mutations.",
    )

    args = parser.parse_args()

    lang_list = [l.strip().lower() for l in args.languages.split(",") if l.strip()]

    print(f"[*] Initializing DatasetGenerator with languages: {', '.join(lang_list)}")
    generator = DatasetGenerator(
        languages=lang_list,
        seed=args.seed,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        filter_symbolic_gate=args.filter_symbolic_gate,
    )

    print(f"[*] Generating dataset (target: {args.num_samples} samples, val_ratio: {args.val_ratio})...")
    train_count, val_count = generator.generate_and_export(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        val_ratio=args.val_ratio,
        repo_path=args.repo,
        include_subtle=args.include_subtle,
    )

    print(f"[+] Dataset successfully generated at {args.output_dir}:")
    print(f"    - dataset_train.jsonl: {train_count} samples")
    print(f"    - dataset_val.jsonl:   {val_count} samples")
    print(f"    - Total:               {train_count + val_count} samples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
