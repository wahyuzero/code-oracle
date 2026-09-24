#!/usr/bin/env python3
"""
CLI Tool for Dataset Verification and Quality Inspection.

Verifies:
1. JSONL Schema & Data Formatting Integrity
2. Class Distribution (PASS / REJECT balance)
3. Language Breakdown (Python, TypeScript, Go, Rust representation)
4. Mutation Category Distribution
5. Token Length Guarantees (strictly < 400 tokens matching ModernBERT context window)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to sys.path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from code_oracle.linearizer import estimate_tokens

VALID_CATEGORIES = {
    "clean_pass",
    "arity_breaking",
    "keyword_changes",
    "circular_imports",
    "deleted_symbol",
}

VALID_LANGUAGES = {"python", "typescript", "go", "rust"}


def inspect_file(
    file_path: Path,
    max_token_limit: int = 400,
) -> Dict[str, Any]:
    """
    Inspect a single JSONL dataset file.
    Returns structured metrics and a list of detected violations.
    """
    file_path = file_path.resolve()
    if not file_path.exists():
        return {
            "file": str(file_path),
            "error": f"File does not exist: {file_path}",
            "passed": False,
        }

    total_records = 0
    violations: List[str] = []
    labels_count: Dict[int, int] = {0: 0, 1: 0}
    languages_count: Dict[str, int] = {}
    categories_count: Dict[str, int] = {}
    token_lengths: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as err:
                violations.append(f"Line {idx}: Invalid JSON syntax ({err})")
                continue

            total_records += 1

            # 1. Schema Validation
            required_keys = {"input_dsl", "label", "risk_score", "category", "language"}
            missing_keys = required_keys - set(record.keys())
            if missing_keys:
                violations.append(f"Line {idx}: Missing schema keys: {sorted(list(missing_keys))}")

            # 2. Label Check
            lbl = record.get("label")
            if lbl not in (0, 1):
                violations.append(f"Line {idx}: Invalid label {lbl} (must be 0 or 1)")
            else:
                labels_count[lbl] = labels_count.get(lbl, 0) + 1

            # 3. Risk Score Check
            risk = record.get("risk_score")
            if not isinstance(risk, (int, float)) or not (0.0 <= risk <= 1.0):
                violations.append(f"Line {idx}: Invalid risk_score {risk} (must be 0.0-1.0)")
            elif lbl == 1 and risk > 0.25:
                violations.append(f"Line {idx}: Positive sample has excessive risk_score ({risk})")
            elif lbl == 0 and risk < 0.75:
                violations.append(f"Line {idx}: Negative sample has insufficient risk_score ({risk})")

            # 4. Category Check
            cat = record.get("category")
            if cat not in VALID_CATEGORIES:
                violations.append(f"Line {idx}: Invalid category '{cat}'")
            else:
                categories_count[cat] = categories_count.get(cat, 0) + 1

            # 5. Language Check
            lang = record.get("language")
            if lang not in VALID_LANGUAGES:
                violations.append(f"Line {idx}: Invalid language '{lang}'")
            else:
                languages_count[lang] = languages_count.get(lang, 0) + 1

            # 6. Token Count Check
            dsl = record.get("input_dsl", "")
            if not isinstance(dsl, str) or not dsl.strip():
                violations.append(f"Line {idx}: Empty or invalid input_dsl string")
                token_len = 0
            else:
                token_len = estimate_tokens(dsl)
                token_lengths.append(token_len)
                if token_len > max_token_limit:
                    violations.append(
                        f"Line {idx}: Token length {token_len} exceeds max limit {max_token_limit}"
                    )

    # Compute Token Stats
    token_stats = {
        "min": min(token_lengths) if token_lengths else 0,
        "max": max(token_lengths) if token_lengths else 0,
        "mean": round(statistics.mean(token_lengths), 1) if token_lengths else 0.0,
        "median": statistics.median(token_lengths) if token_lengths else 0,
    }

    # Class Balance Evaluation
    pos_count = labels_count.get(1, 0)
    neg_count = labels_count.get(0, 0)
    balance_delta = abs(pos_count - neg_count)
    if total_records > 0 and balance_delta > max(2, total_records * 0.05):
        violations.append(
            f"Class imbalance detected: {pos_count} PASS vs {neg_count} REJECT (delta: {balance_delta})"
        )

    passed = len(violations) == 0

    return {
        "file": str(file_path),
        "total_records": total_records,
        "labels": labels_count,
        "languages": languages_count,
        "categories": categories_count,
        "token_stats": token_stats,
        "violations": violations[:20],  # cap reporting violations
        "violations_count": len(violations),
        "passed": passed,
    }


def format_report_text(summary: Dict[str, Any]) -> str:
    """Format inspection summary into clear, readable terminal output."""
    lines = []
    lines.append("================================================================================")
    lines.append(f" DATASET QUALITY INSPECTION REPORT: {Path(summary['file']).name}")
    lines.append("================================================================================")
    lines.append(f"File Path:         {summary['file']}")
    lines.append(f"Total Samples:     {summary['total_records']}")
    lines.append(f"Validation Status: {'[✓] PASSED' if summary['passed'] else '[✗] FAILED'}")

    # Labels
    pos = summary["labels"].get(1, 0)
    neg = summary["labels"].get(0, 0)
    tot = max(1, summary["total_records"])
    lines.append("\nClass Balance:")
    lines.append(f"  - PASS   (Label 1): {pos:>6} ({pos / tot * 100:.1f}%)")
    lines.append(f"  - REJECT (Label 0): {neg:>6} ({neg / tot * 100:.1f}%)")

    # Languages
    lines.append("\nLanguage Breakdown:")
    for lang in sorted(VALID_LANGUAGES):
        cnt = summary["languages"].get(lang, 0)
        lines.append(f"  - {lang.capitalize():<12}: {cnt:>6} ({cnt / tot * 100:.1f}%)")

    # Categories
    lines.append("\nMutation Categories:")
    for cat in sorted(VALID_CATEGORIES):
        cnt = summary["categories"].get(cat, 0)
        lines.append(f"  - {cat:<18}: {cnt:>6} ({cnt / tot * 100:.1f}%)")

    # Token Metrics
    t = summary["token_stats"]
    lines.append("\nToken Length Metrics (< 400 tokens requirement):")
    lines.append(f"  - Minimum Tokens:  {t['min']:>6}")
    lines.append(f"  - Maximum Tokens:  {t['max']:>6} {'[✓ OK]' if t['max'] <= 400 else '[✗ VIOLATION]'}")
    lines.append(f"  - Mean Tokens:     {t['mean']:>6.1f}")
    lines.append(f"  - Median Tokens:   {t['median']:>6}")

    # Violations
    if summary["violations_count"] > 0:
        lines.append(f"\n[!] Detected Violations ({summary['violations_count']} total):")
        for v in summary["violations"]:
            lines.append(f"  ✖ {v}")
        if summary["violations_count"] > len(summary["violations"]):
            lines.append(f"  ... and {summary['violations_count'] - len(summary['violations'])} more")
    else:
        lines.append("\n[✓] Zero violations detected. Schema, class balance, and token limits verified.")

    lines.append("================================================================================\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and verify class distribution, language breakdown, token lengths, and formatting of Code Oracle datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing dataset_train.jsonl and dataset_val.jsonl.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Inspect a specific JSONL dataset file.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Maximum allowed token length per sample (default: 400).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )

    args = parser.parse_args()

    files_to_inspect: List[Path] = []
    if args.file:
        files_to_inspect.append(args.file)
    else:
        train_file = args.data_dir / "dataset_train.jsonl"
        val_file = args.data_dir / "dataset_val.jsonl"
        if train_file.exists():
            files_to_inspect.append(train_file)
        if val_file.exists():
            files_to_inspect.append(val_file)

    if not files_to_inspect:
        print(f"[!] No dataset files found in {args.data_dir} or specified file.")
        return 1

    overall_passed = True
    summaries = []

    for fpath in files_to_inspect:
        summary = inspect_file(fpath, max_token_limit=args.max_tokens)
        summaries.append(summary)
        if not summary.get("passed", False):
            overall_passed = False

        if not args.json:
            print(format_report_text(summary))

    if args.json:
        out = {
            "status": "PASSED" if overall_passed else "FAILED",
            "files": summaries,
        }
        print(json.dumps(out, indent=2))

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
