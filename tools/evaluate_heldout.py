#!/usr/bin/env python3
"""
Independent Evaluation Script for Held-Out Benchmark Dataset.

Evaluates Code Oracle verification pipeline across unseen real-world repositories
(e.g. pallets/flask, encode/httpx, fastify/fastify, go-chi/chi, serde-rs/serde).
Computes and publishes independent precision, recall, F1, accuracy, and confusion matrix.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to sys.path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from code_oracle.decision import LayaDecisionHead


def evaluate_dataset(
    dataset_path: Path,
    weights_path: Optional[Path] = None,
    risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate a dataset file against Code Oracle decision engine.
    Calculates precision, recall, F1, accuracy, and 2x2 confusion matrix.
    """
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    head = LayaDecisionHead(weights_path=weights_path, enabled=bool(weights_path), risk_threshold=risk_threshold)

    records: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                records.append(json.loads(line_s))

    if not records:
        raise ValueError(f"No records found in {dataset_path}")

    tp = 0  # True Positive: Actual PASS (1), Predicted PASS (1)
    fp = 0  # False Positive: Actual REJECT (0), Predicted PASS (1)
    tn = 0  # True Negative: Actual REJECT (0), Predicted REJECT (0)
    fn = 0  # False Negative: Actual PASS (1), Predicted REJECT (0)

    by_lang: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}

    for rec in records:
        true_label = rec.get("label", 1)
        lang = rec.get("language", "unknown")
        cat = rec.get("category", "unknown")
        dsl = rec.get("input_dsl", "")

        # Default symbolic gate status from record DSL
        has_violations = "STATUS: REJECTED" in dsl
        symbolic_status = "REJECTED" if has_violations else "APPROVED"
        symbolic_conf = 0.95 if has_violations else 0.90

        pred_res = head.predict_multi_task(
            linearized_dsl=dsl,
            symbolic_status=symbolic_status,
            symbolic_confidence=symbolic_conf,
            has_violations=has_violations,
            risk_threshold=risk_threshold,
        )

        pred_label = 1 if pred_res.status == "APPROVED" and pred_res.risk_score < risk_threshold else 0

        # Update confusion matrix
        if true_label == 1 and pred_label == 1:
            tp += 1
            outcome = "TP"
        elif true_label == 0 and pred_label == 1:
            fp += 1
            outcome = "FP"
        elif true_label == 0 and pred_label == 0:
            tn += 1
            outcome = "TN"
        else:
            fn += 1
            outcome = "FN"

        by_lang.setdefault(lang, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})[outcome] += 1
        by_category.setdefault(cat, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})[outcome] += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "dataset_path": str(dataset_path),
        "total_samples": total,
        "is_neural_enabled": head.is_neural_enabled,
        "risk_threshold": risk_threshold,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "specificity": round(specificity, 4),
        },
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "table": [
                [tn, fp],  # Row 0: Actual REJECT -> [Pred REJECT, Pred PASS]
                [fn, tp],  # Row 1: Actual PASS   -> [Pred REJECT, Pred PASS]
            ],
        },
        "breakdown_by_language": by_lang,
        "breakdown_by_category": by_category,
    }


def format_evaluation_report(eval_res: Dict[str, Any]) -> str:
    """Format evaluation results into a clean markdown report."""
    m = eval_res["metrics"]
    cm = eval_res["confusion_matrix"]

    lines = [
        "=" * 80,
        " INDEPENDENT HELD-OUT EVALUATION REPORT",
        "=" * 80,
        f"Dataset:         {eval_res['dataset_path']}",
        f"Total Samples:   {eval_res['total_samples']}",
        f"Neural Head:     {'Active (Fine-Tuned)' if eval_res['is_neural_enabled'] else 'Deterministic Symbolic Fallback'}",
        f"Risk Threshold:  {eval_res['risk_threshold']}",
        "-" * 80,
        "Overall Performance Metrics:",
        f"  - Accuracy:     {m['accuracy'] * 100:.2f}%",
        f"  - Precision:    {m['precision'] * 100:.2f}%",
        f"  - Recall:       {m['recall'] * 100:.2f}%",
        f"  - F1 Score:     {m['f1'] * 100:.2f}%",
        f"  - Specificity:  {m['specificity'] * 100:.2f}%",
        "-" * 80,
        "Confusion Matrix:",
        f"                        Predicted REJECT (0)    Predicted PASS (1)",
        f"  Actual REJECT (0):    {cm['TN']:<23} {cm['FP']:<20}",
        f"  Actual PASS (1):      {cm['FN']:<23} {cm['TP']:<20}",
        "-" * 80,
        "Breakdown by Language:",
    ]

    for lang, counts in sorted(eval_res["breakdown_by_language"].items()):
        sub_total = sum(counts.values())
        sub_acc = (counts["TP"] + counts["TN"]) / sub_total if sub_total > 0 else 0.0
        lines.append(f"  - {lang.capitalize():<12} ({sub_total} samples): Acc={sub_acc * 100:.1f}%, TP={counts['TP']}, TN={counts['TN']}, FP={counts['FP']}, FN={counts['FN']}")

    lines.extend([
        "-" * 80,
        "Breakdown by Category:",
    ])
    for cat, counts in sorted(eval_res["breakdown_by_category"].items()):
        sub_total = sum(counts.values())
        lines.append(f"  - {cat:<24} ({sub_total} samples): TP={counts['TP']}, TN={counts['TN']}, FP={counts['FP']}, FN={counts['FN']}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Code Oracle on held-out test datasets and report precision, recall, F1, and confusion matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=REPO_ROOT / "data" / "dataset_heldout_eval.jsonl",
        help="Path to JSONL dataset file to evaluate.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=Path,
        default=None,
        help="Path to fine-tuned Laya model directory with model.safetensors.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Decision boundary risk threshold (default: 0.5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results instead of formatted text.",
    )

    args = parser.parse_args()

    try:
        eval_res = evaluate_dataset(
            dataset_path=args.file,
            weights_path=args.weights,
            risk_threshold=args.threshold,
        )
    except Exception as e:
        print(f"[!] Evaluation failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(eval_res, indent=2))
    else:
        print(format_evaluation_report(eval_res))

    return 0


if __name__ == "__main__":
    sys.exit(main())
