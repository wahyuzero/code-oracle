#!/usr/bin/env python3
"""
CLI utility to export ModernBERT PyTorch weights to ONNX format
and generate dynamic INT8 quantized models for sub-50ms CPU inference.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to sys.path to allow execution without prior pip install
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from code_oracle.export_onnx import export_and_quantize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export PyTorch ModernBERT weights to ONNX FP32 and dynamic INT8 formats."
    )
    parser.add_argument(
        "--weights-dir",
        "-w",
        default="weights_base",
        help="Source directory containing PyTorch model weights (default: weights_base)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Destination directory for exported ONNX models (default: same as weights-dir)",
    )
    parser.add_argument(
        "--no-int8",
        dest="quantize_int8",
        action="store_false",
        default=True,
        help="Skip dynamic INT8 quantization",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify_parity",
        action="store_false",
        default=True,
        help="Skip numeric parity verification against PyTorch",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    weights_dir = Path(args.weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = (repo_root / weights_dir).resolve()

    if not weights_dir.exists():
        print(f"Error: Weights directory not found at {weights_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else weights_dir
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    try:
        summary = export_and_quantize(
            weights_path=weights_dir,
            output_dir=out_dir,
            quantize_int8=args.quantize_int8,
            verify_parity=args.verify_parity,
            opset_version=args.opset,
        )

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("====================================================")
            print(" ONNX EXPORT & QUANTIZATION REPORT")
            print("====================================================")
            print(f" Output Directory:    {summary['output_dir']}")
            print(f" Model ONNX (FP32):   {summary['model_onnx']} ({summary['fp32_size_mb']:.2f} MB)")
            if summary.get("model_int8_onnx"):
                print(f" Model ONNX (INT8):   {summary['model_int8_onnx']} ({summary['int8_size_mb']:.2f} MB)")
            print(f" Total Elapsed Time:  {summary['elapsed_seconds']:.2f} s")

            parity = summary.get("parity")
            if parity:
                print("----------------------------------------------------")
                p_status = parity.get("status", "UNKNOWN")
                print(f" Parity Status:       {p_status}")
                print(f" FP32 Parity Pass:    {parity.get('fp32_parity_pass')}")
                print(f" Max FP32 Risk Diff:  {parity.get('max_fp32_risk_diff'):.6e}")
                if summary.get("model_int8_onnx"):
                    print(f" INT8 Parity Pass:    {parity.get('int8_parity_pass')}")
                    print(f" Max INT8 Risk Diff:  {parity.get('max_int8_risk_diff'):.6f}")
                print(f" Samples Evaluated:   {parity.get('samples_tested')}")
            print("====================================================")
        return 0
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        else:
            print(f"Error during ONNX export/quantization: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
