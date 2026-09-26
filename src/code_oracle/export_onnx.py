"""
ONNX Export and Dynamic INT8 Quantization Pipeline for Code Oracle.

Exports ModernBERT multi-task decision models to ONNX format with dynamic axes
and performs dynamic INT8 quantization for sub-50ms CPU inference.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

OUTPUT_NAMES = [
    "risk_score",
    "risk_logits",
    "taxonomy_logits",
    "taxonomy_probs",
    "log_variance",
    "variance",
    "confidence",
]


class OnnxExportWrapper:
    """Wrapper to adapt dictionary output to tuple for clean ONNX export."""

    def __init__(self, base_model: Any):
        super().__init__()
        self.base_model = base_model

    def __call__(self, input_ids: Any, attention_mask: Any) -> Tuple[Any, ...]:
        return self.forward(input_ids, attention_mask)

    def forward(self, input_ids: Any, attention_mask: Any) -> Tuple[Any, ...]:
        res = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return (
            res["risk_score"],
            res["risk_logits"],
            res["taxonomy_logits"],
            res["taxonomy_probs"],
            res["log_variance"],
            res["variance"],
            res["confidence"],
        )


def export_model_to_onnx(
    weights_path: Union[Path, str],
    output_path: Union[Path, str],
    opset_version: int = 17,
    model: Optional[Any] = None,
) -> Path:
    """
    Export ModernBERT multi-task model to ONNX format with dynamic batch & sequence axes.

    Args:
        weights_path: Directory containing PyTorch model weights (e.g. model.safetensors).
        output_path: Destination file path for model.onnx.
        opset_version: Target ONNX opset version (default: 17).
        model: Optional pre-loaded PyTorch model instance.

    Returns:
        Path to exported ONNX model file.
    """
    import torch
    from safetensors.torch import load_file
    from code_oracle.decision import ModernBERTWithMultiTaskHead

    weights_path = Path(weights_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        safetensors_file = weights_path / "model.safetensors"
        if not safetensors_file.exists():
            raise FileNotFoundError(f"Model safetensors weights not found at {safetensors_file}")

        config_file = weights_path / "config.json"
        encoder_name = "answerdotai/ModernBERT-base"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f_cfg:
                    cfg_data = json.load(f_cfg)
                    encoder_name = cfg_data.get("encoder", encoder_name)
            except Exception as e_cfg:
                logger.debug(f"Could not read config.json encoder: {e_cfg}")

        logger.info(f"Loading PyTorch weights from {safetensors_file} with encoder {encoder_name}")
        model = ModernBERTWithMultiTaskHead(encoder_name=encoder_name)
        sd = load_file(str(safetensors_file))
        model.load_state_dict(sd)

    model.eval()

    # Create export wrapper subclassing torch.nn.Module
    class _ExportModule(torch.nn.Module):
        def __init__(self, inner_model: torch.nn.Module):
            super().__init__()
            self.inner = inner_model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            res = self.inner(input_ids, attention_mask)
            return (
                res["risk_score"],
                res["risk_logits"],
                res["taxonomy_logits"],
                res["taxonomy_probs"],
                res["log_variance"],
                res["variance"],
                res["confidence"],
            )

    export_module = _ExportModule(model)
    export_module.eval()

    dummy_input_ids = torch.ones((1, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "risk_score": {0: "batch_size"},
        "risk_logits": {0: "batch_size"},
        "taxonomy_logits": {0: "batch_size"},
        "taxonomy_probs": {0: "batch_size"},
        "log_variance": {0: "batch_size"},
        "variance": {0: "batch_size"},
        "confidence": {0: "batch_size"},
    }

    logger.info(f"Exporting ONNX model to {output_path} (opset {opset_version})...")
    torch.onnx.export(
        export_module,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    logger.info(f"ONNX export completed: {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return output_path


def quantize_onnx_int8(
    onnx_path: Union[Path, str],
    output_path: Union[Path, str],
    per_channel: bool = True,
    reduce_range: bool = True,
    op_types: Optional[List[str]] = None,
) -> Path:
    """
    Apply dynamic INT8 quantization to an ONNX model, targeting MatMul / Gemm operations.

    Args:
        onnx_path: Path to FP32 ONNX model.
        output_path: Path for output INT8 quantized model.
        per_channel: Whether to quantize weights per-channel (default: True).
        reduce_range: Whether to use 7-bit quantization for non-VNNI hardware (default: True for overflow protection).
        op_types: Optional list of operator types to quantize (default: all supported linear ops).

    Returns:
        Path to quantized model_int8.onnx.
    """
    import onnxruntime.quantization as oq

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {onnx_path}")

    logger.info(f"Applying dynamic INT8 quantization (per_channel={per_channel}) to {onnx_path}...")

    quant_kwargs: Dict[str, Any] = {
        "model_input": str(onnx_path),
        "model_output": str(output_path),
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "weight_type": oq.QuantType.QInt8,
    }
    if op_types:
        quant_kwargs["op_types_to_quantize"] = op_types

    oq.quantize_dynamic(**quant_kwargs)

    orig_size = onnx_path.stat().st_size / (1024 * 1024)
    quant_size = output_path.stat().st_size / (1024 * 1024)
    reduction = ((orig_size - quant_size) / orig_size) * 100.0 if orig_size > 0 else 0.0
    logger.info(f"INT8 quantization completed: {output_path} ({quant_size:.2f} MB, {reduction:.1f}% reduction)")
    return output_path


def verify_numeric_parity(
    weights_path: Union[Path, str],
    onnx_fp32_path: Union[Path, str],
    onnx_int8_path: Optional[Union[Path, str]] = None,
    sample_texts: Optional[List[str]] = None,
    max_diff_fp32: float = 1e-4,
    max_diff_int8: float = 0.05,
) -> Dict[str, Any]:
    """
    Verify numeric parity between PyTorch outputs and ONNX (FP32 & INT8) outputs.

    Args:
        weights_path: Directory with PyTorch weights and tokenizer.
        onnx_fp32_path: Path to model.onnx.
        onnx_int8_path: Optional path to model_int8.onnx.
        sample_texts: Optional list of sample linearized DSL texts.
        max_diff_fp32: Maximum acceptable absolute diff for FP32 ONNX vs PyTorch.
        max_diff_int8: Maximum acceptable absolute diff for INT8 ONNX vs PyTorch.

    Returns:
        Structured parity evaluation dictionary.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from code_oracle.decision import ModernBERTWithMultiTaskHead

    weights_path = Path(weights_path)
    onnx_fp32_path = Path(onnx_fp32_path)

    default_samples = [
        "[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED\nDEF add(a, b) -> RETURN a + b",
        "[DIFF_TARGET] user_service.py\n[GATE]\nSTATUS: REJECTED\nARITY_MISMATCH: compute() takes 2 arguments but 3 were given",
        "[DIFF_TARGET] auth.py\n[GATE]\nSTATUS: APPROVED\nCALL authenticate(user, password) -> VALID",
        "[DIFF_TARGET] pipeline.py\n[GATE]\nSTATUS: REJECTED\nCIRCULAR_DEPENDENCY: cycle detected [A -> B -> A]",
    ]
    samples = sample_texts or default_samples

    # Load PyTorch model
    sd = load_file(str(weights_path / "model.safetensors"))
    pt_model = ModernBERTWithMultiTaskHead()
    pt_model.load_state_dict(sd)
    pt_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(weights_path))

    # Initialize ONNX sessions
    session_fp32 = ort.InferenceSession(str(onnx_fp32_path), providers=["CPUExecutionProvider"])
    session_int8 = None
    if onnx_int8_path and Path(onnx_int8_path).exists():
        session_int8 = ort.InferenceSession(str(onnx_int8_path), providers=["CPUExecutionProvider"])

    max_fp32_risk_diff = 0.0
    max_fp32_conf_diff = 0.0
    max_fp32_tax_diff = 0.0

    max_int8_risk_diff = 0.0
    max_int8_conf_diff = 0.0
    max_int8_tax_diff = 0.0

    sample_results = []

    for text in samples:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # PyTorch inference
        with torch.no_grad():
            pt_out = pt_model(tokens["input_ids"], tokens["attention_mask"])

        pt_risk = float(pt_out["risk_score"].item())
        pt_conf = float(pt_out["confidence"].item())
        pt_tax = [float(x) for x in pt_out["taxonomy_probs"].squeeze(0).tolist()]

        # ONNX FP32 inference
        ort_inputs = {
            "input_ids": tokens["input_ids"].numpy().astype(np.int64),
            "attention_mask": tokens["attention_mask"].numpy().astype(np.int64),
        }
        fp32_outs = session_fp32.run(None, ort_inputs)
        fp32_map = dict(zip(OUTPUT_NAMES, fp32_outs))

        fp32_risk = float(fp32_map["risk_score"][0][0])
        fp32_conf = float(fp32_map["confidence"][0][0])
        fp32_tax = [float(x) for x in fp32_map["taxonomy_probs"][0]]

        d_fp32_risk = abs(fp32_risk - pt_risk)
        d_fp32_conf = abs(fp32_conf - pt_conf)
        d_fp32_tax = max(abs(a - b) for a, b in zip(fp32_tax, pt_tax))

        max_fp32_risk_diff = max(max_fp32_risk_diff, d_fp32_risk)
        max_fp32_conf_diff = max(max_fp32_conf_diff, d_fp32_conf)
        max_fp32_tax_diff = max(max_fp32_tax_diff, d_fp32_tax)

        sample_entry: Dict[str, Any] = {
            "text": text[:40] + "...",
            "pt_risk": pt_risk,
            "fp32_risk": fp32_risk,
            "fp32_risk_diff": d_fp32_risk,
        }

        # ONNX INT8 inference if available
        if session_int8 is not None:
            int8_outs = session_int8.run(None, ort_inputs)
            int8_map = dict(zip(OUTPUT_NAMES, int8_outs))

            int8_risk = float(int8_map["risk_score"][0][0])
            int8_conf = float(int8_map["confidence"][0][0])
            int8_tax = [float(x) for x in int8_map["taxonomy_probs"][0]]

            d_int8_risk = abs(int8_risk - pt_risk)
            d_int8_conf = abs(int8_conf - pt_conf)
            d_int8_tax = max(abs(a - b) for a, b in zip(int8_tax, pt_tax))

            max_int8_risk_diff = max(max_int8_risk_diff, d_int8_risk)
            max_int8_conf_diff = max(max_int8_conf_diff, d_int8_conf)
            max_int8_tax_diff = max(max_int8_tax_diff, d_int8_tax)

            sample_entry["int8_risk"] = int8_risk
            sample_entry["int8_risk_diff"] = d_int8_risk

        sample_results.append(sample_entry)

    fp32_pass = (
        max_fp32_risk_diff <= max_diff_fp32
        and max_fp32_conf_diff <= max_diff_fp32
        and max_fp32_tax_diff <= max_diff_fp32
    )

    int8_pass = True
    if session_int8 is not None:
        int8_pass = (
            max_int8_risk_diff <= max_diff_int8
            and max_int8_conf_diff <= max_diff_int8
            and max_int8_tax_diff <= max_diff_int8
        )

    return {
        "status": "PASS" if (fp32_pass and int8_pass) else "FAIL",
        "fp32_parity_pass": fp32_pass,
        "int8_parity_pass": int8_pass,
        "max_fp32_risk_diff": max_fp32_risk_diff,
        "max_fp32_conf_diff": max_fp32_conf_diff,
        "max_fp32_tax_diff": max_fp32_tax_diff,
        "max_int8_risk_diff": max_int8_risk_diff,
        "max_int8_conf_diff": max_int8_conf_diff,
        "max_int8_tax_diff": max_int8_tax_diff,
        "samples_tested": len(samples),
        "sample_results": sample_results,
    }


def export_and_quantize(
    weights_path: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    quantize_int8: bool = True,
    verify_parity: bool = True,
    opset_version: int = 17,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: Export PyTorch weights to ONNX FP32 and dynamic INT8,
    verify numeric parity, and synchronize configuration and tokenizer files.

    Args:
        weights_path: Source directory containing PyTorch weights (e.g. weights_base).
        output_dir: Output directory (defaults to weights_path if None).
        quantize_int8: Whether to generate model_int8.onnx.
        verify_parity: Whether to run numeric parity verification against PyTorch.
        opset_version: Target ONNX opset version.

    Returns:
        Structured dictionary summarizing export results.
    """
    weights_path = Path(weights_path).resolve()
    out_dir = Path(output_dir or weights_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # 1. Export FP32 ONNX
    onnx_fp32_path = out_dir / "model.onnx"
    export_model_to_onnx(
        weights_path=weights_path,
        output_path=onnx_fp32_path,
        opset_version=opset_version,
    )
    fp32_size_mb = onnx_fp32_path.stat().st_size / (1024 * 1024)

    # 2. Dynamic INT8 Quantization
    onnx_int8_path = None
    int8_size_mb = None
    if quantize_int8:
        onnx_int8_path = out_dir / "model_int8.onnx"
        quantize_onnx_int8(
            onnx_path=onnx_fp32_path,
            output_path=onnx_int8_path,
        )
        int8_size_mb = onnx_int8_path.stat().st_size / (1024 * 1024)

    # 3. Synchronize ancillary assets (config.json, tokenizer.json, tokenizer_config.json)
    if out_dir != weights_path:
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            src_f = weights_path / fname
            if src_f.exists():
                shutil.copy2(src_f, out_dir / fname)
        # Also copy tokenizer dir if present
        tok_dir = weights_path / "tokenizer"
        if tok_dir.is_dir() and not (out_dir / "tokenizer").exists():
            shutil.copytree(tok_dir, out_dir / "tokenizer")

    # 4. Numeric Parity Verification
    parity_report = None
    if verify_parity:
        logger.info("Running numeric parity verification against PyTorch...")
        parity_report = verify_numeric_parity(
            weights_path=weights_path,
            onnx_fp32_path=onnx_fp32_path,
            onnx_int8_path=onnx_int8_path,
        )

    elapsed_s = time.perf_counter() - t0

    result = {
        "status": "SUCCESS",
        "output_dir": str(out_dir),
        "model_onnx": str(onnx_fp32_path),
        "model_int8_onnx": str(onnx_int8_path) if onnx_int8_path else None,
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_mb": round(int8_size_mb, 2) if int8_size_mb is not None else None,
        "elapsed_seconds": round(elapsed_s, 2),
        "parity": parity_report,
    }
    return result
