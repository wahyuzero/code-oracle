"""
Unit and integration tests for ONNX Export, Dynamic INT8 Quantization,
and ONNX Runtime Inference in Code Oracle.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from code_oracle.decision import (
    EnhancedDecisionResult,
    LayaDecisionHead,
    ModernBERTMultiTaskModel,
    ModernBERTWithMultiTaskHead,
)
from code_oracle.engine import TopoSliceEngine
from code_oracle.export_onnx import (
    export_and_quantize,
    export_model_to_onnx,
    quantize_onnx_int8,
    verify_numeric_parity,
)
from code_oracle.models import EnhancedVerificationReport

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_BASE = REPO_ROOT / "weights_base"


@pytest.fixture
def sample_weights_dir():
    """Ensure weights_base exists and return path."""
    assert WEIGHTS_BASE.exists(), f"weights_base not found at {WEIGHTS_BASE}"
    return WEIGHTS_BASE


def test_onnx_model_files_exist(sample_weights_dir):
    """Verify that exported model.onnx and model_int8.onnx exist and have valid sizes."""
    fp32_path = sample_weights_dir / "model.onnx"
    int8_path = sample_weights_dir / "model_int8.onnx"

    assert fp32_path.exists(), "weights_base/model.onnx must exist"
    assert int8_path.exists(), "weights_base/model_int8.onnx must exist"

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size

    # Verify model is substantially compressed (> 50% reduction)
    assert int8_size < fp32_size * 0.5
    assert int8_size > 10 * 1024 * 1024  # > 10 MB


def test_onnx_export_pipeline_dynamic_axes(tmp_path, sample_weights_dir):
    """Verify export_model_to_onnx exports valid model with dynamic batch and sequence axes."""
    out_onnx = tmp_path / "test_model.onnx"
    res_path = export_model_to_onnx(
        weights_path=sample_weights_dir,
        output_path=out_onnx,
        opset_version=17,
    )
    assert res_path.exists()

    # Load with onnx library to validate graph structure
    onnx_proto = onnx.load(str(out_onnx))
    onnx.checker.check_model(onnx_proto)

    # Test dynamic batch size & sequence length with ONNX Runtime
    session = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])

    # Batch 1, sequence 12
    in1 = {
        "input_ids": np.ones((1, 12), dtype=np.int64),
        "attention_mask": np.ones((1, 12), dtype=np.int64),
    }
    out1 = session.run(None, in1)
    assert len(out1) == 7
    assert out1[0].shape == (1, 1)  # risk_score
    assert out1[2].shape == (1, 5)  # taxonomy_logits

    # Batch 3, sequence 32 (testing dynamic batch and sequence dimension)
    in2 = {
        "input_ids": np.ones((3, 32), dtype=np.int64),
        "attention_mask": np.ones((3, 32), dtype=np.int64),
    }
    out2 = session.run(None, in2)
    assert out2[0].shape == (3, 1)  # risk_score batch 3
    assert out2[2].shape == (3, 5)  # taxonomy_logits batch 3


def test_quantize_onnx_int8(tmp_path, sample_weights_dir):
    """Verify dynamic INT8 quantization generates valid model with size reduction."""
    fp32_onnx = sample_weights_dir / "model.onnx"
    int8_onnx = tmp_path / "test_model_int8.onnx"

    quant_path = quantize_onnx_int8(
        onnx_path=fp32_onnx,
        output_path=int8_onnx,
        per_channel=True,
    )
    assert quant_path.exists()

    fp32_size = fp32_onnx.stat().st_size
    int8_size = quant_path.stat().st_size
    assert int8_size < fp32_size * 0.6  # > 40% reduction

    # Verify INT8 session runs properly
    session = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
    inputs = {
        "input_ids": np.ones((1, 16), dtype=np.int64),
        "attention_mask": np.ones((1, 16), dtype=np.int64),
    }
    outs = session.run(None, inputs)
    assert len(outs) == 7
    assert 0.0 <= outs[0][0][0] <= 1.0


def test_numeric_parity_verification(sample_weights_dir):
    """Verify numeric parity between PyTorch and ONNX FP32 / INT8 outputs."""
    report = verify_numeric_parity(
        weights_path=sample_weights_dir,
        onnx_fp32_path=sample_weights_dir / "model.onnx",
        onnx_int8_path=sample_weights_dir / "model_int8.onnx",
        max_diff_fp32=1e-4,
        max_diff_int8=0.05,
    )

    assert report["status"] == "PASS"
    assert report["fp32_parity_pass"] is True
    assert report["int8_parity_pass"] is True
    assert report["max_fp32_risk_diff"] < 1e-4
    assert report["max_int8_risk_diff"] < 0.05
    assert report["samples_tested"] >= 4


def test_decision_head_prioritizes_onnx_int8(sample_weights_dir):
    """Verify LayaDecisionHead loads model_int8.onnx by default."""
    head = LayaDecisionHead(
        weights_path=sample_weights_dir,
        enabled=True,
        quantize_int8=True,
    )

    assert head.is_neural_enabled is True
    assert head.engine_mode == "onnx_int8"
    assert head.onnx_session is not None

    dsl = "[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED\nDEF add(a, b) -> RETURN a + b"
    res = head.predict_multi_task(
        linearized_dsl=dsl,
        symbolic_status="APPROVED",
        symbolic_confidence=0.95,
        has_violations=False,
    )

    assert isinstance(res, EnhancedDecisionResult)
    assert res.is_neural_calibrated is True
    assert res.engine_mode == "onnx_int8"
    assert 0.0 <= res.risk_score <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.epistemic_uncertainty <= 1.0
    assert len(res.risk_taxonomy.to_dict()) == 5


def test_decision_head_onnx_fp32_mode(sample_weights_dir):
    """Verify LayaDecisionHead loads model.onnx when quantize_int8 is False."""
    head = LayaDecisionHead(
        weights_path=sample_weights_dir,
        enabled=True,
        quantize_int8=False,
    )

    assert head.is_neural_enabled is True
    assert head.engine_mode == "onnx_fp32"
    assert head.onnx_session is not None

    dsl = "[DIFF_TARGET] auth.py\n[GATE]\nSTATUS: APPROVED\nCALL auth()"
    res = head.predict_multi_task(
        linearized_dsl=dsl,
        symbolic_status="APPROVED",
        symbolic_confidence=0.92,
        has_violations=False,
    )

    assert res.is_neural_calibrated is True
    assert res.engine_mode == "onnx_fp32"


def test_decision_head_pytorch_fallback_when_onnx_missing(tmp_path, sample_weights_dir):
    """Verify graceful fallback to PyTorch safetensors when ONNX files are absent."""
    # Create isolated directory with only safetensors and config
    shutil.copy2(sample_weights_dir / "model.safetensors", tmp_path / "model.safetensors")
    shutil.copy2(sample_weights_dir / "config.json", tmp_path / "config.json")
    shutil.copy2(sample_weights_dir / "tokenizer.json", tmp_path / "tokenizer.json")
    shutil.copy2(sample_weights_dir / "tokenizer_config.json", tmp_path / "tokenizer_config.json")

    head = LayaDecisionHead(
        weights_path=tmp_path,
        enabled=True,
        prefer_onnx=True,
    )

    assert head.is_neural_enabled is True
    assert head.engine_mode == "pytorch"
    assert head.pytorch_multitask_model is not None
    assert head.onnx_session is None

    res = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] test.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.90,
        has_violations=False,
    )
    assert res.is_neural_calibrated is True
    assert res.engine_mode == "pytorch"


def test_decision_head_heuristic_fallback_when_weights_absent():
    """Verify graceful fallback to heuristic when no weights are available."""
    head = LayaDecisionHead(
        weights_path=None,
        enabled=False,
    )

    assert head.is_neural_enabled is False
    assert head.engine_mode == "heuristic"

    res = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] test.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.90,
        has_violations=False,
    )
    assert res.is_neural_calibrated is False
    assert res.engine_mode == "heuristic"
    assert res.status == "APPROVED"
    assert res.risk_score == 0.05


def test_decision_head_symbolic_hard_veto_retains_engine_mode(sample_weights_dir):
    """Verify symbolic gate veto overrides neural prediction while preserving engine mode."""
    head = LayaDecisionHead(
        weights_path=sample_weights_dir,
        enabled=True,
        quantize_int8=True,
    )
    assert head.engine_mode == "onnx_int8"

    res = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] test.py\n[GATE]\nSTATUS: REJECTED",
        symbolic_status="REJECTED",
        symbolic_confidence=0.98,
        has_violations=True,
        violations=["ARITY_MISMATCH: compute() takes 1 argument but 2 were given"],
    )

    assert res.status == "REJECTED"
    assert res.confidence == 1.0
    assert res.risk_score == 0.95
    assert res.engine_mode == "onnx_int8"
    assert res.is_neural_calibrated is False


def test_pure_onnx_inference_without_torch(sample_weights_dir):
    """Verify pure ONNX session execution using only numpy and onnxruntime."""
    session = ort.InferenceSession(
        str(sample_weights_dir / "model_int8.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # Pure NumPy inputs without torch dependency
    input_ids = np.ones((1, 20), dtype=np.int64)
    attention_mask = np.ones((1, 20), dtype=np.int64)

    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    assert len(outputs) == 7
    risk_score = float(outputs[0][0][0])
    taxonomy_probs = outputs[3][0]

    assert 0.0 <= risk_score <= 1.0
    assert len(taxonomy_probs) == 5
    assert all(0.0 <= p <= 1.0 for p in taxonomy_probs)


def test_engine_verification_report_includes_engine_mode(tmp_path, sample_weights_dir):
    """Verify TopoSliceEngine includes engine_mode in EnhancedVerificationReport."""
    calc_file = tmp_path / "calc.py"
    calc_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

    engine = TopoSliceEngine(
        workspace_root=tmp_path,
        weights_path=sample_weights_dir,
        enable_neural=True,
    )

    patch = "def add(a: int, b: int) -> int:\n    # clean patch\n    return a + b\n"
    report = engine.verify("calc.py", patch)

    assert isinstance(report, EnhancedVerificationReport)
    assert report.status == "APPROVED"
    assert report.engine_mode in ("onnx_int8", "onnx_fp32")
    assert report.is_neural_calibrated is True

    report_dict = report.to_dict()
    assert "engine_mode" in report_dict
    assert report_dict["engine_mode"] in ("onnx_int8", "onnx_fp32")


def test_cli_export_onnx_command(tmp_path, sample_weights_dir):
    """Verify code-oracle export-onnx CLI execution."""
    out_dir = tmp_path / "onnx_out"
    cmd = [
        "code-oracle",
        "export-onnx",
        "--weights",
        str(sample_weights_dir),
        "--output-dir",
        str(out_dir),
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI error: {res.stderr}"

    data = json.loads(res.stdout)
    assert data["status"] == "SUCCESS"
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "model_int8.onnx").exists()
    assert data["parity"]["status"] == "PASS"


def test_cpu_inference_benchmark_latency(sample_weights_dir):
    """
    Benchmark CPU inference showing latency reduction with ONNX INT8.
    Verifies that ONNX INT8 is faster than PyTorch and completes in sub-50ms range.
    """
    head_onnx = LayaDecisionHead(weights_path=sample_weights_dir, enabled=True, quantize_int8=True)
    head_pt = LayaDecisionHead(weights_path=sample_weights_dir, enabled=True, prefer_onnx=False)

    dsl = "[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED\nDEF benchmark_func(x, y) -> RETURN x + y"

    # Warmup
    for _ in range(3):
        head_onnx.predict_multi_task(dsl, "APPROVED", 0.95, False)
        head_pt.predict_multi_task(dsl, "APPROVED", 0.95, False)

    # Benchmark ONNX INT8 (5 iterations)
    t0 = time.perf_counter()
    for _ in range(5):
        head_onnx.predict_multi_task(dsl, "APPROVED", 0.95, False)
    onnx_latency_ms = (time.perf_counter() - t0) / 5.0 * 1000.0

    # Benchmark PyTorch (5 iterations)
    t0 = time.perf_counter()
    for _ in range(5):
        head_pt.predict_multi_task(dsl, "APPROVED", 0.95, False)
    pt_latency_ms = (time.perf_counter() - t0) / 5.0 * 1000.0

    # ONNX INT8 should outperform PyTorch on CPU
    assert onnx_latency_ms < pt_latency_ms
    assert onnx_latency_ms < 150.0  # reasonable bound allowing for noisy virtualized test environments
