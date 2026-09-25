"""
Tests for tools/error_analysis.py and configurable risk threshold functionality.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from error_analysis import (
    EvalSample,
    TemperatureScaler,
    apply_decision_threshold,
    classify_semantic_root_cause,
    compute_metrics,
    fit_temperature_scaling,
    parse_dsl_summary,
    sweep_thresholds,
)
from code_oracle.decision import LayaDecisionHead


def test_parse_dsl_summary():
    dsl = (
        "[DIFF_TARGET] src/auth.py::login (MODIFIED)\n"
        "[METADATA] File: src/auth.py\n"
        "[NODES]\n"
        "N0: src/auth.py::login [def login()] (SEED, MODIFIED)\n"
        "N1: src/db.py::get_user [def get_user()]\n"
    )
    diff_target, nodes_summary = parse_dsl_summary(dsl)
    assert "src/auth.py::login" in diff_target
    assert "N0: src/auth.py::login" in nodes_summary


def test_classify_semantic_root_cause():
    sample_fp = EvalSample(
        index=0,
        true_label=0,
        risk_score=0.3,
        confidence=0.8,
        epistemic_uncertainty=0.1,
        taxonomy_scores={},
        language="typescript",
        category="security_surface",
        source_type="mutation",
        input_dsl="[DIFF_TARGET] auth.ts\n[NODES]\nN0: auth [func auth()]\n",
        outcome="FP",
    )
    rc = classify_semantic_root_cause(sample_fp)
    assert "Security Surface Expansion" in rc

    sample_revert = EvalSample(
        index=1,
        true_label=0,
        risk_score=0.2,
        confidence=0.8,
        epistemic_uncertainty=0.1,
        taxonomy_scores={},
        language="python",
        category="real_revert",
        source_type="real_hotfix",
        input_dsl="[DIFF_TARGET] util.py\n",
        outcome="FP",
    )
    rc_revert = classify_semantic_root_cause(sample_revert)
    assert "Revert of Hotfix/Bugfix" in rc_revert


def test_apply_decision_threshold_and_metrics():
    samples = [
        # Safe sample, risk 0.2
        EvalSample(
            index=0, true_label=1, risk_score=0.2, confidence=0.9, epistemic_uncertainty=0.05,
            taxonomy_scores={}, language="go", category="clean_pass", source_type="clean", input_dsl=""
        ),
        # Buggy sample, risk 0.35
        EvalSample(
            index=1, true_label=0, risk_score=0.35, confidence=0.8, epistemic_uncertainty=0.1,
            taxonomy_scores={}, language="python", category="silent_logic_drift", source_type="mutation", input_dsl=""
        ),
        # Buggy sample, risk 0.8
        EvalSample(
            index=2, true_label=0, risk_score=0.8, confidence=0.95, epistemic_uncertainty=0.02,
            taxonomy_scores={}, language="rust", category="concurrency_hazard", source_type="mutation", input_dsl=""
        ),
    ]

    # At threshold 0.50:
    # sample 0 (safe, 0.2 < 0.5): PASS (TP)
    # sample 1 (bug, 0.35 < 0.5): PASS (FP - missed bug!)
    # sample 2 (bug, 0.8 >= 0.5): REJECT (TN - caught bug!)
    apply_decision_threshold(samples, 0.50)
    m50 = compute_metrics(samples)
    assert m50["TP"] == 1
    assert m50["FP"] == 1
    assert m50["TN"] == 1
    assert m50["FN"] == 0

    # At threshold 0.30:
    # sample 0 (safe, 0.2 < 0.3): PASS (TP)
    # sample 1 (bug, 0.35 >= 0.3): REJECT (TN - caught bug!)
    # sample 2 (bug, 0.8 >= 0.3): REJECT (TN - caught bug!)
    apply_decision_threshold(samples, 0.30)
    m30 = compute_metrics(samples)
    assert m30["TP"] == 1
    assert m30["FP"] == 0
    assert m30["TN"] == 2
    assert m30["FN"] == 0


def test_sweep_thresholds():
    samples = [
        EvalSample(index=0, true_label=1, risk_score=0.15, confidence=0.9, epistemic_uncertainty=0.05, taxonomy_scores={}, language="go", category="clean_pass", source_type="clean", input_dsl=""),
        EvalSample(index=1, true_label=0, risk_score=0.42, confidence=0.8, epistemic_uncertainty=0.1, taxonomy_scores={}, language="python", category="silent_logic_drift", source_type="mutation", input_dsl=""),
    ]
    rows = sweep_thresholds(samples, [0.30, 0.40, 0.50])
    assert len(rows) == 3
    # At 0.40, risk 0.42 >= 0.40 -> caught!
    assert rows[1]["TN"] == 1
    # At 0.50, risk 0.42 < 0.50 -> missed!
    assert rows[2]["FP"] == 1


def test_temperature_scaler():
    scaler = TemperatureScaler(init_temperature=1.5)
    logits = torch.tensor([[1.5], [-2.0], [0.5]], dtype=torch.float32)
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 1.5)


def test_laya_decision_head_configurable_threshold():
    head = LayaDecisionHead(weights_path=None, risk_threshold=0.40)
    assert head.risk_threshold == 0.40

    # Test predict_multi_task threshold override
    res_high = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.95,
        has_violations=False,
        risk_threshold=0.01,  # Lower than default 0.05 risk
    )
    # Since risk=0.05 >= 0.01 threshold, status should be REJECTED
    assert res_high.status == "REJECTED"

    res_normal = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.95,
        has_violations=False,
        risk_threshold=0.50,
    )
    assert res_normal.status == "APPROVED"


def test_explain_why_missed():
    from error_analysis import explain_why_missed
    sample_revert = EvalSample(
        index=0, true_label=0, risk_score=0.25, confidence=0.8, epistemic_uncertainty=0.1,
        taxonomy_scores={}, language="python", category="real_revert", source_type="real_hotfix", input_dsl=""
    )
    explanation = explain_why_missed(sample_revert)
    assert "exact revert of a previous hotfix/bugfix" in explanation

    sample_sec = EvalSample(
        index=1, true_label=0, risk_score=0.35, confidence=0.8, epistemic_uncertainty=0.1,
        taxonomy_scores={}, language="typescript", category="security_surface", source_type="mutation", input_dsl=""
    )
    assert "authorization guards" in explain_why_missed(sample_sec)


def test_generate_markdown_report_sequential_headers(tmp_path):
    from error_analysis import generate_markdown_report
    samples = [
        EvalSample(index=0, true_label=1, risk_score=0.1, confidence=0.9, epistemic_uncertainty=0.05, taxonomy_scores={}, language="python", category="clean_pass", source_type="clean", input_dsl="[DIFF_TARGET] app.py\nN0: app.py [func()]\n"),
        EvalSample(index=1, true_label=0, risk_score=0.3, confidence=0.8, epistemic_uncertainty=0.1, taxonomy_scores={}, language="python", category="real_revert", source_type="real_hotfix", input_dsl="[DIFF_TARGET] app.py\nN0: app.py [func()]\n"),
        EvalSample(index=2, true_label=0, risk_score=0.4, confidence=0.8, epistemic_uncertainty=0.1, taxonomy_scores={}, language="typescript", category="security_surface", source_type="mutation", input_dsl="[DIFF_TARGET] sec.ts\nN0: sec.ts [func()]\n"),
    ]
    sweep_results = [
        {"threshold": 0.40, "accuracy": 0.67, "precision": 0.5, "recall": 1.0, "specificity": 0.5, "f1": 0.67, "TP": 1, "FP": 1, "TN": 1, "FN": 0},
    ]
    report_file = tmp_path / "test_report.md"
    content = generate_markdown_report(
        samples=samples,
        threshold=0.50,
        sweep_results=sweep_results,
        fitted_T=1.93,
        ece_pre=0.14,
        ece_post=0.10,
        output_path=report_file,
    )
    assert "### 4.1 " in content
    assert "### 4.2 " in content
    assert "4.58" not in content  # Confirms no line-number bug
    assert "4.67" not in content
    assert report_file.exists()


def test_laya_decision_head_config_loading_and_precedence(tmp_path, monkeypatch):
    import json
    weights_dir = tmp_path / "fake_weights"
    weights_dir.mkdir()
    cfg = {
        "default_decision_threshold": 0.42,
        "calibrated_temperature": 1.85,
    }
    with open(weights_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    # 1. Config loading when no explicit param or env var
    head_cfg = LayaDecisionHead(weights_path=weights_dir, enabled=True)
    assert head_cfg.risk_threshold == 0.42
    assert head_cfg.temperature == 1.85

    # 2. Env var overrides config.json
    monkeypatch.setenv("CODE_ORACLE_RISK_THRESHOLD", "0.38")
    monkeypatch.setenv("CODE_ORACLE_TEMPERATURE", "2.10")
    head_env = LayaDecisionHead(weights_path=weights_dir, enabled=True)
    assert head_env.risk_threshold == 0.38
    assert head_env.temperature == 2.10

    # 3. Explicit argument overrides env var
    head_arg = LayaDecisionHead(weights_path=weights_dir, enabled=True, risk_threshold=0.25, temperature=1.2)
    assert head_arg.risk_threshold == 0.25
    assert head_arg.temperature == 1.2

