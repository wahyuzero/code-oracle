"""
Tests for Role 4: Multi-Task Risk Taxonomy with Epistemic Uncertainty.
Tests ModernBERTMultiTaskModel architecture, bounded damping, loss functions,
EnhancedDecisionResult, EnhancedVerificationReport, TopoSliceEngine integration, and CLI.
"""

import math
from pathlib import Path
import pytest
import torch

from code_oracle.decision import (
    EnhancedDecisionResult,
    LayaDecisionHead,
    ModernBERTMultiTaskModel,
)
from code_oracle.engine import TopoSliceEngine
from code_oracle.models import EnhancedVerificationReport, RiskTaxonomyScores
from code_oracle.server import verify_patch


def test_multitask_model_initialization():
    model = ModernBERTMultiTaskModel(hidden_size=768, num_taxonomy_classes=5)
    assert model.hidden_size == 768
    assert model.num_taxonomy_classes == 5
    assert len(model.TAXONOMY_CLASSES) == 5
    assert "BreakingPublicAPI" in model.TAXONOMY_CLASSES
    assert "SecuritySurface" in model.TAXONOMY_CLASSES
    assert "ConcurrencyHazard" in model.TAXONOMY_CLASSES
    assert "PerformanceRegression" in model.TAXONOMY_CLASSES
    assert "SilentLogicDrift" in model.TAXONOMY_CLASSES


def test_multitask_model_forward_shapes():
    model = ModernBERTMultiTaskModel(hidden_size=768, num_taxonomy_classes=5)
    model.eval()

    batch_size = 4
    h_pool = torch.randn(batch_size, 768)

    with torch.no_grad():
        out = model(h_pool)

    # Head 1: Continuous Risk
    assert out["risk_score"].shape == (batch_size, 1)
    assert (out["risk_score"] >= 0.0).all() and (out["risk_score"] <= 1.0).all()

    # Head 2: Multi-Label Taxonomy
    assert out["taxonomy_logits"].shape == (batch_size, 5)
    assert out["taxonomy_probs"].shape == (batch_size, 5)
    assert (out["taxonomy_probs"] >= 0.0).all() and (out["taxonomy_probs"] <= 1.0).all()

    # Head 3: Epistemic Uncertainty
    assert out["log_variance"].shape == (batch_size, 1)
    assert (out["log_variance"] >= -6.0).all() and (out["log_variance"] <= 6.0).all()
    assert (out["variance"] > 0.0).all()
    assert out["confidence"].shape == (batch_size, 1)
    assert (out["confidence"] >= 0.0).all() and (out["confidence"] <= 1.0).all()


def test_bounded_damping_clamps_extreme_values():
    model = ModernBERTMultiTaskModel(hidden_size=768, num_taxonomy_classes=5)
    model.eval()

    # Force large weights on uncertainty head to produce extreme logits
    with torch.no_grad():
        for p in model.uncertainty_head.parameters():
            p.fill_(100.0)

    h_pool = torch.ones(2, 768)
    with torch.no_grad():
        out = model(h_pool)

    # Must be clamped to 6.0
    assert torch.allclose(out["log_variance"], torch.tensor([[6.0], [6.0]]))
    assert not torch.isinf(out["variance"]).any()
    assert not torch.isnan(out["variance"]).any()


def test_multitask_model_compute_loss():
    model = ModernBERTMultiTaskModel(hidden_size=768, num_taxonomy_classes=5)

    batch_size = 8
    risk_pred = torch.tensor([[0.2], [0.8], [0.5], [0.1], [0.9], [0.4], [0.3], [0.7]])
    risk_target = torch.tensor([[0.0], [1.0], [0.5], [0.0], [1.0], [0.5], [0.0], [1.0]])

    taxonomy_logits = torch.randn(batch_size, 5)
    taxonomy_target = torch.randint(0, 2, (batch_size, 5)).float()

    log_variance = torch.zeros(batch_size, 1)

    loss_dict = model.compute_loss(
        risk_pred=risk_pred,
        risk_target=risk_target,
        taxonomy_logits=taxonomy_logits,
        taxonomy_target=taxonomy_target,
        log_variance=log_variance,
    )

    assert "loss_total" in loss_dict
    assert "loss_risk" in loss_dict
    assert "loss_taxonomy" in loss_dict
    assert "loss_uncertainty" in loss_dict

    assert loss_dict["loss_total"].item() > 0.0
    assert loss_dict["loss_risk"].item() >= 0.0
    assert loss_dict["loss_taxonomy"].item() >= 0.0
    assert loss_dict["loss_uncertainty"].item() >= 0.0

    # Test with homoscedastic weights
    weighted_loss = model.compute_loss(
        risk_pred=risk_pred,
        risk_target=risk_target,
        taxonomy_logits=taxonomy_logits,
        taxonomy_target=taxonomy_target,
        log_variance=log_variance,
        homoscedastic_weights=(1.0, 1.0, 1.0),
    )
    assert weighted_loss["loss_total"].item() > 0.0


def test_decision_head_predict_multi_task_approved():
    head = LayaDecisionHead(weights_path=None)

    result = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] calc.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.98,
        has_violations=False,
    )

    assert isinstance(result, EnhancedDecisionResult)
    assert result.status == "APPROVED"
    assert result.confidence == 0.98
    assert result.risk_score == 0.05
    assert result.epistemic_uncertainty == 0.02
    assert result.active_risk_categories == []
    assert not result.is_neural_calibrated


def test_decision_head_predict_multi_task_rejected_arity():
    head = LayaDecisionHead(weights_path=None)

    result = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] billing.py\n[GATE]\nSTATUS: REJECTED",
        symbolic_status="REJECTED",
        symbolic_confidence=0.95,
        has_violations=True,
        violations=["ARITY_MISMATCH: charge() requires at least 2 arguments, 1 provided"],
    )

    assert isinstance(result, EnhancedDecisionResult)
    assert result.status == "REJECTED"
    assert result.confidence == 1.0
    assert result.risk_score == 0.95
    assert "BreakingPublicAPI" in result.active_risk_categories


def test_decision_head_predict_multi_task_rejected_cycle():
    head = LayaDecisionHead(weights_path=None)

    result = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] graph.py\n[GATE]\nSTATUS: REJECTED",
        symbolic_status="REJECTED",
        symbolic_confidence=0.95,
        has_violations=True,
        violations=["CIRCULAR_DEPENDENCY detected: a -> b -> a"],
        cycles=[["a", "b", "a"]],
    )

    assert isinstance(result, EnhancedDecisionResult)
    assert result.status == "REJECTED"
    assert "ConcurrencyHazard" in result.active_risk_categories


def test_decision_head_backward_compatible_tuple_unpacking():
    head = LayaDecisionHead(weights_path=None)

    res = head.predict_multi_task(
        linearized_dsl="[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.98,
        has_violations=False,
    )

    # Test unpacking directly
    status, conf, risk = res
    assert status == "APPROVED"
    assert conf == 0.98
    assert risk == 0.05


def test_enhanced_verification_report_serialization():
    tax = RiskTaxonomyScores(
        breaking_public_api=0.95,
        security_surface=0.10,
        concurrency_hazard=0.05,
        performance_regression=0.05,
        silent_logic_drift=0.85,
    )
    report = EnhancedVerificationReport(
        status="REJECTED",
        confidence=1.0,
        risk_score=0.95,
        epistemic_uncertainty=0.01,
        risk_taxonomy=tax,
        active_risk_categories=tax.active_categories(threshold=0.5),
        cycles_detected=[],
        invariant_violations=["ARITY_MISMATCH: unexpected keyword 'foo'"],
        linearized_subgraph="[DIFF_TARGET] foo.py\n[GATE]\nSTATUS: REJECTED",
        affected_symbols=["foo.py::bar"],
        latency_ms=12.34,
        is_neural_calibrated=False,
    )

    d = report.to_dict()
    assert d["status"] == "REJECTED"
    assert d["confidence"] == 1.0
    assert d["risk_score"] == 0.95
    assert d["epistemic_uncertainty"] == 0.01
    assert "risk_taxonomy" in d
    assert d["risk_taxonomy"]["breaking_public_api"] == 0.95
    assert "BreakingPublicAPI" in d["active_risk_categories"]
    assert "SilentLogicDrift" in d["active_risk_categories"]
    assert d["latency_ms"] == 12.34
    assert d["is_neural_calibrated"] is False


def test_engine_verify_multi_task_integration(tmp_path: Path):
    file_path = tmp_path / "calc.py"
    file_path.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

    engine = TopoSliceEngine(workspace_root=tmp_path)
    clean_patch = "def add(a: int, b: int) -> int:\n    # clean patch\n    return a + b\n"
    rep = engine.verify("calc.py", clean_patch, taxonomy_threshold=0.5)

    assert isinstance(rep, EnhancedVerificationReport)
    assert rep.status == "APPROVED"
    assert rep.confidence >= 0.95
    assert rep.risk_score == 0.05
    assert rep.epistemic_uncertainty == 0.02
    assert rep.active_risk_categories == []

    d = rep.to_dict()
    assert "risk_taxonomy" in d
    assert "epistemic_uncertainty" in d
    assert "active_risk_categories" in d


def test_server_verify_patch_taxonomy_endpoint(tmp_path: Path):
    (tmp_path / "service.py").write_text("def run():\n    return True\n")
    patch = "def run(flag: bool = True):\n    return flag\n"

    res = verify_patch(
        file_path="service.py",
        patch_content=patch,
        workspace_dir=str(tmp_path),
        taxonomy_threshold=0.5,
    )

    assert isinstance(res, dict)
    assert res["status"] == "APPROVED"
    assert "risk_taxonomy" in res
    assert "epistemic_uncertainty" in res
    assert "active_risk_categories" in res


def test_multitask_model_compute_loss_shape_resilience():
    model = ModernBERTMultiTaskModel(hidden_size=768, num_taxonomy_classes=5)

    # 1D risk_pred and 1D risk_target, with 2D log_variance (model output)
    pred_1d = torch.tensor([0.2, 0.4])
    target_1d = torch.tensor([0.0, 1.0])
    logits = torch.randn(2, 5)
    t_target = torch.zeros(2, 5, dtype=torch.long)  # test integer target casting
    log_var_2d = torch.zeros(2, 1)

    loss_dict = model.compute_loss(
        risk_pred=pred_1d,
        risk_target=target_1d,
        taxonomy_logits=logits,
        taxonomy_target=t_target,
        log_variance=log_var_2d,
        delta=0.1,
    )

    assert "loss_total" in loss_dict
    assert "loss_risk" in loss_dict
    # Mathematically exact Huber loss for [0.2, 0.0] and [0.4, 1.0] with delta=0.1 is 0.035
    assert abs(loss_dict["loss_risk"].item() - 0.035) < 1e-5


def test_multitask_model_homoscedastic_autograd():
    model = ModernBERTMultiTaskModel(hidden_size=768, num_taxonomy_classes=5)

    p1 = torch.nn.Parameter(torch.tensor(1.0))
    p2 = torch.nn.Parameter(torch.tensor(1.0))
    p3 = torch.nn.Parameter(torch.tensor(1.0))

    pred = torch.tensor([[0.3], [0.7]])
    target = torch.tensor([[0.0], [1.0]])
    logits = torch.randn(2, 5)
    t_target = torch.zeros(2, 5)
    log_var = torch.zeros(2, 1)

    loss_dict = model.compute_loss(
        risk_pred=pred,
        risk_target=target,
        taxonomy_logits=logits,
        taxonomy_target=t_target,
        log_variance=log_var,
        homoscedastic_weights=(p1, p2, p3),
    )

    loss_dict["loss_total"].backward()
    assert p1.grad is not None and p2.grad is not None and p3.grad is not None
    assert not torch.isnan(p1.grad) and not torch.isnan(p2.grad) and not torch.isnan(p3.grad)

