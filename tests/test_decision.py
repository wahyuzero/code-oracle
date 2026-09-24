"""
Tests for LayaDecisionHead, risk calibration, and neural decision integration.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from code_oracle.decision import LayaDecisionHead, VERIFICATION_QUESTIONS
from code_oracle.engine import TopoSliceEngine
from code_oracle.models import VerificationReport


def test_decision_head_fallback_without_weights():
    head = LayaDecisionHead(weights_path=None)
    assert not head.is_neural_enabled

    # Approved case
    status, conf, risk = head.predict(
        linearized_dsl="[DIFF_TARGET] app.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.98,
        has_violations=False,
    )
    assert status == "APPROVED"
    assert conf == 0.98
    assert risk == 0.05

    # Rejected case
    status, conf, risk = head.predict(
        linearized_dsl="[DIFF_TARGET] app.py\n[GATE]\nSTATUS: REJECTED",
        symbolic_status="REJECTED",
        symbolic_confidence=0.95,
        has_violations=True,
    )
    assert status == "REJECTED"
    assert conf == 1.0
    assert risk == 0.95


def test_decision_head_hard_veto_rule():
    head = LayaDecisionHead(weights_path=None)

    # Even if mock agent would say APPROVED, symbolic violations enforce REJECTED
    mock_agent = MagicMock()
    mock_agent.predict.return_value = {
        "answers": {
            "status": {"choice": "APPROVED", "confidence": 0.99},
            "risk": {"score": 0.1},
        }
    }
    head.agent = mock_agent
    head._loaded = True
    assert head.is_neural_enabled

    # Case with symbolic violations
    status, conf, risk = head.predict(
        linearized_dsl="[DIFF_TARGET] foo.py\n[GATE]\nSTATUS: REJECTED",
        symbolic_status="REJECTED",
        symbolic_confidence=0.9,
        has_violations=True,
    )
    assert status == "REJECTED"
    assert conf == 1.0
    assert risk == 0.95
    # Mock should not even need to be called due to hard veto
    mock_agent.predict.assert_not_called()


def test_decision_head_neural_inference_flow():
    head = LayaDecisionHead(weights_path=None)

    mock_agent = MagicMock()
    mock_agent.predict.return_value = {
        "answers": {
            "status": {"choice": "APPROVED", "confidence": 0.992},
            "risk": {"score": 0.4},
        }
    }
    head.agent = mock_agent
    head._loaded = True

    status, conf, risk = head.predict(
        linearized_dsl="[DIFF_TARGET] foo.py\n[GATE]\nSTATUS: APPROVED",
        symbolic_status="APPROVED",
        symbolic_confidence=0.95,
        has_violations=False,
    )

    mock_agent.predict.assert_called_once()
    assert status == "APPROVED"
    assert conf == 0.992
    assert risk == 0.1  # 0.4 / 4.0 = 0.1


def test_verification_report_includes_risk_score():
    report = VerificationReport(
        status="APPROVED",
        confidence=0.98,
        cycles_detected=[],
        invariant_violations=[],
        linearized_subgraph="[DIFF_TARGET] foo.py\n[GATE]\nSTATUS: APPROVED",
        affected_symbols=["foo"],
        latency_ms=1.25,
        risk_score=0.05,
    )

    d = report.to_dict()
    assert "risk_score" in d
    assert d["risk_score"] == 0.05
    assert d["status"] == "APPROVED"


def test_engine_verify_populates_risk_score(tmp_path: Path):
    file_path = tmp_path / "calc.py"
    file_path.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

    engine = TopoSliceEngine(workspace_root=tmp_path)
    clean_patch = "def add(a: int, b: int) -> int:\n    # clean patch\n    return a + b\n"
    rep = engine.verify("calc.py", clean_patch)

    assert rep.status == "APPROVED"
    assert rep.risk_score == 0.05
    assert rep.to_dict()["risk_score"] == 0.05
