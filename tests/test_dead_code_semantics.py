"""
Tests for Role 3: Embedded Dead Code Semantics Classifier.
Tests Stage 1 deterministic AST pruner, Stage 2 neural/heuristic classifier,
detector integration, suppression, CLI, and FastMCP server interface.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from code_oracle.dead_code.detector import DeadCodeDetector, detect_dead_code
from code_oracle.dead_code.models import (
    DeadCodeReport,
    DeadSymbol,
    SemanticClassification,
    SemanticDeadSymbol,
)
from code_oracle.dead_code.semantics import (
    DeadCodeSemanticsClassifier,
    vectorize_symbol,
)
from code_oracle.models import Symbol
from code_oracle.server import run_dead_code_detection


def test_vectorize_symbol():
    sym = Symbol(
        name="Contact",
        qualname="fastapi.openapi.models.Contact",
        file_path="fastapi/openapi/models.py",
        kind="class",
        lineno=10,
        end_lineno=25,
        signature="class Contact(BaseModel):",
        docstring="Contact information for the exposed API.",
        is_exported=True,
        visibility="public",
    )
    vec = vectorize_symbol(sym)
    assert "[SYM] fastapi.openapi.models.Contact" in vec
    assert "[KIND] class" in vec
    assert "[SIG] class Contact(BaseModel):" in vec
    assert "[FILE] fastapi/openapi/models.py" in vec
    assert "[VIS] export" in vec
    assert "[DOC] Contact information for the exposed API." in vec


def test_stage1_pruner_go_exported():
    classifier = DeadCodeSemanticsClassifier(enabled=False)

    go_sym = Symbol(
        name="MIMEJSON",
        qualname="binding.MIMEJSON",
        file_path="binding/binding.go",
        kind="constant",
        lineno=12,
        end_lineno=12,
        signature='const MIMEJSON = "application/json"',
        is_exported=True,
        visibility="public",
    )

    res = classifier.prune_stage1_public(go_sym)
    assert res is not None
    assert res.semantic_classification == SemanticClassification.PUBLIC_API_SURFACE
    assert res.calibrated_confidence == 0.05
    assert res.suppressed is True
    assert "Go exported identifier" in res.reason


def test_stage1_pruner_typescript_exported():
    classifier = DeadCodeSemanticsClassifier(enabled=False)

    ts_sym = Symbol(
        name="ServeStaticOptions",
        qualname="ServeStaticOptions",
        file_path="src/middleware/serve-static/index.ts",
        kind="interface",
        lineno=5,
        end_lineno=15,
        signature="export interface ServeStaticOptions",
        is_exported=True,
        visibility="public",
    )

    res = classifier.prune_stage1_public(ts_sym)
    assert res is not None
    assert res.semantic_classification == SemanticClassification.PUBLIC_API_SURFACE
    assert res.calibrated_confidence == 0.05
    assert res.suppressed is True


def test_stage1_pruner_rust_exported():
    classifier = DeadCodeSemanticsClassifier(enabled=False)

    rs_sym = Symbol(
        name="ServerConfig",
        qualname="ServerConfig",
        file_path="src/config.rs",
        kind="struct",
        lineno=10,
        end_lineno=20,
        signature="pub struct ServerConfig",
        is_exported=True,
        visibility="public",
    )

    res = classifier.prune_stage1_public(rs_sym)
    assert res is not None
    assert res.semantic_classification == SemanticClassification.PUBLIC_API_SURFACE
    assert res.calibrated_confidence == 0.05
    assert res.suppressed is True


def test_stage1_pruner_python_framework_and_public_path():
    classifier = DeadCodeSemanticsClassifier(enabled=False)

    # Class inheriting from BaseModel in models.py
    py_sym = Symbol(
        name="UserSchema",
        qualname="models.UserSchema",
        file_path="myapp/models.py",
        kind="class",
        lineno=20,
        end_lineno=30,
        signature="class UserSchema(BaseModel):",
        bases=["BaseModel"],
        is_exported=True,
        visibility="public",
    )

    res = classifier.prune_stage1_public(py_sym)
    assert res is not None
    assert res.semantic_classification == SemanticClassification.PUBLIC_API_SURFACE
    assert res.calibrated_confidence == 0.05
    assert res.suppressed is True


def test_stage2_cruft_and_internal_orphan_classification():
    classifier = DeadCodeSemanticsClassifier(enabled=False)

    # Genuine cruft symbol with "legacy" or "deprecated" in name/docstring
    dead_info = DeadSymbol(
        id="calc.py::_old_unused_calc",
        name="_old_unused_calc",
        qualname="_old_unused_calc",
        file_path="calc.py",
        kind="function",
        lineno=50,
        end_lineno=60,
        is_orphan=True,
        confidence=1.0,
    )
    cruft_sym = Symbol(
        name="_old_unused_calc",
        qualname="_old_unused_calc",
        file_path="calc.py",
        kind="function",
        lineno=50,
        end_lineno=60,
        docstring="legacy deprecated helper to be deleted",
        is_exported=False,
        visibility="internal",
    )

    res_cruft = classifier.classify_ambiguous_symbol(cruft_sym, dead_info)
    assert res_cruft.semantic_classification == SemanticClassification.GENUINE_CRUFT
    assert res_cruft.calibrated_confidence >= 0.90
    assert not res_cruft.suppressed

    # Internal orphan symbol without cruft clues
    orphan_dead_info = DeadSymbol(
        id="calc.py::_helper",
        name="_helper",
        qualname="_helper",
        file_path="calc.py",
        kind="function",
        lineno=70,
        end_lineno=75,
        is_orphan=True,
        confidence=1.0,
        reason="Unreferenced symbol with 0 incoming calls",
    )
    orphan_sym = Symbol(
        name="_helper",
        qualname="_helper",
        file_path="calc.py",
        kind="function",
        lineno=70,
        end_lineno=75,
        is_exported=False,
        visibility="internal",
    )

    res_orphan = classifier.classify_ambiguous_symbol(orphan_sym, orphan_dead_info)
    assert res_orphan.semantic_classification == SemanticClassification.INTERNAL_ORPHAN
    assert res_orphan.calibrated_confidence == 0.70
    assert not res_orphan.suppressed


def test_dead_code_detector_semantic_suppression_e2e(tmp_path: Path):
    # Create workspace with:
    # 1. Public API surface in models.py (FastAPI pattern)
    # 2. Genuine cruft in utils.py
    models_file = tmp_path / "models.py"
    models_file.write_text(
        """class PublicItem:
    \"\"\"Public schema for external clients.\"\"\"
    def __init__(self, name: str):
        self.name = name
""",
        encoding="utf-8",
    )

    utils_file = tmp_path / "utils.py"
    utils_file.write_text(
        """def _deprecated_old_helper():
    return 42
""",
        encoding="utf-8",
    )

    detector = DeadCodeDetector(workspace_root=tmp_path)

    # 1. Run without semantics (standard static reachability)
    rep_raw = detector.detect(include_unexported=True, semantic=False)
    raw_names = [s.name for s in rep_raw.dead_symbols]
    assert "PublicItem" in raw_names
    assert "_deprecated_old_helper" in raw_names
    assert len(rep_raw.suppressed_symbols) == 0

    # 2. Run with semantic classification and suppression
    rep_semantic = detector.detect(
        include_unexported=True,
        semantic=True,
        suppress_api=True,
    )
    active_names = [s.name for s in rep_semantic.dead_symbols]
    suppressed_names = [s.name for s in rep_semantic.suppressed_symbols]

    # PublicItem should be suppressed from active dead symbols
    assert "PublicItem" not in active_names
    assert "PublicItem" in suppressed_names

    # _deprecated_old_helper should remain active dead code
    assert "_deprecated_old_helper" in active_names

    # Check serialization
    d = rep_semantic.to_dict()
    assert "suppressed_symbols_count" in d
    assert d["suppressed_symbols_count"] == 1
    assert d["suppressed_symbols"][0]["name"] == "PublicItem"
    assert d["suppressed_symbols"][0]["semantic_classification"] == "PUBLIC_API_SURFACE"


def test_dead_code_report_formatting():
    sem_sym = SemanticDeadSymbol(
        id="app.py::dead_func",
        name="dead_func",
        qualname="dead_func",
        file_path="app.py",
        kind="function",
        lineno=10,
        end_lineno=20,
        is_orphan=True,
        is_transitive=False,
        cluster_id=None,
        raw_reachability_confidence=1.0,
        semantic_classification=SemanticClassification.GENUINE_CRUFT,
        calibrated_confidence=0.95,
        semantic_probabilities={"PUBLIC_API_SURFACE": 0.01, "INTERNAL_ORPHAN": 0.04, "GENUINE_CRUFT": 0.95},
        suppressed=False,
        reason="Genuine cruft / obsolete code",
    )
    suppressed_sym = SemanticDeadSymbol(
        id="app.py::public_api",
        name="public_api",
        qualname="public_api",
        file_path="app.py",
        kind="function",
        lineno=30,
        end_lineno=40,
        is_orphan=True,
        is_transitive=False,
        cluster_id=None,
        raw_reachability_confidence=1.0,
        semantic_classification=SemanticClassification.PUBLIC_API_SURFACE,
        calibrated_confidence=0.05,
        semantic_probabilities={"PUBLIC_API_SURFACE": 0.95, "INTERNAL_ORPHAN": 0.04, "GENUINE_CRUFT": 0.01},
        suppressed=True,
        reason="Stage 1 Pruner: Public interface",
    )

    report = DeadCodeReport(
        workspace_root="/test",
        total_symbols_scanned=10,
        dead_symbols=[sem_sym],
        suppressed_symbols=[suppressed_sym],
        roots_count=2,
        scanned_files_count=1,
        latency_ms=15.5,
    )

    # Check table format
    table = report.format_table()
    assert "GENUINE_CRUFT" in table
    assert "1 public API symbols suppressed" in table

    # Check text format
    text = report.format_text()
    assert "[GENUINE_CRUFT]" in text
    assert "1 public API symbols were suppressed" in text


def test_server_run_dead_code_detection(tmp_path: Path):
    (tmp_path / "api.py").write_text(
        """class ExternalConfig:
    \"\"\"Public API export.\"\"\"
    pass
""",
        encoding="utf-8",
    )

    res = run_dead_code_detection(
        workspace_dir=str(tmp_path),
        neural_semantics=True,
        suppress_public_api=True,
    )

    assert isinstance(res, dict)
    assert "dead_symbols" in res
    assert "suppressed_symbols" in res
    assert any(s["name"] == "ExternalConfig" for s in res["suppressed_symbols"])
