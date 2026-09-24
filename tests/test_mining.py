"""
Unit and integration tests for real-world repository mining tool,
dataset inspection tool, and generated ModernBERT training data.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from code_oracle.dataset import DatasetGenerator, DatasetRecord
from code_oracle.linearizer import estimate_tokens
from inspect_dataset import inspect_file, VALID_CATEGORIES, VALID_LANGUAGES
from mine_top_repos import balance_dataset, mine_all_top_repos


@pytest.fixture
def mini_repo_fixture(tmp_path: Path) -> Path:
    """Create a multi-file mini repository with caller-callee structure across files."""
    # callee module
    (tmp_path / "math_mod.py").write_text(
        "def compute_sum(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def helper_func(x: int) -> int:\n"
        "    return x * 2\n",
        encoding="utf-8",
    )
    # caller module
    (tmp_path / "app_main.py").write_text(
        "from math_mod import compute_sum\n\n"
        "def run_app():\n"
        "    total = compute_sum(10, 20)\n"
        "    return total\n",
        encoding="utf-8",
    )
    return tmp_path


def test_mine_repository_all_categories(mini_repo_fixture: Path):
    """Verify that mine_repository discovers and generates all mutation categories."""
    gen = DatasetGenerator(languages=["python"], seed=42)
    records = gen.mine_repository(mini_repo_fixture, max_samples=20)

    assert len(records) > 0
    categories = {r.category for r in records}
    assert "clean_pass" in categories
    assert "arity_breaking" in categories
    assert "deleted_symbol" in categories
    assert "keyword_changes" in categories
    assert "circular_imports" in categories

    for r in records:
        assert r.language == "python"
        assert estimate_tokens(r.input_dsl) <= 400
        if r.category == "clean_pass":
            assert r.label == 1
            assert 0.0 <= r.risk_score <= 0.25
        else:
            assert r.label == 0
            assert 0.75 <= r.risk_score <= 1.0


def test_inspect_dataset_clean_sample(tmp_path: Path):
    """Test that inspect_file returns passed=True for compliant JSONL data."""
    jsonl_path = tmp_path / "test_data.jsonl"
    records = [
        DatasetRecord(
            input_dsl="[DIFF_TARGET] a.py::f1 (MODIFIED)\n[GATE]\nSTATUS: APPROVED",
            label=1,
            risk_score=0.05,
            category="clean_pass",
            language="python",
        ),
        DatasetRecord(
            input_dsl="[DIFF_TARGET] a.py::f1 (MODIFIED)\n[GATE]\nSTATUS: REJECTED\nVIOLATIONS:\n  - ARITY_MISMATCH",
            label=0,
            risk_score=0.92,
            category="arity_breaking",
            language="python",
        ),
    ]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    summary = inspect_file(jsonl_path, max_token_limit=400)
    assert summary["passed"] is True
    assert summary["total_records"] == 2
    assert summary["violations_count"] == 0
    assert summary["labels"][1] == 1
    assert summary["labels"][0] == 1
    assert summary["token_stats"]["max"] <= 400


def test_inspect_dataset_catches_token_overflow(tmp_path: Path):
    """Test that inspect_file detects and flags token lengths exceeding 400."""
    jsonl_path = tmp_path / "overflow.jsonl"
    long_dsl = "[DIFF_TARGET] foo.py::bar\n" + " ".join(["token_item"] * 450)
    record = {
        "input_dsl": long_dsl,
        "label": 1,
        "risk_score": 0.05,
        "category": "clean_pass",
        "language": "python",
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    summary = inspect_file(jsonl_path, max_token_limit=400)
    assert summary["passed"] is False
    assert any("Token length" in v and "exceeds" in v for v in summary["violations"])


def test_inspect_dataset_catches_schema_and_value_violations(tmp_path: Path):
    """Test that inspect_file catches missing keys and invalid labels/scores."""
    jsonl_path = tmp_path / "bad_schema.jsonl"
    bad_records = [
        {"input_dsl": "test", "label": 99, "risk_score": 0.5, "category": "clean_pass", "language": "python"},
        {"input_dsl": "test", "label": 1, "risk_score": 0.99, "category": "clean_pass", "language": "python"},
        {"input_dsl": "test", "label": 0, "risk_score": 0.01, "category": "unknown_cat", "language": "python"},
        {"input_dsl": "", "label": 1, "risk_score": 0.1, "category": "clean_pass", "language": "brainfuck"},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in bad_records:
            f.write(json.dumps(r) + "\n")

    summary = inspect_file(jsonl_path, max_token_limit=400)
    assert summary["passed"] is False
    assert summary["violations_count"] >= 4


def test_balance_dataset_parity():
    """Test balance_dataset ensures exact 50% PASS / REJECT parity."""
    records = []
    for i in range(10):
        records.append(
            DatasetRecord(
                input_dsl="dsl_pass",
                label=1,
                risk_score=0.05,
                category="clean_pass",
                language="python",
            )
        )
    for i in range(30):
        records.append(
            DatasetRecord(
                input_dsl="dsl_reject",
                label=0,
                risk_score=0.95,
                category="arity_breaking",
                language="python",
            )
        )

    balanced = balance_dataset(records, target_count=16, seed=42)
    pos = sum(1 for r in balanced if r.label == 1)
    neg = sum(1 for r in balanced if r.label == 0)
    assert pos == neg
    assert pos + neg == 16


def test_inspect_dataset_cli(tmp_path: Path):
    """Test running tools/inspect_dataset.py via subprocess CLI."""
    jsonl_path = tmp_path / "sample.jsonl"
    records = [
        DatasetRecord(
            input_dsl="[DIFF_TARGET] a.py::f1 (MODIFIED)\n[GATE]\nSTATUS: APPROVED",
            label=1,
            risk_score=0.05,
            category="clean_pass",
            language="python",
        ),
        DatasetRecord(
            input_dsl="[DIFF_TARGET] a.py::f1 (MODIFIED)\n[GATE]\nSTATUS: REJECTED",
            label=0,
            risk_score=0.95,
            category="arity_breaking",
            language="python",
        ),
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    script_path = Path(__file__).resolve().parent.parent / "tools" / "inspect_dataset.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--file", str(jsonl_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "DATASET QUALITY INSPECTION REPORT" in res.stdout
    assert "PASSED" in res.stdout


def test_mine_top_repos_pipeline_execution(tmp_path: Path):
    """Test end-to-end execution of mine_all_top_repos with synthetic fallback."""
    out_dir = tmp_path / "data_out"
    train_n, val_n = mine_all_top_repos(
        cache_dir=tmp_path / "cache",
        output_dir=out_dir,
        target_samples=40,
        val_ratio=0.25,
        seed=100,
        custom_repos=[],  # forces synthetic generation across Tier 1 languages
    )

    assert train_n + val_n == 40
    assert (out_dir / "dataset_train.jsonl").exists()
    assert (out_dir / "dataset_val.jsonl").exists()

    summary_train = inspect_file(out_dir / "dataset_train.jsonl")
    summary_val = inspect_file(out_dir / "dataset_val.jsonl")
    assert summary_train["passed"] is True
    assert summary_val["passed"] is True


def test_production_dataset_files():
    """Verify that generated data/ datasets conform to all ModernBERT specifications."""
    data_dir = REPO_ROOT / "data"
    train_file = data_dir / "dataset_train.jsonl"
    val_file = data_dir / "dataset_val.jsonl"

    assert train_file.exists(), f"Missing production dataset: {train_file}"
    assert val_file.exists(), f"Missing production dataset: {val_file}"

    summary_train = inspect_file(train_file, max_token_limit=400)
    summary_val = inspect_file(val_file, max_token_limit=400)

    assert summary_train["passed"] is True, f"Train dataset violations: {summary_train['violations']}"
    assert summary_val["passed"] is True, f"Val dataset violations: {summary_val['violations']}"

    total_samples = summary_train["total_records"] + summary_val["total_records"]
    assert 2000 <= total_samples <= 4000, f"Expected 2000-4000 total samples, got {total_samples}"

    # Verify token length guarantees (< 400 tokens)
    assert summary_train["token_stats"]["max"] <= 400
    assert summary_val["token_stats"]["max"] <= 400

    # Verify class balance
    assert summary_train["labels"][1] == summary_train["labels"][0]
    assert summary_val["labels"][1] == summary_val["labels"][0]

    # Verify all 4 Tier 1 languages are represented
    for lang in ("python", "typescript", "go", "rust"):
        assert summary_train["languages"].get(lang, 0) > 0
        assert summary_val["languages"].get(lang, 0) > 0
