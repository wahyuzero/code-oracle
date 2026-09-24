"""
Unit and integration tests for dataset mining, synthetic mutation engine,
and JSONL dataset generation across Tier 1 languages.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from code_oracle.dataset import DatasetGenerator, DatasetRecord, TEMPLATES


def test_dataset_record_schema():
    rec = DatasetRecord(
        input_dsl="[DIFF_TARGET] foo.py::bar (MODIFIED)\n[GATE]\nSTATUS: APPROVED",
        label=1,
        risk_score=0.05,
        category="clean_pass",
        language="python",
    )
    d = rec.to_dict()
    assert set(d.keys()) == {"input_dsl", "label", "risk_score", "category", "language"}
    assert d["label"] == 1
    assert d["risk_score"] == 0.05
    assert d["category"] == "clean_pass"
    assert d["language"] == "python"
    assert isinstance(d["input_dsl"], str)


def test_synthetic_mutation_categories():
    gen = DatasetGenerator(languages=["python"], seed=100)
    template = TEMPLATES["python"][0]
    records = gen.generate_pairs_for_template(template, "python")

    categories = {r.category for r in records}
    assert "clean_pass" in categories
    assert "arity_breaking" in categories
    assert "circular_imports" in categories
    assert "deleted_symbol" in categories

    for r in records:
        if r.category == "clean_pass":
            assert r.label == 1
            assert 0.0 <= r.risk_score <= 0.2
            assert "STATUS: APPROVED" in r.input_dsl
        else:
            assert r.label == 0
            assert 0.8 <= r.risk_score <= 1.0
            assert "STATUS: REJECTED" in r.input_dsl


def test_multi_language_synthetic_coverage():
    languages = ["python", "typescript", "go", "rust"]
    gen = DatasetGenerator(languages=languages, seed=42)
    records = gen.generate_synthetic_dataset(num_samples=40)

    covered_langs = {r.language for r in records}
    for lang in languages:
        assert lang in covered_langs

    # Check categories across all languages
    categories = {r.category for r in records}
    assert "clean_pass" in categories
    assert "arity_breaking" in categories
    assert "deleted_symbol" in categories


def test_dataset_balance():
    gen = DatasetGenerator(languages=["python", "typescript", "go", "rust"], seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        train_n, val_n = gen.generate_and_export(tmp, num_samples=30, val_ratio=0.2)

        train_file = tmp / "dataset_train.jsonl"
        val_file = tmp / "dataset_val.jsonl"

        assert train_file.exists()
        assert val_file.exists()

        all_records = []
        for fpath in (train_file, val_file):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))

        positives = [r for r in all_records if r["label"] == 1]
        negatives = [r for r in all_records if r["label"] == 0]

        # Positives and negatives should be balanced
        assert abs(len(positives) - len(negatives)) <= 1


def test_dataset_jsonl_schema_validation():
    gen = DatasetGenerator(languages=["python", "typescript", "go", "rust"], seed=123)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        gen.generate_and_export(tmp, num_samples=25, val_ratio=0.2)

        for fname in ("dataset_train.jsonl", "dataset_val.jsonl"):
            fpath = tmp / fname
            with open(fpath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    # Verify schema keys
                    assert set(data.keys()) == {
                        "input_dsl",
                        "label",
                        "risk_score",
                        "category",
                        "language",
                    }, f"Mismatch in {fname} line {idx}"
                    # Verify types
                    assert isinstance(data["input_dsl"], str)
                    assert len(data["input_dsl"]) > 0
                    assert data["label"] in (0, 1)
                    assert isinstance(data["risk_score"], (int, float))
                    assert 0.0 <= data["risk_score"] <= 1.0
                    assert isinstance(data["category"], str)
                    assert data["language"] in ("python", "typescript", "go", "rust")


def test_dataset_generator_cli_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        cli_script = Path(__file__).resolve().parent.parent / "tools" / "dataset_generator.py"

        cmd = [
            sys.executable,
            str(cli_script),
            "--output-dir",
            str(out_dir),
            "--num-samples",
            "16",
            "--val-ratio",
            "0.25",
            "--languages",
            "python,typescript,go,rust",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Dataset successfully generated" in res.stdout

        train_file = out_dir / "dataset_train.jsonl"
        val_file = out_dir / "dataset_val.jsonl"
        assert train_file.exists()
        assert val_file.exists()


def test_code_oracle_cli_dataset_subcommand():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "subcommand_out"
        cmd = [
            sys.executable,
            "-m",
            "code_oracle.cli",
            "dataset",
            "--output-dir",
            str(out_dir),
            "--num-samples",
            "12",
            "--json",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0

        data = json.loads(res.stdout)
        assert data["status"] == "SUCCESS"
        assert data["train_samples"] > 0
        assert data["val_samples"] > 0


def test_mine_repository():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create a mini repository with caller-callee structure
        (tmp / "calc.py").write_text("def multiply(a: int, b: int) -> int:\n    return a * b\n")
        (tmp / "app.py").write_text("from calc import multiply\n\ndef run():\n    return multiply(2, 3)\n")

        gen = DatasetGenerator(languages=["python"], seed=42)
        records = gen.mine_repository(tmp, max_samples=10)

        assert len(records) > 0
        categories = {r.category for r in records}
        assert "clean_pass" in categories
        assert "arity_breaking" in categories or "deleted_symbol" in categories
        # Verify labels
        positives = [r for r in records if r.label == 1]
        negatives = [r for r in records if r.label == 0]
        assert len(positives) > 0
        assert len(negatives) > 0


def test_mine_multilang_repository():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Go project
        (tmp / "calc.go").write_text("package main\n\nfunc Add(a, b int) int {\n    return a + b\n}\n")
        (tmp / "main.go").write_text("package main\n\nfunc Run() int {\n    return Add(1, 2)\n}\n")

        gen = DatasetGenerator(languages=["go"], seed=42)
        records = gen.mine_repository(tmp, max_samples=10)

        assert len(records) >= 2
        categories = {r.category for r in records}
        assert "clean_pass" in categories
        assert "arity_breaking" in categories
        assert all(r.language == "go" for r in records)

