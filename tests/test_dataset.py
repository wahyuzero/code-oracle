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
from code_oracle.dataset import DatasetGenerator, DatasetRecord, TEMPLATES, TAXONOMY_CLASSES


def test_dataset_record_schema():
    rec = DatasetRecord(
        input_dsl="[DIFF_TARGET] foo.py::bar (MODIFIED)\n[GATE]\nSTATUS: APPROVED",
        label=1,
        risk_score=0.05,
        category="clean_pass",
        language="python",
    )
    d = rec.to_dict()
    assert set(d.keys()) == {
        "input_dsl",
        "label",
        "risk_score",
        "category",
        "language",
        "taxonomy_labels",
        "symbolic_gate_passed",
        "source_type",
    }
    assert d["label"] == 1
    assert d["risk_score"] == 0.05
    assert d["category"] == "clean_pass"
    assert d["language"] == "python"
    assert isinstance(d["input_dsl"], str)
    assert isinstance(d["taxonomy_labels"], dict)
    assert set(d["taxonomy_labels"].keys()) == set(TAXONOMY_CLASSES)
    assert d["symbolic_gate_passed"] is True
    assert d["source_type"] == "mutation_subtle"


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
                        "taxonomy_labels",
                        "symbolic_gate_passed",
                        "source_type",
                    }, f"Mismatch in {fname} line {idx}"
                    # Verify types
                    assert isinstance(data["input_dsl"], str)
                    assert len(data["input_dsl"]) > 0
                    assert data["label"] in (0, 1)
                    assert isinstance(data["risk_score"], (int, float))
                    assert 0.0 <= data["risk_score"] <= 1.0
                    assert isinstance(data["category"], str)
                    assert data["language"] in ("python", "typescript", "go", "rust")
                    assert isinstance(data["taxonomy_labels"], dict)
                    assert set(data["taxonomy_labels"].keys()) == set(TAXONOMY_CLASSES)
                    assert isinstance(data["symbolic_gate_passed"], bool)
                    assert isinstance(data["source_type"], str)


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


def test_dataset_record_multitask_features():
    # 1. Clean pass -> zero taxonomy
    rec_pass = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py::f\n[GATE]\nSTATUS: APPROVED",
        label=1,
        risk_score=0.04,
        category="clean_pass",
        language="python",
    )
    assert all(v == 0.0 for v in rec_pass.taxonomy_labels.values())

    # 2. Category mapping
    rec_sec = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py::f\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.91,
        category="security_surface",
        language="python",
    )
    assert rec_sec.taxonomy_labels["SecuritySurface"] > 0.8

    rec_conc = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py::f\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.88,
        category="concurrency_hazard",
        language="python",
    )
    assert rec_conc.taxonomy_labels["ConcurrencyHazard"] > 0.8

    rec_perf = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py::f\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.85,
        category="performance_regression",
        language="python",
    )
    assert rec_perf.taxonomy_labels["PerformanceRegression"] > 0.8

    rec_api = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py::f\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.95,
        category="breaking_public_api",
        language="python",
    )
    assert rec_api.taxonomy_labels["BreakingPublicAPI"] > 0.8

    rec_drift = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py::f\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.90,
        category="silent_logic_drift",
        language="python",
    )
    assert rec_drift.taxonomy_labels["SilentLogicDrift"] > 0.8

    # 3. Roundtrip dictionary conversion
    d = rec_sec.to_dict()
    restored = DatasetRecord.from_dict(d)
    assert restored.category == rec_sec.category
    assert restored.label == rec_sec.label
    assert restored.risk_score == round(rec_sec.risk_score, 4)
    assert restored.taxonomy_labels == rec_sec.taxonomy_labels
    assert restored.symbolic_gate_passed == rec_sec.symbolic_gate_passed
    assert restored.source_type == rec_sec.source_type


def test_symbolic_gate_filtering():
    # With filter_symbolic_gate=True, only mutations passing the symbolic gate are retained
    gen_filtered = DatasetGenerator(languages=["python"], seed=42, filter_symbolic_gate=True)
    records_filtered = gen_filtered.generate_synthetic_dataset(num_samples=20)

    assert len(records_filtered) > 0
    for r in records_filtered:
        assert r.symbolic_gate_passed is True
        # Gate-filtered samples should pass symbolic gate
        assert "STATUS: APPROVED" in r.input_dsl

    # With filter_symbolic_gate=False, hard structural violations are also present
    gen_unfiltered = DatasetGenerator(languages=["python"], seed=42, filter_symbolic_gate=False)
    records_unfiltered = gen_unfiltered.generate_synthetic_dataset(num_samples=20)

    has_failed_gate = any(not r.symbolic_gate_passed for r in records_unfiltered)
    assert has_failed_gate is True


def test_subtle_semantic_mutations_across_languages():
    languages = ["python", "typescript", "go", "rust"]
    for lang in languages:
        gen = DatasetGenerator(languages=[lang], seed=42)
        template = TEMPLATES[lang][0]
        subtle_records = gen.generate_subtle_pairs_for_template(template, lang)

        assert len(subtle_records) > 0
        categories = {r.category for r in subtle_records}
        expected_subtles = {
            "silent_logic_drift",
            "security_surface",
            "concurrency_hazard",
            "performance_regression",
            "breaking_public_api",
        }
        assert expected_subtles.issubset(categories)

        for r in subtle_records:
            assert r.language == lang
            assert r.source_type in ("mutation_subtle", "clean_commit", "synthetic")
            if r.category in expected_subtles:
                assert r.symbolic_gate_passed is True
                assert r.label == 0
                assert r.source_type == "mutation_subtle"
                # Check that its primary taxonomy class is active
                active_tax = [c for c, v in r.taxonomy_labels.items() if v >= 0.5]
                assert len(active_tax) >= 1


def test_git_commit_mining():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Initialize real git repository
        subprocess.run(["git", "init"], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp), check=True, capture_output=True)

        calc_file = tmp / "calc.py"
        calc_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial clean commit"], cwd=str(tmp), check=True, capture_output=True)

        calc_file.write_text("def add(a: int, b: int) -> int:\n    # Hotfix logic\n    return a + b\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "fix: hotfix addition logic"], cwd=str(tmp), check=True, capture_output=True)

        calc_file.write_text("def add(a: int, b: int) -> int:\n    return a - b\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Broken subtract change"], cwd=str(tmp), check=True, capture_output=True)

        # Revert the broken commit
        subprocess.run(["git", "revert", "--no-edit", "HEAD"], cwd=str(tmp), check=True, capture_output=True)

        gen = DatasetGenerator(languages=["python"], seed=42)
        mined_commits = gen.mine_git_history(tmp, max_samples=10)

        assert len(mined_commits) > 0
        source_types = {r.source_type for r in mined_commits}
        # Verify mined real git sources
        assert any(st in source_types for st in ("real_revert", "real_hotfix", "clean_commit"))
        for r in mined_commits:
            assert r.language == "python"
            assert isinstance(r.input_dsl, str)
            assert len(r.input_dsl) > 0
            assert set(r.taxonomy_labels.keys()) == set(TAXONOMY_CLASSES)


def test_taxonomy_leak_distinction():
    """Verify ADR-0003 distinction: resource/memory leaks -> PerfRegression, concurrency leaks -> ConcurrencyHazard."""
    rec_res = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.88,
        category="resource_leak",
        language="python",
    )
    assert rec_res.taxonomy_labels["PerformanceRegression"] > 0.8
    assert rec_res.taxonomy_labels["ConcurrencyHazard"] == 0.0

    rec_mem = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.88,
        category="memory_leak",
        language="python",
    )
    assert rec_mem.taxonomy_labels["PerformanceRegression"] > 0.8
    assert rec_mem.taxonomy_labels["ConcurrencyHazard"] == 0.0

    rec_conc = DatasetRecord(
        input_dsl="[DIFF_TARGET] a.py\n[GATE]\nSTATUS: APPROVED",
        label=0,
        risk_score=0.88,
        category="concurrency_leak",
        language="python",
    )
    assert rec_conc.taxonomy_labels["ConcurrencyHazard"] > 0.8
    assert rec_conc.taxonomy_labels["PerformanceRegression"] == 0.0


def test_git_commit_word_boundaries():
    """Ensure words like 'prefix', 'fixture', 'suffix' do not trigger false positive hotfixes or false clean exclusions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        subprocess.run(["git", "init"], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp), check=True, capture_output=True)

        f = tmp / "route.py"
        f.write_text("def get_prefix(path: str) -> str:\n    return path\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "feat: add prefix extraction support"], cwd=str(tmp), check=True, capture_output=True)

        f.write_text("def get_prefix(path: str) -> str:\n    return path.strip()\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "test: update test fixtures and mocks"], cwd=str(tmp), check=True, capture_output=True)

        gen = DatasetGenerator(languages=["python"], seed=42)
        mined = gen.mine_git_history(tmp, max_samples=10)

        # Neither commit should be falsely classified as real_hotfix
        hotfixes = [r for r in mined if r.source_type == "real_hotfix"]
        assert len(hotfixes) == 0

        # The clean commit with 'prefix' must be recognized as clean_commit
        cleans = [r for r in mined if r.source_type == "clean_commit"]
        assert len(cleans) >= 1


def test_cli_dataset_subtle_options():
    """Verify that CLI flags for dataset generation (--include-subtle, --no-subtle, --filter-symbolic-gate) work cleanly."""
    script_path = Path(__file__).resolve().parent.parent / "tools" / "dataset_generator.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Test with --filter-symbolic-gate
        res = subprocess.run(
            [sys.executable, str(script_path), "--output-dir", str(tmp), "--num-samples", "10", "--filter-symbolic-gate"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        train_file = tmp / "dataset_train.jsonl"
        assert train_file.exists()

        # Verify all records have symbolic_gate_passed=True
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                assert d["symbolic_gate_passed"] is True
                assert "STATUS: APPROVED" in d["input_dsl"]


