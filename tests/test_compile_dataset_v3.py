"""
Unit and integration tests for Dataset v3 compilation pipeline and artifacts.

Verifies:
1. v3-medium: 5,000 samples total (4,000 train, 1,000 val), exact 50% PASS / 50% REJECT
   per split and per Tier 1 language (1,000/lang train, 250/lang val).
2. v3-full: 10,000 samples total (8,000 train, 2,000 val), exact 50% PASS / 50% REJECT
   per split and per Tier 1 language (2,000/lang train, 500/lang val).
3. Strict Quality Control Gates:
   - 100% symbolic_gate_passed == True.
   - estimate_tokens(input_dsl) <= 400.
   - Full ADR-0003 multi-task risk taxonomy present and valid.
   - Bounded risk scores matching ground truth labels.
4. Packaging: Zip files data/code_oracle_dataset_v3_medium.zip and
   data/code_oracle_dataset_v3_full.zip contain valid train, val, and heldout files.
5. Colab Notebooks: Laya_Code_Oracle_MultiTask_*.ipynb integrity and parameters.
6. Pipeline unit functions: validate_qc_gate, stratify_and_balance_dataset, and compile_v3_datasets.
"""

import base64
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from code_oracle.dataset import (
    TAXONOMY_CLASSES,
    DatasetRecord,
)
from code_oracle.linearizer import estimate_tokens
from compile_dataset_v3 import (
    TIER1_LANGUAGES,
    compile_hybrid_dataset,
    compile_v3_datasets,
    harvest_typescript_authentic_records,
    stratify_and_balance_dataset,
    validate_qc_gate,
)

DATA_DIR = REPO_ROOT / "data"
HYBRID_DIR = DATA_DIR / "v3_hybrid"
MEDIUM_DIR = DATA_DIR / "v3_medium"
FULL_DIR = DATA_DIR / "v3_full"
HYBRID_ZIP = DATA_DIR / "code_oracle_dataset_v3_hybrid.zip"
MEDIUM_ZIP = DATA_DIR / "code_oracle_dataset_v3_medium.zip"
FULL_ZIP = DATA_DIR / "code_oracle_dataset_v3_full.zip"


# ============================================================================
# Helpers
# ============================================================================


def load_jsonl(fpath: Path) -> List[Dict]:
    assert fpath.exists(), f"File {fpath} does not exist"
    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# 1. Artifact Existence & Zip Packaging Tests
# ============================================================================


def test_v3_hybrid_files_exist():
    """Verify v3-hybrid directory and zip archive exist and are non-empty."""
    assert HYBRID_DIR.exists() and HYBRID_DIR.is_dir()
    assert (HYBRID_DIR / "dataset_train.jsonl").exists()
    assert (HYBRID_DIR / "dataset_val.jsonl").exists()
    assert (HYBRID_DIR / "dataset_heldout_eval.jsonl").exists()

    assert HYBRID_ZIP.exists()
    assert HYBRID_ZIP.stat().st_size > 50_000  # at least ~50 KB

    with zipfile.ZipFile(HYBRID_ZIP, "r") as zf:
        namelist = zf.namelist()
        assert "dataset_train.jsonl" in namelist
        assert "dataset_val.jsonl" in namelist
        assert "dataset_heldout_eval.jsonl" in namelist


def test_v3_medium_files_exist():
    """Verify v3-medium directory and zip archive exist and are non-empty."""
    assert MEDIUM_DIR.exists() and MEDIUM_DIR.is_dir()
    assert (MEDIUM_DIR / "dataset_train.jsonl").exists()
    assert (MEDIUM_DIR / "dataset_val.jsonl").exists()
    assert (MEDIUM_DIR / "dataset_heldout_eval.jsonl").exists()

    assert MEDIUM_ZIP.exists()
    assert MEDIUM_ZIP.stat().st_size > 50_000  # at least ~50 KB

    with zipfile.ZipFile(MEDIUM_ZIP, "r") as zf:
        namelist = zf.namelist()
        assert "dataset_train.jsonl" in namelist
        assert "dataset_val.jsonl" in namelist
        assert "dataset_heldout_eval.jsonl" in namelist


def test_v3_full_files_exist():
    """Verify v3-full directory and zip archive exist and are non-empty."""
    assert FULL_DIR.exists() and FULL_DIR.is_dir()
    assert (FULL_DIR / "dataset_train.jsonl").exists()
    assert (FULL_DIR / "dataset_val.jsonl").exists()
    assert (FULL_DIR / "dataset_heldout_eval.jsonl").exists()

    assert FULL_ZIP.exists()
    assert FULL_ZIP.stat().st_size > 100_000  # at least ~100 KB

    with zipfile.ZipFile(FULL_ZIP, "r") as zf:
        namelist = zf.namelist()
        assert "dataset_train.jsonl" in namelist
        assert "dataset_val.jsonl" in namelist
        assert "dataset_heldout_eval.jsonl" in namelist


# ============================================================================
# 2. Scale, Stratification & 50/50 Balance Tests
# ============================================================================


def test_v3_hybrid_sample_counts_and_balance():
    """Verify v3-hybrid has exact counts (3,920 train, 980 val, 4,900 total) and 50/50 balance."""
    train_recs = load_jsonl(HYBRID_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(HYBRID_DIR / "dataset_val.jsonl")

    assert len(train_recs) == 3920
    assert len(val_recs) == 980
    assert len(train_recs) + len(val_recs) == 4900

    # 50/50 overall balance
    train_pass = sum(1 for r in train_recs if r["label"] == 1)
    train_reject = sum(1 for r in train_recs if r["label"] == 0)
    assert train_pass == 1960
    assert train_reject == 1960

    val_pass = sum(1 for r in val_recs if r["label"] == 1)
    val_reject = sum(1 for r in val_recs if r["label"] == 0)
    assert val_pass == 490
    assert val_reject == 490

    # Expected per-language distribution (Golden Hybrid Configuration):
    # Total: Python=1000, Go=1000, TS=1100, Rust=1800
    # Train (80%): Python=800 (400P/400R), Go=800 (400P/400R), TS=880 (440P/440R), Rust=1440 (720P/720R)
    # Val (20%): Python=200 (100P/100R), Go=200 (100P/100R), TS=220 (110P/110R), Rust=360 (180P/180R)
    expected_train = {
        "python": (800, 400, 400),
        "go": (800, 400, 400),
        "typescript": (880, 440, 440),
        "rust": (1440, 720, 720),
    }
    expected_val = {
        "python": (200, 100, 100),
        "go": (200, 100, 100),
        "typescript": (220, 110, 110),
        "rust": (360, 180, 180),
    }

    for lang, (tot, p, r) in expected_train.items():
        l_train = [rec for rec in train_recs if rec["language"] == lang]
        assert len(l_train) == tot, f"Mismatch for {lang} train total: got {len(l_train)}, expected {tot}"
        assert sum(1 for rec in l_train if rec["label"] == 1) == p
        assert sum(1 for rec in l_train if rec["label"] == 0) == r

    for lang, (tot, p, r) in expected_val.items():
        l_val = [rec for rec in val_recs if rec["language"] == lang]
        assert len(l_val) == tot, f"Mismatch for {lang} val total: got {len(l_val)}, expected {tot}"
        assert sum(1 for rec in l_val if rec["label"] == 1) == p
        assert sum(1 for rec in l_val if rec["label"] == 0) == r


def test_v3_medium_sample_counts_and_balance():
    """Verify v3-medium has exact counts and 50/50 balance across splits & languages."""
    train_recs = load_jsonl(MEDIUM_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(MEDIUM_DIR / "dataset_val.jsonl")

    assert len(train_recs) == 4000
    assert len(val_recs) == 1000
    assert len(train_recs) + len(val_recs) == 5000

    # 50/50 overall balance
    train_pass = sum(1 for r in train_recs if r["label"] == 1)
    train_reject = sum(1 for r in train_recs if r["label"] == 0)
    assert train_pass == 2000
    assert train_reject == 2000

    val_pass = sum(1 for r in val_recs if r["label"] == 1)
    val_reject = sum(1 for r in val_recs if r["label"] == 0)
    assert val_pass == 500
    assert val_reject == 500

    # Equal representation per language
    for lang in TIER1_LANGUAGES:
        l_train = [r for r in train_recs if r["language"] == lang]
        assert len(l_train) == 1000
        assert sum(1 for r in l_train if r["label"] == 1) == 500
        assert sum(1 for r in l_train if r["label"] == 0) == 500

        l_val = [r for r in val_recs if r["language"] == lang]
        assert len(l_val) == 250
        assert sum(1 for r in l_val if r["label"] == 1) == 125
        assert sum(1 for r in l_val if r["label"] == 0) == 125


def test_v3_full_sample_counts_and_balance():
    """Verify v3-full has exact counts and 50/50 balance across splits & languages."""
    train_recs = load_jsonl(FULL_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(FULL_DIR / "dataset_val.jsonl")

    assert len(train_recs) == 8000
    assert len(val_recs) == 2000
    assert len(train_recs) + len(val_recs) == 10000

    # 50/50 overall balance
    train_pass = sum(1 for r in train_recs if r["label"] == 1)
    train_reject = sum(1 for r in train_recs if r["label"] == 0)
    assert train_pass == 4000
    assert train_reject == 4000

    val_pass = sum(1 for r in val_recs if r["label"] == 1)
    val_reject = sum(1 for r in val_recs if r["label"] == 0)
    assert val_pass == 1000
    assert val_reject == 1000

    # Equal representation per language
    for lang in TIER1_LANGUAGES:
        l_train = [r for r in train_recs if r["language"] == lang]
        assert len(l_train) == 2000
        assert sum(1 for r in l_train if r["label"] == 1) == 1000
        assert sum(1 for r in l_train if r["label"] == 0) == 1000

        l_val = [r for r in val_recs if r["language"] == lang]
        assert len(l_val) == 500
        assert sum(1 for r in l_val if r["label"] == 1) == 250
        assert sum(1 for r in l_val if r["label"] == 0) == 250


# ============================================================================
# 3. Quality Control Invariants & Taxonomy Coverage
# ============================================================================


def test_quality_control_invariants_hybrid():
    """Check QC gate invariants on v3-hybrid train and val records."""
    train_recs = load_jsonl(HYBRID_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(HYBRID_DIR / "dataset_val.jsonl")

    all_recs = train_recs + val_recs
    for r in all_recs:
        # Symbolic gate must be passed
        assert r.get("symbolic_gate_passed") is True, f"Gate failed for {r}"

        # Token ceiling <= 400
        tok_len = estimate_tokens(r["input_dsl"])
        assert 0 < tok_len <= 400, f"Token ceiling violated: {tok_len}"

        # Language must be Tier 1
        assert r["language"] in TIER1_LANGUAGES

        # ADR-0003 taxonomy labels
        tax = r.get("taxonomy_labels")
        assert isinstance(tax, dict)
        for cat in TAXONOMY_CLASSES:
            assert cat in tax
            assert 0.0 <= tax[cat] <= 1.0

        # Risk score and label consistency
        if r["label"] == 1:
            assert 0.0 <= r["risk_score"] <= 0.25
        else:
            assert 0.65 <= r["risk_score"] <= 1.0
            assert max(tax.values()) >= 0.50


def test_quality_control_invariants_medium():
    """Check QC gate invariants on v3-medium train and val records."""
    train_recs = load_jsonl(MEDIUM_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(MEDIUM_DIR / "dataset_val.jsonl")

    all_recs = train_recs + val_recs
    for r in all_recs:
        # Symbolic gate must be passed
        assert r.get("symbolic_gate_passed") is True, f"Gate failed for {r}"

        # Token ceiling <= 400
        tok_len = estimate_tokens(r["input_dsl"])
        assert 0 < tok_len <= 400, f"Token ceiling violated: {tok_len}"

        # Language must be Tier 1
        assert r["language"] in TIER1_LANGUAGES

        # ADR-0003 taxonomy labels
        tax = r.get("taxonomy_labels")
        assert isinstance(tax, dict)
        for cat in TAXONOMY_CLASSES:
            assert cat in tax
            assert 0.0 <= tax[cat] <= 1.0

        # Risk score and label consistency
        if r["label"] == 1:
            assert 0.0 <= r["risk_score"] <= 0.25
        else:
            assert 0.65 <= r["risk_score"] <= 1.0
            assert max(tax.values()) >= 0.50


def test_quality_control_invariants_full():
    """Check QC gate invariants on v3-full train and val records."""
    train_recs = load_jsonl(FULL_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(FULL_DIR / "dataset_val.jsonl")

    all_recs = train_recs + val_recs
    for r in all_recs:
        assert r.get("symbolic_gate_passed") is True
        tok_len = estimate_tokens(r["input_dsl"])
        assert 0 < tok_len <= 400
        assert r["language"] in TIER1_LANGUAGES

        tax = r.get("taxonomy_labels")
        assert isinstance(tax, dict)
        for cat in TAXONOMY_CLASSES:
            assert cat in tax
            assert 0.0 <= tax[cat] <= 1.0

        if r["label"] == 1:
            assert 0.0 <= r["risk_score"] <= 0.25
        else:
            assert 0.65 <= r["risk_score"] <= 1.0
            assert max(tax.values()) >= 0.50


def test_heldout_eval_dataset():
    """Verify independent held-out evaluation dataset structure."""
    heldout_file = MEDIUM_DIR / "dataset_heldout_eval.jsonl"
    recs = load_jsonl(heldout_file)
    assert len(recs) == 400
    for r in recs:
        assert r.get("symbolic_gate_passed") is True
        assert estimate_tokens(r["input_dsl"]) <= 400
        assert r["language"] in TIER1_LANGUAGES


# ============================================================================
# 4. Colab Notebook Integrity
# ============================================================================


def test_colab_notebooks_structure():
    """Verify generated Google Colab notebooks exist, are valid JSON, and configure ADR-0003."""
    notebook_files = [
        "Laya_Code_Oracle_MultiTask_Base.ipynb",
        "Laya_Code_Oracle_MultiTask_v3_Hybrid.ipynb",
        "Laya_Code_Oracle_MultiTask_v3_Medium.ipynb",
        "Laya_Code_Oracle_MultiTask_v3_Full.ipynb",
    ]

    for fname in notebook_files:
        nb_path = REPO_ROOT / fname
        assert nb_path.exists(), f"Missing notebook {fname}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert "cells" in nb
        assert len(nb["cells"]) >= 15

        # Combine all source code cells
        all_code = "".join(
            "".join(c.get("source", []))
            for c in nb["cells"]
            if c.get("cell_type") == "code"
        )

        # Check required training parameters & self-healing cell guards
        assert "pos_weight" in all_code
        assert "torch.tensor([2.0" in all_code or "pos_weight = 2.0" in all_code or "pos_weight" in all_code
        assert "temperature" in all_code
        assert "0.8" in all_code and "2.5" in all_code  # temperature clamp range
        assert "EarlyStopping" in all_code or "early_stopping" in all_code or "patience" in all_code

        # Cell 4 explicit device definition
        assert 'device = torch.device(' in all_code

        # Cell 5 & 7 auto-loading fallback guards
        assert "if 'train_records' not in globals()" in all_code
        assert "if 'heldout_records' not in globals()" in all_code

        # Held-out threshold sweep across [0.25 .. 0.60]
        assert "0.25" in all_code and "0.60" in all_code and "DEFAULT_THRESHOLD" in all_code

        # Automatic weights export
        assert "files.download" in all_code and "code_oracle_laya_multitask_weights" in all_code


def test_colab_embedded_zip_validity():
    """Verify embedded base64 zip payloads in variant notebooks decode properly."""
    import io
    import re

    for fname in (
        "Laya_Code_Oracle_MultiTask_v3_Hybrid.ipynb",
        "Laya_Code_Oracle_MultiTask_v3_Medium.ipynb",
        "Laya_Code_Oracle_MultiTask_v3_Full.ipynb",
    ):
        nb_path = REPO_ROOT / fname
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        found_b64 = False
        for c in nb["cells"]:
            source = "".join(c.get("source", []))
            match = re.search(r'EMBEDDED_ZIP_B64\s*=\s*["\']([A-Za-z0-9+/=]+)["\']', source)
            if match:
                found_b64 = True
                b64_content = match.group(1).strip()
                zip_bytes = base64.b64decode(b64_content)
                assert len(zip_bytes) > 50_000
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    assert "dataset_train.jsonl" in zf.namelist()
                    assert "dataset_val.jsonl" in zf.namelist()
                    assert "dataset_heldout_eval.jsonl" in zf.namelist()
                break
        assert found_b64, f"Could not find EMBEDDED_ZIP_B64 in {fname}"


# ============================================================================
# 5. Unit Tests for Pipeline Functions
# ============================================================================


def test_validate_qc_gate_unit():
    """Unit tests for validate_qc_gate edge cases."""
    good_rec = DatasetRecord(
        input_dsl="[DIFF_TARGET] test.py (MODIFIED)",
        label=1,
        risk_score=0.10,
        category="clean_pass",
        language="python",
        symbolic_gate_passed=True,
        taxonomy_labels={c: 0.05 for c in TAXONOMY_CLASSES},
    )
    assert validate_qc_gate(good_rec, max_tokens=400) is True

    # Failed symbolic gate
    bad_gate = DatasetRecord.from_dict(good_rec.to_dict())
    bad_gate.symbolic_gate_passed = False
    assert validate_qc_gate(bad_gate) is False

    # Exceeded token ceiling
    bad_tokens = DatasetRecord.from_dict(good_rec.to_dict())
    bad_tokens.input_dsl = "word " * 500
    assert validate_qc_gate(bad_tokens, max_tokens=100) is False

    # Unsupported language
    bad_lang = DatasetRecord.from_dict(good_rec.to_dict())
    bad_lang.language = "ruby"
    assert validate_qc_gate(bad_lang) is False

    # Mismatched risk score for PASS
    bad_pass_risk = DatasetRecord.from_dict(good_rec.to_dict())
    bad_pass_risk.risk_score = 0.50
    assert validate_qc_gate(bad_pass_risk) is False

    # REJECT with no high taxonomy class
    bad_reject = DatasetRecord.from_dict(good_rec.to_dict())
    bad_reject.label = 0
    bad_reject.risk_score = 0.85
    bad_reject.taxonomy_labels = {c: 0.1 for c in TAXONOMY_CLASSES}
    assert validate_qc_gate(bad_reject) is False


def test_stratify_and_balance_dataset_unit():
    """Unit tests for stratify_and_balance_dataset balancing logic."""
    records = []
    for lang in TIER1_LANGUAGES:
        for i in range(30):
            # 30 pass
            records.append(
                DatasetRecord(
                    input_dsl=f"{lang} pass {i}",
                    label=1,
                    risk_score=0.05,
                    category="clean_pass",
                    language=lang,
                    symbolic_gate_passed=True,
                    taxonomy_labels={c: 0.05 for c in TAXONOMY_CLASSES},
                )
            )
            # 30 reject
            records.append(
                DatasetRecord(
                    input_dsl=f"{lang} reject {i}",
                    label=0,
                    risk_score=0.90,
                    category="targeted_mutation",
                    language=lang,
                    symbolic_gate_passed=True,
                    taxonomy_labels={
                        TAXONOMY_CLASSES[0]: 0.9,
                        TAXONOMY_CLASSES[1]: 0.1,
                        TAXONOMY_CLASSES[2]: 0.1,
                        TAXONOMY_CLASSES[3]: 0.1,
                        TAXONOMY_CLASSES[4]: 0.1,
                    },
                )
            )

    train, val = stratify_and_balance_dataset(records, target_total=40, val_ratio=0.2, seed=123)
    assert len(train) == 32
    assert len(val) == 8

    # Each language has target_per_lang = 10, val=2, train=8
    for lang in TIER1_LANGUAGES:
        l_train = [r for r in train if r.language == lang]
        assert len(l_train) == 8
        assert sum(1 for r in l_train if r.label == 1) == 4
        assert sum(1 for r in l_train if r.label == 0) == 4

        l_val = [r for r in val if r.language == lang]
        assert len(l_val) == 2
        assert sum(1 for r in l_val if r.label == 1) == 1
        assert sum(1 for r in l_val if r.label == 0) == 1


def test_compile_v3_datasets_small_run(tmp_path):
    """End-to-end integration test of compile_v3_datasets on a small target in temporary dir."""
    res = compile_v3_datasets(
        scale="medium",
        medium_total=40,
        val_ratio=0.2,
        data_dir=tmp_path,
        seed=999,
        max_tokens=400,
    )
    assert "medium" in res
    med = res["medium"]
    assert med["total"] == 40
    assert med["train"] == 32
    assert med["val"] == 8
    assert Path(med["zip"]).exists()
    assert (tmp_path / "v3_medium" / "dataset_train.jsonl").exists()
    assert (tmp_path / "v3_medium" / "dataset_val.jsonl").exists()


def test_compile_hybrid_dataset_small_run(tmp_path):
    """End-to-end integration test of compile_hybrid_dataset on a small target in temporary dir."""
    res = compile_hybrid_dataset(
        target_total=40,
        val_ratio=0.2,
        data_dir=tmp_path,
        seed=101,
        max_tokens=400,
    )
    assert res["total"] == 40
    assert res["train"] == 32
    assert res["val"] == 8
    assert Path(res["zip"]).exists()
    assert (tmp_path / "v3_hybrid" / "dataset_train.jsonl").exists()
    assert (tmp_path / "v3_hybrid" / "dataset_val.jsonl").exists()
    assert (tmp_path / "v3_hybrid" / "dataset_heldout_eval.jsonl").exists()


def test_v3_hybrid_heldout_integrity():
    """Verify v3-hybrid heldout evaluation file exists, has 400 samples, passes QC, and is disjoint from train/val."""
    heldout_recs = load_jsonl(HYBRID_DIR / "dataset_heldout_eval.jsonl")
    train_recs = load_jsonl(HYBRID_DIR / "dataset_train.jsonl")
    val_recs = load_jsonl(HYBRID_DIR / "dataset_val.jsonl")

    assert len(heldout_recs) == 400
    for r in heldout_recs:
        assert r.get("symbolic_gate_passed") is True
        assert estimate_tokens(r["input_dsl"]) <= 400
        assert r["label"] in (0, 1)

    train_dsls = {r["input_dsl"] for r in train_recs}
    val_dsls = {r["input_dsl"] for r in val_recs}
    heldout_dsls = {r["input_dsl"] for r in heldout_recs}

    assert len(train_dsls.intersection(heldout_dsls)) == 0, "Data leakage detected: train and heldout overlap!"
    assert len(val_dsls.intersection(heldout_dsls)) == 0, "Data leakage detected: val and heldout overlap!"


def test_compile_hybrid_odd_target_safety(tmp_path):
    """Verify compile_hybrid_dataset handles odd target total gracefully and maintains 50/50 balance."""
    res = compile_hybrid_dataset(
        target_total=39,  # Odd target total
        val_ratio=0.2,
        data_dir=tmp_path,
        seed=101,
        max_tokens=400,
    )
    # Should round to 40 (even)
    assert res["total"] == 40
    assert res["train_pass"] == res["train_reject"]
    assert res["val_pass"] == res["val_reject"]


