#!/usr/bin/env python3
"""
Dataset v3 Compilation Pipeline for Code Oracle / Laya ModernBERT Fine-Tuning.

Compiles two balanced, high-craft dataset variants:
1. Medium Scale (v3-medium): 5,000 samples total (1,250 per language across Python, TypeScript, Go, Rust).
   - 80% train (4,000 samples: 1,000 per language, 500 PASS / 500 REJECT).
   - 20% val (1,000 samples: 250 per language, 125 PASS / 125 REJECT).
   - Packaged as data/code_oracle_dataset_v3_medium.zip and in data/v3_medium/.

2. Full Scale (v3-full): 10,000 samples total (2,500 per language across Python, TypeScript, Go, Rust).
   - 80% train (8,000 samples: 2,000 per language, 1,000 PASS / 1,000 REJECT).
   - 20% val (2,000 samples: 500 per language, 250 PASS / 250 REJECT).
   - Packaged as data/code_oracle_dataset_v3_full.zip and in data/v3_full/.

Strict Quality Control Gates:
- All negative/hard samples strictly pass Stage 1 (AST) & Stage 2 (Tarjan cycle) symbolic checks (symbolic_gate_passed=True).
- Fully labeled with ADR-0003 multi-task risk taxonomy (BreakingPublicAPI, SecuritySurface, ConcurrencyHazard, PerformanceRegression, SilentLogicDrift).
- Token length ceiling <= 400 tokens per sample (estimate_tokens).
- Exact 50% PASS / 50% REJECT ratio across train, val, and per language.
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from code_oracle.dataset import (
    DEFAULT_HELDOUT_REPOS,
    TARGETED_MUTATIONS_MAP,
    TAXONOMY_CLASSES,
    TEMPLATES,
    DatasetGenerator,
    DatasetRecord,
)
from code_oracle.linearizer import estimate_tokens
from harvest_mutants import MutantHarvester
from mine_git_reverts import GitRevertMiner, find_cached_repository

TIER1_LANGUAGES = ["python", "typescript", "go", "rust"]


# ============================================================================
# 1. Quality Control & Validation Gate
# ============================================================================


def validate_qc_gate(record: DatasetRecord, max_tokens: int = 400) -> bool:
    """
    Strict Quality Control Gate:
    1. symbolic_gate_passed must be True (no trivial syntax / cyclic errors).
    2. estimate_tokens(input_dsl) <= max_tokens (strict 400 ceiling).
    3. label must be 0 or 1.
    4. risk_score must match label bounds:
       - PASS (1): 0.0 <= risk_score <= 0.25
       - REJECT (0): 0.65 <= risk_score <= 1.0
    5. taxonomy_labels must contain all 5 ADR-0003 classes with valid float values.
    6. language must be one of Tier 1 (python, typescript, go, rust).
    """
    if record.symbolic_gate_passed is not True:
        return False

    tok_len = estimate_tokens(record.input_dsl)
    if tok_len <= 0 or tok_len > max_tokens:
        return False

    if record.label not in (0, 1):
        return False

    if record.language not in TIER1_LANGUAGES:
        return False

    if record.taxonomy_labels is None:
        return False

    for c in TAXONOMY_CLASSES:
        if c not in record.taxonomy_labels:
            return False
        val = record.taxonomy_labels[c]
        if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
            return False

    if record.label == 1:
        if record.risk_score < 0.0 or record.risk_score > 0.25:
            return False
    else:
        if record.risk_score < 0.65 or record.risk_score > 1.0:
            return False
        if max(record.taxonomy_labels.values()) < 0.50:
            return False

    return True


# ============================================================================
# 2. Data Ingestion: Existing Curated Base Records
# ============================================================================


def load_existing_curated_records(
    data_dir: Path,
    max_tokens: int = 400,
) -> List[DatasetRecord]:
    """Load and validate existing curated dataset records from train and val files."""
    loaded: List[DatasetRecord] = []

    for fname in ("dataset_train.jsonl", "dataset_val.jsonl"):
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rec = DatasetRecord.from_dict(data)
                    if validate_qc_gate(rec, max_tokens=max_tokens):
                        loaded.append(rec)
                except Exception:
                    continue

    print(f"[+] Loaded {len(loaded)} existing curated records passing QC gates.")
    return loaded


# ============================================================================
# 3. Data Ingestion: Real-World Reverts & CVEs
# ============================================================================


def mine_real_world_reverts(
    languages: List[str],
    max_samples_per_repo: int = 10,
    seed: int = 42,
    max_tokens: int = 400,
) -> List[DatasetRecord]:
    """Mine real-world git revert and CVE commit pairs from selected available repositories."""
    miner = GitRevertMiner(languages=languages, filter_symbolic_gate=True, seed=seed)
    records: List[DatasetRecord] = []

    # Selected repositories with quick history checks (excluding held-out repos)
    target_repos = [
        ("pydantic", "python"),
        ("gin", "go"),
        ("tokio", "rust"),
        ("hono", "typescript"),
    ]

    print("[*] Mining real-world git reverts from local repositories...")
    for repo_name, lang in target_repos:
        if lang not in languages:
            continue
        repo_path = find_cached_repository(repo_name)
        if not repo_path:
            continue

        try:
            rev_pairs = miner.extract_revert_pairs_from_repo(repo_path, max_samples=max_samples_per_repo)
            for r in rev_pairs:
                if validate_qc_gate(r, max_tokens=max_tokens):
                    records.append(r)
        except Exception:
            continue

    print(f"[+] Extracted {len(records)} real-world revert records passing QC gates.")
    return records


# ============================================================================
# 4. Data Ingestion: Surviving Mutants from Production Code
# ============================================================================


def harvest_production_mutants(
    languages: List[str],
    max_samples: int = 50,
    seed: int = 42,
    max_tokens: int = 400,
) -> List[DatasetRecord]:
    """Harvest genuine surviving mutants from key production code files."""
    records: List[DatasetRecord] = []

    target_files = [
        ("python", REPO_ROOT / "benchmarks_repos" / "fastapi" / "fastapi" / "concurrency.py"),
        ("python", REPO_ROOT / "benchmarks_repos" / "fastapi" / "fastapi" / "datastructures.py"),
        ("typescript", REPO_ROOT / "benchmarks_repos" / "hono" / "src" / "utils" / "url.ts"),
        ("typescript", REPO_ROOT / "benchmarks_repos" / "hono" / "src" / "utils" / "cookie.ts"),
        ("go", Path("/tmp/code_oracle_mined_repos") / "gin" / "binding" / "default_validator.go"),
        ("rust", Path("/tmp/code_oracle_mined_repos") / "tokio" / "tokio" / "src" / "sync" / "barrier.rs"),
    ]

    print("[*] Harvesting surviving mutants from key production code files...")
    for lang, fpath in target_files:
        if lang not in languages or not fpath.exists():
            continue

        ws_root = fpath.parent
        curr = fpath.parent
        while curr != curr.parent:
            if (curr / ".git").exists() or (curr / "go.mod").exists() or (curr / "Cargo.toml").exists() or (curr / "package.json").exists():
                ws_root = curr
                break
            curr = curr.parent

        try:
            harvester = MutantHarvester(
                workspace_root=ws_root,
                languages=[lang],
                filter_symbolic_gate=True,
                seed=seed,
            )
            pairs = harvester.harvest_file(fpath)
            for neg, pos in pairs:
                for r in (neg, pos):
                    if validate_qc_gate(r, max_tokens=max_tokens):
                        records.append(r)
        except Exception:
            continue

        if len(records) >= max_samples:
            break

    print(f"[+] Harvested {len(records)} surviving mutant records passing QC gates.")
    return records


# ============================================================================
# 4b. Data Ingestion: Authentic TypeScript Commits & Mutants (Hono / Zod)
# ============================================================================


def harvest_typescript_authentic_records(
    target_count: int = 100,
    seed: int = 42,
    max_tokens: int = 400,
) -> List[DatasetRecord]:
    """
    Harvest authentic TypeScript real-world commit pairs and surviving mutants
    from local repositories (Hono and Zod) to enrich the TypeScript dataset without generic template dilution.
    Guarantees 50% PASS / 50% REJECT and 100% symbolic gate pass.
    """
    records: List[DatasetRecord] = []

    # 1. Mine commit revert pairs from Zod if available
    zod_path = find_cached_repository("zod") or Path("/tmp/code_oracle_mined_repos/zod")
    if zod_path.exists():
        miner = GitRevertMiner(languages=["typescript"], filter_symbolic_gate=True, seed=seed)
        try:
            rev_pairs = miner.extract_revert_pairs_from_repo(zod_path, max_samples=10)
            for r in rev_pairs:
                if validate_qc_gate(r, max_tokens=max_tokens):
                    records.append(r)
        except Exception:
            pass

    # 2. Harvest surviving mutants from real TypeScript files in Hono and Zod
    hono_path = find_cached_repository("hono") or (REPO_ROOT / "benchmarks_repos" / "hono")
    target_ts_files = [
        # Hono utils & helpers
        (hono_path, hono_path / "src" / "utils" / "url.ts"),
        (hono_path, hono_path / "src" / "utils" / "cookie.ts"),
        (hono_path, hono_path / "src" / "utils" / "crypto.ts"),
        (hono_path, hono_path / "src" / "utils" / "encode.ts"),
        (hono_path, hono_path / "src" / "utils" / "html.ts"),
        (hono_path, hono_path / "src" / "utils" / "ipaddr.ts"),
        (hono_path, hono_path / "src" / "utils" / "headers.ts"),
        (hono_path, hono_path / "src" / "utils" / "jwt" / "jwt.ts"),
        (hono_path, hono_path / "src" / "utils" / "mime.ts"),
        (hono_path, hono_path / "src" / "utils" / "buffer.ts"),
        (hono_path, hono_path / "src" / "helper" / "cookie" / "index.ts"),
        (hono_path, hono_path / "src" / "helper" / "accepts" / "accepts.ts"),
        (hono_path, hono_path / "src" / "validator" / "validator.ts"),
        # Hono middleware
        (hono_path, hono_path / "src" / "middleware" / "basic-auth" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "bearer-auth" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "csrf" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "cors" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "method-override" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "powered-by" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "pretty-json" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "secure-headers" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "body-limit" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "cache" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "request-id" / "request-id.ts"),
        (hono_path, hono_path / "src" / "middleware" / "timing" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "trailing-slash" / "index.ts"),
        (hono_path, hono_path / "src" / "middleware" / "timeout" / "index.ts"),
        # Zod core schemas and parsers
        (zod_path, zod_path / "packages" / "zod" / "src" / "v4" / "mini" / "schemas.ts"),
        (zod_path, zod_path / "packages" / "zod" / "src" / "v4" / "mini" / "parse.ts"),
        (zod_path, zod_path / "packages" / "zod" / "src" / "v4" / "mini" / "checks.ts"),
        (zod_path, zod_path / "packages" / "zod" / "src" / "v4" / "mini" / "iso.ts"),
        (zod_path, zod_path / "packages" / "zod" / "src" / "v4" / "mini" / "external.ts"),
        (zod_path, zod_path / "packages" / "zod" / "src" / "v4" / "mini" / "in-out.ts"),
    ]

    harvesters: Dict[Path, MutantHarvester] = {}
    for ws, fpath in target_ts_files:
        if not ws.exists() or not fpath.exists():
            continue
        if ws not in harvesters:
            try:
                harvesters[ws] = MutantHarvester(
                    workspace_root=ws,
                    languages=["typescript"],
                    filter_symbolic_gate=True,
                    seed=seed,
                )
            except Exception:
                continue
        h = harvesters[ws]
        try:
            pairs = h.harvest_file(fpath)
            for neg, pos in pairs:
                if validate_qc_gate(neg, max_tokens=max_tokens) and validate_qc_gate(pos, max_tokens=max_tokens):
                    records.extend([neg, pos])
        except Exception:
            continue

        pass_cnt = sum(1 for r in records if r.label == 1)
        neg_cnt = sum(1 for r in records if r.label == 0)
        if min(pass_cnt, neg_cnt) >= target_count // 2:
            break

    pass_recs = [r for r in records if r.label == 1]
    neg_recs = [r for r in records if r.label == 0]
    needed_each = target_count // 2

    # Supplement with targeted TS mutations if authentic yield is insufficient
    gen: Optional[DatasetGenerator] = None
    while len(pass_recs) < needed_each or len(neg_recs) < needed_each:
        missing_pass = max(0, needed_each - len(pass_recs))
        missing_neg = max(0, needed_each - len(neg_recs))
        if gen is None:
            gen = DatasetGenerator(languages=["typescript"], seed=seed, filter_symbolic_gate=True)
        count_per_type = max(1, (max(missing_pass, missing_neg) + 3) // 4)
        synth = gen.generate_targeted_typescript_mutations(count_per_type=count_per_type)
        added = 0
        for r in synth:
            if validate_qc_gate(r, max_tokens=max_tokens):
                if r.label == 1 and len(pass_recs) < needed_each:
                    pass_recs.append(r)
                    added += 1
                elif r.label == 0 and len(neg_recs) < needed_each:
                    neg_recs.append(r)
                    added += 1
        if added == 0:
            break

    rng = random.Random(seed)
    rng.shuffle(pass_recs)
    rng.shuffle(neg_recs)
    selected = pass_recs[:needed_each] + neg_recs[:needed_each]
    rng.shuffle(selected)
    print(f"[+] Harvested {len(selected)} authentic TypeScript records ({needed_each} PASS, {needed_each} REJECT).")
    return selected



# ============================================================================
# 5. Data Ingestion: Targeted Semantic Mutations
# ============================================================================


def generate_targeted_semantic_mutations(
    languages: List[str],
    needed_per_language: Dict[str, int],
    seed: int = 42,
    max_tokens: int = 400,
) -> List[DatasetRecord]:
    """
    Generate targeted subtle mutations across all Tier 1 languages using
    DatasetGenerator to guarantee 100% symbolic gate compliance and balanced taxonomy.
    """
    gen = DatasetGenerator(languages=languages, seed=seed, filter_symbolic_gate=True)
    records: List[DatasetRecord] = []

    print("[*] Generating targeted semantic mutations across Tier 1 languages...")
    for lang in languages:
        target_count = needed_per_language.get(lang, 0)
        if target_count <= 0:
            continue

        # Each type generates 1 PASS + 1 REJECT per template
        # 5 templates for Go (5 PASS + 5 REJECT per cpt), 4 templates for others (4 PASS + 4 REJECT per cpt)
        count_per_type = max(1, (target_count + 4) // 5 if lang == "go" else (target_count + 3) // 4)

        if lang == "typescript":
            lang_records = gen.generate_targeted_typescript_mutations(count_per_type=count_per_type)
        elif lang == "python":
            lang_records = gen.generate_targeted_python_mutations(count_per_type=count_per_type)
        elif lang == "go":
            lang_records = gen.generate_targeted_go_mutations(count_per_type=count_per_type)
        elif lang == "rust":
            lang_records = gen.generate_targeted_rust_mutations(count_per_type=count_per_type)
        else:
            lang_records = []

        compliant = [r for r in lang_records if validate_qc_gate(r, max_tokens=max_tokens)]
        records.extend(compliant)
        print(f"    [+] Generated {len(compliant)} compliant records for {lang} (count_per_type={count_per_type})")

    return records


# ============================================================================
# 6. Stratified Balancing & Split Partitioning
# ============================================================================


def stratify_and_balance_dataset(
    records: List[DatasetRecord],
    target_total: int,
    val_ratio: float = 0.2,
    languages: Optional[List[str]] = None,
    seed: int = 42,
    targets_per_language: Optional[Dict[str, int]] = None,
) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
    """
    Stratifies records into train and val splits with:
    1. Equal or targeted representation across languages.
    2. Exact 50% PASS (label=1) / 50% REJECT (label=0) ratio in both train and val splits.
    3. Deterministic shuffling and reproducible allocation.
    """
    langs = languages or (list(targets_per_language.keys()) if targets_per_language else TIER1_LANGUAGES)

    rng = random.Random(seed)
    train_all: List[DatasetRecord] = []
    val_all: List[DatasetRecord] = []

    for lang in langs:
        if targets_per_language and lang in targets_per_language:
            target_per_lang = targets_per_language[lang]
        else:
            target_per_lang = target_total // len(langs)
        val_per_lang = int(round(target_per_lang * val_ratio / 2) * 2)
        train_per_lang = target_per_lang - val_per_lang

        target_pass_train = train_per_lang // 2
        target_reject_train = train_per_lang - target_pass_train

        target_pass_val = val_per_lang // 2
        target_reject_val = val_per_lang - target_pass_val

        lang_recs = [r for r in records if r.language == lang]
        pass_recs = [r for r in lang_recs if r.label == 1]
        reject_recs = [r for r in lang_recs if r.label == 0]

        rng.shuffle(pass_recs)
        rng.shuffle(reject_recs)

        needed_pass = target_pass_train + target_pass_val
        needed_reject = target_reject_train + target_reject_val

        if len(pass_recs) < needed_pass or len(reject_recs) < needed_reject:
            raise ValueError(
                f"Insufficient samples for {lang}: need {needed_pass} PASS, got {len(pass_recs)}; "
                f"need {needed_reject} REJECT, got {len(reject_recs)}"
            )

        # Train split
        lang_train = pass_recs[:target_pass_train] + reject_recs[:target_reject_train]
        # Val split
        lang_val = (
            pass_recs[target_pass_train : target_pass_train + target_pass_val]
            + reject_recs[target_reject_train : target_reject_train + target_reject_val]
        )

        train_all.extend(lang_train)
        val_all.extend(lang_val)

    rng.shuffle(train_all)
    rng.shuffle(val_all)

    return train_all, val_all


# ============================================================================
# 7. Packaging & Archive Creation
# ============================================================================


def package_dataset_variant(
    variant_name: str,
    train_records: List[DatasetRecord],
    val_records: List[DatasetRecord],
    heldout_source_path: Path,
    output_dir: Path,
    zip_path: Path,
) -> Dict[str, Any]:
    """
    Write JSONL files into output_dir and package into zip archive.
    Includes heldout evaluation dataset if available.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "dataset_train.jsonl"
    val_file = output_dir / "dataset_val.jsonl"
    heldout_dest = output_dir / "dataset_heldout_eval.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    if heldout_source_path.exists():
        shutil.copy2(heldout_source_path, heldout_dest)

    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(train_file, arcname="dataset_train.jsonl")
        zf.write(val_file, arcname="dataset_val.jsonl")
        if heldout_dest.exists():
            zf.write(heldout_dest, arcname="dataset_heldout_eval.jsonl")

    stats = {
        "variant": variant_name,
        "total": len(train_records) + len(val_records),
        "train": len(train_records),
        "val": len(val_records),
        "train_pass": sum(1 for r in train_records if r.label == 1),
        "train_reject": sum(1 for r in train_records if r.label == 0),
        "val_pass": sum(1 for r in val_records if r.label == 1),
        "val_reject": sum(1 for r in val_records if r.label == 0),
        "dir": str(output_dir),
        "zip": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
    }
    return stats


# ============================================================================
# 8. Golden Hybrid Dataset Compilation Pipeline
# ============================================================================


def compile_hybrid_dataset(
    target_total: int = 4900,
    val_ratio: float = 0.2,
    data_dir: Optional[Path] = None,
    seed: int = 42,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    """
    Compile the Golden Hybrid Dataset (4,900 samples, default 4,900):
    1. Python: High-density targeted mutation samples from 3.2k dataset (data/dataset_train.jsonl and data/dataset_val.jsonl).
       1,000 samples (500 PASS / 500 REJECT).
    2. Go: Balanced samples enriched with 5 idiomatic Go semantic mutations.
       1,000 samples (500 PASS / 500 REJECT).
    3. Rust: Comprehensive multi-class samples from 10k dataset (data/v3_full/).
       1,800 samples (900 PASS / 900 REJECT).
    4. TypeScript: Base samples from 3.2k dataset (1,000 samples) enriched with authentic
       real-world commit pairs and surviving mutants from real TypeScript repos (Hono/Zod/Fastify).
       1,100 samples (550 PASS / 550 REJECT).

    Exact Quality Gates:
    - 50% PASS / 50% REJECT ratio across all splits (3,920 train, 980 val) and per language.
    - 100% symbolic_gate_passed == True.
    - estimate_tokens <= 400.
    - Full ADR-0003 multi-task risk taxonomy.
    - Packaged into data/v3_hybrid/ and data/code_oracle_dataset_v3_hybrid.zip.
    """
    data_dir = (data_dir or REPO_ROOT / "data").resolve()
    source_dir = REPO_ROOT / "data"
    heldout_path = (data_dir / "dataset_heldout_eval.jsonl") if (data_dir / "dataset_heldout_eval.jsonl").exists() else (source_dir / "dataset_heldout_eval.jsonl")

    print("=" * 80)
    print(" 🚀 STARTING CODE ORACLE GOLDEN HYBRID DATASET COMPILATION PIPELINE")
    print(f" Target Total: {target_total} | Val Ratio: {val_ratio}")
    print("=" * 80)

    # Ensure target_total is even to allow 50% PASS / 50% REJECT
    if target_total % 2 != 0:
        target_total += 1

    # Compute target language allocations in Golden Hybrid Configuration
    if target_total == 4900:
        targets_per_language = {
            "python": 1000,
            "go": 1000,
            "typescript": 1100,
            "rust": 1800,
        }
    elif target_total == 4500:
        targets_per_language = {
            "python": 1000,
            "go": 600,
            "typescript": 1100,
            "rust": 1800,
        }
    elif target_total < 100 and target_total % 4 == 0:
        q = target_total // 4
        if q % 2 != 0:
            targets_per_language = {
                "python": q - 1,
                "go": q - 1,
                "typescript": q + 1,
                "rust": q + 1,
            }
        else:
            targets_per_language = {
                "python": q,
                "go": q,
                "typescript": q,
                "rust": q,
            }
    else:
        # Scale proportionally to 4900 (Python: 10/49, Go: 10/49, TS: 11/49, Rust: 18/49)
        scale_f = target_total / 4900.0
        py_t = max(2, int(round(1000 * scale_f / 2) * 2))
        go_t = max(2, int(round(1000 * scale_f / 2) * 2))
        ts_t = max(2, int(round(1100 * scale_f / 2) * 2))
        rust_t = max(2, target_total - (py_t + go_t + ts_t))
        targets_per_language = {
            "python": py_t,
            "go": go_t,
            "typescript": ts_t,
            "rust": rust_t,
        }

    # 1. Base records (Python, Go, TypeScript base)
    base_train = (data_dir / "dataset_train.jsonl") if (data_dir / "dataset_train.jsonl").exists() else (source_dir / "dataset_train.jsonl")
    base_val = (data_dir / "dataset_val.jsonl") if (data_dir / "dataset_val.jsonl").exists() else (source_dir / "dataset_val.jsonl")

    py_records: List[DatasetRecord] = []
    go_records: List[DatasetRecord] = []
    ts_base_records: List[DatasetRecord] = []

    for fpath in (base_train, base_val):
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = DatasetRecord.from_dict(json.loads(line))
                    if validate_qc_gate(r, max_tokens=max_tokens):
                        if r.language == "python":
                            py_records.append(r)
                        elif r.language == "go":
                            go_records.append(r)
                        elif r.language == "typescript":
                            ts_base_records.append(r)
                except Exception:
                    continue

    print(f"[+] Loaded base records: Python={len(py_records)}, Go={len(go_records)}, TS={len(ts_base_records)}")

    # 2. Enrich TypeScript with authentic real-world commit pairs and surviving mutants
    ts_harvest_target = min(100, max(2, targets_per_language.get("typescript", 1100) // 10))
    ts_authentic = harvest_typescript_authentic_records(target_count=ts_harvest_target, seed=seed, max_tokens=max_tokens)
    ts_records = ts_base_records + ts_authentic

    # 3. Rust records from 10k dataset (data/v3_full/)
    v3_full_dir = (data_dir / "v3_full") if (data_dir / "v3_full").exists() else (source_dir / "v3_full")
    rust_records: List[DatasetRecord] = []
    for fpath in (v3_full_dir / "dataset_train.jsonl", v3_full_dir / "dataset_val.jsonl"):
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = DatasetRecord.from_dict(json.loads(line))
                    if r.language == "rust" and validate_qc_gate(r, max_tokens=max_tokens):
                        rust_records.append(r)
                except Exception:
                    continue

    print(f"[+] Loaded Rust records from v3_full: {len(rust_records)}")

    hybrid_pool = py_records + go_records + ts_records + rust_records

    # 4. Fallback synthesis for any language that needs more records
    gen: Optional[DatasetGenerator] = None
    for lang, tgt in targets_per_language.items():
        curr_pass = sum(1 for r in hybrid_pool if r.language == lang and r.label == 1)
        curr_rej = sum(1 for r in hybrid_pool if r.language == lang and r.label == 0)
        needed = max(tgt // 2 - curr_pass, tgt // 2 - curr_rej, 0)
        if needed > 0:
            if gen is None:
                gen = DatasetGenerator(languages=TIER1_LANGUAGES, seed=seed, filter_symbolic_gate=True)
            cpt = max(1, (needed + 4) // 5 if lang == "go" else (needed + 3) // 4)
            if lang == "python":
                syn = gen.generate_targeted_python_mutations(count_per_type=cpt)
            elif lang == "go":
                syn = gen.generate_targeted_go_mutations(count_per_type=cpt)
            elif lang == "typescript":
                syn = gen.generate_targeted_typescript_mutations(count_per_type=cpt)
            elif lang == "rust":
                syn = gen.generate_targeted_rust_mutations(count_per_type=cpt)
            else:
                syn = []
            for r in syn:
                if validate_qc_gate(r, max_tokens=max_tokens):
                    hybrid_pool.append(r)

    print(f"[+] Unified Golden Hybrid pool: {len(hybrid_pool)} records.")

    hybrid_train, hybrid_val = stratify_and_balance_dataset(
        hybrid_pool,
        target_total=target_total,
        val_ratio=val_ratio,
        seed=seed,
        targets_per_language=targets_per_language,
    )

    stats = package_dataset_variant(
        variant_name="v3_hybrid",
        train_records=hybrid_train,
        val_records=hybrid_val,
        heldout_source_path=heldout_path,
        output_dir=data_dir / "v3_hybrid",
        zip_path=data_dir / "code_oracle_dataset_v3_hybrid.zip",
    )
    print(f"[✓] Golden Hybrid complete: {stats['total']} samples ({stats['train']} train, {stats['val']} val)")
    print(f"    Zip: {stats['zip']} ({stats['zip_size_bytes'] / 1024:.1f} KB)")
    return stats


# ============================================================================
# 9. Main Compilation Pipeline Orchestrator
# ============================================================================


def compile_v3_datasets(
    scale: str = "both",
    medium_total: int = 5000,
    full_total: int = 10000,
    hybrid_total: int = 4900,
    val_ratio: float = 0.2,
    data_dir: Optional[Path] = None,
    seed: int = 42,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    """
    Main compilation workflow executing the compilation pipelines:
    - medium: v3-medium (5,000 samples)
    - full: v3-full (10,000 samples)
    - hybrid: Golden Hybrid (4,900 samples)
    - both: medium and full
    - all: medium, full, and hybrid
    """
    data_dir = (data_dir or REPO_ROOT / "data").resolve()
    heldout_path = data_dir / "dataset_heldout_eval.jsonl"

    print("=" * 80)
    print(" 🚀 STARTING CODE ORACLE DATASET V3 COMPILATION PIPELINE")
    print(f" Target Scale: {scale} | Medium: {medium_total} | Full: {full_total} | Hybrid: {hybrid_total} | Val Ratio: {val_ratio}")
    print("=" * 80)

    results: Dict[str, Any] = {}

    if scale in ("hybrid", "all"):
        hybrid_stats = compile_hybrid_dataset(
            target_total=hybrid_total,
            val_ratio=val_ratio,
            data_dir=data_dir,
            seed=seed,
            max_tokens=max_tokens,
        )
        results["hybrid"] = hybrid_stats

    if scale in ("medium", "full", "both", "all"):
        # 1. Ingest existing curated base
        base_records = load_existing_curated_records(data_dir, max_tokens=max_tokens)

        # 2. Mine real-world reverts
        revert_records = mine_real_world_reverts(TIER1_LANGUAGES, max_samples_per_repo=10, seed=seed, max_tokens=max_tokens)

        # 3. Harvest surviving mutants from production code
        mutant_records = harvest_production_mutants(TIER1_LANGUAGES, max_samples=40, seed=seed, max_tokens=max_tokens)

        # Pool empirical records
        pool: List[DatasetRecord] = list(base_records) + revert_records + mutant_records

        # Count empirical records per language and class
        max_target = full_total if scale in ("full", "both", "all") else medium_total
        target_per_lang = max_target // len(TIER1_LANGUAGES)

        needed_per_lang: Dict[str, int] = {}
        for lang in TIER1_LANGUAGES:
            current_pass = sum(1 for r in pool if r.language == lang and r.label == 1)
            current_neg = sum(1 for r in pool if r.language == lang and r.label == 0)
            needed = max(0, target_per_lang - min(current_pass, current_neg)) + 40
            needed_per_lang[lang] = needed

        # 4. Generate targeted subtle mutations to fill requirements
        synth_records = generate_targeted_semantic_mutations(
            languages=TIER1_LANGUAGES,
            needed_per_language=needed_per_lang,
            seed=seed,
            max_tokens=max_tokens,
        )
        pool.extend(synth_records)

        print(f"[✓] Total unified dataset pool size: {len(pool)} records.")

        # Compile Medium Scale (v3-medium)
        if scale in ("medium", "both", "all"):
            print("\n[*] Stratifying and balancing Medium Scale (v3-medium)...")
            med_train, med_val = stratify_and_balance_dataset(
                pool,
                target_total=medium_total,
                val_ratio=val_ratio,
                languages=TIER1_LANGUAGES,
                seed=seed,
            )
            med_stats = package_dataset_variant(
                variant_name="v3_medium",
                train_records=med_train,
                val_records=med_val,
                heldout_source_path=heldout_path,
                output_dir=data_dir / "v3_medium",
                zip_path=data_dir / "code_oracle_dataset_v3_medium.zip",
            )
            results["medium"] = med_stats
            print(f"[✓] v3-medium complete: {med_stats['total']} samples ({med_stats['train']} train, {med_stats['val']} val)")
            print(f"    Zip: {med_stats['zip']} ({med_stats['zip_size_bytes'] / 1024:.1f} KB)")

        # Compile Full Scale (v3-full)
        if scale in ("full", "both", "all"):
            print("\n[*] Stratifying and balancing Full Scale (v3-full)...")
            full_train, full_val = stratify_and_balance_dataset(
                pool,
                target_total=full_total,
                val_ratio=val_ratio,
                languages=TIER1_LANGUAGES,
                seed=seed,
            )
            full_stats = package_dataset_variant(
                variant_name="v3_full",
                train_records=full_train,
                val_records=full_val,
                heldout_source_path=heldout_path,
                output_dir=data_dir / "v3_full",
                zip_path=data_dir / "code_oracle_dataset_v3_full.zip",
            )
            results["full"] = full_stats
            print(f"[✓] v3-full complete: {full_stats['total']} samples ({full_stats['train']} train, {full_stats['val']} val)")
            print(f"    Zip: {full_stats['zip']} ({full_stats['zip_size_bytes'] / 1024:.1f} KB)")

    print("\n" + "=" * 80)
    print(" 🎉 DATASET COMPILATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dataset v3 Compilation Pipeline for Code Oracle / Laya ModernBERT Fine-Tuning."
    )
    parser.add_argument(
        "--scale",
        choices=["medium", "full", "hybrid", "both", "all"],
        default="hybrid",
        help="Dataset scale variant to compile (default: hybrid).",
    )
    parser.add_argument(
        "--hybrid-total",
        type=int,
        default=4900,
        help="Target total sample count for v3-hybrid (default: 4900).",
    )
    parser.add_argument(
        "--medium-total",
        type=int,
        default=5000,
        help="Target total sample count for v3-medium (default: 5000).",
    )
    parser.add_argument(
        "--full-total",
        type=int,
        default=10000,
        help="Target total sample count for v3-full (default: 10000).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction allocated to validation split (default: 0.2 -> 80%% train / 20%% val).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Target directory for compiled datasets and zip files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic compilation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Maximum token ceiling per sample (default: 400).",
    )

    args = parser.parse_args()
    compile_v3_datasets(
        scale=args.scale,
        medium_total=args.medium_total,
        full_total=args.full_total,
        hybrid_total=args.hybrid_total,
        val_ratio=args.val_ratio,
        data_dir=args.data_dir,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )



if __name__ == "__main__":
    main()
