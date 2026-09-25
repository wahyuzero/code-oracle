#!/usr/bin/env python3
"""
Mining Tool for Top Open-Source Repositories across Tier 1 Languages.

Shallow-clones top clean repositories across Python, TypeScript, Go, and Rust,
and leverages DatasetGenerator.mine_repository to extract real-world Micro-DSL pairs
for Laya ModernBERT fine-tuning. Merges and balances into train and validation JSONL datasets.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to sys.path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from code_oracle.dataset import (
    DEFAULT_HELDOUT_REPOS,
    DEFAULT_TOP_REPOS,
    FALLBACK_REPOS,
    KNOWN_REPOS,
    DatasetGenerator,
    DatasetRecord,
)
from code_oracle.linearizer import estimate_tokens


def resolve_repo_spec(spec: str) -> Dict[str, str]:
    """Resolve a repo name, owner/repo, or git URL to a structured repo dict."""
    cleaned = spec.strip().lower()
    if cleaned in KNOWN_REPOS:
        return KNOWN_REPOS[cleaned]

    # Full git URL
    if spec.startswith("http://") or spec.startswith("https://") or spec.startswith("git@"):
        url = spec
        repo_name = Path(url.rstrip("/")).stem.removesuffix(".git")
    elif "/" in spec:
        # e.g. owner/repo
        repo_name = spec.split("/")[-1]
        url = f"https://github.com/{spec}.git" if not spec.endswith(".git") else f"https://github.com/{spec}"
    else:
        repo_name = spec
        url = f"https://github.com/{spec}/{spec}.git"

    # Infer language from name if possible
    for k, v in KNOWN_REPOS.items():
        if v["name"] == repo_name.lower():
            return {"language": v["language"], "name": repo_name, "url": url}

    return {"language": "python", "name": repo_name, "url": url}


def find_local_cached_repo(repo_name: str, cache_dir: Path) -> Optional[Path]:
    """
    Search local cache directories for an existing clone:
    1. cache_dir / repo_name
    2. REPO_ROOT / "benchmarks_repos" / repo_name
    """
    candidates = [
        cache_dir / repo_name,
        REPO_ROOT / "benchmarks_repos" / repo_name,
    ]
    for cand in candidates:
        if cand.exists() and (cand / ".git").exists():
            return cand
    return None


def _attempt_fetch_depth(repo_path: Path, depth: int = 250) -> None:
    """Attempt deepening clone history if network is available; fail silently if offline."""
    try:
        count_res = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=3,
        )
        if count_res.returncode == 0:
            count = int(count_res.stdout.strip())
            if count < depth:
                subprocess.run(
                    ["git", "fetch", "--depth", str(depth)],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
    except Exception:
        pass


def shallow_clone_repo(url: str, dest_dir: Path, depth: int = 250) -> Path:
    """
    Shallow-clone a git repository into dest_dir if not already present.
    Reuses existing local clone from benchmarks_repos/ or cache_dir to save bandwidth
    and operate gracefully in offline environments.
    Returns the path to the cloned repository.
    """
    dest_dir = dest_dir.resolve()
    if dest_dir.exists() and (dest_dir / ".git").exists():
        print(f"[*] Repository already cached in target: {dest_dir}")
        _attempt_fetch_depth(dest_dir, depth)
        return dest_dir

    # Check local benchmarks_repos/ cache
    local_cached = find_local_cached_repo(dest_dir.name, dest_dir.parent)
    if local_cached and local_cached.exists() and (local_cached / ".git").exists():
        print(f"[*] Reusing locally cached repository: {local_cached}")
        _attempt_fetch_depth(local_cached, depth)
        return local_cached

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Cloning {url} (shallow, --depth {depth}) into {dest_dir}...")
    cmd = ["git", "clone", "--depth", str(depth), url, str(dest_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to clone {url}: {res.stderr.strip()}")
    print(f"[+] Successfully cloned: {dest_dir.name}")
    return dest_dir



def balance_dataset(
    records: List[DatasetRecord],
    target_count: int,
    positive_label: int = 1,
    negative_label: int = 0,
    seed: int = 42,
) -> List[DatasetRecord]:
    """
    Balance dataset across positive/negative labels and languages.
    Ensures exact 50% PASS and 50% REJECT ratio with equal language representation.
    """
    rng = random.Random(seed)

    by_lang_label: Dict[str, Dict[int, List[DatasetRecord]]] = {}
    for r in records:
        by_lang_label.setdefault(r.language, {}).setdefault(r.label, []).append(r)

    languages = sorted(list(by_lang_label.keys()))
    if not languages:
        return []

    per_lang_target = target_count // len(languages)
    per_lang_pos = per_lang_target // 2
    per_lang_neg = per_lang_target - per_lang_pos

    balanced: List[DatasetRecord] = []
    for lang in languages:
        pos_list = by_lang_label[lang].get(positive_label, [])
        neg_list = by_lang_label[lang].get(negative_label, [])

        rng.shuffle(pos_list)
        rng.shuffle(neg_list)

        # Take up to target count
        chosen_pos = pos_list[:per_lang_pos]
        chosen_neg = neg_list[:per_lang_neg]

        # Balance between chosen pos and neg
        min_label = min(len(chosen_pos), len(chosen_neg))
        balanced.extend(chosen_pos[:min_label])
        balanced.extend(chosen_neg[:min_label])

    rng.shuffle(balanced)
    return balanced


def mine_heldout_eval_dataset(
    cache_dir: Path,
    output_dir: Path,
    target_samples: int = 400,
    seed: int = 42,
    custom_repos: Optional[List[Dict[str, str]]] = None,
    mined_samples_per_repo: int = 50,
    filter_symbolic_gate: bool = True,
    heldout_filename: str = "dataset_heldout_eval.jsonl",
) -> int:
    """
    Mine designated held-out evaluation repositories (unseen during training)
    across Python, TypeScript, Go, and Rust.
    Strictly filters via symbolic gate (filter_symbolic_gate=True) and balances
    PASS/REJECT classes to enable rigorous, independent generalization benchmarking.
    """
    cache_dir = Path(cache_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    heldout_repos = custom_repos if custom_repos is not None else DEFAULT_HELDOUT_REPOS
    all_mined_records: List[DatasetRecord] = []

    print(f"\n[*] Starting Held-Out Evaluation Dataset Mining (Target: {target_samples} samples)...")
    for repo_info in heldout_repos:
        lang = repo_info["language"]
        name = repo_info["name"]
        url = repo_info["url"]
        dest = cache_dir / name

        try:
            repo_path = shallow_clone_repo(url, dest, depth=250)
            generator = DatasetGenerator(languages=[lang], seed=seed, filter_symbolic_gate=filter_symbolic_gate)
            print(f"[*] Mining held-out {lang} repository ({name}) from {repo_path}...")
            mined = generator.mine_repository(repo_path, max_samples=mined_samples_per_repo)
            print(f"    [+] Extracted {len(mined)} held-out samples from {name} ({lang})")
            all_mined_records.extend(mined)
        except Exception as e:
            print(f"[!] Warning: Mining held-out repo {name} ({url}) failed: {e}. Supplementing...")

    # Group mined records by language
    records_by_lang: Dict[str, List[DatasetRecord]] = {}
    for r in all_mined_records:
        records_by_lang.setdefault(r.language, []).append(r)

    languages = ["python", "typescript", "go", "rust"]
    per_lang_target = max(4, target_samples // len(languages))
    per_lang_pos_target = per_lang_target // 2
    per_lang_neg_target = per_lang_target - per_lang_pos_target

    final_pool: List[DatasetRecord] = []
    for lang in languages:
        lang_records = records_by_lang.get(lang, [])
        positives = [r for r in lang_records if r.label == 1]
        negatives = [r for r in lang_records if r.label == 0]

        # If needed, supplement with subtle gate-passing synthetic samples
        needed_pos = max(0, per_lang_pos_target - len(positives))
        needed_neg = max(0, per_lang_neg_target - len(negatives))
        needed_synth = max(0, (needed_pos + needed_neg) * 2)

        if needed_synth > 0:
            synth_gen = DatasetGenerator(languages=[lang], seed=seed + 99, filter_symbolic_gate=filter_symbolic_gate)
            synth_records = synth_gen.generate_synthetic_dataset(num_samples=needed_synth, include_subtle=True)
            positives.extend([r for r in synth_records if r.label == 1 and r.language == lang])
            negatives.extend([r for r in synth_records if r.label == 0 and r.language == lang])

        lang_pos = positives[:per_lang_pos_target]
        lang_neg = negatives[:per_lang_neg_target]

        parity = min(len(lang_pos), len(lang_neg))
        final_pool.extend(lang_pos[:parity])
        final_pool.extend(lang_neg[:parity])

    rng = random.Random(seed)
    rng.shuffle(final_pool)

    out_file = output_dir / heldout_filename
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in final_pool:
            f.write(json.dumps(rec.to_dict()) + "\n")

    print(f"[+] Held-out evaluation dataset generated: {out_file} ({len(final_pool)} samples)")
    return len(final_pool)


def mine_all_top_repos(
    cache_dir: Path,
    output_dir: Path,
    target_samples: int = 2400,
    val_ratio: float = 0.2,
    seed: int = 42,
    custom_repos: Optional[List[Dict[str, str]]] = None,
    mined_samples_per_repo: int = 50,
    filter_symbolic_gate: bool = True,
    generate_heldout: bool = True,
    heldout_samples: int = 400,
) -> Tuple[int, int]:
    """
    Main orchestration routine:
    1. Shallow clone top repositories across Python, TypeScript, Go, Rust.
    2. Mine real-world mutation pairs using DatasetGenerator.mine_repository.
    3. Supplement with synthetic pairs where needed to achieve full balance and target size.
    4. Balance classes and languages evenly.
    5. Write dataset_train.jsonl and dataset_val.jsonl into output_dir.
    6. Optionally mine dedicated held-out evaluation dataset (dataset_heldout_eval.jsonl).
    Returns: (train_count, val_count)
    """
    cache_dir = Path(cache_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repos_to_mine = custom_repos if custom_repos is not None else DEFAULT_TOP_REPOS
    all_mined_records: List[DatasetRecord] = []

    per_repo_target = max(10, mined_samples_per_repo)

    print(f"[*] Starting multi-language repository mining pipeline (Target: {target_samples} samples)...")
    for repo_info in repos_to_mine:
        lang = repo_info["language"]
        name = repo_info["name"]
        url = repo_info["url"]
        dest = cache_dir / name

        try:
            repo_path = shallow_clone_repo(url, dest, depth=250)
            generator = DatasetGenerator(languages=[lang], seed=seed, filter_symbolic_gate=filter_symbolic_gate)
            print(f"[*] Mining {lang} repository ({name}) from {repo_path}...")
            mined = generator.mine_repository(repo_path, max_samples=per_repo_target)
            print(f"    [+] Extracted {len(mined)} raw samples from {name} ({lang})")
            all_mined_records.extend(mined)
        except Exception as e:
            fallback_url = FALLBACK_REPOS.get(lang)
            if fallback_url and fallback_url != url:
                print(f"[!] Warning: Mining {name} failed: {e}. Trying fallback repository: {fallback_url}...")
                try:
                    fallback_name = Path(fallback_url.rstrip("/")).stem.removesuffix(".git")
                    fb_dest = cache_dir / fallback_name
                    fb_path = shallow_clone_repo(fallback_url, fb_dest, depth=250)
                    fb_generator = DatasetGenerator(languages=[lang], seed=seed, filter_symbolic_gate=filter_symbolic_gate)
                    mined = fb_generator.mine_repository(fb_path, max_samples=per_repo_target)
                    print(f"    [+] Extracted {len(mined)} raw samples from fallback {fallback_name} ({lang})")
                    all_mined_records.extend(mined)
                    continue
                except Exception as fb_err:
                    print(f"[!] Fallback mining failed: {fb_err}")
            print(f"[!] Warning: Mining {name} failed: {e}. Trying fallback synthetic generation...")

    # Group mined records by language
    records_by_lang: Dict[str, List[DatasetRecord]] = {}
    for r in all_mined_records:
        records_by_lang.setdefault(r.language, []).append(r)

    # Check if any language needs synthetic supplement to hit per-language target
    languages = [r["language"] for r in repos_to_mine] if repos_to_mine else ["python", "typescript", "go", "rust"]
    per_lang_target = target_samples // len(languages)
    per_lang_pos_target = per_lang_target // 2
    per_lang_neg_target = per_lang_target - per_lang_pos_target

    final_pool: List[DatasetRecord] = []
    for lang in languages:
        lang_records = records_by_lang.get(lang, [])
        positives = [r for r in lang_records if r.label == 1]
        negatives = [r for r in lang_records if r.label == 0]

        # If needed, supplement with synthetic samples for balance and diversity
        needed_pos = max(0, per_lang_pos_target - len(positives))
        needed_neg = max(0, per_lang_neg_target - len(negatives))
        needed_synth = max(0, (needed_pos + needed_neg) * 2)

        if needed_synth > 0:
            synth_gen = DatasetGenerator(languages=[lang], seed=seed + 10, filter_symbolic_gate=filter_symbolic_gate)
            synth_records = synth_gen.generate_synthetic_dataset(num_samples=needed_synth, include_subtle=True)
            positives.extend([r for r in synth_records if r.label == 1 and r.language == lang])
            negatives.extend([r for r in synth_records if r.label == 0 and r.language == lang])

        # Take target counts
        lang_pos = positives[:per_lang_pos_target]
        lang_neg = negatives[:per_lang_neg_target]

        # Exact parity
        parity = min(len(lang_pos), len(lang_neg))
        final_pool.extend(lang_pos[:parity])
        final_pool.extend(lang_neg[:parity])

    # Final shuffle
    rng = random.Random(seed)
    rng.shuffle(final_pool)

    # Stratified split to ensure both train and val are exactly balanced
    positives = [r for r in final_pool if r.label == 1]
    negatives = [r for r in final_pool if r.label == 0]

    val_pos_n = max(1, int(len(positives) * val_ratio))
    val_neg_n = max(1, int(len(negatives) * val_ratio))

    val_records = positives[:val_pos_n] + negatives[:val_neg_n]
    train_records = positives[val_pos_n:] + negatives[val_neg_n:]

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    train_file = output_dir / "dataset_train.jsonl"
    val_file = output_dir / "dataset_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    print("\n[+] Dataset Mining & Merging Complete:")
    print(f"    - Output Directory:    {output_dir}")
    print(f"    - dataset_train.jsonl: {len(train_records)} samples")
    print(f"    - dataset_val.jsonl:   {len(val_records)} samples")
    print(f"    - Total Balanced:      {len(final_pool)} samples")

    if generate_heldout:
        heldout_repos = [] if custom_repos == [] else None
        mine_heldout_eval_dataset(
            cache_dir=cache_dir,
            output_dir=output_dir,
            target_samples=heldout_samples,
            seed=seed + 777,
            custom_repos=heldout_repos,
            mined_samples_per_repo=mined_samples_per_repo,
            filter_symbolic_gate=filter_symbolic_gate,
        )

    return len(train_records), len(val_records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine top real-world open-source repositories to generate balanced ModernBERT training datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        type=Path,
        default=Path("/tmp/code_oracle_mined_repos"),
        help="Local directory to store shallow git clones.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=REPO_ROOT / "data",
        help="Destination directory for dataset_train.jsonl and dataset_val.jsonl.",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=2400,
        help="Target total number of balanced samples across all languages (2,000 to 4,000 recommended).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation and shuffling.",
    )
    parser.add_argument(
        "--repos",
        "-r",
        nargs="*",
        default=None,
        help="List of repository names or URLs to mine (e.g. psf/requests colinhacks/zod or fastapi trpc cobra clap).",
    )
    parser.add_argument(
        "--mine-samples-per-repo",
        type=int,
        default=150,
        help="Number of real-world samples to mine per cloned repository before synthetic balancing.",
    )
    parser.add_argument(
        "--filter-symbolic-gate",
        action="store_true",
        default=True,
        help="Filter out mutations that fail symbolic gate (keep only gray-area hard negatives).",
    )
    parser.add_argument(
        "--no-filter-symbolic-gate",
        action="store_false",
        dest="filter_symbolic_gate",
        help="Disable symbolic gate filtering (allow mutations failing Stage 1 & 2 AST checks).",
    )
    parser.add_argument(
        "--generate-heldout",
        action="store_true",
        default=True,
        help="Also generate dataset_heldout_eval.jsonl from unseen evaluation repositories.",
    )
    parser.add_argument(
        "--no-heldout",
        action="store_false",
        dest="generate_heldout",
        help="Skip generating dataset_heldout_eval.jsonl.",
    )
    parser.add_argument(
        "--heldout-samples",
        type=int,
        default=400,
        help="Number of balanced samples to generate for dataset_heldout_eval.jsonl.",
    )
    parser.add_argument(
        "--heldout-only",
        action="store_true",
        default=False,
        help="Only mine and generate held-out evaluation dataset (skip train/val).",
    )

    args = parser.parse_args()

    if args.heldout_only:
        heldout_count = mine_heldout_eval_dataset(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            target_samples=args.heldout_samples,
            seed=args.seed,
            mined_samples_per_repo=args.mine_samples_per_repo,
            filter_symbolic_gate=args.filter_symbolic_gate,
        )
        print(f"[✓] Held-out evaluation dataset generated with {heldout_count} samples.")
        return 0

    custom_repos = None
    if args.repos:
        repo_specs = []
        for r_arg in args.repos:
            for part in r_arg.replace(",", " ").split():
                if part.strip():
                    repo_specs.append(part.strip())
        custom_repos = [resolve_repo_spec(s) for s in repo_specs]

    train_n, val_n = mine_all_top_repos(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        target_samples=args.num_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        custom_repos=custom_repos,
        mined_samples_per_repo=args.mine_samples_per_repo,
        filter_symbolic_gate=args.filter_symbolic_gate,
        generate_heldout=args.generate_heldout,
        heldout_samples=args.heldout_samples,
    )

    if train_n + val_n < 2000 or train_n + val_n > 4000:
        print(f"[!] Warning: Total sample count {train_n + val_n} outside 2,000-4,000 recommended window.")
    else:
        print(f"[✓] Total sample count ({train_n + val_n}) conforms with 2,000-4,000 fine-tuning specifications.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

