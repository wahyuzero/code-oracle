#!/usr/bin/env python3
"""
Mining Revert & GHSA/CVE Tool for Code Oracle Dataset Curation.

Extracts real-world reverted commit pairs and CVE / GHSA security advisory patches
across top-tier repositories in Python, TypeScript, Go, and Rust.
Strictly filters candidates via TopoSliceEngine.verify() to guarantee all samples
pass Stage 1 (AST) & Stage 2 (Tarjan cycle) symbolic checks (symbolic_gate_passed=True),
producing balanced DatasetRecord instances with ADR-0003 multi-task taxonomy labels.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to sys.path if running directly or from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from code_oracle.dataset import (
    DEFAULT_HELDOUT_REPOS,
    DEFAULT_TOP_REPOS,
    KNOWN_REPOS,
    TAXONOMY_CLASSES,
    DatasetRecord,
)
from code_oracle.engine import TopoSliceEngine
from code_oracle.languages import detect_language
from code_oracle.linearizer import estimate_tokens

# Comprehensive catalogue of target repositories across Tier 1 languages
TARGET_REVERT_REPOS: List[Dict[str, str]] = [
    # Python
    {
        "language": "python",
        "name": "fastapi",
        "url": "https://github.com/tiangolo/fastapi.git",
        "pkg_name": "fastapi",
        "ecosystem": "PyPI",
    },
    {
        "language": "python",
        "name": "pydantic",
        "url": "https://github.com/pydantic/pydantic.git",
        "pkg_name": "pydantic",
        "ecosystem": "PyPI",
    },
    {
        "language": "python",
        "name": "requests",
        "url": "https://github.com/psf/requests.git",
        "pkg_name": "requests",
        "ecosystem": "PyPI",
    },
    {
        "language": "python",
        "name": "werkzeug",
        "url": "https://github.com/pallets/werkzeug.git",
        "pkg_name": "werkzeug",
        "ecosystem": "PyPI",
    },
    # TypeScript
    {
        "language": "typescript",
        "name": "hono",
        "url": "https://github.com/honojs/hono.git",
        "pkg_name": "hono",
        "ecosystem": "npm",
    },
    {
        "language": "typescript",
        "name": "zod",
        "url": "https://github.com/colinhacks/zod.git",
        "pkg_name": "zod",
        "ecosystem": "npm",
    },
    {
        "language": "typescript",
        "name": "trpc",
        "url": "https://github.com/trpc/trpc.git",
        "pkg_name": "@trpc/server",
        "ecosystem": "npm",
    },
    {
        "language": "typescript",
        "name": "fastify",
        "url": "https://github.com/fastify/fastify.git",
        "pkg_name": "fastify",
        "ecosystem": "npm",
    },
    # Go
    {
        "language": "go",
        "name": "gin",
        "url": "https://github.com/gin-gonic/gin.git",
        "pkg_name": "github.com/gin-gonic/gin",
        "ecosystem": "Go",
    },
    {
        "language": "go",
        "name": "fiber",
        "url": "https://github.com/gofiber/fiber.git",
        "pkg_name": "github.com/gofiber/fiber/v2",
        "ecosystem": "Go",
    },
    {
        "language": "go",
        "name": "cobra",
        "url": "https://github.com/spf13/cobra.git",
        "pkg_name": "github.com/spf13/cobra",
        "ecosystem": "Go",
    },
    {
        "language": "go",
        "name": "chi",
        "url": "https://github.com/go-chi/chi.git",
        "pkg_name": "github.com/go-chi/chi/v5",
        "ecosystem": "Go",
    },
    # Rust
    {
        "language": "rust",
        "name": "tokio",
        "url": "https://github.com/tokio-rs/tokio.git",
        "pkg_name": "tokio",
        "ecosystem": "crates.io",
    },
    {
        "language": "rust",
        "name": "axum",
        "url": "https://github.com/tokio-rs/axum.git",
        "pkg_name": "axum",
        "ecosystem": "crates.io",
    },
    {
        "language": "rust",
        "name": "ripgrep",
        "url": "https://github.com/BurntSushi/ripgrep.git",
        "pkg_name": "ripgrep",
        "ecosystem": "crates.io",
    },
    {
        "language": "rust",
        "name": "clap",
        "url": "https://github.com/clap-rs/clap.git",
        "pkg_name": "clap",
        "ecosystem": "crates.io",
    },
    {
        "language": "rust",
        "name": "serde",
        "url": "https://github.com/serde-rs/serde.git",
        "pkg_name": "serde",
        "ecosystem": "crates.io",
    },
]


def find_cached_repository(repo_name: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Find an existing local clone across common cache paths."""
    candidates = []
    if cache_dir is not None:
        candidates.append(cache_dir / repo_name)
    candidates.extend(
        [
            REPO_ROOT / "benchmarks_repos" / repo_name,
            REPO_ROOT / ".cache" / "repos" / repo_name,
            Path("/tmp/code_oracle_repos") / repo_name,
        ]
    )
    for cand in candidates:
        if cand.exists() and (cand / ".git").exists():
            return cand.resolve()
    return None


def ensure_repository_cloned(
    repo_info: Dict[str, str],
    cache_dir: Path,
    depth: int = 250,
    offline: bool = False,
) -> Optional[Path]:
    """
    Ensure the target repository is available locally.
    Reuses existing local clones in benchmarks_repos/ or cache_dir.
    Clones shallowly if missing and online.
    """
    repo_name = repo_info["name"]
    local_cached = find_cached_repository(repo_name, cache_dir)
    if local_cached is not None:
        return local_cached

    dest_dir = cache_dir / repo_name
    if dest_dir.exists() and (dest_dir / ".git").exists():
        return dest_dir.resolve()

    if offline:
        print(f"[!] Offline mode: Skipping clone for {repo_name} ({repo_info['url']})")
        return None

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Cloning {repo_name} ({repo_info['url']}) into {dest_dir} (depth {depth})...")
    try:
        res = subprocess.run(
            ["git", "clone", "--depth", str(depth), repo_info["url"], str(dest_dir)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if res.returncode == 0 and (dest_dir / ".git").exists():
            print(f"[+] Cloned {repo_name} successfully.")
            return dest_dir.resolve()
        else:
            print(f"[!] Warning: Git clone failed for {repo_name}: {res.stderr.strip()}")
            return None
    except Exception as exc:
        print(f"[!] Warning: Exception cloning {repo_name}: {exc}")
        return None


def query_osv_advisories(
    pkg_name: str,
    ecosystem: str,
    timeout: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Query the public OSV API (https://api.osv.dev/v1/query) for package vulnerabilities.
    Returns list of advisory dicts, or empty list on offline / error conditions.
    """
    if not pkg_name or not ecosystem:
        return []

    url = "https://api.osv.dev/v1/query"
    payload = json.dumps({"package": {"name": pkg_name, "ecosystem": ecosystem}}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "CodeOracle-Miner/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("vulns", [])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception):
        # Graceful degradation on offline or network timeouts
        pass
    return []


class GitRevertMiner:
    """
    Mines real-world reverted commit pairs and CVE / GHSA security advisory patches.
    Verifies every candidate through TopoSliceEngine, strictly enforcing symbolic_gate_passed=True.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        filter_symbolic_gate: bool = True,
        seed: int = 42,
    ):
        self.languages = languages or ["python", "typescript", "go", "rust"]
        self.filter_symbolic_gate = filter_symbolic_gate
        self.rng = random.Random(seed)

    def extract_revert_pairs_from_repo(
        self,
        repo_path: Path,
        max_samples: int = 30,
        engine: Optional[TopoSliceEngine] = None,
    ) -> List[DatasetRecord]:
        """
        Extract reverted commit pairs from git log:
        1. Find commits referencing 'revert' in subject.
        2. Extract referenced original buggy commit hash 'This reverts commit <hash>'.
        3. Extract the original buggy diff and clean base state.
        4. When original hash is unavailable, extract reverse diff of the revert.
        5. Verify diffs via TopoSliceEngine.verify() to guarantee symbolic_gate_passed=True.
        6. Pair with clean revert patch as positive sample.
        """
        repo_path = repo_path.resolve()
        if not (repo_path / ".git").exists():
            return []

        if engine is None:
            engine = TopoSliceEngine(workspace_root=repo_path)
            engine.indexer.scan_workspace()

        records: List[DatasetRecord] = []

        try:
            cmd = ["git", "log", "--grep=revert", "-i", "-n", "180", "--format=%H|%s"]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=12)
            if res.returncode != 0 or not res.stdout.strip():
                return []

            for line in res.stdout.strip().splitlines():
                if "|" not in line:
                    continue
                rev_hash, subject = line.split("|", 1)
                if not re.search(r"(?i)\b(revert|reverted|reverting)\b|^revert\b", subject):
                    continue

                # Retrieve full commit message body to find referenced buggy commit hash
                b_msg_cmd = ["git", "show", "-s", "--format=%B", rev_hash]
                b_msg_res = subprocess.run(b_msg_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                commit_body = b_msg_res.stdout if b_msg_res.returncode == 0 else ""

                rev_match = re.search(
                    r"(?i)(?:this reverts commit|reverts commit|reverting commit)\s+([0-9a-f]{7,40})",
                    commit_body,
                )
                bug_hash = rev_match.group(1) if rev_match else None

                bug_body = ""
                bug_exists = False
                if bug_hash:
                    # Check if bug commit exists in this clone
                    chk_cmd = ["git", "cat-file", "-e", f"{bug_hash}^{{commit}}"]
                    chk_res = subprocess.run(chk_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=3)
                    if chk_res.returncode == 0:
                        bug_exists = True
                        b_ctx_cmd = ["git", "show", "-s", "--format=%B", bug_hash]
                        b_ctx_res = subprocess.run(b_ctx_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if b_ctx_res.returncode == 0:
                            bug_body = b_ctx_res.stdout.strip()

                context_text = f"{subject}\n{commit_body}\n{bug_body}"

                # Collect files to check: (fname, buggy_diff, clean_base_content, clean_diff, clean_base_for_fix)
                file_tuples: List[Tuple[str, str, Optional[str], Optional[str], Optional[str]]] = []

                if bug_exists and bug_hash:
                    f_cmd = ["git", "show", "--name-only", "--format=", bug_hash]
                    f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    if f_res.returncode == 0:
                        for fn in f_res.stdout.splitlines():
                            fn = fn.strip()
                            if not fn:
                                continue
                            d_cmd = ["git", "show", "-p", bug_hash, "--", fn]
                            d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                            if d_res.returncode == 0 and d_res.stdout.strip():
                                o_cmd = ["git", "show", f"{bug_hash}~1:{fn}"]
                                o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                orig_c = o_res.stdout if o_res.returncode == 0 else None

                                # Clean fix diff from revert commit
                                rev_diff_cmd = ["git", "show", "-p", rev_hash, "--", fn]
                                rev_diff_res = subprocess.run(rev_diff_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                rev_diff = rev_diff_res.stdout if rev_diff_res.returncode == 0 else None
                                rev_orig_cmd = ["git", "show", f"{rev_hash}~1:{fn}"]
                                rev_orig_res = subprocess.run(rev_orig_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                rev_orig = rev_orig_res.stdout if rev_orig_res.returncode == 0 else None

                                file_tuples.append((fn, d_res.stdout, orig_c, rev_diff, rev_orig))

                # Fallback: reverse diff of revert commit
                if not file_tuples:
                    f_cmd = ["git", "show", "--name-only", "--format=", rev_hash]
                    f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    if f_res.returncode == 0:
                        for fn in f_res.stdout.splitlines():
                            fn = fn.strip()
                            if not fn:
                                continue
                            d_cmd = ["git", "show", "-R", "-p", rev_hash, "--", fn]
                            d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                            if d_res.returncode == 0 and d_res.stdout.strip():
                                o_cmd = ["git", "show", f"{rev_hash}:{fn}"]
                                o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                orig_c = o_res.stdout if o_res.returncode == 0 else None

                                # Clean fix diff is direct revert commit diff
                                rev_diff_cmd = ["git", "show", "-p", rev_hash, "--", fn]
                                rev_diff_res = subprocess.run(rev_diff_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                rev_diff = rev_diff_res.stdout if rev_diff_res.returncode == 0 else None
                                rev_orig_cmd = ["git", "show", f"{rev_hash}~1:{fn}"]
                                rev_orig_res = subprocess.run(rev_orig_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                rev_orig = rev_orig_res.stdout if rev_orig_res.returncode == 0 else None

                                file_tuples.append((fn, d_res.stdout, orig_c, rev_diff, rev_orig))

                for fn, bug_diff, clean_base, fix_diff, fix_base in file_tuples:
                    lang = detect_language(fn)
                    if not lang or lang not in self.languages:
                        continue

                    # 1. Negative buggy sample
                    try:
                        rep_bug = engine.verify(fn, bug_diff, original_content=clean_base)
                        if self.filter_symbolic_gate and rep_bug.status != "APPROVED":
                            continue

                        tax = DatasetRecord.assign_taxonomy_labels(
                            category="real_revert",
                            label=0,
                            risk_score=0.88,
                            context_text=context_text,
                            invariant_violations=rep_bug.invariant_violations,
                        )
                        record_neg = DatasetRecord(
                            input_dsl=rep_bug.linearized_subgraph,
                            label=0,
                            risk_score=round(self.rng.uniform(0.85, 0.95), 4),
                            category="real_revert",
                            language=lang,
                            taxonomy_labels=tax,
                            symbolic_gate_passed=True,
                            source_type="real_revert",
                        )
                        records.append(record_neg)

                        # 2. Paired positive sample: Clean revert commit diff
                        if fix_diff:
                            rep_fix = engine.verify(fn, fix_diff, original_content=fix_base)
                            if rep_fix.status == "APPROVED":
                                record_pos = DatasetRecord(
                                    input_dsl=rep_fix.linearized_subgraph,
                                    label=1,
                                    risk_score=round(self.rng.uniform(0.02, 0.12), 4),
                                    category="clean_commit",
                                    language=lang,
                                    taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                                    symbolic_gate_passed=True,
                                    source_type="real_revert_fix",
                                )
                                records.append(record_pos)

                        if len(records) >= max_samples:
                            return records[:max_samples]
                    except Exception:
                        continue
        except Exception:
            pass

        return records[:max_samples]

    def extract_cve_ghsa_pairs_from_repo(
        self,
        repo_path: Path,
        repo_info: Dict[str, str],
        max_samples: int = 30,
        engine: Optional[TopoSliceEngine] = None,
        query_osv: bool = True,
        offline: bool = False,
    ) -> List[DatasetRecord]:
        """
        Extract CVE and GHSA security advisory patches:
        1. Query OSV API for package vulnerabilities to identify fix commit hashes.
        2. Query local git log for commits with CVE-, GHSA-, or security advisory keywords.
        3. Extract the pre-fix vulnerable state (reverse diff of fix) and fix patch.
        4. Strictly filter via TopoSliceEngine.verify() (symbolic_gate_passed=True).
        5. Map to ADR-0003 SecuritySurface and ConcurrencyHazard taxonomy classes.
        """
        repo_path = repo_path.resolve()
        if not (repo_path / ".git").exists():
            return []

        if engine is None:
            engine = TopoSliceEngine(workspace_root=repo_path)
            engine.indexer.scan_workspace()

        records: List[DatasetRecord] = []
        fix_commit_targets: List[Tuple[str, str, str]] = []  # (commit_hash, vuln_id, context_text)

        # 1. OSV API Advisories (if online and configured)
        if query_osv and not offline:
            pkg_name = repo_info.get("pkg_name", "")
            ecosystem = repo_info.get("ecosystem", "")
            advisories = query_osv_advisories(pkg_name, ecosystem)
            for adv in advisories:
                adv_id = adv.get("id", "SECURITY-ADV")
                summary = adv.get("summary", "")
                details = adv.get("details", "")
                ctx = f"{adv_id} {summary}\n{details}"

                # Discover fix commits in references
                for ref in adv.get("references", []):
                    ref_url = ref.get("url", "")
                    match = re.search(r"github\.com/[^/]+/[^/]+/commit/([0-9a-f]{7,40})", ref_url, re.I)
                    if match:
                        c_hash = match.group(1)
                        # Verify commit exists in local git history
                        chk = subprocess.run(["git", "cat-file", "-e", f"{c_hash}^{{commit}}"], cwd=str(repo_path), capture_output=True)
                        if chk.returncode == 0:
                            fix_commit_targets.append((c_hash, adv_id, ctx))

                # Discover fix commits in affected ranges
                for affected in adv.get("affected", []):
                    for r_range in affected.get("ranges", []):
                        for event in r_range.get("events", []):
                            if "fixed" in event:
                                f_hash = event["fixed"]
                                chk = subprocess.run(["git", "cat-file", "-e", f"{f_hash}^{{commit}}"], cwd=str(repo_path), capture_output=True)
                                if chk.returncode == 0:
                                    fix_commit_targets.append((f_hash, adv_id, ctx))

        # 2. Local Git Log CVE / GHSA Security Mining (offline-capable)
        try:
            cmd = [
                "git",
                "log",
                "--grep=CVE-",
                "--grep=GHSA-",
                "--grep=security",
                "--grep=vulnerability",
                "--grep=advisory",
                "-i",
                "-n",
                "150",
                "--format=%H|%s",
            ]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=12)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    if "|" not in line:
                        continue
                    h_val, subj = line.split("|", 1)
                    if re.search(r"(?i)\b(cve-\d{4}-\d+|ghsa-[a-z0-9-]+|vulnerability|security advisory)\b", subj):
                        b_cmd = ["git", "show", "-s", "--format=%B", h_val]
                        b_res = subprocess.run(b_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        b_body = b_res.stdout if b_res.returncode == 0 else ""
                        fix_commit_targets.append((h_val, "CVE/GHSA-LOCAL", f"{subj}\n{b_body}"))
        except Exception:
            pass

        seen_commits = set()
        for fix_hash, adv_id, context_text in fix_commit_targets:
            if fix_hash in seen_commits:
                continue
            seen_commits.add(fix_hash)

            f_cmd = ["git", "show", "--name-only", "--format=", fix_hash]
            f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
            if f_res.returncode != 0:
                continue

            for fn in f_res.stdout.splitlines():
                fn = fn.strip()
                if not fn:
                    continue
                lang = detect_language(fn)
                if not lang or lang not in self.languages:
                    continue

                # a) Negative sample: Vulnerable state (reverse diff of fix commit)
                rev_cmd = ["git", "show", "-R", "-p", fix_hash, "--", fn]
                rev_res = subprocess.run(rev_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                if rev_res.returncode == 0 and rev_res.stdout.strip():
                    o_cmd = ["git", "show", f"{fix_hash}:{fn}"]
                    o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    orig_c = o_res.stdout if o_res.returncode == 0 else None

                    try:
                        rep_vuln = engine.verify(fn, rev_res.stdout, original_content=orig_c)
                        if self.filter_symbolic_gate and rep_vuln.status != "APPROVED":
                            continue

                        # Determine primary category and taxonomy
                        ctx_lower = context_text.lower()
                        is_concurrency = bool(
                            re.search(r"\b(race|deadlock|mutex|channel|goroutine|thread|concurr\w*)\b", ctx_lower)
                        )
                        category_name = "concurrency_hazard" if is_concurrency else "security_surface"

                        tax = DatasetRecord.assign_taxonomy_labels(
                            category=category_name,
                            label=0,
                            risk_score=0.92,
                            context_text=context_text,
                            invariant_violations=rep_vuln.invariant_violations,
                        )

                        # Enforce explicit security and concurrency activation
                        if not is_concurrency:
                            tax["SecuritySurface"] = max(tax["SecuritySurface"], 0.90)
                        else:
                            tax["ConcurrencyHazard"] = max(tax["ConcurrencyHazard"], 0.90)
                            tax["SecuritySurface"] = max(tax["SecuritySurface"], 0.85)

                        rec_neg = DatasetRecord(
                            input_dsl=rep_vuln.linearized_subgraph,
                            label=0,
                            risk_score=round(self.rng.uniform(0.88, 0.96), 4),
                            category=category_name,
                            language=lang,
                            taxonomy_labels=tax,
                            symbolic_gate_passed=True,
                            source_type="cve_ghsa",
                        )
                        records.append(rec_neg)

                        # b) Positive sample: Security fix patch
                        d_cmd = ["git", "show", "-p", fix_hash, "--", fn]
                        d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if d_res.returncode == 0 and d_res.stdout.strip():
                            o_fix_cmd = ["git", "show", f"{fix_hash}~1:{fn}"]
                            o_fix_res = subprocess.run(o_fix_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                            orig_fix_c = o_fix_res.stdout if o_fix_res.returncode == 0 else None

                            rep_fix = engine.verify(fn, d_res.stdout, original_content=orig_fix_c)
                            if rep_fix.status == "APPROVED":
                                rec_pos = DatasetRecord(
                                    input_dsl=rep_fix.linearized_subgraph,
                                    label=1,
                                    risk_score=round(self.rng.uniform(0.02, 0.12), 4),
                                    category="real_hotfix",
                                    language=lang,
                                    taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                                    symbolic_gate_passed=True,
                                    source_type="cve_ghsa_fix",
                                )
                                records.append(rec_pos)

                        if len(records) >= max_samples:
                            return records[:max_samples]
                    except Exception:
                        continue

        return records[:max_samples]


def balance_revert_dataset(
    records: List[DatasetRecord],
    target_count: int,
    seed: int = 42,
) -> List[DatasetRecord]:
    """
    Balance dataset across positive/negative labels and languages.
    Ensures 50% PASS (label 1) and 50% REJECT (label 0) with equal representation.
    """
    rng = random.Random(seed)
    by_lang_label: Dict[str, Dict[int, List[DatasetRecord]]] = {}
    for r in records:
        by_lang_label.setdefault(r.language, {}).setdefault(r.label, []).append(r)

    languages = sorted(list(by_lang_label.keys()))
    if not languages:
        return []

    per_lang_target = max(2, target_count // len(languages))
    per_lang_pos = per_lang_target // 2
    per_lang_neg = per_lang_target - per_lang_pos

    balanced: List[DatasetRecord] = []
    for lang in languages:
        pos_list = by_lang_label[lang].get(1, [])
        neg_list = by_lang_label[lang].get(0, [])
        rng.shuffle(pos_list)
        rng.shuffle(neg_list)

        chosen_pos = pos_list[:per_lang_pos]
        chosen_neg = neg_list[:per_lang_neg]
        min_label = min(len(chosen_pos), len(chosen_neg))
        balanced.extend(chosen_pos[:min_label])
        balanced.extend(chosen_neg[:min_label])

    rng.shuffle(balanced)
    return balanced


def run_mining_pipeline(
    cache_dir: Path,
    output_path: Path,
    repos_to_mine: Optional[List[Dict[str, str]]] = None,
    languages: Optional[List[str]] = None,
    max_samples_per_repo: int = 40,
    target_samples: int = 400,
    include_osv: bool = True,
    offline: bool = False,
    seed: int = 42,
) -> List[DatasetRecord]:
    """
    Executes the end-to-end revert & GHSA/CVE mining pipeline.
    """
    cache_dir = Path(cache_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_repos = repos_to_mine or TARGET_REVERT_REPOS
    target_languages = languages or ["python", "typescript", "go", "rust"]

    miner = GitRevertMiner(languages=target_languages, filter_symbolic_gate=True, seed=seed)
    all_records: List[DatasetRecord] = []

    print(f"[*] Starting Git Revert & GHSA/CVE Mining Pipeline across {len(target_repos)} repositories...")

    for repo_info in target_repos:
        name = repo_info["name"]
        lang = repo_info["language"]
        if lang not in target_languages:
            continue

        repo_path = ensure_repository_cloned(repo_info, cache_dir=cache_dir, offline=offline)
        if not repo_path or not (repo_path / ".git").exists():
            print(f"[!] Warning: Repository {name} not available locally, skipping...")
            continue

        print(f"[*] Mining {name} ({lang}) from {repo_path}...")
        engine = TopoSliceEngine(workspace_root=repo_path)
        engine.indexer.scan_workspace()

        # 1. Mine real reverts
        reverts = miner.extract_revert_pairs_from_repo(
            repo_path=repo_path,
            max_samples=max_samples_per_repo // 2,
            engine=engine,
        )
        print(f"    [+] Extracted {len(reverts)} revert records from {name}")
        all_records.extend(reverts)

        # 2. Mine CVE / GHSA advisories
        cves = miner.extract_cve_ghsa_pairs_from_repo(
            repo_path=repo_path,
            repo_info=repo_info,
            max_samples=max_samples_per_repo // 2,
            engine=engine,
            query_osv=include_osv,
            offline=offline,
        )
        print(f"    [+] Extracted {len(cves)} CVE/GHSA records from {name}")
        all_records.extend(cves)

    # Balance across classes and languages
    balanced_records = balance_revert_dataset(all_records, target_count=target_samples, seed=seed)

    # If dataset has fewer records than target, supplement with subtle gate-passing synthetic samples
    if len(balanced_records) < target_samples:
        from code_oracle.dataset import DatasetGenerator
        needed = target_samples - len(balanced_records)
        print(f"[*] Supplementing {needed} balanced subtle samples to achieve target {target_samples}...")
        synth_gen = DatasetGenerator(languages=target_languages, seed=seed, filter_symbolic_gate=True)
        synth_samples = synth_gen.generate_synthetic_dataset(num_samples=needed, include_subtle=True)
        all_pool = balanced_records + synth_samples
        balanced_records = balance_revert_dataset(all_pool, target_count=target_samples, seed=seed)

    # Enforce token length ceiling (< 400 tokens)
    compliant_records = [r for r in balanced_records if estimate_tokens(r.input_dsl) <= 400]

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in compliant_records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    print(f"[+] Output written to {output_path} ({len(compliant_records)} balanced records)")
    return compliant_records


def main():
    parser = argparse.ArgumentParser(
        description="Mining Revert & GHSA/CVE Tool for Code Oracle Dataset Curation."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks_repos",
        help="Directory where target repositories are cloned or cached.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "dataset_mined_reverts_cve.jsonl",
        help="Path for output JSONL dataset.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=200,
        help="Target total balanced samples across PASS/REJECT classes.",
    )
    parser.add_argument(
        "--max-samples-per-repo",
        type=int,
        default=30,
        help="Maximum samples extracted per repository.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,typescript,go,rust",
        help="Comma-separated target languages.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Run in offline mode without attempting git clone or remote OSV network queries.",
    )
    parser.add_argument(
        "--no-osv",
        action="store_false",
        dest="include_osv",
        help="Disable remote OSV API query.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing.")

    args = parser.parse_args()
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]

    run_mining_pipeline(
        cache_dir=args.cache_dir,
        output_path=args.output,
        languages=langs,
        max_samples_per_repo=args.max_samples_per_repo,
        target_samples=args.target_samples,
        include_osv=args.include_osv,
        offline=args.offline,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
