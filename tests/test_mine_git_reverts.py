"""
Unit and integration tests for tools/mine_git_reverts.py.

Covers:
1. Target repository catalogue resolution and local cache lookup.
2. Revert commit parsing from git history and original buggy diff resolution.
3. CVE / GHSA advisory mining (OSV API mock & local commit log mining).
4. Strict TopoSliceEngine symbolic gate filtering (symbolic_gate_passed=True).
5. ADR-0003 multi-task risk taxonomy score assignment.
6. Dataset balancing and inspect_file validation.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from inspect_dataset import inspect_file
from mine_git_reverts import (
    TARGET_REVERT_REPOS,
    GitRevertMiner,
    balance_revert_dataset,
    ensure_repository_cloned,
    find_cached_repository,
    query_osv_advisories,
    run_mining_pipeline,
)
from code_oracle.dataset import DatasetRecord


@pytest.fixture
def git_repo_with_reverts(tmp_path: Path) -> Path:
    """
    Construct a real local git repository with:
    - Base commit (v1)
    - Buggy commit (v2) introducing subtle logic regression
    - Revert commit (v3) referencing 'This reverts commit <hash>'
    - Security advisory commit (v4) with CVE / race hazard fix
    """
    repo_dir = tmp_path / "test_git_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Committer"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True)

    # Commit 1: Clean base state
    service_file = repo_dir / "calc_service.py"
    service_file.write_text(
        "def compute_discount(price: float, is_vip: bool) -> float:\n"
        "    if is_vip:\n"
        "        return price * 0.8\n"
        "    return price * 1.0\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "feat: initial discount service"], cwd=str(repo_dir), check=True)

    # Commit 2: Buggy commit (inverts discount logic silently)
    service_file.write_text(
        "def compute_discount(price: float, is_vip: bool) -> float:\n"
        "    if not is_vip:\n"
        "        return price * 0.8\n"
        "    return price * 1.0\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: update VIP discount policy with bug"],
        cwd=str(repo_dir),
        check=True,
    )

    bug_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Commit 3: Revert commit
    service_file.write_text(
        "def compute_discount(price: float, is_vip: bool) -> float:\n"
        "    if is_vip:\n"
        "        return price * 0.8\n"
        "    return price * 1.0\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    revert_msg = f"Revert 'feat: update VIP discount policy with bug'\n\nThis reverts commit {bug_hash}."
    subprocess.run(["git", "commit", "-m", revert_msg], cwd=str(repo_dir), check=True)

    # Commit 4: Security advisory commit (CVE race condition fix)
    auth_file = repo_dir / "auth_lock.py"
    # Base state for auth_lock
    auth_file.write_text(
        "import threading\n\n"
        "_LOCK = threading.Lock()\n\n"
        "def update_session(token: str) -> None:\n"
        "    # Missing lock acquisition causing data race\n"
        "    global _SHARED_SESSION\n"
        "    _SHARED_SESSION = token\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "feat: initial session updater"], cwd=str(repo_dir), check=True)

    auth_file.write_text(
        "import threading\n\n"
        "_LOCK = threading.Lock()\n\n"
        "def update_session(token: str) -> None:\n"
        "    with _LOCK:\n"
        "        global _SHARED_SESSION\n"
        "        _SHARED_SESSION = token\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    cve_msg = "fix(security): CVE-2024-5555 resolve race condition in mutex session lock"
    subprocess.run(["git", "commit", "-m", cve_msg], cwd=str(repo_dir), check=True)

    return repo_dir


def test_target_repos_catalogue():
    """Verify target repo catalogue includes required Tier 1 repositories."""
    names = {r["name"] for r in TARGET_REVERT_REPOS}
    # Python
    assert "fastapi" in names
    assert "pydantic" in names
    assert "requests" in names
    assert "werkzeug" in names
    # TypeScript
    assert "hono" in names
    assert "zod" in names
    assert "trpc" in names
    assert "fastify" in names
    # Go
    assert "gin" in names
    assert "fiber" in names
    assert "cobra" in names
    assert "chi" in names
    # Rust
    assert "tokio" in names
    assert "axum" in names
    assert "ripgrep" in names
    assert "clap" in names
    assert "serde" in names


def test_find_cached_repository():
    """Verify local cache discovery finds existing clones in benchmarks_repos/."""
    fastapi_p = find_cached_repository("fastapi")
    assert fastapi_p is not None
    assert (fastapi_p / ".git").exists()

    gin_p = find_cached_repository("gin")
    assert gin_p is not None
    assert (gin_p / ".git").exists()

    nonexistent = find_cached_repository("nonexistent_repo_xyz_123")
    assert nonexistent is None


def test_ensure_repository_cloned_offline(tmp_path: Path):
    """Verify offline mode skips network clone and returns None for missing repos."""
    info = {"name": "ghost_repo", "url": "https://github.com/example/ghost.git", "language": "python"}
    res = ensure_repository_cloned(info, cache_dir=tmp_path / "cache", offline=True)
    assert res is None


def test_query_osv_advisories_offline():
    """Verify query_osv_advisories handles network failures gracefully without raising."""
    # When pointed at invalid address or offline
    res = query_osv_advisories("nonexistent-pkg-999", "PyPI", timeout=0.5)
    assert isinstance(res, list)


def test_query_osv_advisories_mock():
    """Verify query_osv_advisories correctly unpacks vulnerabilities."""
    mock_payload = {
        "vulns": [
            {
                "id": "GHSA-1234-5678-9012",
                "summary": "Race condition in session handler",
                "details": "A concurrency hazard allows race conditions.",
                "references": [
                    {"type": "FIX", "url": "https://github.com/pallets/werkzeug/commit/abcdef1234567890"}
                ],
            }
        ]
    }
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        vulns = query_osv_advisories("werkzeug", "PyPI")
        assert len(vulns) == 1
        assert vulns[0]["id"] == "GHSA-1234-5678-9012"


def test_extract_revert_pairs(git_repo_with_reverts: Path):
    """Verify GitRevertMiner extracts reverted commit pairs and resolves buggy diff."""
    miner = GitRevertMiner(languages=["python"], filter_symbolic_gate=True)
    records = miner.extract_revert_pairs_from_repo(git_repo_with_reverts, max_samples=10)

    assert len(records) >= 2
    negatives = [r for r in records if r.label == 0]
    positives = [r for r in records if r.label == 1]

    assert len(negatives) >= 1
    assert len(positives) >= 1

    # Check negative record
    neg = negatives[0]
    assert neg.category == "real_revert"
    assert neg.symbolic_gate_passed is True
    assert neg.source_type == "real_revert"
    assert neg.risk_score >= 0.75
    assert "STATUS: APPROVED" in neg.input_dsl
    assert neg.taxonomy_labels["SilentLogicDrift"] >= 0.70

    # Check positive record
    pos = positives[0]
    assert pos.label == 1
    assert pos.risk_score <= 0.25
    assert pos.symbolic_gate_passed is True
    assert "STATUS: APPROVED" in pos.input_dsl


def test_extract_cve_ghsa_pairs(git_repo_with_reverts: Path):
    """Verify extraction of CVE / GHSA security commits and mapping to SecuritySurface and ConcurrencyHazard."""
    miner = GitRevertMiner(languages=["python"], filter_symbolic_gate=True)
    repo_info = {"name": "test_git_repo", "language": "python", "pkg_name": "test_pkg", "ecosystem": "PyPI"}
    records = miner.extract_cve_ghsa_pairs_from_repo(
        git_repo_with_reverts,
        repo_info=repo_info,
        max_samples=10,
        query_osv=False,
        offline=True,
    )

    assert len(records) >= 2
    negatives = [r for r in records if r.label == 0]
    positives = [r for r in records if r.label == 1]

    assert len(negatives) >= 1
    assert len(positives) >= 1

    neg = negatives[0]
    assert neg.category in ("concurrency_hazard", "security_surface")
    assert neg.symbolic_gate_passed is True
    assert neg.source_type == "cve_ghsa"
    # ConcurrencyHazard and SecuritySurface must be activated
    assert neg.taxonomy_labels["ConcurrencyHazard"] >= 0.85
    assert neg.taxonomy_labels["SecuritySurface"] >= 0.80

    pos = positives[0]
    assert pos.label == 1
    assert pos.risk_score <= 0.25
    assert pos.symbolic_gate_passed is True


def test_balance_revert_dataset():
    """Verify balance_revert_dataset produces exact 50% PASS / 50% REJECT split."""
    records = [
        DatasetRecord(input_dsl="dsl_pass_1", label=1, risk_score=0.05, category="clean_commit", language="python"),
        DatasetRecord(input_dsl="dsl_pass_2", label=1, risk_score=0.06, category="clean_commit", language="python"),
        DatasetRecord(input_dsl="dsl_neg_1", label=0, risk_score=0.91, category="real_revert", language="python"),
        DatasetRecord(input_dsl="dsl_pass_ts", label=1, risk_score=0.05, category="clean_commit", language="typescript"),
        DatasetRecord(input_dsl="dsl_neg_ts", label=0, risk_score=0.92, category="security_surface", language="typescript"),
    ]

    balanced = balance_revert_dataset(records, target_count=4)
    pos = sum(1 for r in balanced if r.label == 1)
    neg = sum(1 for r in balanced if r.label == 0)
    assert pos == neg
    assert len(balanced) == 4


def test_inspect_file_on_mined_dataset(git_repo_with_reverts: Path, tmp_path: Path):
    """Verify that mined records pass inspect_dataset.py inspection without violations."""
    out_file = tmp_path / "mined_dataset.jsonl"
    miner = GitRevertMiner(languages=["python"], filter_symbolic_gate=True)
    records = miner.extract_revert_pairs_from_repo(git_repo_with_reverts, max_samples=10)
    records.extend(
        miner.extract_cve_ghsa_pairs_from_repo(
            git_repo_with_reverts,
            repo_info={"name": "test_git_repo", "language": "python"},
            max_samples=10,
            query_osv=False,
            offline=True,
        )
    )

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    summary = inspect_file(out_file, max_token_limit=400, require_symbolic_gate=True)
    assert summary["passed"] is True, f"Inspect file reported violations: {summary['violations']}"
    assert summary["symbolic_gate"]["failed"] == 0
    assert summary["total_records"] == len(records)


def test_extract_cve_body_match(tmp_path: Path):
    """Verify that CVE identifiers in commit bodies (not just subject) are discovered."""
    repo_dir = tmp_path / "cve_body_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True)

    f = repo_dir / "service.py"
    f.write_text("def sanitize(x: str) -> str:\n    return x\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True)

    f.write_text("def sanitize(x: str) -> str:\n    return x.strip()\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    # Generic subject, but CVE identifier in commit body
    body_msg = "fix: update input filter\n\nResolves CVE-2024-9999 security vulnerability in parameter parser"
    subprocess.run(["git", "commit", "-m", body_msg], cwd=str(repo_dir), check=True)

    miner = GitRevertMiner(languages=["python"], filter_symbolic_gate=True)
    records = miner.extract_cve_ghsa_pairs_from_repo(
        repo_dir,
        repo_info={"name": "cve_body_repo", "language": "python"},
        max_samples=5,
        query_osv=False,
        offline=True,
    )
    assert len(records) >= 1
    neg = next(r for r in records if r.label == 0)
    assert neg.category == "security_surface"
    assert neg.taxonomy_labels["SecuritySurface"] >= 0.85


def test_cli_depth_parameter(monkeypatch, tmp_path: Path):
    """Verify that --depth CLI parameter is accepted and forwarded."""
    from mine_git_reverts import main
    out_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_git_reverts.py",
            "--cache-dir",
            str(tmp_path),
            "--output",
            str(out_file),
            "--target-samples",
            "2",
            "--depth",
            "50",
            "--offline",
        ],
    )
    # Should run pipeline without raising argument errors
    main()
    assert out_file.exists()

