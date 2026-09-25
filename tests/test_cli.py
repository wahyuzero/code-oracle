"""
Tests for Code Oracle CLI commands.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def cli_workspace(tmp_path):
    (tmp_path / "calc.py").write_text(
        """def add(a: int) -> int:
    return a + 1

def compute(x: int):
    return add(x)
""",
        encoding="utf-8",
    )
    return tmp_path


def test_cli_version():
    res = subprocess.run(["code-oracle", "--version"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "code-oracle 0.1.0" in res.stdout


def test_cli_index_and_clean(cli_workspace):
    # Index
    res_index = subprocess.run(
        ["code-oracle", "index", str(cli_workspace)],
        capture_output=True,
        text=True,
    )
    assert res_index.returncode == 0
    assert "Workspace indexed successfully" in res_index.stdout
    assert (cli_workspace / ".code_oracle" / "index.json").exists()

    # Clean
    res_clean = subprocess.run(
        ["code-oracle", "clean", str(cli_workspace)],
        capture_output=True,
        text=True,
    )
    assert res_clean.returncode == 0
    assert "cleaned successfully" in res_clean.stdout
    assert not (cli_workspace / ".code_oracle").exists()


def test_cli_verify_approved(cli_workspace):
    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
"""
    res = subprocess.run(
        [
            "code-oracle",
            "verify",
            "calc.py",
            "--patch",
            patch,
            "-w",
            str(cli_workspace),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "VERDICT: APPROVED" in res.stdout


def test_cli_verify_rejected(cli_workspace):
    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int) -> int:
"""
    res = subprocess.run(
        [
            "code-oracle",
            "verify",
            "calc.py",
            "--patch",
            patch,
            "-w",
            str(cli_workspace),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "VERDICT: REJECTED" in res.stdout
    assert "ARITY_MISMATCH" in res.stdout


def test_cli_verify_json_output(cli_workspace):
    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
"""
    res = subprocess.run(
        [
            "code-oracle",
            "verify",
            "calc.py",
            "--patch",
            patch,
            "-w",
            str(cli_workspace),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"
    assert "confidence" in data
    assert "latency_ms" in data
    assert "linearized_subgraph" in data


def test_cli_slice_command(cli_workspace):
    res = subprocess.run(
        [
            "code-oracle",
            "slice",
            "calc.py",
            "--symbol",
            "add",
            "-w",
            str(cli_workspace),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[DIFF_TARGET]" in res.stdout
    assert "add" in res.stdout


def test_cli_verify_no_neural_flag(cli_workspace):
    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
"""
    res = subprocess.run(
        [
            "code-oracle",
            "verify",
            "calc.py",
            "--patch",
            patch,
            "-w",
            str(cli_workspace),
            "--no-neural",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"
    assert data["risk_score"] == 0.05


def test_cli_verify_neural_flag(cli_workspace):
    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
"""
    res = subprocess.run(
        [
            "code-oracle",
            "verify",
            "calc.py",
            "--patch",
            patch,
            "-w",
            str(cli_workspace),
            "--neural",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0

