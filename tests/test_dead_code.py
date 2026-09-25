"""
Comprehensive test suite for the Dead Code & Orphan Symbol Detection Engine.
Validates multi-language reachability across Python, TypeScript, Go, and Rust,
as well as entrypoint heuristics, CLI options, and FastMCP server endpoints.
"""

import json
from pathlib import Path
import subprocess
import pytest

from code_oracle.dead_code import (
    DeadCodeDetector,
    EntrypointDetector,
    detect_dead_code,
    is_entrypoint,
)
from code_oracle.dead_code.models import DeadCodeReport, DeadSymbol
from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import Symbol
from code_oracle.server import detect_dead_code as server_detect_dead_code


# ============================================================================
# Python Dead Code & Entrypoint Tests
# ============================================================================

def test_python_direct_orphan_and_used_functions(tmp_path):
    """Verify that an uncalled function is detected as an orphan, while called functions are alive."""
    (tmp_path / "main.py").write_text(
        """def main():
    used_helper()

def used_helper():
    return 42

def dead_orphan():
    return 99
""",
        encoding="utf-8",
    )

    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    detector = DeadCodeDetector(workspace_root=tmp_path, indexer=indexer)
    report = detector.detect()

    dead_names = [s.name for s in report.dead_symbols]
    assert "dead_orphan" in dead_names
    assert "used_helper" not in dead_names
    assert "main" not in dead_names

    orphan_sym = next(s for s in report.dead_symbols if s.name == "dead_orphan")
    assert orphan_sym.is_orphan is True
    assert orphan_sym.is_transitive is False
    assert orphan_sym.cluster_id is None
    assert "0 incoming calls" in orphan_sym.reason


def test_python_transitive_dead_cluster(tmp_path):
    """Verify transitive dead cluster detection where dead_a calls dead_b which calls dead_c."""
    (tmp_path / "app.py").write_text(
        """def main():
    print("alive")

def dead_a():
    dead_b()

def dead_b():
    dead_c()

def dead_c():
    return 100
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)

    dead_map = {s.name: s for s in report.dead_symbols}
    assert "dead_a" in dead_map
    assert "dead_b" in dead_map
    assert "dead_c" in dead_map

    # dead_a has in-degree 0 (orphan)
    assert dead_map["dead_a"].is_orphan is True
    assert dead_map["dead_a"].is_transitive is False

    # dead_b and dead_c are transitive dead symbols
    assert dead_map["dead_b"].is_transitive is True
    assert dead_map["dead_c"].is_transitive is True
    assert dead_map["dead_b"].cluster_id == dead_map["dead_a"].id
    assert dead_map["dead_c"].cluster_id == dead_map["dead_a"].id


def test_python_dead_mutual_cycle(tmp_path):
    """Verify mutual recursion dead code where cycle_a and cycle_b only call each other."""
    (tmp_path / "cycle.py").write_text(
        """def main():
    pass

def cycle_a():
    cycle_b()

def cycle_b():
    cycle_a()
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]
    assert "cycle_a" in dead_names
    assert "cycle_b" in dead_names
    for s in report.dead_symbols:
        assert s.is_transitive is True
        assert "Dead cycle" in s.reason or "Transitive dead" in s.reason


def test_python_framework_route_decorators(tmp_path):
    """Verify that route handlers like @app.get and @router.post are treated as entrypoint roots."""
    (tmp_path / "routes.py").write_text(
        """from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter()

@app.get("/items")
def get_items():
    return internal_query()

@router.post("/items")
def create_item():
    return 1

def internal_query():
    return [1, 2, 3]

def dead_unreferenced():
    return 0
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "dead_unreferenced" in dead_names
    assert "get_items" not in dead_names
    assert "create_item" not in dead_names
    assert "internal_query" not in dead_names


def test_python_cli_decorators_and_click(tmp_path):
    """Verify that CLI commands decorated with @click.command are recognized as entrypoints."""
    (tmp_path / "cli.py").write_text(
        """import click

@click.command()
def run_backup():
    execute_backup()

def execute_backup():
    return True

def unused_cli_helper():
    return False
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "run_backup" not in dead_names
    assert "execute_backup" not in dead_names
    assert "unused_cli_helper" in dead_names


def test_python_test_suite_and_fixtures(tmp_path):
    """Verify test files, test functions, and fixtures are recognized as roots."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)

    (tmp_path / "service.py").write_text(
        """def compute_tax(amount):
    return amount * 0.1

def dead_logic():
    return 0
""",
        encoding="utf-8",
    )

    (tests_dir / "test_service.py").write_text(
        """from service import compute_tax

def test_tax_calculation():
    assert compute_tax(100) == 10
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "dead_logic" in dead_names
    assert "compute_tax" not in dead_names
    assert "test_tax_calculation" not in dead_names


def test_python_dunder_and_magic_methods(tmp_path):
    """Verify magic methods (__init__, __str__, __repr__) on an active class are preserved."""
    (tmp_path / "model.py").write_text(
        """class Item:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Item({self.name})"

    def dead_method(self):
        return "unused"

def main():
    item = Item("book")
    print(item)
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "dead_method" in dead_names
    assert "__init__" not in dead_names
    assert "__str__" not in dead_names
    assert "__repr__" not in dead_names
    assert "Item" not in dead_names


def test_python_root_export_init(tmp_path):
    """Verify symbols exported in __init__.py are treated as roots."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()

    (pkg / "core.py").write_text(
        """def exported_api():
    return 1

def internal_unused():
    return 2
""",
        encoding="utf-8",
    )

    (pkg / "__init__.py").write_text(
        """from mypkg.core import exported_api
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "internal_unused" in dead_names
    assert "exported_api" not in dead_names


def test_python_include_unexported_flag(tmp_path):
    """Verify private/unexported symbols (_func) are only included when requested."""
    (tmp_path / "helpers.py").write_text(
        """def public_unused():
    return 1

def _private_unused():
    return 2
""",
        encoding="utf-8",
    )

    # Without include_unexported: only public_unused
    report_default = detect_dead_code(workspace_root=tmp_path, include_unexported=False)
    names_default = [s.name for s in report_default.dead_symbols]
    assert "public_unused" in names_default
    assert "_private_unused" not in names_default

    # With include_unexported: both
    report_all = detect_dead_code(workspace_root=tmp_path, include_unexported=True)
    names_all = [s.name for s in report_all.dead_symbols]
    assert "public_unused" in names_all
    assert "_private_unused" in names_all


# ============================================================================
# TypeScript / JavaScript Dead Code Tests
# ============================================================================

def test_typescript_dead_code_and_spec_tests(tmp_path):
    """Verify TypeScript dead code detection and .spec.ts root heuristic."""
    (tmp_path / "math.ts").write_text(
        """export function add(a: number, b: number): number {
    return a + b;
}

export function unusedTsFunc(): number {
    return 42;
}

function privateUnusedTs(): void {}
""",
        encoding="utf-8",
    )

    (tmp_path / "math.spec.ts").write_text(
        """import { add } from './math';

describe('math', () => {
    it('adds', () => {
        expect(add(1, 2)).toBe(3);
    });
});
""",
        encoding="utf-8",
    )

    # Default: exported unusedTsFunc detected, privateUnusedTs omitted
    report = detect_dead_code(workspace_root=tmp_path, include_unexported=False)
    dead_names = [s.name for s in report.dead_symbols]
    assert "unusedTsFunc" in dead_names
    assert "add" not in dead_names
    assert "privateUnusedTs" not in dead_names

    # With include_unexported: privateUnusedTs is detected
    report_all = detect_dead_code(workspace_root=tmp_path, include_unexported=True)
    dead_names_all = [s.name for s in report_all.dead_symbols]
    assert "unusedTsFunc" in dead_names_all
    assert "privateUnusedTs" in dead_names_all


def test_typescript_index_root_exports(tmp_path):
    """Verify symbols exported in index.ts are treated as entrypoint roots."""
    (tmp_path / "calc.ts").write_text(
        """export function calculateMetrics(): number {
    return 100;
}

export function deadMetrics(): number {
    return 0;
}
""",
        encoding="utf-8",
    )

    (tmp_path / "index.ts").write_text(
        """export { calculateMetrics } from './calc';
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]
    assert "deadMetrics" in dead_names
    assert "calculateMetrics" not in dead_names


# ============================================================================
# Go Dead Code Tests
# ============================================================================

def test_go_dead_code_and_root_package_exports(tmp_path):
    """Verify Go uppercase symbols in root package are roots, while unused symbols in subpackages are dead."""
    (tmp_path / "main.go").write_text(
        """package main

func main() {
    activeRootFunc()
}

func activeRootFunc() {}

// Public Go export in root package
func PublicRootExport() {}
""",
        encoding="utf-8",
    )

    sub_dir = tmp_path / "pkg" / "math"
    sub_dir.mkdir(parents=True)
    (sub_dir / "calc.go").write_text(
        """package math

// Exported but in subpackage and never called
func UnusedSubpackageFunc() int {
    return 10
}

func unexportedPrivateGo() int {
    return 20
}
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path, include_unexported=False)
    dead_names = [s.name for s in report.dead_symbols]

    assert "UnusedSubpackageFunc" in dead_names
    assert "unexportedPrivateGo" not in dead_names
    assert "PublicRootExport" not in dead_names
    assert "activeRootFunc" not in dead_names
    assert "main" not in dead_names

    # Test with include_unexported=True
    report_all = detect_dead_code(workspace_root=tmp_path, include_unexported=True)
    dead_names_all = [s.name for s in report_all.dead_symbols]
    assert "unexportedPrivateGo" in dead_names_all


def test_go_test_files_exemption(tmp_path):
    """Verify that Go test files (*_test.go) and Test* functions act as roots."""
    svc_dir = tmp_path / "pkg" / "service"
    svc_dir.mkdir(parents=True)

    (svc_dir / "service.go").write_text(
        """package service

func ProcessData() bool {
    return true
}

func DeadGoFunc() bool {
    return false
}
""",
        encoding="utf-8",
    )

    (svc_dir / "service_test.go").write_text(
        """package service

import "testing"

func TestProcessData(t *testing.T) {
    if !ProcessData() {
        t.Fail()
    }
}
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "DeadGoFunc" in dead_names
    assert "ProcessData" not in dead_names
    assert "TestProcessData" not in dead_names


# ============================================================================
# Rust Dead Code Tests
# ============================================================================

def test_rust_dead_code_and_tests(tmp_path):
    """Verify Rust dead code detection, root exports in lib.rs, and tests/ root discovery."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "lib.rs").write_text(
        """pub mod utils;

pub fn lib_entrypoint() {
    utils::active_util();
}
""",
        encoding="utf-8",
    )

    (src_dir / "utils.rs").write_text(
        """pub fn active_util() {}

pub fn dead_rust_func() {
    dead_sub_rust();
}

pub fn dead_sub_rust() {}
""",
        encoding="utf-8",
    )

    report = detect_dead_code(workspace_root=tmp_path)
    dead_names = [s.name for s in report.dead_symbols]

    assert "dead_rust_func" in dead_names
    assert "dead_sub_rust" in dead_names
    assert "active_util" not in dead_names
    assert "lib_entrypoint" not in dead_names

    # Check transitive relationship
    dead_map = {s.name: s for s in report.dead_symbols}
    assert dead_map["dead_rust_func"].is_orphan is True
    assert dead_map["dead_sub_rust"].is_transitive is True


# ============================================================================
# Filtering & Formatting Tests
# ============================================================================

def test_min_lines_filtering(tmp_path):
    """Verify --min-lines excludes small dead symbols."""
    (tmp_path / "sample.py").write_text(
        """def small_dead():
    pass

def large_dead():
    a = 1
    b = 2
    c = 3
    d = 4
    return a + b + c + d
""",
        encoding="utf-8",
    )

    # Filter with min_lines=5
    report = detect_dead_code(workspace_root=tmp_path, min_lines=5)
    dead_names = [s.name for s in report.dead_symbols]
    assert "large_dead" in dead_names
    assert "small_dead" not in dead_names


def test_paths_filtering(tmp_path):
    """Verify paths filter targets specific directories or files."""
    dir_a = tmp_path / "pkg_a"
    dir_b = tmp_path / "pkg_b"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "a.py").write_text("def dead_a(): pass\n", encoding="utf-8")
    (dir_b / "b.py").write_text("def dead_b(): pass\n", encoding="utf-8")

    report_a = detect_dead_code(workspace_root=tmp_path, paths=["pkg_a"])
    dead_names = [s.name for s in report_a.dead_symbols]
    assert "dead_a" in dead_names
    assert "dead_b" not in dead_names


def test_report_formats(tmp_path):
    """Verify table and text format generation."""
    (tmp_path / "app.py").write_text("def unused(): pass\n", encoding="utf-8")

    report = detect_dead_code(workspace_root=tmp_path)
    assert report.dead_symbols_count == 1

    tbl = report.format_table()
    assert "Symbol" in tbl
    assert "unused" in tbl
    assert "ORPHAN" in tbl

    txt = report.format_text()
    assert "Dead Code Report" in txt
    assert "unused" in txt

    d = report.to_dict()
    assert d["dead_symbols_count"] == 1
    assert d["latency_ms"] >= 0.0


# ============================================================================
# CLI Command Tests
# ============================================================================

def test_cli_dead_code_subcommand(tmp_path):
    """Verify code-oracle dead-code execution via subprocess."""
    (tmp_path / "service.py").write_text(
        """def main():
    pass

def dead_fn():
    return 10
""",
        encoding="utf-8",
    )

    # Table format
    res = subprocess.run(
        ["code-oracle", "dead-code", "--workspace", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1  # 1 because dead code was found
    assert "dead_fn" in res.stdout
    assert "ORPHAN" in res.stdout

    # JSON format
    res_json = subprocess.run(
        ["code-oracle", "dead-code", "--workspace", str(tmp_path), "--format", "json"],
        capture_output=True,
        text=True,
    )
    assert res_json.returncode == 1
    data = json.loads(res_json.stdout)
    assert data["dead_symbols_count"] == 1
    assert data["dead_symbols"][0]["name"] == "dead_fn"


def test_cli_dead_code_clean_exit_code(tmp_path):
    """Verify exit code 0 when no dead code is present."""
    (tmp_path / "clean.py").write_text(
        """def main():
    helper()

def helper():
    return 1
""",
        encoding="utf-8",
    )

    res = subprocess.run(
        ["code-oracle", "dead-code", "--workspace", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "No dead code detected" in res.stdout


# ============================================================================
# FastMCP Server Endpoint Tests
# ============================================================================

def test_fastmcp_detect_dead_code_endpoint(tmp_path):
    """Verify FastMCP server endpoint detect_dead_code execution."""
    (tmp_path / "worker.py").write_text(
        """def main():
    pass

def dead_worker():
    return -1
""",
        encoding="utf-8",
    )

    result = server_detect_dead_code(workspace_dir=str(tmp_path))
    assert isinstance(result, dict)
    assert result["dead_symbols_count"] == 1
    assert result["dead_symbols"][0]["name"] == "dead_worker"
    assert result["latency_ms"] < 50.0
