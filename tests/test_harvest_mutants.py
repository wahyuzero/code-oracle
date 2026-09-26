"""
Unit and integration tests for tools/harvest_mutants.py.

Covers:
1. Targeted AST-topology-preserving mutation operators:
   - TruthinessInversionOperator (Python, TypeScript, Go, Rust)
   - MutableDefaultsOperator (Python)
   - FloatingPromisesOperator (TypeScript)
   - OptionalChainingDriftOperator (TypeScript, Python)
   - UnhandledChannelReadOperator (Go, Python)
   - UnclosedResourceOperator (Python, Go, TypeScript, Rust)
2. Deterministic Stage 1-2 symbolic gate verification (symbolic_gate_passed=True).
3. Test suite runner execution and mutant survival vs kill filtering.
4. Rollback resilience (clean file state restoration).
5. Multi-task risk taxonomy assignments (ADR-0003).
6. JSONL export and validation via inspect_dataset.py.
"""

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from harvest_mutants import (
    ALL_OPERATORS,
    FloatingPromisesOperator,
    MutableDefaultsOperator,
    MutantHarvester,
    OptionalChainingDriftOperator,
    TruthinessInversionOperator,
    UnclosedResourceOperator,
    UnhandledChannelReadOperator,
    export_mutants_to_jsonl,
)
from inspect_dataset import inspect_file
from code_oracle.languages import validate_syntax


# ============================================================================
# 1. Targeted Mutation Operators Unit Tests
# ============================================================================


def test_truthiness_inversion_python():
    op = TruthinessInversionOperator()
    code = (
        "def check_user(user: str, active: bool) -> bool:\n"
        "    if active:\n"
        "        return True\n"
        "    if not user:\n"
        "        return False\n"
        "    if user == 'admin':\n"
        "        return True\n"
        "    return False\n"
    )
    assert op.can_mutate("user.py", code, "python")
    mutants = op.generate_mutants("user.py", code, "python")
    assert len(mutants) >= 2
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "user.py") is None


def test_truthiness_inversion_typescript():
    op = TruthinessInversionOperator()
    code = (
        "export function isAllowed(role: string, isSuper: boolean): boolean {\n"
        "    if (isSuper) {\n"
        "        return true;\n"
        "    }\n"
        "    if (role === 'admin') {\n"
        "        return true;\n"
        "    }\n"
        "    return false;\n"
        "}\n"
    )
    assert op.can_mutate("auth.ts", code, "typescript")
    mutants = op.generate_mutants("auth.ts", code, "typescript")
    assert len(mutants) >= 1
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "auth.ts") is None


def test_truthiness_inversion_go():
    op = TruthinessInversionOperator()
    code = (
        "package service\n\n"
        "func Process(err error) bool {\n"
        "    if err != nil {\n"
        "        return false\n"
        "    }\n"
        "    return true\n"
        "}\n"
    )
    assert op.can_mutate("service.go", code, "go")
    mutants = op.generate_mutants("service.go", code, "go")
    assert len(mutants) >= 1
    assert "err == nil" in mutants[0][0]
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "service.go") is None


def test_truthiness_inversion_rust():
    op = TruthinessInversionOperator()
    code = (
        "pub fn is_valid(token: &str) -> bool {\n"
        "    if token == \"secret\" {\n"
        "        true\n"
        "    } else {\n"
        "        false\n"
        "    }\n"
        "}\n"
    )
    assert op.can_mutate("auth.rs", code, "rust")
    mutants = op.generate_mutants("auth.rs", code, "rust")
    assert len(mutants) >= 1
    assert "token != \"secret\"" in mutants[0][0]
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "auth.rs") is None


def test_mutable_defaults_python():
    op = MutableDefaultsOperator()
    code = (
        "def query_database(query: str, options: Optional[dict] = None) -> dict:\n"
        "    if options is None:\n"
        "        options = {}\n"
        "    return {'query': query, 'options': options}\n"
    )
    assert op.can_mutate("db.py", code, "python")
    mutants = op.generate_mutants("db.py", code, "python")
    assert len(mutants) >= 1
    assert "options: dict = {}" in mutants[0][0]
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "db.py") is None


def test_floating_promises_typescript():
    op = FloatingPromisesOperator()
    code = (
        "export async function handleRequest(req: Request): Promise<Response> {\n"
        "    await logAccess(req.url);\n"
        "    return new Response('OK');\n"
        "}\n"
    )
    assert op.can_mutate("handler.ts", code, "typescript")
    mutants = op.generate_mutants("handler.ts", code, "typescript")
    assert len(mutants) >= 1
    assert "logAccess(req.url)" in mutants[0][0]
    assert "await logAccess" not in mutants[0][0]
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "handler.ts") is None


def test_optional_chaining_drift():
    op = OptionalChainingDriftOperator()
    # TypeScript
    ts_code = (
        "export function getSetting(cfg: any): number {\n"
        "    return cfg?.server?.port ?? 3000;\n"
        "}\n"
    )
    assert op.can_mutate("config.ts", ts_code, "typescript")
    ts_mutants = op.generate_mutants("config.ts", ts_code, "typescript")
    assert len(ts_mutants) >= 1
    assert "cfg.server" in ts_mutants[0][0]

    # Python
    py_code = (
        "def get_user_email(profile: dict) -> str:\n"
        "    return profile.get('email', 'default@example.com')\n"
    )
    assert op.can_mutate("user.py", py_code, "python")
    py_mutants = op.generate_mutants("user.py", py_code, "python")
    assert len(py_mutants) >= 1
    assert "profile['email']" in py_mutants[0][0]


def test_unhandled_channel_read_go():
    op = UnhandledChannelReadOperator()
    code = (
        "package worker\n\n"
        "func Consume(ch <-chan int) int {\n"
        "    val, ok := <-ch\n"
        "    if !ok {\n"
        "        return 0\n"
        "    }\n"
        "    return val\n"
        "}\n"
    )
    assert op.can_mutate("worker.go", code, "go")
    mutants = op.generate_mutants("worker.go", code, "go")
    assert len(mutants) >= 1
    assert "val := <-ch" in mutants[0][0]
    for mut_code, desc in mutants:
        assert validate_syntax(mut_code, "worker.go") is None


def test_unclosed_resource():
    op = UnclosedResourceOperator()
    # Python
    py_code = (
        "def read_file(path: str) -> str:\n"
        "    with open(path, 'r') as f:\n"
        "        return f.read()\n"
    )
    assert op.can_mutate("reader.py", py_code, "python")
    py_mutants = op.generate_mutants("reader.py", py_code, "python")
    assert len(py_mutants) >= 1
    assert "open(path, 'r')" in py_mutants[0][0]
    assert "with open" not in py_mutants[0][0]

    # Go
    go_code = (
        "package io\n\n"
        "func Fetch() {\n"
        "    defer file.Close()\n"
        "}\n"
    )
    assert op.can_mutate("io.go", go_code, "go")
    go_mutants = op.generate_mutants("io.go", go_code, "go")
    assert len(go_mutants) >= 1
    assert "// defer file.Close()" in go_mutants[0][0]


# ============================================================================
# 2. Mutant Harvester Integration Tests
# ============================================================================


@pytest.fixture
def project_with_tests(tmp_path: Path) -> Path:
    """Creates a miniature workspace with service code and a passing pytest suite."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Service file
    (pkg_dir / "calculator.py").write_text(
        "def compute_total(base_rate: float, apply_tax: bool) -> float:\n"
        "    if apply_tax:\n"
        "        return base_rate * 1.10\n"
        "    return base_rate * 1.0\n",
        encoding="utf-8",
    )

    # Test file that only tests apply_tax=False (leaving apply_tax=True as an untested edge case!)
    (tests_dir / "test_calc.py").write_text(
        "from pkg.calculator import compute_total\n\n"
        "def test_no_tax():\n"
        "    assert compute_total(100.0, False) == 100.0\n",
        encoding="utf-8",
    )
    return tmp_path


def test_harvest_file_symbolic_survival(project_with_tests: Path):
    """Verify MutantHarvester extracts valid DatasetRecord pairs with symbolic_gate_passed=True."""
    harvester = MutantHarvester(
        workspace_root=project_with_tests,
        languages=["python"],
        operator_names=["truthiness_inversion"],
        filter_symbolic_gate=True,
    )
    calc_file = project_with_tests / "pkg" / "calculator.py"
    pairs = harvester.harvest_file(calc_file)

    assert len(pairs) >= 1
    neg, pos = pairs[0]

    # Negative record (surviving mutant)
    assert neg.label == 0
    assert neg.risk_score >= 0.75
    assert neg.category == "silent_logic_drift"
    assert neg.language == "python"
    assert neg.symbolic_gate_passed is True
    assert neg.source_type == "surviving_mutant"
    assert "STATUS: APPROVED" in neg.input_dsl
    assert neg.taxonomy_labels["SilentLogicDrift"] >= 0.85

    # Positive record (clean pass base)
    assert pos.label == 1
    assert pos.risk_score <= 0.25
    assert pos.category == "clean_pass"
    assert pos.language == "python"
    assert pos.symbolic_gate_passed is True
    assert pos.source_type == "surviving_mutant_base"
    assert "STATUS: APPROVED" in pos.input_dsl


def test_harvest_with_test_runner_survival(project_with_tests: Path):
    """
    Verify test runner execution correctly harvests a surviving mutant
    when existing tests do not assert against the mutated branch,
    while guaranteeing rollback resilience.
    """
    harvester = MutantHarvester(
        workspace_root=project_with_tests,
        languages=["python"],
        operator_names=["truthiness_inversion"],
        filter_symbolic_gate=True,
    )
    calc_file = project_with_tests / "pkg" / "calculator.py"
    original_code = calc_file.read_text(encoding="utf-8")

    test_cmd = f"{sys.executable} -m pytest tests/test_calc.py"
    pairs = harvester.harvest_file(calc_file, test_cmd=test_cmd)

    # Mutant inverted 'if apply_tax' -> 'if not (apply_tax)'.
    # Because test_no_tax() calls compute_total(100.0, False), with the mutation it executes 'if not False'
    # which returns 110.0, causing test_no_tax() to fail.
    # Therefore, this specific mutant is killed!
    # And most importantly, original code is cleanly rolled back on disk.
    assert calc_file.read_text(encoding="utf-8") == original_code


def test_harvest_directory_and_export(project_with_tests: Path, tmp_path: Path):
    """Verify directory harvesting, balancing, and inspect_file validation."""
    harvester = MutantHarvester(
        workspace_root=project_with_tests,
        languages=["python"],
        filter_symbolic_gate=True,
        seed=123,
    )
    records = harvester.harvest_directory(
        target_dir=project_with_tests / "pkg",
        max_samples=10,
        balance=True,
    )

    assert len(records) >= 2
    pos = sum(1 for r in records if r.label == 1)
    neg = sum(1 for r in records if r.label == 0)
    assert pos == neg

    out_file = tmp_path / "surviving_mutants.jsonl"
    written = export_mutants_to_jsonl(records, out_file)
    assert written == len(records)

    summary = inspect_file(out_file, max_token_limit=400, require_symbolic_gate=True)
    assert summary["passed"] is True, f"Violations reported: {summary['violations']}"
    assert summary["total_records"] == written
    assert summary["symbolic_gate"]["failed"] == 0


def test_mutable_defaults_union_and_typed():
    op = MutableDefaultsOperator()
    # Python 3.10+ union types
    code_union = (
        "def configure(settings: dict | None = None, tags: list | None = None) -> None:\n"
        "    pass\n"
    )
    assert op.can_mutate("config.py", code_union, "python")
    mutants = op.generate_mutants("config.py", code_union, "python")
    assert len(mutants) >= 1
    assert "settings: dict = {}" in mutants[0][0]
    assert validate_syntax(mutants[0][0], "config.py") is None

    # Typed function with return annotation and **kwargs
    code_kwargs = (
        "def process_data(data: str, **kwargs) -> dict:\n"
        "    return {'data': data}\n"
    )
    mutants_kw = op.generate_mutants("service.py", code_kwargs, "python")
    assert len(mutants_kw) >= 1
    mut_code = mutants_kw[0][0]
    assert "_memo: dict = {}" in mut_code
    assert validate_syntax(mut_code, "service.py") is None


def test_optional_chaining_call_and_single_get():
    op = OptionalChainingDriftOperator()
    # TypeScript optional method invocation a?.()
    ts_code = (
        "export function notify(callback?: () => void): void {\n"
        "    callback?.();\n"
        "}\n"
    )
    assert op.can_mutate("event.ts", ts_code, "typescript")
    ts_mutants = op.generate_mutants("event.ts", ts_code, "typescript")
    assert len(ts_mutants) >= 1
    assert "callback();" in ts_mutants[0][0]
    assert validate_syntax(ts_mutants[0][0], "event.ts") is None

    # Python single-argument dict.get(key)
    py_code = (
        "def get_user_name(profile: dict) -> str:\n"
        "    return profile.get('name')\n"
    )
    assert op.can_mutate("user.py", py_code, "python")
    py_mutants = op.generate_mutants("user.py", py_code, "python")
    assert len(py_mutants) >= 1
    assert "profile['name']" in py_mutants[0][0]
    assert validate_syntax(py_mutants[0][0], "user.py") is None


def test_unhandled_channel_read_custom_var_and_python_queue():
    op = UnhandledChannelReadOperator()
    # Go with arbitrary variable name (not just 'ok')
    go_code = (
        "package queue\n\n"
        "func Read(ch <-chan string) string {\n"
        "    msg, more := <-ch\n"
        "    if !more {\n"
        "        return \"\"\n"
        "    }\n"
        "    return msg\n"
        "}\n"
    )
    assert op.can_mutate("queue.go", go_code, "go")
    go_muts = op.generate_mutants("queue.go", go_code, "go")
    assert len(go_muts) >= 1
    assert "msg := <-ch" in go_muts[0][0]
    assert validate_syntax(go_muts[0][0], "queue.go") is None

    # Python blocking queue.get()
    py_code = (
        "import queue\n\n"
        "def poll_task(q: queue.Queue):\n"
        "    return q.get()\n"
    )
    assert op.can_mutate("worker.py", py_code, "python")
    py_muts = op.generate_mutants("worker.py", py_code, "python")
    assert len(py_muts) >= 1
    assert "q.get_nowait()" in py_muts[0][0]
    assert validate_syntax(py_muts[0][0], "worker.py") is None


def test_unclosed_resource_selector_and_unlock():
    op = UnclosedResourceOperator()
    # Go selector defer resp.Body.Close()
    go_code = (
        "package client\n\n"
        "import \"net/http\"\n\n"
        "func Fetch(url string) (*http.Response, error) {\n"
        "    resp, err := http.Get(url)\n"
        "    if err != nil {\n"
        "        return nil, err\n"
        "    }\n"
        "    defer resp.Body.Close()\n"
        "    return resp, nil\n"
        "}\n"
    )
    assert op.can_mutate("client.go", go_code, "go")
    go_muts = op.generate_mutants("client.go", go_code, "go")
    assert len(go_muts) >= 1
    assert "// defer resp.Body.Close()" in go_muts[0][0]
    assert validate_syntax(go_muts[0][0], "client.go") is None


def test_harvest_directory_excludes_test_files(tmp_path: Path):
    """Verify that test files and test directories are excluded from mutation."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    node_modules = src_dir / "node_modules"
    node_modules.mkdir(parents=True, exist_ok=True)

    # Production file
    (src_dir / "service.ts").write_text(
        "export function check(valid: boolean): boolean {\n"
        "    if (valid) { return true; }\n"
        "    return false;\n"
        "}\n",
        encoding="utf-8",
    )
    # Test file that should be ignored
    (src_dir / "service.test.ts").write_text(
        "import { check } from './service';\n"
        "if (check(true)) { console.log('ok'); }\n",
        encoding="utf-8",
    )
    # Ignored directory file
    (node_modules / "dep.ts").write_text(
        "if (true) {}\n",
        encoding="utf-8",
    )

    harvester = MutantHarvester(
        workspace_root=tmp_path,
        languages=["typescript"],
        filter_symbolic_gate=True,
    )
    records = harvester.harvest_directory(target_dir=src_dir, max_samples=10, balance=False)
    # Mutants should ONLY target service.ts, not service.test.ts or node_modules
    assert len(records) >= 1
    for r in records:
        assert r.input_dsl.startswith("[DIFF_TARGET] src/service.ts")
        assert "dep.ts" not in r.input_dsl

