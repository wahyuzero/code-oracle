"""
Comprehensive test suite for the Static Performance Anti-Patterns & Resource Leak Detector.
Validates PERF001, PERF002, PERF003, and PERF004 across Python, TypeScript, Go, and Rust,
as well as CLI options, inline suppressions, and FastMCP server endpoints.
"""

import json
from pathlib import Path
import subprocess
import pytest

from code_oracle.perf_lint import (
    PerfDiagnostic,
    PerfLintEngine,
    PerfLintVisitor,
    PerfReport,
    PerfRule,
    Severity,
    lint_performance,
    lint_performance_patterns,
)
from code_oracle.server import lint_performance_patterns as server_lint_performance_patterns


# ============================================================================
# PERF001: Nested Loops Complexity Tests
# ============================================================================

def test_perf001_python_loop_depths():
    """Verify O(N^2) warning at depth 2 and O(N^3) error at depth 3 in Python."""
    code = """
def test_loops():
    for i in range(10):  # depth 1: safe
        for j in range(10):  # depth 2: warn
            for k in range(10):  # depth 3: error
                pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 2

    d2 = next(d for d in diags if d.lineno == 4)
    assert d2.rule_id == PerfRule.PERF001.value
    assert d2.severity == Severity.WARN
    assert "O(N^2)" in d2.message
    assert "depth 2" in d2.message

    d3 = next(d for d in diags if d.lineno == 5)
    assert d3.rule_id == PerfRule.PERF001.value
    assert d3.severity == Severity.ERROR
    assert "O(N^3)" in d3.message
    assert "depth 3" in d3.message


def test_perf001_typescript_loop_depths():
    """Verify loop complexity escalation across for, while, and do-while in TypeScript."""
    code = """
function processGrid(matrix: number[][][]) {
    for (let i = 0; i < matrix.length; i++) {
        while (matrix[i].length > 0) { // depth 2: warn
            for (const val of matrix[i][0]) { // depth 3: error
                console.log(val);
            }
        }
    }
}
"""
    v = PerfLintVisitor(code, "test.ts", "typescript")
    diags = v.run()
    assert len(diags) == 2

    assert diags[0].severity == Severity.WARN
    assert "depth 2" in diags[0].message
    assert diags[1].severity == Severity.ERROR
    assert "depth 3" in diags[1].message


def test_perf001_go_loop_depths():
    """Verify loop complexity detection in Go."""
    code = """
package main

func matrixMultiply() {
    for i := 0; i < 10; i++ {
        for j := 0; j < 10; j++ { // depth 2: warn
            for k := 0; k < 10; k++ { // depth 3: error
                _ = i * j * k
            }
        }
    }
}
"""
    v = PerfLintVisitor(code, "main.go", "go")
    diags = v.run()
    assert len(diags) == 2
    assert diags[0].severity == Severity.WARN
    assert diags[1].severity == Severity.ERROR


def test_perf001_rust_loop_depths():
    """Verify loop complexity detection in Rust for for, while, and loop."""
    code = """
fn compute() {
    for i in 0..10 {
        while true { // depth 2: warn
            loop { // depth 3: error
                break;
            }
            break;
        }
    }
}
"""
    v = PerfLintVisitor(code, "main.rs", "rust")
    diags = v.run()
    assert len(diags) == 2
    assert diags[0].severity == Severity.WARN
    assert diags[1].severity == Severity.ERROR


def test_perf001_sibling_loops_safe():
    """Verify sequential sibling loops do not escalate depth."""
    code = """
def process():
    for x in range(10):
        pass
    for y in range(10):
        pass
    for z in range(10):
        pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_perf001_inner_function_resets_depth():
    """Verify nested helper function definitions reset loop depth."""
    code = """
def outer():
    for x in range(10):
        def inner():
            for y in range(10): # depth 1 inside inner
                pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_perf001_inline_suppression_same_line():
    """Verify inline suppression comment on same line suppresses diagnostic."""
    code = """
def matrix():
    for i in range(10):
        for j in range(10): # code-oracle: ignore-perf
            pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_perf001_inline_suppression_preceding_line():
    """Verify inline suppression comment on preceding line suppresses diagnostic."""
    code = """
def matrix():
    for i in range(10):
        # code-oracle: ignore-perf
        for j in range(10):
            pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_perf001_root_loop_suppression_entire_nest():
    """Verify suppression on root loop suppresses entire nested loop structure."""
    code = """
# code-oracle: ignore-perf
for i in range(3):
    for j in range(3):
        for k in range(3):
            pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_perf001_rule_specific_suppression():
    """Verify rule-specific suppression ignore-perf[PERF001] works, while other rules do not suppress."""
    code_match = """
for i in range(10):
    for j in range(10): # code-oracle: ignore-perf[PERF001]
        pass
"""
    v1 = PerfLintVisitor(code_match, "test.py", "python")
    assert len(v1.run()) == 0

    code_mismatch = """
for i in range(10):
    for j in range(10): # code-oracle: ignore-perf[PERF002]
        pass
"""
    v2 = PerfLintVisitor(code_mismatch, "test.py", "python")
    assert len(v2.run()) == 1


def test_perf001_max_depth_threshold():
    """Verify configurable max_depth filters lower depth loops."""
    code = """
for i in range(10):
    for j in range(10):  # depth 2
        for k in range(10):  # depth 3
            pass
"""
    # max_depth=3 should ignore depth 2 and only report depth 3
    v = PerfLintVisitor(code, "test.py", "python", max_depth=3)
    diags = v.run()
    assert len(diags) == 1
    assert diags[0].severity == Severity.ERROR
    assert "depth 3" in diags[0].message


# ============================================================================
# PERF002: N+1 I/O & Database in Loops Tests
# ============================================================================

def test_perf002_python_db_and_http_in_loop():
    """Verify N+1 query warning for database and HTTP calls inside loops in Python."""
    code = """
def fetch_users(user_ids):
    for uid in user_ids:
        user = db.query(uid)
        record = cursor.execute("SELECT * FROM items WHERE id = ?", uid)
        item = User.find(uid)
        detail = repo.find_one(uid)
        res = requests.get(f"https://api.example.com/user/{uid}")
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 5
    for d in diags:
        assert d.severity == Severity.WARN
        assert "Possible N+1 query" in d.message


def test_perf002_python_calls_outside_loop_safe():
    """Verify database calls outside loops do not trigger PERF002."""
    code = """
def fetch_bulk():
    users = db.query("SELECT * FROM users")
    res = requests.get("https://api.example.com/all")
    for u in users:
        print(u)
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 0


def test_perf002_typescript_n_plus_one():
    """Verify N+1 query warning in TypeScript loop bodies."""
    code = """
async function syncUsers(ids: string[]) {
    for (const id of ids) {
        const u = await db.query(id);
        const item = await repo.find_one({ id });
        const res = await fetch(`https://api.com/${id}`);
    }
}
"""
    v = PerfLintVisitor(code, "test.ts", "typescript")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 3


def test_perf002_go_n_plus_one():
    """Verify N+1 database queries and HTTP calls in Go loops."""
    code = """
package main

func processItems(ids []int) {
    for _, id := range ids {
        db.Query("SELECT * FROM items WHERE id = ?", id)
        db.Exec("UPDATE items SET checked = 1 WHERE id = ?", id)
        http.Get("http://example.com")
    }
}
"""
    v = PerfLintVisitor(code, "main.go", "go")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 3


def test_perf002_rust_n_plus_one():
    """Verify N+1 database and network calls in Rust loops."""
    code = """
fn sync_data(ids: Vec<i32>) {
    for id in ids {
        db.query("SELECT * FROM users WHERE id = $1", &[&id]);
        reqwest::get(format!("http://example.com/{}", id));
    }
}
"""
    v = PerfLintVisitor(code, "main.rs", "rust")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 2


def test_perf002_suppression():
    """Verify PERF002 can be suppressed inline."""
    code = """
for id in ids:
    db.query(id) # code-oracle: ignore-perf
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 0


# ============================================================================
# PERF003: Resource Leak / Unclosed Descriptors Tests
# ============================================================================

def test_perf003_python_unclosed_descriptor():
    """Verify unclosed file and connection handles without scoped 'with' in Python."""
    code = """
def read_data():
    f = open("data.txt")  # leak
    s = socket.socket()  # leak
    conn = sqlite3.connect("db.sqlite")  # leak

def safe_read():
    with open("data.txt") as f:
        pass
    with socket.socket() as s:
        pass
    with sqlite3.connect("db.sqlite") as conn:
        pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 3
    for d in diags:
        assert d.severity == Severity.ERROR
        assert "without scoped 'with'" in d.message


def test_perf003_go_unclosed_descriptor():
    """Verify unclosed file or connection in Go without defer ...Close()."""
    code = """
package main
import "os"

func leaky() {
    f, _ := os.Open("file.txt")
    _ = f
}

func clean() {
    f, _ := os.Open("file.txt")
    defer f.Close()
}
"""
    v = PerfLintVisitor(code, "main.go", "go")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 6
    assert "defer ...Close()" in diags[0].message


def test_perf003_typescript_unclosed_descriptor():
    """Verify unclosed descriptor in TypeScript without try/finally."""
    code = """
import * as fs from "fs";

function leaky() {
    const fd = fs.openSync("data.txt", "r");
    fs.closeSync(fd);
}

function clean() {
    let fd: number;
    try {
        fd = fs.openSync("data.txt", "r");
    } finally {
        if (fd) fs.closeSync(fd);
    }
}
"""
    v = PerfLintVisitor(code, "test.ts", "typescript")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 5
    assert "try/finally" in diags[0].message


def test_perf003_rust_resource_leak():
    """Verify Box::leak and mem::forget in Rust."""
    code = """
fn leak_memory() {
    let b = Box::new(100);
    Box::leak(b);

    let f = File::open("a.txt");
    std::mem::forget(f);
}
"""
    v = PerfLintVisitor(code, "main.rs", "rust")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 2
    for d in diags:
        assert d.severity == Severity.ERROR
        assert "leak" in d.message


# ============================================================================
# PERF004: Blocking Synchronous Calls in Async Context Tests
# ============================================================================

def test_perf004_python_blocking_calls_in_async():
    """Verify blocking calls inside async def in Python."""
    code = """
import asyncio
import time
import requests

async def worker():
    time.sleep(2)  # error
    requests.get("https://api.com")  # error
    open("file.txt")  # error
    await asyncio.sleep(2)  # safe

def sync_worker():
    time.sleep(2)  # safe in sync function
    requests.get("https://api.com")  # safe in sync function
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF004.value]
    assert len(diags) == 3
    for d in diags:
        assert d.severity == Severity.ERROR
        assert "inside async function 'worker'" in d.message


def test_perf004_typescript_blocking_calls_in_async():
    """Verify blocking fs.*Sync calls inside async functions in TypeScript."""
    code = """
import * as fs from "fs";

async function loadConfig() {
    const raw = fs.readFileSync("config.json", "utf-8"); // error
}

const arrowLoad = async () => {
    fs.openSync("file.txt", "r"); // error
};

class Service {
    async run() {
        fs.existsSync("path"); // error
    }
}

function syncOk() {
    fs.readFileSync("config.json", "utf-8"); // safe
}
"""
    v = PerfLintVisitor(code, "test.ts", "typescript")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF004.value]
    assert len(diags) == 3
    for d in diags:
        assert d.severity == Severity.ERROR
        assert "inside async function" in d.message


def test_perf004_rust_blocking_calls_in_async():
    """Verify thread::sleep and blocking fs inside async fn in Rust."""
    code = """
async fn handle() {
    std::thread::sleep(std::time::Duration::from_millis(500)); // error
    std::fs::read("file.txt"); // error
}

fn sync_handle() {
    std::thread::sleep(std::time::Duration::from_millis(500)); // safe
}
"""
    v = PerfLintVisitor(code, "main.rs", "rust")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF004.value]
    assert len(diags) == 2
    for d in diags:
        assert d.severity == Severity.ERROR
        assert "inside async function 'handle'" in d.message


# ============================================================================
# Engine & CLI Integration Tests
# ============================================================================

def test_engine_workspace_scan_and_filtering(tmp_path):
    """Verify workspace linting with file discovery and severity filtering."""
    py_file = tmp_path / "app.py"
    py_file.write_text(
        """
for i in range(10):
    for j in range(10):
        pass
f = open("leak.txt")
""",
        encoding="utf-8",
    )

    engine = PerfLintEngine(workspace_root=tmp_path)

    # Default severity warn: includes both warn and error
    report_warn = engine.lint_workspace()
    assert report_warn.total_diagnostics_count == 2
    assert report_warn.warnings_count == 1
    assert report_warn.errors_count == 1

    # Filtered severity error: includes only error
    report_err = engine.lint_workspace(severity="error")
    assert report_err.total_diagnostics_count == 1
    assert report_err.warnings_count == 0
    assert report_err.errors_count == 1


def test_cli_perf_lint_json_output(tmp_path):
    """Verify CLI perf-lint command with --json flag."""
    test_file = tmp_path / "service.py"
    test_file.write_text(
        """
async def task():
    time.sleep(1)
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1  # failed on error
    data = json.loads(result.stdout)
    assert data["total_diagnostics_count"] >= 1
    assert data["errors_count"] >= 1
    assert data["diagnostics"][0]["rule_id"] == "PERF004"


def test_cli_perf_lint_fail_on_flags(tmp_path):
    """Verify --fail-on {warn, error, none} exit code semantics."""
    test_file = tmp_path / "nested.py"
    # Only produces a warning (PERF001 depth 2)
    test_file.write_text(
        """
for i in range(10):
    for j in range(10):
        pass
""",
        encoding="utf-8",
    )

    # --fail-on error should exit 0 because only warning exists
    r1 = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--fail-on", "error"],
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0

    # --fail-on warn should exit 1 because warning exists
    r2 = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--fail-on", "warn"],
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 1

    # --fail-on none should exit 0
    r3 = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--fail-on", "none"],
        capture_output=True,
        text=True,
    )
    assert r3.returncode == 0


def test_cli_perf_lint_format_table_and_text(tmp_path):
    """Verify format table and text output strings."""
    test_file = tmp_path / "leak.py"
    test_file.write_text("f = open('data.txt')\n", encoding="utf-8")

    # Table format
    r_table = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--format", "table"],
        capture_output=True,
        text=True,
    )
    assert "PERF003" in r_table.stdout
    assert "ERROR" in r_table.stdout

    # Text format
    r_text = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--format", "text"],
        capture_output=True,
        text=True,
    )
    assert "PERF003" in r_text.stdout
    assert "Resource 'open' opened without scoped 'with'" in r_text.stdout


# ============================================================================
# FastMCP Server Endpoint Tests
# ============================================================================

def test_fastmcp_server_lint_patterns_on_disk(tmp_path):
    """Verify FastMCP endpoint linting existing file on disk."""
    py_file = tmp_path / "handler.py"
    py_file.write_text(
        """
async def process():
    time.sleep(1)
""",
        encoding="utf-8",
    )

    res = server_lint_performance_patterns(
        file_path=str(py_file),
        workspace_dir=str(tmp_path),
    )
    assert res["total_diagnostics_count"] == 1
    assert res["diagnostics"][0]["rule_id"] == "PERF004"


def test_fastmcp_server_lint_patterns_with_patch(tmp_path):
    """Verify FastMCP endpoint evaluating in-memory patch content."""
    py_file = tmp_path / "algo.py"
    py_file.write_text("def solve():\n    pass\n", encoding="utf-8")

    patch = """def solve():
    for x in range(10):
        for y in range(10):
            pass
"""
    res = server_lint_performance_patterns(
        file_path=str(py_file),
        patch_content=patch,
        workspace_dir=str(tmp_path),
    )
    assert res["total_diagnostics_count"] == 1
    assert res["diagnostics"][0]["rule_id"] == "PERF001"
    assert res["diagnostics"][0]["severity"] == "warn"


def test_rules_direct_module_imports():
    """Verify direct imports and instantiation from code_oracle.perf_lint.rules."""
    from code_oracle.perf_lint.rules import (
        AsyncBlockingRule,
        NPlusOneRule,
        NestedLoopsRule,
        UnclosedResourceRule,
        check_async_blocking,
        check_n_plus_one,
        check_nested_loop,
        check_unclosed_resource,
    )

    assert NestedLoopsRule.RULE_ID == "PERF001"
    assert NPlusOneRule.RULE_ID == "PERF002"
    assert UnclosedResourceRule.RULE_ID == "PERF003"
    assert AsyncBlockingRule.RULE_ID == "PERF004"

    # Test NPlusOne helper detection
    assert NPlusOneRule.is_io_call("db.query")
    assert NPlusOneRule.is_io_call("requests.get")
    assert not NPlusOneRule.is_io_call("math.sqrt")

    # Test UnclosedResource helper detection
    assert UnclosedResourceRule.is_open_call("open", "python")
    assert UnclosedResourceRule.is_open_call("os.Open", "go")
    assert UnclosedResourceRule.is_open_call("fs.openSync", "typescript")
    assert UnclosedResourceRule.is_open_call("Box::leak", "rust")
    assert not UnclosedResourceRule.is_open_call("print", "python")

    # Test AsyncBlocking helper detection
    assert AsyncBlockingRule.is_blocking_call("time.sleep", "python")
    assert AsyncBlockingRule.is_blocking_call("fs.readFileSync", "typescript")
    assert AsyncBlockingRule.is_blocking_call("std::thread::sleep", "rust")
    assert not AsyncBlockingRule.is_blocking_call("asyncio.sleep", "python")


def test_perf001_closure_resets_loop_depth():
    """Verify that inner function closures reset loop depth calculation."""
    code = """
def outer():
    for i in range(10):  # depth 1 in outer
        def inner():
            for j in range(10):  # depth 1 in inner (should NOT trigger depth 2)
                pass
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_perf_preceding_comment_and_specific_rule_suppression():
    """Verify suppression via comment on preceding line and rule-specific tags."""
    code = """
def test_suppression():
    # code-oracle: ignore-perf(PERF001)
    for i in range(10):
        for j in range(10):
            pass

    # Suppressed with generic tag on previous line
    # code-oracle: ignore-perf
    f = open("unclosed.txt")
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    assert len(diags) == 0


def test_cli_perf_lint_max_loop_depth_flag(tmp_path):
    """Verify CLI --max-loop-depth flag adjusts loop complexity threshold."""
    test_file = tmp_path / "matrix.py"
    test_file.write_text(
        """
for i in range(10):
    for j in range(10):
        pass
""",
        encoding="utf-8",
    )

    # With default depth 2, it should detect depth 2 warning
    r_def = subprocess.run(
        ["code-oracle", "perf-lint", "--workspace", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    data_def = json.loads(r_def.stdout)
    assert data_def["total_diagnostics_count"] == 1

    # With --max-loop-depth 3, depth 2 is permitted
    r_depth3 = subprocess.run(
        [
            "code-oracle",
            "perf-lint",
            "--workspace",
            str(tmp_path),
            "--max-loop-depth",
            "3",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    data_depth3 = json.loads(r_depth3.stdout)
    assert data_depth3["total_diagnostics_count"] == 0


def test_cli_perf_lint_paths_filter(tmp_path):
    """Verify CLI perf-lint targets specific paths when specified."""
    file_a = tmp_path / "file_a.py"
    file_b = tmp_path / "file_b.py"

    file_a.write_text("f = open('leak.txt')\n", encoding="utf-8")
    file_b.write_text("f2 = open('leak2.txt')\n", encoding="utf-8")

    # Target only file_a.py
    r = subprocess.run(
        [
            "code-oracle",
            "perf-lint",
            "--workspace",
            str(tmp_path),
            str(file_a),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(r.stdout)
    assert data["scanned_files_count"] == 1
    assert data["total_diagnostics_count"] == 1
    assert "file_a.py" in data["diagnostics"][0]["file_path"]


def test_fastmcp_server_lint_patterns_with_options(tmp_path):
    """Verify FastMCP server lint endpoint with severity and max_depth options."""
    py_file = tmp_path / "service.py"
    py_file.write_text(
        """
for i in range(5):
    for j in range(5):
        pass
""",
        encoding="utf-8",
    )

    # Calling with severity="error" should ignore warnings
    res_err = server_lint_performance_patterns(
        file_path=str(py_file),
        workspace_dir=str(tmp_path),
        severity="error",
    )
    assert res_err["total_diagnostics_count"] == 0

    # Calling with severity="warn" should include warnings
    res_warn = server_lint_performance_patterns(
        file_path=str(py_file),
        workspace_dir=str(tmp_path),
        severity="warn",
    )
    assert res_warn["total_diagnostics_count"] == 1


# ============================================================================
# Deep Edge-Case Tests (Comprehensions, Precise Scoping, Node Sync, N+1 Post)
# ============================================================================

def test_perf001_comprehension_nested_loop_depths():
    """Verify nested loop complexity detection inside Python list/dict/set comprehensions."""
    code = """
flat = [x for row in matrix for x in row]  # depth 2
cubic = [x for i in I for j in J for k in K]  # depth 2 and depth 3
nested = [[x for x in row] for row in matrix]  # depth 2
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF001.value]
    assert len(diags) == 4
    # flat has depth 2
    assert any("depth 2" in d.message and d.severity == Severity.WARN for d in diags)
    # cubic has depth 3
    assert any("depth 3" in d.message and d.severity == Severity.ERROR for d in diags)


def test_perf002_comprehension_n_plus_one_detection():
    """Verify N+1 database queries and HTTP requests inside comprehensions."""
    code = """
users = [db.query(uid) for uid in user_ids]
responses = [requests.get(url) for url in urls]
safe_dict = [item.get("key") for item in items]
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 2
    assert any("db.query" in d.message for d in diags)
    assert any("requests.get" in d.message for d in diags)
    # item.get("key") MUST NOT be flagged as N+1
    assert not any("item.get" in d.message for d in diags)


def test_perf002_post_and_network_receiver_methods():
    """Verify .post and network client .get inside loops while avoiding dict/set false positives."""
    code_py = """
for item in items:
    api.post("/items", item)
    session.get(f"/items/{item.id}")
    db.update(item)
    # in-memory operations: safe
    target_files.update(item)
    val = record.get("id")
"""
    v_py = PerfLintVisitor(code_py, "test.py", "python")
    diags_py = [d for d in v_py.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags_py) == 3
    assert any("api.post" in d.message for d in diags_py)
    assert any("session.get" in d.message for d in diags_py)
    assert any("db.update" in d.message for d in diags_py)
    assert not any("target_files.update" in d.message for d in diags_py)
    assert not any("record.get" in d.message for d in diags_py)

    code_ts = """
for (const item of items) {
    await api.post('/items', item);
    await httpClient.get('/items/' + item.id);
}
"""
    v_ts = PerfLintVisitor(code_ts, "test.ts", "typescript")
    diags_ts = [d for d in v_ts.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags_ts) == 2


def test_perf003_python_strongconnect_and_bare_identifiers_not_leaks():
    """Verify strongconnect, reconnect, and disconnect are not falsely flagged as unclosed DB connections."""
    code = """
def tarjan():
    strongconnect(neighbor)
    reconnect()
    disconnect()
    # actual leak:
    conn = sqlite3.connect("db.sqlite")
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert "sqlite3.connect" in diags[0].message
    assert not any("strongconnect" in d.message for d in diags)
    assert not any("reconnect" in d.message for d in diags)
    assert not any("disconnect" in d.message for d in diags)


def test_perf003_typescript_sibling_try_finally_and_pipe_scoped():
    """Verify TS streams followed by try/finally or chained with pipe are recognized as safely scoped."""
    code = """
function processFile() {
    const s1 = fs.createReadStream("a.txt");
    try {
        work(s1);
    } finally {
        s1.close();
    }

    // chained pipe
    fs.createReadStream("b.txt").pipe(dest);

    // variable pipe
    const s3 = fs.createReadStream("c.txt");
    s3.pipe(dest);

    // actual unclosed leak
    const s4 = fs.createReadStream("leak.txt");
}
"""
    v = PerfLintVisitor(code, "test.ts", "typescript")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 18  # s4 leak line


def test_perf003_go_multi_variable_defer_tracking():
    """Verify Go resource tracking accurately catches unclosed descriptor when another variable has defer."""
    code = """
package main
import "os"

func handler() {
    f1, _ := os.Open("safe.txt")
    defer f1.Close()

    f2, _ := os.Open("leak.txt") // f2 is not closed!
}
"""
    v = PerfLintVisitor(code, "main.go", "go")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 9


def test_perf003_rust_raw_pointer_leaks():
    """Verify Rust Box::into_raw and CString::into_raw are flagged under PERF003."""
    code = """
fn leak_ptrs() {
    let raw = Box::into_raw(b);
    let raw_str = CString::into_raw(s);
}
"""
    v = PerfLintVisitor(code, "main.rs", "rust")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 2


def test_perf004_async_blocking_expanded():
    """Verify bare sleep in Python and all *Sync methods in TypeScript."""
    code_py = """
from time import sleep
async def worker():
    sleep(1)
    os.popen("ls")
"""
    v_py = PerfLintVisitor(code_py, "test.py", "python")
    diags_py = [d for d in v_py.run() if d.rule_id == PerfRule.PERF004.value]
    assert len(diags_py) == 2

    code_ts = """
import { writeFileSync, existsSync } from 'fs';
import { spawnSync } from 'child_process';

async function syncWorker() {
    writeFileSync('a.txt', 'data');
    existsSync('a.txt');
    spawnSync('ls');
}
"""
    v_ts = PerfLintVisitor(code_ts, "test.ts", "typescript")
    diags_ts = [d for d in v_ts.run() if d.rule_id == PerfRule.PERF004.value]
    assert len(diags_ts) == 3


def test_suppression_comma_separated_and_intermediate_loops():
    """Verify comma-separated rules in suppression tag and intermediate loop suppression."""
    code = """
def test_nested():
    for i in range(10):
        # code-oracle: ignore-perf[PERF001, PERF002]
        for j in range(10):
            for k in range(10):
                db.query(k)
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = v.run()
    # The middle loop suppresses PERF001 for loop k and PERF002 for db.query(k)
    assert len(diags) == 0


# ============================================================================
# Regression Tests for Priorities 1 & 2 (Keyword Masking & Linter Cleanups)
# ============================================================================

def test_perf003_keyword_substring_masking_go_and_ts():
    """Verify single-letter variables f, r, d are not falsely marked closed by defer/finally."""
    go_code = """
package main
import "os"

func testF() {
    f, _ := os.Open("a.txt")
    defer other.Close()
}

func testR() {
    r, _ := os.Open("b.txt")
    defer other.Close()
}

func testD() {
    d, _ := os.Open("c.txt")
    defer other.Close()
}

func testSafe() {
    f, _ := os.Open("safe.txt")
    defer f.Close()
}
"""
    v_go = PerfLintVisitor(go_code, "main.go", "go")
    diags_go = [d for d in v_go.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_go) == 3
    assert [d.lineno for d in diags_go] == [6, 11, 16]

    ts_code = """
import * as fs from "fs";

function testF() {
    const f = fs.openSync("a.txt", "r");
    try {
        work();
    } finally {
        other.destroy();
    }
}

function testSafe() {
    const f = fs.openSync("safe.txt", "r");
    try {
        work();
    } finally {
        fs.closeSync(f);
    }
}
"""
    v_ts = PerfLintVisitor(ts_code, "test.ts", "typescript")
    diags_ts = [d for d in v_ts.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_ts) == 1
    assert diags_ts[0].lineno == 5


def test_perf002_n_plus_one_dict_get_not_flagged():
    """Verify dictionary .get() lookups (c.get, hc.get, config.get) are not flagged as N+1 queries."""
    code = """
def test_loop(items):
    for item in items:
        val1 = c.get("key")
        val2 = hc.get("key")
        val3 = config.get("timeout")
        val4 = params.get("user_id")
        resp = http_client.get("https://api.com")
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 1
    assert diags[0].lineno == 8
    assert "http_client.get" in diags[0].message


def test_perf002_n_plus_one_camel_case_and_go_context_methods():
    """Verify camelCase Prisma/Mongo queries and Go context db methods are flagged inside loops."""
    ts_code = """
async function processUsers(ids: string[]) {
    for (const id of ids) {
        const u1 = await prisma.user.findUnique({ where: { id } });
        const u2 = await prisma.user.findFirst({ where: { id } });
        const posts = await prisma.post.findMany({ where: { authorId: id } });
        const doc = await mongo.collection.findOne({ _id: id });
    }
}
"""
    v_ts = PerfLintVisitor(ts_code, "test.ts", "typescript")
    diags_ts = [d for d in v_ts.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags_ts) == 4
    for d in diags_ts:
        assert d.severity == Severity.WARN

    go_code = """
package main
import "context"

func queryLoop(ctx context.Context, ids []int) {
    for _, id := range ids {
        _ = db.QueryContext(ctx, "SELECT * FROM users WHERE id = ?", id)
        _ = db.ExecContext(ctx, "UPDATE users SET active = 1 WHERE id = ?", id)
        _ = db.QueryRowContext(ctx, "SELECT name FROM users WHERE id = ?", id)
    }
}
"""
    v_go = PerfLintVisitor(go_code, "main.go", "go")
    diags_go = [d for d in v_go.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags_go) == 3
    for d in diags_go:
        assert d.severity == Severity.WARN


def test_perf004_async_blocking_await_sleep_and_httpx_async_client():
    """Verify await sleep and httpx.AsyncClient / httpx.Timeout are not flagged as blocking."""
    code = """
import asyncio
from time import sleep

async def worker():
    await sleep(1)
    await asyncio.sleep(1)
    client = httpx.AsyncClient(timeout=httpx.Timeout(10))

    sleep(1)
    httpx.get("https://example.com")
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF004.value]
    assert len(diags) == 2
    assert [d.lineno for d in diags] == [10, 11]
    assert any("sleep" in d.message for d in diags)
    assert any("httpx.get" in d.message for d in diags)


def test_perf003_factory_return_ownership_transfer():
    """Verify returned opened resources (direct or assigned variable) are exempted as ownership transfer."""
    py_code = """
def f_direct():
    return open("a.txt")

def f_var():
    f = open("b.txt")
    return f

def f_tuple():
    f = open("c.txt")
    return f, "ok"

def f_leak():
    f = open("d.txt")
    return f.read()
"""
    v_py = PerfLintVisitor(py_code, "test.py", "python")
    diags_py = [d for d in v_py.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_py) == 1
    assert diags_py[0].lineno == 14

    go_code = """
package main
import "os"

func f_direct() (*os.File, error) {
    return os.Open("a.txt")
}

func f_var() (*os.File, error) {
    f, err := os.Open("b.txt")
    if err != nil {
        return nil, err
    }
    return f, nil
}

func f_leak() error {
    f, _ := os.Open("c.txt")
    _ = f
    return nil
}
"""
    v_go = PerfLintVisitor(go_code, "main.go", "go")
    diags_go = [d for d in v_go.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_go) == 1
    assert diags_go[0].lineno == 18

    ts_code = """
import * as fs from "fs";

function f_direct() {
    return fs.createReadStream("a.txt");
}

function f_var() {
    const s = fs.createReadStream("b.txt");
    return s;
}

function f_leak() {
    const s = fs.createReadStream("c.txt");
}
"""
    v_ts = PerfLintVisitor(ts_code, "test.ts", "typescript")
    diags_ts = [d for d in v_ts.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_ts) == 1
    assert diags_ts[0].lineno == 14


def test_perf003_python_try_finally_close():
    """Verify Python try ... finally: ...close() properly marks resources as scoped."""
    py_code = """
def test_before_try():
    f = open("data1.txt")
    try:
        data = f.read()
    finally:
        f.close()

def test_inside_try():
    try:
        f = open("data2.txt")
        data = f.read()
    finally:
        f.close()

def test_no_close():
    f = open("data3.txt")
    try:
        data = f.read()
    finally:
        print("cleanup finished")
"""
    v = PerfLintVisitor(py_code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 17


def test_perf003_typescript_stream_pipe_argument():
    """Verify TypeScript stream.pipe(writable) recognizes writable argument as scoped."""
    ts_code = """
import * as fs from "fs";

function test_pipe_arg(reader: any) {
    const writable = fs.createWriteStream("out1.txt");
    reader.pipe(writable);
}

function test_pipe_inline(reader: any) {
    reader.pipe(fs.createWriteStream("out2.txt"));
}

function test_unpiped() {
    const leaked = fs.createWriteStream("out3.txt");
}
"""
    v = PerfLintVisitor(ts_code, "test.ts", "typescript")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 14


def test_perf002_n_plus_one_dict_token_boundaries_no_false_positives():
    """Verify dictionary lookups like feedback, interest, report do not trigger false N+1 positives."""
    code = """
def process(items):
    for item in items:
        # In-memory dictionary and set methods that contain substrings like db, rest, repo, table
        fb = feedback.get(item.id)
        rate = interest.get(item.id)
        r = restaurant.get(item.id)
        bag = handbag.get(item.id)
        feedback.update(item)
        report.update(item)
        vegetable.update(item)

        # Real DB and network operations
        db.update(item)
        user_db.update(item)
        userRepo.update(item)
        httpClient.get("https://api.com")
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF002.value]
    assert len(diags) == 4
    callees = [d.message for d in diags]
    assert any("db.update" in m for m in callees)
    assert any("user_db.update" in m for m in callees)
    assert any("userRepo.update" in m for m in callees)
    assert any("httpClient.get" in m for m in callees)
    assert not any("feedback" in m for m in callees)
    assert not any("interest" in m for m in callees)
    assert not any("restaurant" in m for m in callees)
    assert not any("handbag" in m for m in callees)
    assert not any("report" in m for m in callees)
    assert not any("vegetable" in m for m in callees)


def test_perf003_resource_assigned_method_call_and_property_returns():
    """Verify returning properties or results of methods on unclosed resources flags a leak."""
    code = """
def test_leak_read():
    content = open("a.txt").read()
    return content

def test_leak_prop():
    f = open("b.txt")
    return f.name

def test_leak_direct_prop():
    return open("c.txt").name

def test_safe_tuple_unpack():
    x, f = "meta", open("d.txt")
    return f
"""
    v = PerfLintVisitor(code, "test.py", "python")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 3
    # Lines for test_leak_read, test_leak_prop, test_leak_direct_prop
    assert [d.lineno for d in diags] == [3, 7, 11]


def test_perf003_unassigned_resource_not_masked_by_unrelated_finally():
    """Verify unassigned resource calls are flagged even if an unrelated finally block is present."""
    py_code = """
def test():
    open("a.txt")
    try:
        pass
    finally:
        socket.close()
"""
    v_py = PerfLintVisitor(py_code, "test.py", "python")
    diags_py = [d for d in v_py.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_py) == 1
    assert diags_py[0].lineno == 3

    ts_code = """
import * as fs from "fs";
function test() {
    try {
        fs.openSync("a.txt", "r");
    } finally {
        console.log("cleanup done");
    }
}
"""
    v_ts = PerfLintVisitor(ts_code, "test.ts", "typescript")
    diags_ts = [d for d in v_ts.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags_ts) == 1
    assert diags_ts[0].lineno == 5


def test_perf003_typescript_res_pipe_no_substring_masking():
    """Verify single-letter stream variable 's' is not falsely masked by res.pipe."""
    ts_code = """
import * as fs from "fs";

function testLeakyS(res: any, something: any) {
    const s = fs.createReadStream("a.txt");
    res.pipe(something);
}

function testSafeS(something: any) {
    const s = fs.createReadStream("safe.txt");
    s.pipe(something);
}
"""
    v = PerfLintVisitor(ts_code, "test.ts", "typescript")
    diags = [d for d in v.run() if d.rule_id == PerfRule.PERF003.value]
    assert len(diags) == 1
    assert diags[0].lineno == 5



