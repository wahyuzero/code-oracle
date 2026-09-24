"""
Comprehensive scenario tests validating all agent workflow modes:
1. Safe / approved patch
2. Arity & parameter contract violations (missing required args, extra args, positional-only args, kwargs)
3. Direct & multi-hop circular call/import cycles
4. Deleted symbol breakage (callers and importers, including same-file active callers)
5. CLI stdin / diff file / replacement modes and JSON contracts
"""

import json
import subprocess
from pathlib import Path
import pytest

from code_oracle.engine import TopoSliceEngine


# ==============================================================================
# Scenario 1: Safe / Approved Patch
# ==============================================================================

def test_scenario1_safe_patch_unified_diff(tmp_path):
    (tmp_path / "math_ops.py").write_text(
        """def add(a: int, b: int) -> int:
    return a + b
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Adding safe optional parameter with default
    diff = """@@ -1,2 +1,3 @@
-def add(a: int, b: int) -> int:
+def add(a: int, b: int, multiplier: int = 1) -> int:
+    return (a + b) * multiplier
"""
    report = engine.verify(file_path="math_ops.py", patch_content=diff)
    assert report.status == "APPROVED"
    assert report.confidence >= 0.95
    assert report.invariant_violations == []
    assert report.cycles_detected == []


def test_scenario1_safe_patch_full_replacement(tmp_path):
    (tmp_path / "calc.py").write_text(
        """def compute(x: int) -> int:
    return x * 2
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    replacement = """def compute(x: int) -> int:
    # Optimized implementation
    return x << 1
"""
    report = engine.verify(file_path="calc.py", patch_content=replacement)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_scenario1_safe_patch_new_file(tmp_path):
    engine = TopoSliceEngine(workspace_root=tmp_path)

    new_file_content = """def brand_new_function(val: str) -> str:
    return val.strip().lower()
"""
    report = engine.verify(file_path="utils/helpers.py", patch_content=new_file_content)
    assert report.status == "APPROVED"
    assert "helpers.py::brand_new_function" in report.linearized_subgraph


# ==============================================================================
# Scenario 2: Arity & Parameter Contract Violations
# ==============================================================================

def test_scenario2_caller_missing_required_args(tmp_path):
    # Workspace defines target
    (tmp_path / "math_lib.py").write_text(
        """def divide(a: float, b: float) -> float:
    return a / b
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from math_lib import divide

def run():
    return 0
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch in app.py calls divide with only 1 argument (missing required 'b')
    patch = """from math_lib import divide

def run():
    return divide(10.0)
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'b'" in v for v in report.invariant_violations)


def test_scenario2_caller_extra_positional_args(tmp_path):
    (tmp_path / "math_lib.py").write_text(
        """def multiply(a: int, b: int) -> int:
    return a * b
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from math_lib import multiply

def run():
    return multiply(2, 3)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch provides 4 arguments where multiply accepts at most 2
    patch = """from math_lib import multiply

def run():
    return multiply(2, 3, 4, 5)
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("accepts at most 2 positional arguments" in v for v in report.invariant_violations)


def test_scenario2_caller_positional_only_violation(tmp_path):
    (tmp_path / "service.py").write_text(
        """def configure(timeout: int, /, retries: int = 3):
    return timeout
""",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        """from service import configure

def start():
    return 0
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch calls timeout as keyword argument
    patch = """from service import configure

def start():
    return configure(timeout=30)
"""
    report = engine.verify(file_path="main.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("KEYWORD_MISMATCH" in v for v in report.invariant_violations)
    assert any("positional-only argument 'timeout'" in v for v in report.invariant_violations)


def test_scenario2_caller_missing_kwonly_arg(tmp_path):
    (tmp_path / "db.py").write_text(
        """def query(sql: str, *, transaction: bool, timeout: int = 30):
    return sql
""",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        """from db import query

def fetch():
    return 0
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch supplies positional 'sql' and optional kwarg 'timeout', but misses required kwonly 'transaction'
    patch = """from db import query

def fetch():
    return query("SELECT 1", timeout=10)
"""
    report = engine.verify(file_path="main.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("KEYWORD_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required keyword argument 'transaction'" in v for v in report.invariant_violations)


def test_scenario2_caller_duplicate_argument(tmp_path):
    (tmp_path / "api.py").write_text(
        """def post_data(endpoint: str, payload: dict = None):
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "client.py").write_text(
        """from api import post_data

def send():
    pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch passes endpoint positionally AND as keyword
    patch = """from api import post_data

def send():
    post_data("/v1/data", endpoint="/v2/override")
"""
    report = engine.verify(file_path="client.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("DUPLICATE_ARGUMENT" in v for v in report.invariant_violations)
    assert any("multiple values for argument 'endpoint'" in v for v in report.invariant_violations)


def test_scenario2_callee_signature_change_breaks_existing_caller(tmp_path):
    (tmp_path / "server.py").write_text(
        """def start_server(host: str) -> None:
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        """from server import start_server

def run():
    start_server("127.0.0.1")
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch to server.py adds required 'port' without default
    patch = """def start_server(host: str, port: int) -> None:
    pass
"""
    report = engine.verify(file_path="server.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'port'" in v for v in report.invariant_violations)


# ==============================================================================
# Scenario 3: Direct & Multi-Hop Circular Call / Import Cycles
# ==============================================================================

def test_scenario3_direct_circular_import(tmp_path):
    (tmp_path / "mod_a.py").write_text(
        """from mod_b import b_func

def a_func():
    return b_func()
""",
        encoding="utf-8",
    )
    (tmp_path / "mod_b.py").write_text(
        """def b_func():
    return 42
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch introduces import from mod_a into mod_b -> direct import cycle
    patch = """from mod_a import a_func

def b_func():
    return a_func()
"""
    report = engine.verify(file_path="mod_b.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("CIRCULAR_DEPENDENCY" in v for v in report.invariant_violations)
    assert any("Detected import cycle" in v for v in report.invariant_violations)
    assert len(report.cycles_detected) >= 1


def test_scenario3_multihop_circular_import_3_nodes(tmp_path):
    # A -> B -> C
    (tmp_path / "a.py").write_text(
        """from b import b_func
def a_func(): pass
""",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        """from c import c_func
def b_func(): pass
""",
        encoding="utf-8",
    )
    (tmp_path / "c.py").write_text(
        """def c_func(): pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch introduces C -> A import -> cycle: A -> B -> C -> A
    patch = """from a import a_func

def c_func():
    return a_func()
"""
    report = engine.verify(file_path="c.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("CIRCULAR_DEPENDENCY" in v for v in report.invariant_violations)
    assert any("Detected import cycle" in v for v in report.invariant_violations)


def test_scenario3_relative_import_cycle(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "alpha.py").write_text(
        """from .beta import beta_func
def alpha_func(): pass
""",
        encoding="utf-8",
    )
    (pkg / "beta.py").write_text(
        """def beta_func(): pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """from .alpha import alpha_func

def beta_func():
    return alpha_func()
"""
    report = engine.verify(file_path="pkg/beta.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("CIRCULAR_DEPENDENCY" in v for v in report.invariant_violations)


# ==============================================================================
# Scenario 4: Deleted Symbol Breakage
# ==============================================================================

def test_scenario4_deleted_symbol_called_by_another_file(tmp_path):
    (tmp_path / "utils.py").write_text(
        """def format_data(d):
    return str(d)
""",
        encoding="utf-8",
    )
    (tmp_path / "view.py").write_text(
        """from utils import format_data

def render():
    return format_data({"key": "val"})
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch deletes format_data from utils.py
    patch = """# utils.py without format_data
def other_util():
    pass
"""
    report = engine.verify(file_path="utils.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in report.invariant_violations)
    assert any("format_data" in v for v in report.invariant_violations)


def test_scenario4_deleted_symbol_called_by_surviving_same_file_function(tmp_path):
    (tmp_path / "service.py").write_text(
        """def active_handler():
    return internal_helper()

def internal_helper():
    return 100
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch deletes internal_helper, but active_handler still calls it
    patch = """def active_handler():
    return internal_helper()
"""
    report = engine.verify(file_path="service.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in report.invariant_violations)
    assert any("internal_helper" in v for v in report.invariant_violations)


def test_scenario4_nonexistent_symbol_imported_from_workspace_module(tmp_path):
    (tmp_path / "auth.py").write_text(
        """def verify_token(token: str) -> bool:
    return True
""",
        encoding="utf-8",
    )
    (tmp_path / "routes.py").write_text(
        """def index():
    return "ok"
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch in routes.py imports nonexistent symbol 'generate_jwt' from workspace module auth
    patch = """from auth import generate_jwt

def login():
    return generate_jwt()
"""
    report = engine.verify(file_path="routes.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in report.invariant_violations)
    assert any("generate_jwt" in v for v in report.invariant_violations)


# ==============================================================================
# Scenario 5: CLI Stdin / Diff File / Replacement Modes
# ==============================================================================

def test_scenario5_cli_stdin_pipe(tmp_path):
    (tmp_path / "math.py").write_text(
        """def add(a: int) -> int:
    return a + 1
""",
        encoding="utf-8",
    )
    patch_diff = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
"""
    res = subprocess.run(
        ["code-oracle", "verify", "math.py", "-w", str(tmp_path), "--json"],
        input=patch_diff,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"


def test_scenario5_cli_patch_dash_stdin(tmp_path):
    (tmp_path / "math.py").write_text(
        """def add(a: int) -> int:
    return a + 1
""",
        encoding="utf-8",
    )
    patch_diff = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
"""
    res = subprocess.run(
        ["code-oracle", "verify", "math.py", "-p", "-", "-w", str(tmp_path), "--json"],
        input=patch_diff,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"


def test_scenario5_cli_patch_diff_file(tmp_path):
    (tmp_path / "math.py").write_text(
        """def add(a: int) -> int:
    return a + 1
""",
        encoding="utf-8",
    )
    diff_file = tmp_path / "patch.diff"
    diff_file.write_text(
        """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 1) -> int:
""",
        encoding="utf-8",
    )
    res = subprocess.run(
        ["code-oracle", "verify", "math.py", "--patch", str(diff_file), "-w", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "VERDICT: APPROVED" in res.stdout


def test_scenario5_cli_replacement_mode(tmp_path):
    (tmp_path / "greeting.py").write_text(
        """def greet(name: str):
    return "Hi " + name
""",
        encoding="utf-8",
    )
    replacement = """def greet(name: str):
    return f"Hello, {name}!"
"""
    res = subprocess.run(
        ["code-oracle", "verify", "greeting.py", "--patch", replacement, "-w", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"


def test_scenario5_cli_json_slice_and_clean(tmp_path):
    (tmp_path / "service.py").write_text(
        """def process(): pass
""",
        encoding="utf-8",
    )
    # Index workspace
    subprocess.run(["code-oracle", "index", str(tmp_path), "--json"], capture_output=True)

    # Slice with --json
    res_slice = subprocess.run(
        ["code-oracle", "slice", "service.py", "--symbol", "process", "-w", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res_slice.returncode == 0
    slice_data = json.loads(res_slice.stdout)
    assert slice_data["symbol"] == "process"
    assert "nodes" in slice_data
    assert "edges" in slice_data

    # Clean with --json
    res_clean = subprocess.run(
        ["code-oracle", "clean", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res_clean.returncode == 0
    clean_data = json.loads(res_clean.stdout)
    assert clean_data["cleaned"] is True


# ==============================================================================
# Additional Deep Edge Case Scenarios
# ==============================================================================

def test_scenario2_class_instantiation_missing_args(tmp_path):
    (tmp_path / "models.py").write_text(
        """class User:
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from models import User

def register():
    return None
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch calls User with only username, missing email
    patch = """from models import User

def register():
    return User("john_doe")
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'email'" in v for v in report.invariant_violations)


def test_scenario2_class_instantiation_extra_args(tmp_path):
    (tmp_path / "models.py").write_text(
        """class Config:
    def __init__(self, debug: bool = False):
        self.debug = debug
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from models import Config

def init_app():
    return None
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch passes 3 positional args to Config.__init__ which accepts at most 1
    patch = """from models import Config

def init_app():
    return Config(True, "extra_arg", 42)
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("accepts at most 1 positional arguments" in v for v in report.invariant_violations)


def test_scenario2_class_instantiation_inheritance(tmp_path):
    (tmp_path / "base.py").write_text(
        """class BaseService:
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
""",
        encoding="utf-8",
    )
    (tmp_path / "derived.py").write_text(
        """from base import BaseService

class CustomService(BaseService):
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from derived import CustomService

def start():
    pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # CustomService inherits BaseService.__init__(api_key, endpoint).
    # Caller supplies api_key but misses endpoint.
    patch = """from derived import CustomService

def start():
    return CustomService("my_secret_key")
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'endpoint'" in v for v in report.invariant_violations)


def test_scenario2_class_init_change_breaks_instantiations(tmp_path):
    (tmp_path / "worker.py").write_text(
        """class Worker:
    def __init__(self, name: str):
        self.name = name
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from worker import Worker

def make():
    return Worker("Worker-1")
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch to worker.py adds required 'role' without default to Worker.__init__
    patch = """class Worker:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
"""
    report = engine.verify(file_path="worker.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'role'" in v for v in report.invariant_violations)


def test_scenario2_call_site_varargs_and_kwargs_dynamic(tmp_path):
    (tmp_path / "calc.py").write_text(
        """def compute_total(a: int, b: int, c: int) -> int:
    return a + b + c

def configure(*, env: str, timeout: int = 30) -> None:
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from calc import compute_total, configure

def run():
    return 0
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch passes *args and **kwargs dynamically; should NOT be falsely rejected
    patch = """from calc import compute_total, configure

def run():
    items = [1, 2, 3]
    opts = {"env": "prod"}
    configure(**opts)
    return compute_total(*items)
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_scenario2_patch_with_function_and_module_call_violation(tmp_path):
    (tmp_path / "math_ops.py").write_text(
        """def multiply(a: int, b: int) -> int:
    return a * b
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """def helper():
    return 1
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch modifies both a function AND introduces module-level call with wrong arity
    patch = """from math_ops import multiply

x = multiply(10)  # Missing 'b'

def helper():
    return 42
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'b'" in v for v in report.invariant_violations)


def test_scenario2_callee_change_breaks_module_level_call(tmp_path):
    (tmp_path / "math_ops.py").write_text(
        """def add(a: int, b: int) -> int:
    return a + b
""",
        encoding="utf-8",
    )
    (tmp_path / "script.py").write_text(
        """from math_ops import add
TOTAL = add(1, 2)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch adds required parameter 'c' to add
    patch = """def add(a: int, b: int, c: int) -> int:
    return a + b + c
"""
    report = engine.verify(file_path="math_ops.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'c'" in v for v in report.invariant_violations)


def test_scenario4_relative_import_of_nonexistent_symbol(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text("def valid_func(): pass\n", encoding="utf-8")
    (pkg / "handler.py").write_text("def handle(): pass\n", encoding="utf-8")
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch in handler.py relative-imports a nonexistent symbol
    patch = """from . import nonexistent_helper

def handle():
    return nonexistent_helper()
"""
    report = engine.verify(file_path="mypkg/handler.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in report.invariant_violations)
    assert any("nonexistent_helper" in v for v in report.invariant_violations)


def test_scenario4_unpacked_constant_assignment_import_approved(tmp_path):
    (tmp_path / "constants.py").write_text(
        """# Unpacked constants
CONST_A, CONST_B = (100, 200)
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text("def run(): pass\n", encoding="utf-8")
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """from constants import CONST_A, CONST_B

def run():
    return CONST_A + CONST_B
"""
    report = engine.verify(file_path="app.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_scenario4_star_import_reexport_approved(tmp_path):
    (tmp_path / "internal.py").write_text("MAGIC_NUM = 777\n", encoding="utf-8")
    (tmp_path / "api.py").write_text("from internal import *\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text("def test(): pass\n", encoding="utf-8")
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Importing MAGIC_NUM from api.py (which has star import)
    patch = """from api import MAGIC_NUM

def test():
    return MAGIC_NUM
"""
    report = engine.verify(file_path="consumer.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_scenario5_diff_with_empty_lines_without_leading_space(tmp_path):
    (tmp_path / "formatter.py").write_text(
        """def format_message():

    return "Hello"
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Diff containing a blank line without leading space
    diff = """@@ -1,4 +1,4 @@
 def format_message():

-    return "Hello"
+    return "Hello World"
"""
    report = engine.verify(file_path="formatter.py", patch_content=diff)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_scenario5_cli_no_subcommand_exit_code_2():
    res = subprocess.run(["code-oracle"], capture_output=True, text=True)
    assert res.returncode == 2


def test_scenario5_cli_patch_file_relative_to_workspace(tmp_path):
    (tmp_path / "calc.py").write_text("def mul(a: int) -> int: return a * 2\n", encoding="utf-8")
    diff_file = tmp_path / "patch.diff"
    diff_file.write_text(
        """@@ -1,1 +1,1 @@
-def mul(a: int) -> int: return a * 2
+def mul(a: int, b: int = 1) -> int: return a * b
""",
        encoding="utf-8",
    )
    res = subprocess.run(
        ["code-oracle", "verify", "calc.py", "--patch", "patch.diff", "-w", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "APPROVED"


def test_scenario5_cli_slice_json_error_when_symbol_not_found(tmp_path):
    (tmp_path / "dummy.py").write_text("def exists(): pass\n", encoding="utf-8")
    res = subprocess.run(
        ["code-oracle", "slice", "dummy.py", "--symbol", "nonexistent", "-w", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert "error" in data
    assert "nonexistent" in data["error"]
