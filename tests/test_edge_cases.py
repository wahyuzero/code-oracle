"""
Comprehensive edge cases for TopoSlice verification engine.
"""

from pathlib import Path
import pytest

from code_oracle.engine import TopoSliceEngine
from code_oracle.symbolic import find_cycles_tarjan


def test_edge_case_required_kwonly_arg_missing(tmp_path):
    (tmp_path / "app.py").write_text(
        """from service import configure

def startup():
    configure("prod")
""",
        encoding="utf-8",
    )
    (tmp_path / "service.py").write_text(
        """def configure(env: str):
    pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch adds required keyword-only argument: *, timeout: int
    patch = """@@ -1,2 +1,2 @@
-def configure(env: str):
+def configure(env: str, *, timeout: int):
"""
    report = engine.verify(file_path="service.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("KEYWORD_MISMATCH" in v for v in report.invariant_violations)
    assert any("timeout" in v for v in report.invariant_violations)


def test_edge_case_required_kwonly_arg_supplied(tmp_path):
    (tmp_path / "app.py").write_text(
        """from service import configure

def startup():
    configure("prod", timeout=30)
""",
        encoding="utf-8",
    )
    (tmp_path / "service.py").write_text(
        """def configure(env: str):
    pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,2 @@
-def configure(env: str):
+def configure(env: str, *, timeout: int):
"""
    report = engine.verify(file_path="service.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_edge_case_varargs_and_kwargs_acceptance(tmp_path):
    (tmp_path / "caller.py").write_text(
        """from flex import dynamic_call

def run():
    dynamic_call(1, 2, 3, 4, 5, mode="fast", debug=True)
""",
        encoding="utf-8",
    )
    (tmp_path / "flex.py").write_text(
        """def dynamic_call(base: int):
    pass
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch upgrades function to accept *args, **kwargs
    patch = """@@ -1,2 +1,2 @@
-def dynamic_call(base: int):
+def dynamic_call(base: int, *args, **kwargs):
"""
    report = engine.verify(file_path="flex.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_edge_case_comment_only_patch(tmp_path):
    (tmp_path / "math_util.py").write_text(
        """def calculate(x: int) -> int:
    return x * 2
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,3 @@
+# Added docstring comment
 def calculate(x: int) -> int:
"""
    report = engine.verify(file_path="math_util.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_edge_case_whitespace_diff(tmp_path):
    (tmp_path / "clean.py").write_text(
        """def foo():
    return 1
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """def foo():
    
    return 1
"""
    report = engine.verify(file_path="clean.py", patch_content=patch)
    assert report.status == "APPROVED"


def test_edge_case_multi_cycle_graph():
    # Two disjoint cycles: A <-> B and C <-> D
    graph = {
        "A": ["B"],
        "B": ["A"],
        "C": ["D"],
        "D": ["C"],
        "E": ["A", "C"],
    }
    cycles = find_cycles_tarjan(graph)
    assert len(cycles) == 2
    cycles_sets = [set(c) for c in cycles]
    assert {"A", "B"} in cycles_sets
    assert {"C", "D"} in cycles_sets


def test_edge_case_5_node_cycle():
    # A -> B -> C -> D -> E -> A
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["D"],
        "D": ["E"],
        "E": ["A"],
    }
    cycles = find_cycles_tarjan(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C", "D", "E"}


def test_edge_case_new_file_creation(tmp_path):
    engine = TopoSliceEngine(workspace_root=tmp_path)

    new_file_content = """def brand_new_feature(a: int) -> int:
    return a + 10
"""
    report = engine.verify(file_path="brand_new.py", patch_content=new_file_content)
    assert report.status == "APPROVED"
    assert report.cycles_detected == []
    assert report.invariant_violations == []


def test_edge_case_method_call_self_binding(tmp_path):
    (tmp_path / "calculator.py").write_text(
        """class Calculator:
    def compute(self, x: int) -> int:
        return x * 2

def run():
    calc = Calculator()
    return calc.compute(5)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -2,2 +2,2 @@
-    def compute(self, x: int) -> int:
+    def compute(self, x: int, factor: int = 2) -> int:
"""
    report = engine.verify(file_path="calculator.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_edge_case_renamed_parameter_keyword_call(tmp_path):
    (tmp_path / "api.py").write_text(
        """def request_data(url: str, timeout_sec: int = 5):
    pass

def caller():
    request_data("http://example.com", timeout_sec=10)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch renames timeout_sec -> timeout
    patch = """@@ -1,2 +1,2 @@
-def request_data(url: str, timeout_sec: int = 5):
+def request_data(url: str, timeout: int = 5):
"""
    report = engine.verify(file_path="api.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("KEYWORD_MISMATCH" in v for v in report.invariant_violations)
    assert any("timeout_sec" in v for v in report.invariant_violations)


def test_edge_case_async_function_support(tmp_path):
    (tmp_path / "async_service.py").write_text(
        """async def fetch_async(url: str) -> str:
    return "ok"

async def handler():
    return await fetch_async("http://example.com")
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,2 @@
-async def fetch_async(url: str) -> str:
+async def fetch_async(url: str, retries: int = 3) -> str:
"""
    report = engine.verify(file_path="async_service.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []
