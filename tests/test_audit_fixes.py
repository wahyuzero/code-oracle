"""
Tests for TopoSlice architecture audit fixes and edge cases:
- @staticmethod and 0-argument method contract resolution
- Method-to-method call cycles (self/cls)
- Multi-node call cycles with k=1 induced subgraph completion
- Method arity mismatch detection
- Missing required positional args despite supplied kwargs/kwonly
- Deleted symbol import detection across files
- Class inheritance edges (INHERITS)
- Positional-only parameter enforcement
- Duplicate argument detection
- Absolute path normalization
- Zero-side-effect isolation rollback
"""

import tempfile
from pathlib import Path
import pytest

from code_oracle.engine import TopoSliceEngine
from code_oracle.locator import extract_symbols_from_ast
from code_oracle.models import SlicedGraph
from code_oracle.slicer import slice_neighborhood


def test_staticmethod_zero_and_multi_args(tmp_path):
    (tmp_path / "math_ops.py").write_text(
        """class MathOps:
    @staticmethod
    def zero_args():
        return 42

    @staticmethod
    def add(a: int, b: int) -> int:
        return a + b

def caller():
    v1 = MathOps.zero_args()
    v2 = MathOps.add(10, 20)
    return v1 + v2
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch: docstring addition to MathOps
    patch = """@@ -1,5 +1,6 @@
 class MathOps:
+    \"\"\"Utility class for static math operations.\"\"\"
     @staticmethod
     def zero_args():
         return 42
"""
    report = engine.verify(file_path="math_ops.py", patch_content=patch)
    assert report.status == "APPROVED"
    assert report.invariant_violations == []


def test_method_mutual_recursion_cycle_detected(tmp_path):
    (tmp_path / "handler.py").write_text(
        """class Handler:
    def step_a(self):
        return 1

    def step_b(self):
        return 2
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch introduces mutual recursion between methods
    patch = """@@ -1,6 +1,6 @@
 class Handler:
     def step_a(self):
-        return 1
+        return self.step_b()
 
     def step_b(self):
-        return 2
+        return self.step_a()
"""
    report = engine.verify(file_path="handler.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("CIRCULAR_DEPENDENCY" in v for v in report.invariant_violations)
    assert len(report.cycles_detected) >= 1


def test_multi_node_cycle_with_k1_induced_subgraph(tmp_path):
    # A -> B -> C -> A cycle with k=1
    (tmp_path / "b_c.py").write_text(
        """def b():
    return c()

def c():
    from patch_target import a
    return a()
""",
        encoding="utf-8",
    )
    (tmp_path / "patch_target.py").write_text(
        """def a():
    return 1
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)
    engine.indexer.scan_workspace()

    # Propose patch to a() calling b()
    patch = """@@ -1,2 +1,3 @@
+from b_c import b
 def a():
-    return 1
+    return b()
"""
    report = engine.verify(file_path="patch_target.py", patch_content=patch, k=1)
    assert report.status == "REJECTED"
    assert any("CIRCULAR_DEPENDENCY" in v for v in report.invariant_violations)
    assert len(report.cycles_detected) >= 1


def test_method_arity_mismatch_detected(tmp_path):
    (tmp_path / "service.py").write_text(
        """class Service:
    def process(self, x: int) -> int:
        return x * 2

def runner():
    svc = Service()
    return svc.process(1, 2, 3)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    # Patch modifies process signature to accept at most 1 positional arg (plus self)
    patch = """@@ -1,3 +1,3 @@
 class Service:
-    def process(self, x: int) -> int:
+    def process(self, x: int, debug: bool = False) -> int:
         return x * 2
"""
    report = engine.verify(file_path="service.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("accepts at most 2 positional arguments" in v or "accepts at most 1 positional arguments" in v for v in report.invariant_violations)


def test_caller_missing_required_arg_when_optional_kwarg_passed(tmp_path):
    (tmp_path / "billing.py").write_text(
        """def foo(a: int, b: int = 10):
    return a + b

def caller():
    return foo(b=20)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,2 @@
-def foo(a: int, b: int = 10):
+def foo(a: int, b: int = 10, note: str = ""):
"""
    report = engine.verify(file_path="billing.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'a'" in v for v in report.invariant_violations)


def test_caller_missing_required_arg_when_kwonly_passed(tmp_path):
    (tmp_path / "service.py").write_text(
        """def process(a: int, b: int, *, verbose: bool = False):
    return a + b

def caller():
    return process(1, verbose=True)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,2 @@
-def process(a: int, b: int, *, verbose: bool = False):
+def process(a: int, b: int, *, verbose: bool = False, retries: int = 3):
"""
    report = engine.verify(file_path="service.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("missing required argument 'b'" in v for v in report.invariant_violations)


def test_deleted_symbol_imported_by_another_file_rejected(tmp_path):
    (tmp_path / "lib.py").write_text(
        """def helper():
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """from lib import helper
# helper is imported
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)
    engine.indexer.scan_workspace()

    # Patch deletes helper from lib.py
    patch = """# lib.py empty
"""
    report = engine.verify(file_path="lib.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in report.invariant_violations)
    assert any("helper" in v for v in report.invariant_violations)


def test_class_inheritance_slicing(tmp_path):
    (tmp_path / "models.py").write_text(
        """class Animal:
    def speak(self): pass

class Dog(Animal):
    def speak(self): return "woof"
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)
    engine.indexer.scan_workspace()

    dog_sym = engine.indexer.get_definition("Dog")
    assert dog_sym is not None

    slice_graph = slice_neighborhood(seeds=[dog_sym], indexer=engine.indexer, k=1)
    inherits_edges = [
        (e.source, e.target, e.relation)
        for e in slice_graph.edges
        if e.relation == "INHERITS"
    ]
    assert ("models.py::Dog", "models.py::Animal", "INHERITS") in inherits_edges


def test_duplicate_argument_rejected(tmp_path):
    (tmp_path / "api.py").write_text(
        """def call_api(endpoint: str, timeout: int = 10):
    pass

def caller():
    call_api("http://test", endpoint="http://duplicate")
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,2 @@
-def call_api(endpoint: str, timeout: int = 10):
+def call_api(endpoint: str, timeout: int = 10, retries: int = 1):
"""
    report = engine.verify(file_path="api.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("DUPLICATE_ARGUMENT" in v for v in report.invariant_violations)


def test_positional_only_keyword_call_rejected(tmp_path):
    (tmp_path / "pos_only.py").write_text(
        """def configure(port: int, /, host: str = "localhost"):
    pass

def caller():
    configure(port=8080, host="0.0.0.0")
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)

    patch = """@@ -1,2 +1,2 @@
-def configure(port: int, /, host: str = "localhost"):
+def configure(port: int, /, host: str = "localhost", ssl: bool = False):
"""
    report = engine.verify(file_path="pos_only.py", patch_content=patch)
    assert report.status == "REJECTED"
    assert any("KEYWORD_MISMATCH" in v for v in report.invariant_violations)
    assert any("positional-only" in v for v in report.invariant_violations)


def test_absolute_path_verification_no_duplicate_violations(tmp_path):
    (tmp_path / "calc.py").write_text(
        """def add(a: int) -> int:
    return a + 1

def compute(x: int):
    return add(x)
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)
    abs_file = str((tmp_path / "calc.py").resolve())

    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int) -> int:
"""
    report = engine.verify(file_path=abs_file, patch_content=patch)
    assert report.status == "REJECTED"
    # Should only report 1 violation, not duplicate
    assert len(report.invariant_violations) == 1


def test_engine_zero_side_effect_isolation(tmp_path):
    target = tmp_path / "isolated.py"
    target.write_text(
        """def original_func():
    return 100
""",
        encoding="utf-8",
    )
    engine = TopoSliceEngine(workspace_root=tmp_path)
    engine.indexer.scan_workspace()

    # Pre-state
    sym_before = engine.indexer.get_definition("original_func")
    assert sym_before is not None

    # Verify a patch that renames original_func -> new_func
    patch = """def new_func():
    return 200
"""
    report = engine.verify(file_path="isolated.py", patch_content=patch)
    assert report.status == "APPROVED"

    # Post-state: Indexer MUST be cleanly restored to disk state
    sym_after = engine.indexer.get_definition("original_func")
    assert sym_after is not None
    assert engine.indexer.get_definition("new_func") is None
