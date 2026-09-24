"""
Tests for Stage 1: Diff Boundary Locator.
"""

import tempfile
from pathlib import Path
import pytest

from code_oracle.locator import (
    apply_patch,
    extract_symbols_from_ast,
    locate_affected_symbols,
    parse_unified_diff,
)


def test_parse_unified_diff_single_hunk():
    diff_text = """--- a/foo.py
+++ b/foo.py
@@ -10,3 +10,4 @@
 def hello():
-    return 1
+    return 2
+    return 3
"""
    hunks = parse_unified_diff(diff_text)
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.old_start == 10
    assert hunk.old_count == 3
    assert hunk.new_start == 10
    assert hunk.new_count == 4
    assert len(hunk.lines) >= 3


def test_parse_unified_diff_multiple_hunks():
    diff_text = """@@ -5,2 +5,2 @@
-a = 1
+a = 2
@@ -20,3 +20,4 @@
 def bar():
-    pass
+    x = 1
+    return x
"""
    hunks = parse_unified_diff(diff_text)
    assert len(hunks) == 2
    assert hunks[0].old_start == 5
    assert hunks[1].old_start == 20


def test_apply_patch_unified_diff():
    orig = """line 1
line 2
def calculate(x):
    return x * 10
line 5
"""
    patch = """@@ -3,2 +3,3 @@
 def calculate(x):
-    return x * 10
+    res = x * 20
+    return res
"""
    patched, old_lines, new_lines = apply_patch(orig, patch)
    assert "return res" in patched
    assert "return x * 10" not in patched
    assert 4 in old_lines
    assert 4 in new_lines or 5 in new_lines


def test_apply_patch_full_replacement():
    orig = "def foo():\n    return 1\n"
    patch = "def foo():\n    return 2\n"
    patched, old_lines, new_lines = apply_patch(orig, patch)
    assert "return 2" in patched
    assert old_lines == {2}
    assert new_lines == {2}


def test_apply_patch_new_file():
    orig = ""
    patch = "def brand_new():\n    return True\n"
    patched, old_lines, new_lines = apply_patch(orig, patch)
    assert "def brand_new():" in patched
    assert old_lines == set()
    assert 1 in new_lines
    assert 2 in new_lines


def test_extract_symbols_from_ast():
    source = """
import os

def standalone(a: int, b: str = "default") -> bool:
    return True

async def async_worker(task_id: int):
    pass

class DataProcessor:
    def __init__(self, name: str):
        self.name = name

    def process(self, data: list) -> int:
        return standalone(len(data))
"""
    symbols = extract_symbols_from_ast(source, "processor.py")
    qualnames = [s.qualname for s in symbols]
    assert "standalone" in qualnames
    assert "async_worker" in qualnames
    assert "DataProcessor" in qualnames
    assert "DataProcessor.__init__" in qualnames
    assert "DataProcessor.process" in qualnames

    # Check standalone params
    standalone_sym = [s for s in symbols if s.name == "standalone"][0]
    assert standalone_sym.min_args == 1
    assert standalone_sym.max_args == 2
    assert standalone_sym.return_type == "bool"
    assert standalone_sym.kind == "function"

    # Check async_worker
    async_sym = [s for s in symbols if s.name == "async_worker"][0]
    assert async_sym.kind == "async_function"

    # Check method
    method_sym = [s for s in symbols if s.qualname == "DataProcessor.process"][0]
    assert method_sym.kind == "method"
    assert method_sym.is_method is True
    assert method_sym.min_args == 2  # self, data
    assert len(method_sym.calls) == 2  # standalone, len


def test_locate_affected_symbols_modified():
    orig = """def add(a: int) -> int:
    return a + 1

def mul(a: int, b: int) -> int:
    return a * b
"""
    patch = """@@ -1,2 +1,2 @@
-def add(a: int) -> int:
+def add(a: int, b: int = 0) -> int:
"""
    res = locate_affected_symbols("math_ops.py", patch, original_content=orig)
    assert res.syntax_error is None
    assert len(res.affected_symbols) == 1
    assert res.affected_symbols[0].name == "add"
    assert res.affected_symbols[0].min_args == 1
    assert res.affected_symbols[0].max_args == 2


def test_locate_affected_symbols_syntax_error():
    orig = "def good(): pass\n"
    patch = "def broken(:\n    pass\n"
    res = locate_affected_symbols("broken.py", patch, original_content=orig)
    assert res.syntax_error is not None
    assert "SyntaxError" in res.syntax_error


def test_locate_affected_symbols_deleted_and_added():
    orig = """def old_func():
    return 1

def keep_func():
    return 2
"""
    patch = """def new_func():
    return 3

def keep_func():
    return 2
"""
    res = locate_affected_symbols("change.py", patch, original_content=orig)
    assert [s.name for s in res.deleted_symbols] == ["old_func"]
    assert [s.name for s in res.added_symbols] == ["new_func"]


def test_locate_affected_symbols_module_level():
    orig = "CONFIG_VALUE = 10\n"
    patch = "CONFIG_VALUE = 20\n"
    res = locate_affected_symbols("config.py", patch, original_content=orig)
    assert len(res.affected_symbols) == 1
    assert res.affected_symbols[0].kind == "module"
