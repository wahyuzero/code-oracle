"""
Tests for Stage 4: Deterministic Symbolic Gate.
Tarjan SCC cycle detection and contract invariant verification.
"""

import pytest

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import CallReference, GateResult, Parameter, PatchResult, SlicedGraph, SliceNode, SliceEdge, Symbol
from code_oracle.symbolic import find_cycles_tarjan, verify_symbolic_gate


def test_tarjan_acyclic():
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    cycles = find_cycles_tarjan(graph)
    assert cycles == []


def test_tarjan_2_node_cycle():
    graph = {
        "A": ["B"],
        "B": ["A"],
    }
    cycles = find_cycles_tarjan(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B"}


def test_tarjan_3_node_cycle():
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
        "D": ["A"],
    }
    cycles = find_cycles_tarjan(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C"}


def test_tarjan_self_loop():
    graph = {
        "A": ["A"],
        "B": [],
    }
    assert find_cycles_tarjan(graph, include_self_loops=False) == []
    cycles = find_cycles_tarjan(graph, include_self_loops=True)
    assert len(cycles) == 1
    assert cycles[0] == ["A"]


def test_gate_syntax_error():
    indexer = WorkspaceIndexer()
    patch_result = PatchResult(
        file_path="foo.py",
        original_content="",
        patched_content="def invalid(:",
        syntax_error="SyntaxError: invalid syntax",
    )
    slice_graph = SlicedGraph()
    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "REJECTED"
    assert res.confidence == 1.0
    assert any("SYNTAX_ERROR" in v for v in res.violations)


def test_gate_detects_call_cycle():
    indexer = WorkspaceIndexer()
    node_a = SliceNode(id="a.py::fn_a", name="fn_a", file_path="a.py", kind="function", signature="def fn_a()")
    node_b = SliceNode(id="b.py::fn_b", name="fn_b", file_path="b.py", kind="function", signature="def fn_b()")

    # Sliced graph with cycle: fn_a -> fn_b -> fn_a
    edges = [
        SliceEdge("a.py::fn_a", "b.py::fn_b", "CALLS"),
        SliceEdge("b.py::fn_b", "a.py::fn_a", "CALLS"),
    ]
    slice_graph = SlicedGraph(
        nodes={"a.py::fn_a": node_a, "b.py::fn_b": node_b},
        edges=edges,
    )
    patch_result = PatchResult(
        file_path="a.py",
        original_content="",
        patched_content="",
        affected_symbols=[],
    )
    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "REJECTED"
    assert len(res.cycles) == 1
    assert any("CIRCULAR_DEPENDENCY" in v for v in res.violations)


def test_gate_arity_mismatch_missing_args(tmp_path):
    # Setup indexer with caller in caller.py calling target(x) with 1 argument
    (tmp_path / "caller.py").write_text(
        """def call_target():
    target(10)
""",
        encoding="utf-8",
    )
    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    # Target symbol modified in patch to require 2 arguments (no defaults)
    patched_target = Symbol(
        name="target",
        qualname="target",
        file_path="target.py",
        kind="function",
        lineno=1,
        end_lineno=3,
        signature="def target(a: int, b: int) -> int",
        min_args=2,
        max_args=2,
        params=[Parameter(name="a"), Parameter(name="b")],
    )
    patch_result = PatchResult(
        file_path="target.py",
        original_content="",
        patched_content="",
        affected_symbols=[patched_target],
    )
    slice_graph = SlicedGraph(nodes={patched_target.id: SliceNode(id=patched_target.id, name="target", file_path="target.py", kind="function", signature=patched_target.signature)})

    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in res.violations)
    assert any("requires at least 2 arguments" in v for v in res.violations)


def test_gate_arity_mismatch_too_many_args(tmp_path):
    (tmp_path / "caller.py").write_text(
        """def call_target():
    target(10, 20, 30)
""",
        encoding="utf-8",
    )
    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    # Target accepts at most 1 argument
    patched_target = Symbol(
        name="target",
        qualname="target",
        file_path="target.py",
        kind="function",
        lineno=1,
        end_lineno=3,
        signature="def target(a: int) -> int",
        min_args=1,
        max_args=1,
        params=[Parameter(name="a")],
    )
    patch_result = PatchResult(
        file_path="target.py",
        original_content="",
        patched_content="",
        affected_symbols=[patched_target],
    )
    slice_graph = SlicedGraph(nodes={patched_target.id: SliceNode(id=patched_target.id, name="target", file_path="target.py", kind="function", signature=patched_target.signature)})

    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in res.violations)
    assert any("accepts at most 1 positional arguments" in v for v in res.violations)


def test_gate_keyword_mismatch(tmp_path):
    (tmp_path / "caller.py").write_text(
        """def call_target():
    target(10, unknown_opt=True)
""",
        encoding="utf-8",
    )
    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    patched_target = Symbol(
        name="target",
        qualname="target",
        file_path="target.py",
        kind="function",
        lineno=1,
        end_lineno=3,
        signature="def target(a: int)",
        min_args=1,
        max_args=1,
        accepted_kwargs={"a"},
        params=[Parameter(name="a")],
    )
    patch_result = PatchResult(
        file_path="target.py",
        original_content="",
        patched_content="",
        affected_symbols=[patched_target],
    )
    slice_graph = SlicedGraph(nodes={patched_target.id: SliceNode(id=patched_target.id, name="target", file_path="target.py", kind="function", signature=patched_target.signature)})

    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "REJECTED"
    assert any("KEYWORD_MISMATCH" in v for v in res.violations)
    assert any("unexpected keyword argument 'unknown_opt'" in v for v in res.violations)


def test_gate_deleted_symbol_with_active_callers(tmp_path):
    (tmp_path / "caller.py").write_text(
        """def call_old():
    deprecated_helper()
""",
        encoding="utf-8",
    )
    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    deleted_sym = Symbol(
        name="deprecated_helper",
        qualname="deprecated_helper",
        file_path="lib.py",
        kind="function",
        lineno=1,
        end_lineno=3,
    )
    patch_result = PatchResult(
        file_path="lib.py",
        original_content="",
        patched_content="",
        deleted_symbols=[deleted_sym],
    )
    slice_graph = SlicedGraph()

    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in res.violations)
    assert any("deprecated_helper" in v for v in res.violations)


def test_gate_approved_clean_patch(tmp_path):
    (tmp_path / "caller.py").write_text(
        """def call_target():
    target(10)
""",
        encoding="utf-8",
    )
    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    # Target modified: added optional param with default
    patched_target = Symbol(
        name="target",
        qualname="target",
        file_path="target.py",
        kind="function",
        lineno=1,
        end_lineno=3,
        signature="def target(a: int, b: int = 1) -> int",
        min_args=1,
        max_args=2,
        accepted_kwargs={"a", "b"},
        params=[Parameter(name="a"), Parameter(name="b", default="1", has_default=True)],
    )
    patch_result = PatchResult(
        file_path="target.py",
        original_content="",
        patched_content="",
        affected_symbols=[patched_target],
    )
    slice_graph = SlicedGraph(nodes={patched_target.id: SliceNode(id=patched_target.id, name="target", file_path="target.py", kind="function", signature=patched_target.signature)})

    res = verify_symbolic_gate(patch_result, slice_graph, indexer)
    assert res.status == "APPROVED"
    assert res.confidence == 0.98
    assert res.violations == []
    assert res.cycles == []
