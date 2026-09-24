"""
Tests for Stage 3: k-Hop Neighborhood Slicer.
"""

from pathlib import Path
import pytest

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import CallReference, Symbol
from code_oracle.slicer import slice_neighborhood


def test_slicer_1_hop(tmp_path):
    # Setup test workspace
    f1 = tmp_path / "a.py"
    f1.write_text(
        """def foo():
    bar()

def bar():
    pass
""",
        encoding="utf-8",
    )

    f2 = tmp_path / "b.py"
    f2.write_text(
        """from a import foo

def caller_of_foo():
    foo()
""",
        encoding="utf-8",
    )

    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    foo_sym = indexer.get_definition("foo")
    assert foo_sym is not None

    # Slice around foo (k=1)
    graph = slice_neighborhood(seeds=[foo_sym], indexer=indexer, k=1)

    node_ids = set(graph.nodes.keys())
    assert foo_sym.id in node_ids
    # bar is callee of foo
    bar_sym = indexer.get_definition("bar")
    assert bar_sym.id in node_ids
    # caller_of_foo is caller of foo
    caller_sym = indexer.get_definition("caller_of_foo")
    assert caller_sym.id in node_ids

    # Check edges
    edge_tuples = [(e.source, e.target, e.relation) for e in graph.edges]
    assert (foo_sym.id, bar_sym.id, "CALLS") in edge_tuples
    assert (caller_sym.id, foo_sym.id, "CALLS") in edge_tuples
    assert graph.truncated is False


def test_slicer_2_hop(tmp_path):
    # Chain: d -> c -> b -> a
    (tmp_path / "chain.py").write_text(
        """def a(): pass
def b(): a()
def c(): b()
def d(): c()
""",
        encoding="utf-8",
    )
    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    c_sym = indexer.get_definition("c")

    # k=1 from c should find b and d, but not a
    graph_k1 = slice_neighborhood(seeds=[c_sym], indexer=indexer, k=1)
    assert "chain.py::b" in graph_k1.nodes
    assert "chain.py::d" in graph_k1.nodes
    assert "chain.py::a" not in graph_k1.nodes

    # k=2 from c should reach a (c -> b -> a)
    graph_k2 = slice_neighborhood(seeds=[c_sym], indexer=indexer, k=2)
    assert "chain.py::a" in graph_k2.nodes


def test_slicer_fanout_protection(tmp_path):
    # Create a utility function called by 30 callers
    caller_defs = "\n".join([f"def caller_{i}():\n    util()\n" for i in range(30)])
    (tmp_path / "util.py").write_text(
        f"""def util():
    pass

{caller_defs}
""",
        encoding="utf-8",
    )

    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    util_sym = indexer.get_definition("util")
    assert len(indexer.get_callers("util")) == 30

    # Slice with max_fanout=20
    graph = slice_neighborhood(seeds=[util_sym], indexer=indexer, k=1, max_fanout=20)

    # Edge count should be capped at 20
    assert len(graph.edges) == 20
    assert graph.truncated is True
    assert graph.nodes[util_sym.id].truncated is True


def test_slicer_max_nodes_boundary(tmp_path):
    # Generate many disconnected or linked nodes
    lines = [f"def node_{i}(): pass" for i in range(70)]
    (tmp_path / "many.py").write_text("\n".join(lines), encoding="utf-8")

    indexer = WorkspaceIndexer(workspace_root=tmp_path)
    indexer.scan_workspace()

    all_syms = indexer.get_file_symbols("many.py")
    # If passed many seeds, total nodes should not exceed max_nodes=50
    graph = slice_neighborhood(seeds=all_syms, indexer=indexer, k=1, max_nodes=50)
    assert len(graph.nodes) <= 50
