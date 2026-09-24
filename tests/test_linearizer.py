"""
Tests for Stage 5: Graph Linearizer.
Compact Domain-Specific Language (< 400 tokens) for Laya decision head.
"""

import pytest

from code_oracle.linearizer import estimate_tokens, linearize_subgraph
from code_oracle.models import (
    GateResult,
    PatchResult,
    SlicedGraph,
    SliceNode,
    SliceEdge,
    Symbol,
)


def test_linearizer_basic_dsl_structure():
    sym = Symbol(
        name="charge",
        qualname="charge",
        file_path="billing.py",
        kind="function",
        lineno=10,
        end_lineno=15,
        signature="def charge(amount: float) -> bool",
    )
    patch_result = PatchResult(
        file_path="billing.py",
        original_content="",
        patched_content="",
        modified_old_lines={10, 11},
        modified_new_lines={10, 11, 12},
        affected_symbols=[sym],
    )
    node_seed = SliceNode(
        id=sym.id,
        name="charge",
        file_path="billing.py",
        kind="function",
        signature=sym.signature,
        is_seed=True,
        is_modified=True,
    )
    node_caller = SliceNode(
        id="api.py::checkout",
        name="checkout",
        file_path="api.py",
        kind="function",
        signature="def checkout(user_id: int)",
    )
    edge = SliceEdge(source="api.py::checkout", target=sym.id, relation="CALLS")

    graph = SlicedGraph(
        nodes={sym.id: node_seed, "api.py::checkout": node_caller},
        edges=[edge],
        seed_ids={sym.id},
    )
    gate = GateResult(status="APPROVED", confidence=0.98, cycles=[], violations=[])

    dsl = linearize_subgraph(patch_result, graph, gate)

    # Validate DSL structure
    assert "[DIFF_TARGET] billing.py::charge (MODIFIED)" in dsl
    assert "[METADATA]" in dsl
    assert "[NODES]" in dsl
    assert "[EDGES]" in dsl
    assert "[GATE]" in dsl
    assert "STATUS: APPROVED (conf: 0.98)" in dsl
    assert "CYCLES: 0" in dsl
    assert "VIOLATIONS: NONE" in dsl

    # Check token budget
    tokens = estimate_tokens(dsl)
    assert tokens < 400
    assert tokens > 20


def test_linearizer_strictly_under_400_tokens_large_graph():
    # Build a huge 60-node graph that would easily exceed 500 tokens if unconstrained
    seed_sym = Symbol(
        name="core_fn",
        qualname="core_fn",
        file_path="core.py",
        kind="function",
        lineno=1,
        end_lineno=5,
        signature="def core_fn()",
    )
    patch_result = PatchResult(
        file_path="core.py",
        original_content="",
        patched_content="",
        affected_symbols=[seed_sym],
    )

    nodes = {
        seed_sym.id: SliceNode(
            id=seed_sym.id,
            name="core_fn",
            file_path="core.py",
            kind="function",
            signature="def core_fn()",
            is_seed=True,
            is_modified=True,
        )
    }
    edges = []

    for i in range(1, 60):
        node_id = f"module_{i}.py::function_with_a_very_long_name_and_signature_{i}"
        nodes[node_id] = SliceNode(
            id=node_id,
            name=f"function_{i}",
            file_path=f"module_{i}.py",
            kind="function",
            signature=f"def function_with_a_very_long_name_{i}(arg1: str, arg2: int = 100, flag: bool = True) -> Optional[Dict[str, Any]]",
        )
        edges.append(SliceEdge(source=node_id, target=seed_sym.id, relation="CALLS"))

    graph = SlicedGraph(nodes=nodes, edges=edges, seed_ids={seed_sym.id})
    gate = GateResult(status="APPROVED", confidence=0.98)

    dsl = linearize_subgraph(patch_result, graph, gate, max_tokens=400)
    tokens = estimate_tokens(dsl)

    # Must strictly satisfy < 400 tokens constraint
    assert tokens <= 400
    assert "[TRUNCATED:" in dsl
    # Seed node must always remain preserved
    assert "core_fn" in dsl


def test_estimate_tokens_empty_and_special():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("def foo(x: int) -> bool:") >= 8
