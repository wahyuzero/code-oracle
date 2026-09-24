"""
Deterministic Symbolic Gate: Graph Cycle Detection and Invariant Checks.
"""

from typing import Any, Dict, List, Optional, Set

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import GateResult, PatchResult, SlicedGraph, Symbol


def find_cycles_tarjan(
    graph: Dict[str, List[str]], include_self_loops: bool = False
) -> List[List[str]]:
    """
    Find strongly connected components with size > 1 (cycles) using Tarjan's algorithm.
    Deterministic O(V + E) runtime with sorted node traversal.
    """
    index = 0
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    cycles: List[List[str]] = []

    def strongconnect(node: str):
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == node:
                    break

            if len(component) > 1:
                # Rotate cycle so minimum element is first for determinism
                min_idx = component.index(min(component))
                rotated = component[min_idx:] + component[:min_idx]
                cycles.append(rotated)
            elif include_self_loops and len(component) == 1:
                if node in graph.get(node, []):
                    cycles.append(component)

    for node in sorted(graph.keys()):
        if node not in indices:
            strongconnect(node)

    # Sort cycles by their starting element
    cycles.sort(key=lambda c: c[0] if c else "")
    return cycles


def verify_symbolic_gate(
    patch_result: PatchResult,
    slice_graph: SlicedGraph,
    indexer: WorkspaceIndexer,
) -> GateResult:
    """
    Deterministic Symbolic Gate (Stage 4):
    Verifies:
      1. Syntax validity of patched content.
      2. Absence of cyclic dependency / circular calls (Tarjan SCC).
      3. Parameter arity and keyword invariants against callers.
      4. Deleted symbol references (callers and importers).
    Runs deterministically in sub-2ms.
    """
    violations: List[str] = []
    cycles_detected: List[List[str]] = []

    # 1. Syntax check
    if patch_result.syntax_error:
        violations.append(f"SYNTAX_ERROR: {patch_result.syntax_error}")
        return GateResult(
            status="REJECTED",
            confidence=1.0,
            cycles=[],
            violations=violations,
            details={"error_type": "SYNTAX_ERROR"},
        )

    # 2. Cycle detection via Tarjan's SCC
    # Build call graph adjacency list from sliced subgraph
    call_graph: Dict[str, List[str]] = {}
    for node_id in slice_graph.nodes:
        call_graph.setdefault(node_id, [])

    for edge in slice_graph.edges:
        if edge.relation == "CALLS":
            call_graph.setdefault(edge.source, []).append(edge.target)

    # Ensure calls inside modified/added symbols are included in graph
    for sym in patch_result.affected_symbols + patch_result.added_symbols:
        u_id = sym.id
        call_graph.setdefault(u_id, [])
        for call in sym.calls:
            callee_def = indexer.resolve_callee(call, caller_sym=sym)
            if callee_def:
                v_id = callee_def.id
                if v_id not in call_graph[u_id]:
                    call_graph[u_id].append(v_id)

    raw_cycles = find_cycles_tarjan(call_graph)
    if raw_cycles:
        cycles_detected = raw_cycles
        for cycle in raw_cycles:
            cycle_repr = " -> ".join([c.split("::")[-1] for c in cycle] + [cycle[0].split("::")[-1]])
            violations.append(f"CIRCULAR_DEPENDENCY: Detected call cycle: {cycle_repr}")

    # 3. Contract / Arity Invariant Checks
    for sym in patch_result.affected_symbols + patch_result.added_symbols:
        if sym.kind not in ("function", "async_function", "method"):
            continue

        # If method (not @staticmethod), the first param ('self'/'cls') is bound at runtime
        if sym.is_method:
            formal_params = sym.params[1:] if len(sym.params) > 0 else []
        else:
            formal_params = sym.params[:]

        pos_params = [
            p for p in formal_params if not p.is_kwonly and not p.is_vararg and not p.is_kwarg
        ]
        kwonly_params = [p for p in formal_params if p.is_kwonly]
        has_vararg = any(p.is_vararg for p in formal_params)
        has_kwarg = any(p.is_kwarg for p in formal_params)

        effective_max_args = None if has_vararg else len(pos_params)
        effective_min_args = len([p for p in pos_params if not p.has_default])

        # Collect callers from indexer and slice
        callers = indexer.get_callers(sym.qualname)
        if sym.qualname != sym.name:
            for c in indexer.get_callers(sym.name):
                if c not in callers:
                    callers.append(c)

        # Deduplicate callers
        seen_calls: Set[tuple] = set()

        for call in callers:
            # Skip self-calls from within the same modified symbol to avoid duplicate checks
            if call.caller == sym.id and call.lineno >= sym.lineno and call.lineno <= sym.end_lineno:
                pass

            call_key = (call.caller, call.lineno, call.args_count, tuple(call.kwargs))
            if call_key in seen_calls:
                continue
            seen_calls.add(call_key)

            caller_label = call.caller or "unknown_caller"

            # Check max positional args
            if effective_max_args is not None and call.args_count > effective_max_args:
                violations.append(
                    f"ARITY_MISMATCH: Caller '{caller_label}' (line {call.lineno}) calls "
                    f"'{sym.qualname}' with {call.args_count} positional arguments, "
                    f"but '{sym.qualname}' accepts at most {effective_max_args} positional arguments."
                )

            # Check which positional arguments were supplied
            supplied_pos_params = pos_params[: min(call.args_count, len(pos_params))]
            unsatisfied_pos_params = pos_params[min(call.args_count, len(pos_params)) :]

            # Check duplicate arguments (passed positionally and by keyword)
            supplied_pos_names = {p.name for p in supplied_pos_params}
            for kw in call.kwargs:
                if kw in supplied_pos_names:
                    violations.append(
                        f"DUPLICATE_ARGUMENT: Caller '{caller_label}' (line {call.lineno}) provides "
                        f"multiple values for argument '{kw}' when calling '{sym.qualname}'."
                    )

            # Check positional-only parameter called as keyword
            for p in pos_params:
                if p.is_posonly and p.name in call.kwargs:
                    violations.append(
                        f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) passed "
                        f"positional-only argument '{p.name}' as keyword when calling '{sym.qualname}'."
                    )

            # Check unexpected keyword arguments
            if not has_kwarg:
                accepted_names = {p.name for p in pos_params if not p.is_posonly} | {
                    p.name for p in kwonly_params
                }
                for kw in call.kwargs:
                    if kw not in accepted_names:
                        violations.append(
                            f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) calls "
                            f"'{sym.qualname}' with unexpected keyword argument '{kw}'."
                        )

            # Check missing required positional arguments
            for p in unsatisfied_pos_params:
                if not p.has_default and p.name not in call.kwargs:
                    violations.append(
                        f"ARITY_MISMATCH: Caller '{caller_label}' (line {call.lineno}) missing "
                        f"required argument '{p.name}' when calling '{sym.qualname}' "
                        f"(requires at least {effective_min_args} arguments)."
                    )

            # Check missing required keyword-only arguments
            for p in kwonly_params:
                if not p.has_default and p.name not in call.kwargs:
                    violations.append(
                        f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) missing "
                        f"required keyword argument '{p.name}' when calling '{sym.qualname}'."
                    )

    # 4. Deleted Symbol Invariant: Check if deleted symbols are called or imported elsewhere
    for del_sym in patch_result.deleted_symbols:
        # Check callers
        callers = indexer.get_callers(del_sym.name)
        if del_sym.qualname != del_sym.name:
            callers.extend(indexer.get_callers(del_sym.qualname))

        active_callers = [
            c for c in callers
            if not (c.caller and c.caller.startswith(f"{patch_result.file_path}::"))
        ]
        for c in active_callers:
            violations.append(
                f"BROKEN_REFERENCE: Symbol '{del_sym.qualname}' was deleted in patch, "
                f"but is called by '{c.caller or 'unknown'}' at line {c.lineno}."
            )

        # Check importers
        importers = indexer.get_importers(del_sym.name)
        if del_sym.qualname != del_sym.name:
            importers.extend(indexer.get_importers(del_sym.qualname))

        active_importers = [
            imp for imp in importers
            if imp.file_path != patch_result.file_path
        ]
        for imp in active_importers:
            violations.append(
                f"BROKEN_REFERENCE: Symbol '{del_sym.qualname}' was deleted in patch, "
                f"but is imported by '{imp.file_path}' at line {imp.lineno}."
            )

    # 5. Verdict
    if violations or cycles_detected:
        status = "REJECTED"
        confidence = 0.95
    else:
        status = "APPROVED"
        confidence = 0.98

    return GateResult(
        status=status,
        confidence=confidence,
        cycles=cycles_detected,
        violations=violations,
        details={
            "nodes_checked": len(slice_graph.nodes),
            "edges_checked": len(slice_graph.edges),
            "cycles_count": len(cycles_detected),
            "violations_count": len(violations),
        },
    )
