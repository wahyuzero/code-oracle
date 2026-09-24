"""
Deterministic Symbolic Gate: Graph Cycle Detection and Invariant Checks.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import CallReference, GateResult, PatchResult, SlicedGraph, Symbol


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


def validate_call_site(
    call: CallReference,
    callee_sym: Symbol,
    caller_label: str,
) -> List[str]:
    """
    Validate a single call site against the target symbol's signature.
    Returns a list of violation messages.
    """
    violations: List[str] = []

    # If method (not @staticmethod), the first param ('self'/'cls') is bound at runtime
    if callee_sym.is_method:
        formal_params = callee_sym.params[1:] if len(callee_sym.params) > 0 else []
    else:
        formal_params = callee_sym.params[:]

    pos_params = [
        p for p in formal_params if not p.is_kwonly and not p.is_vararg and not p.is_kwarg
    ]
    kwonly_params = [p for p in formal_params if p.is_kwonly]
    has_vararg = any(p.is_vararg for p in formal_params)
    has_kwarg = any(p.is_kwarg for p in formal_params)

    effective_max_args = None if has_vararg else len(pos_params)
    effective_min_args = len([p for p in pos_params if not p.has_default])

    # Check max positional args
    if not call.has_vararg and effective_max_args is not None and call.args_count > effective_max_args:
        violations.append(
            f"ARITY_MISMATCH: Caller '{caller_label}' (line {call.lineno}) calls "
            f"'{callee_sym.qualname}' with {call.args_count} positional arguments, "
            f"but '{callee_sym.qualname}' accepts at most {effective_max_args} positional arguments."
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
                f"multiple values for argument '{kw}' when calling '{callee_sym.qualname}'."
            )

    # Check positional-only parameter called as keyword
    posonly_names = {p.name for p in pos_params if p.is_posonly}
    for p in pos_params:
        if p.is_posonly and p.name in call.kwargs:
            violations.append(
                f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) passed "
                f"positional-only argument '{p.name}' as keyword when calling '{callee_sym.qualname}'."
            )

    # Check unexpected keyword arguments
    if not has_kwarg:
        accepted_names = {p.name for p in pos_params if not p.is_posonly} | {
            p.name for p in kwonly_params
        }
        for kw in call.kwargs:
            if kw not in accepted_names and kw not in posonly_names:
                violations.append(
                    f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) calls "
                    f"'{callee_sym.qualname}' with unexpected keyword argument '{kw}'."
                )

    # Check missing required positional arguments (if not satisfied via kwargs or dynamic unpacking)
    if not call.has_vararg:
        for p in unsatisfied_pos_params:
            if not p.has_default and p.name not in call.kwargs:
                if not (call.has_kwarg and not p.is_posonly):
                    violations.append(
                        f"ARITY_MISMATCH: Caller '{caller_label}' (line {call.lineno}) missing "
                        f"required argument '{p.name}' when calling '{callee_sym.qualname}' "
                        f"(requires at least {effective_min_args} arguments)."
                    )

    # Check missing required keyword-only arguments
    if not call.has_kwarg:
        for p in kwonly_params:
            if not p.has_default and p.name not in call.kwargs:
                violations.append(
                    f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) missing "
                    f"required keyword argument '{p.name}' when calling '{callee_sym.qualname}'."
                )

    return violations


def verify_symbolic_gate(
    patch_result: PatchResult,
    slice_graph: SlicedGraph,
    indexer: WorkspaceIndexer,
) -> GateResult:
    """
    Deterministic Symbolic Gate (Stage 4):
    Verifies:
      1. Syntax validity of patched content.
      2. Absence of cyclic dependency / circular calls and imports (Tarjan SCC).
      3. Parameter arity and keyword invariants across callers and callees.
      4. Deleted symbol references (callers and importers) and dangling imports.
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
    # 2A. Call graph cycle detection
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

    raw_call_cycles = find_cycles_tarjan(call_graph)
    if raw_call_cycles:
        cycles_detected.extend(raw_call_cycles)
        for cycle in raw_call_cycles:
            cycle_repr = " -> ".join([c.split("::")[-1] for c in cycle] + [cycle[0].split("::")[-1]])
            violations.append(f"CIRCULAR_DEPENDENCY: Detected call cycle: {cycle_repr}")

    # 2B. Import cycle detection (direct and multi-hop across workspace)
    import_graph: Dict[str, List[str]] = {}
    available_files = set(indexer._file_cache.keys())
    for f_path, f_data in indexer._file_cache.items():
        import_graph.setdefault(f_path, [])
        for imp_data in f_data.get("imports", []):
            imp_obj = indexer._deserialize_import(imp_data)
            target_f = indexer.resolve_import_to_file(imp_obj, f_path)
            if target_f and target_f != f_path:
                if target_f not in import_graph[f_path]:
                    import_graph[f_path].append(target_f)

    raw_import_cycles = find_cycles_tarjan(import_graph)
    for cycle in raw_import_cycles:
        if patch_result.file_path in cycle:
            if cycle not in cycles_detected:
                cycles_detected.append(cycle)
            cycle_repr = " -> ".join(cycle + [cycle[0]])
            violations.append(f"CIRCULAR_DEPENDENCY: Detected import cycle: {cycle_repr}")

    # 3. Contract / Arity Invariant Checks (Bidirectional)
    seen_call_sites: Set[tuple] = set()

    # 3A. Existing callers calling modified/added symbols
    for sym in patch_result.affected_symbols + patch_result.added_symbols:
        if sym.kind not in ("function", "async_function", "method"):
            continue

        callers = indexer.get_callers(sym.qualname)
        if sym.qualname != sym.name:
            for c in indexer.get_callers(sym.name):
                if c not in callers:
                    callers.append(c)

        # If __init__ or constructor method of a class, callers might be instantiating the class by its class name
        if sym.name in ("__init__", "constructor") and "." in sym.qualname:
            class_qualname = sym.qualname.rsplit(".", 1)[0]
            for c in indexer.get_callers(class_qualname):
                if c not in callers:
                    callers.append(c)
            class_simple = class_qualname.split(".")[-1]
            if class_simple != class_qualname:
                for c in indexer.get_callers(class_simple):
                    if c not in callers:
                        callers.append(c)

        for call in callers:
            caller_label = call.caller or "unknown_caller"
            call_key = (caller_label, call.lineno, sym.id, call.args_count, tuple(sorted(call.kwargs)))
            if call_key in seen_call_sites:
                continue
            seen_call_sites.add(call_key)

            violations.extend(validate_call_site(call, sym, caller_label))

    # 3B. Calls MADE BY modified/added symbols (or module level)
    for caller_sym in patch_result.affected_symbols + patch_result.added_symbols:
        caller_label = caller_sym.id
        for call in caller_sym.calls:
            callee_def = indexer.resolve_callee(call, caller_sym=caller_sym)
            if not callee_def:
                continue

            target_sym = callee_def
            if callee_def.kind == "class":
                # Class instantiation invokes __init__
                init_def = indexer.resolve_class_init(callee_def)
                if init_def:
                    target_sym = init_def
                else:
                    # Class without custom __init__ accepts 0 arguments
                    call_key = (caller_label, call.lineno, callee_def.id, call.args_count, tuple(sorted(call.kwargs)))
                    if call_key not in seen_call_sites:
                        seen_call_sites.add(call_key)
                        if not call.has_vararg and call.args_count > 0:
                            violations.append(
                                f"ARITY_MISMATCH: Caller '{caller_label}' (line {call.lineno}) calls "
                                f"'{callee_def.qualname}' with {call.args_count} positional arguments, "
                                f"but '{callee_def.qualname}' accepts at most 0 positional arguments."
                            )
                        if not call.has_kwarg and call.kwargs:
                            for kw in call.kwargs:
                                violations.append(
                                    f"KEYWORD_MISMATCH: Caller '{caller_label}' (line {call.lineno}) calls "
                                    f"'{callee_def.qualname}' with unexpected keyword argument '{kw}'."
                                )
                    continue

            if target_sym.kind in ("function", "async_function", "method"):
                call_key = (caller_label, call.lineno, target_sym.id, call.args_count, tuple(sorted(call.kwargs)))
                if call_key in seen_call_sites:
                    continue
                seen_call_sites.add(call_key)

                violations.extend(validate_call_site(call, target_sym, caller_label))

    # 4. Deleted Symbol & Broken Reference Invariants
    active_caller_ids = {s.id for s in patch_result.all_patched_symbols}
    for del_sym in patch_result.deleted_symbols:
        # Check callers
        callers = indexer.get_callers(del_sym.name)
        if del_sym.qualname != del_sym.name:
            callers.extend(indexer.get_callers(del_sym.qualname))

        active_callers = []
        for c in callers:
            if not c.caller:
                continue
            if c.caller.startswith(f"{patch_result.file_path}::"):
                if c.caller in active_caller_ids:
                    active_callers.append(c)
            else:
                active_callers.append(c)

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

    # 4B. Check if any symbol imported by the patch does not exist in workspace module
    for imp in patch_result.imports:
        if (imp.module or imp.level > 0) and imp.name != "*":
            target_f = indexer.resolve_import_to_file(imp, patch_result.file_path)
            if target_f and target_f in indexer._file_cache:
                cached = indexer._file_cache[target_f]
                defined_names = {s["name"] for s in cached.get("symbols", [])}
                imported_names = {i.get("asname") or i["name"] for i in cached.get("imports", [])}

                # If target module re-exports with star import, allow dynamic symbols
                if "*" in {i["name"] for i in cached.get("imports", [])}:
                    continue

                if imp.name not in defined_names and imp.name not in imported_names:
                    target_full = indexer.workspace_root / target_f
                    target_dir = target_full.parent
                    submod_cands = [
                        target_dir / f"{imp.name}.py",
                        target_dir / imp.name / "__init__.py",
                        target_dir / f"{imp.name}.ts",
                        target_dir / f"{imp.name}.tsx",
                        target_dir / f"{imp.name}.js",
                        target_dir / imp.name / "index.ts",
                        target_dir / imp.name / "index.js",
                        target_dir / f"{imp.name}.rs",
                        target_dir / imp.name / "mod.rs",
                        target_dir / f"{imp.name}.go",
                    ]
                    has_symbol = any(c.exists() for c in submod_cands)
                    if not has_symbol and target_full.exists():
                        if target_f.endswith(".py"):
                            try:
                                src = target_full.read_text(encoding="utf-8", errors="ignore")
                                tree = ast.parse(src)
                                for node in ast.walk(tree):
                                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                                        target_list = node.targets if isinstance(node, ast.Assign) else [node.target]
                                        for t in target_list:
                                            for child in ast.walk(t):
                                                if isinstance(child, ast.Name) and child.id == imp.name:
                                                    has_symbol = True
                                                    break
                                            if has_symbol:
                                                break
                                    elif isinstance(node, ast.NamedExpr):
                                        if isinstance(node.target, ast.Name) and node.target.id == imp.name:
                                            has_symbol = True
                                            break
                                    elif isinstance(node, getattr(ast, "TypeAlias", ())):
                                        if isinstance(node.name, ast.Name) and node.name.id == imp.name:
                                            has_symbol = True
                                            break
                                    if has_symbol:
                                        break
                            except Exception:
                                has_symbol = True
                        else:
                            # For TypeScript, Go, Rust: check extracted symbols from target file
                            try:
                                src = target_full.read_text(encoding="utf-8", errors="ignore")
                                from code_oracle.languages import extract_symbols
                                syms = extract_symbols(src, file_path=target_f)
                                if any(s.name == imp.name for s in syms):
                                    has_symbol = True
                            except Exception:
                                pass
                    if not has_symbol:
                        module_label = imp.module if imp.module else ("." * imp.level)
                        violations.append(
                            f"BROKEN_REFERENCE: Symbol '{imp.name}' imported from '{module_label}' "
                            f"does not exist in '{target_f}' (line {imp.lineno})."
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
