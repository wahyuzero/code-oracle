"""
Stage 3: k-Hop Neighborhood Slicer.
Isolates a compact directed subgraph (10 to 50 nodes) representing immediate callers,
callees, importers, and inheritance (k=1 or k=2), capped by a fan-out threshold to prevent graph explosion.
"""

from typing import Dict, List, Optional, Set

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import SlicedGraph, SliceEdge, SliceNode, Symbol


def slice_neighborhood(
    seeds: List[Symbol],
    indexer: WorkspaceIndexer,
    k: int = 1,
    max_fanout: int = 20,
    max_nodes: int = 50,
) -> SlicedGraph:
    """
    Extract a directed neighborhood subgraph around seed symbols.
    Expands forward (callees, base classes) and backward (callers, subclasses, importers) up to k hops,
    enforcing a strict degree cutoff (max_fanout) per node and computing the induced subgraph.
    """
    nodes: Dict[str, SliceNode] = {}
    edges: List[SliceEdge] = []
    seen_edges: Set[tuple] = set()
    seed_ids: Set[str] = set()
    graph_truncated = False

    # Initialize seed nodes (capped by max_nodes)
    if len(seeds) > max_nodes:
        seeds = seeds[:max_nodes]
        graph_truncated = True

    current_symbols: List[Symbol] = []
    for s in seeds:
        node = SliceNode(
            id=s.id,
            name=s.name,
            file_path=s.file_path,
            kind=s.kind,
            signature=s.signature,
            is_seed=True,
            is_modified=True,
            symbol=s,
        )
        nodes[s.id] = node
        seed_ids.add(s.id)
        current_symbols.append(s)

    visited_symbol_ids: Set[str] = set(seed_ids)

    # Breadth-first expansion up to k hops
    for hop in range(1, k + 1):
        next_symbols: List[Symbol] = []

        for curr_sym in current_symbols:
            u_id = curr_sym.id
            node_degree = 0

            # 1. Forward edges: Callees called by curr_sym
            for call in curr_sym.calls:
                callee_def = indexer.resolve_callee(call, caller_sym=curr_sym)
                if callee_def:
                    v_id = callee_def.id
                    if node_degree >= max_fanout:
                        nodes[u_id].truncated = True
                        graph_truncated = True
                        break

                    edge_key = (u_id, v_id, "CALLS")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(SliceEdge(source=u_id, target=v_id, relation="CALLS"))
                        node_degree += 1

                    if v_id not in nodes and len(nodes) < max_nodes:
                        nodes[v_id] = SliceNode(
                            id=v_id,
                            name=callee_def.name,
                            file_path=callee_def.file_path,
                            kind=callee_def.kind,
                            signature=callee_def.signature,
                            symbol=callee_def,
                        )
                        if v_id not in visited_symbol_ids:
                            visited_symbol_ids.add(v_id)
                            next_symbols.append(callee_def)

            # 2. Forward edges: Base classes (INHERITS)
            for base_name in getattr(curr_sym, "bases", []):
                base_def = indexer.get_definition(base_name)
                if base_def:
                    v_id = base_def.id
                    if node_degree >= max_fanout:
                        nodes[u_id].truncated = True
                        graph_truncated = True
                        break

                    edge_key = (u_id, v_id, "INHERITS")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(SliceEdge(source=u_id, target=v_id, relation="INHERITS"))
                        node_degree += 1

                    if v_id not in nodes and len(nodes) < max_nodes:
                        nodes[v_id] = SliceNode(
                            id=v_id,
                            name=base_def.name,
                            file_path=base_def.file_path,
                            kind=base_def.kind,
                            signature=base_def.signature,
                            symbol=base_def,
                        )
                        if v_id not in visited_symbol_ids:
                            visited_symbol_ids.add(v_id)
                            next_symbols.append(base_def)

            # 3. Backward edges: Callers that call curr_sym
            callers = indexer.get_callers(curr_sym.qualname)
            if curr_sym.name != curr_sym.qualname:
                for c in indexer.get_callers(curr_sym.name):
                    if c not in callers:
                        callers.append(c)

            for call in callers:
                if not call.caller:
                    continue

                caller_def = indexer.get_definition(call.caller)
                v_id = caller_def.id if caller_def else call.caller

                if node_degree >= max_fanout:
                    nodes[u_id].truncated = True
                    graph_truncated = True
                    break

                edge_key = (v_id, u_id, "CALLS")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(SliceEdge(source=v_id, target=u_id, relation="CALLS"))
                    node_degree += 1

                if v_id not in nodes and len(nodes) < max_nodes:
                    node_name = caller_def.name if caller_def else call.caller.split("::")[-1]
                    file_path = caller_def.file_path if caller_def else (
                        call.caller.split("::")[0] if "::" in call.caller else ""
                    )
                    signature = caller_def.signature if caller_def else f"def {node_name}(...)"
                    kind = caller_def.kind if caller_def else "function"

                    nodes[v_id] = SliceNode(
                        id=v_id,
                        name=node_name,
                        file_path=file_path,
                        kind=kind,
                        signature=signature,
                        symbol=caller_def,
                    )
                    if caller_def and v_id not in visited_symbol_ids:
                        visited_symbol_ids.add(v_id)
                        next_symbols.append(caller_def)

            # 4. Backward edges: Subclasses (INHERITS)
            subclasses = indexer.get_subclasses(curr_sym.name)
            if curr_sym.name != curr_sym.qualname:
                subclasses.extend(indexer.get_subclasses(curr_sym.qualname))
            for sub in subclasses:
                v_id = sub.id
                if node_degree >= max_fanout:
                    nodes[u_id].truncated = True
                    graph_truncated = True
                    break

                edge_key = (v_id, u_id, "INHERITS")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(SliceEdge(source=v_id, target=u_id, relation="INHERITS"))
                    node_degree += 1

                if v_id not in nodes and len(nodes) < max_nodes:
                    nodes[v_id] = SliceNode(
                        id=v_id,
                        name=sub.name,
                        file_path=sub.file_path,
                        kind=sub.kind,
                        signature=sub.signature,
                        symbol=sub,
                    )
                    if v_id not in visited_symbol_ids:
                        visited_symbol_ids.add(v_id)
                        next_symbols.append(sub)

        current_symbols = next_symbols
        if not current_symbols or len(nodes) >= max_nodes:
            break

    # 5. Induced Subgraph Completion: Connect any calls/inheritance between nodes already in the slice
    for u_id, u_node in list(nodes.items()):
        if not u_node.symbol:
            continue
        # Check calls
        for call in u_node.symbol.calls:
            target_def = indexer.resolve_callee(call, caller_sym=u_node.symbol)
            if target_def and target_def.id in nodes:
                edge_key = (u_id, target_def.id, "CALLS")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(SliceEdge(source=u_id, target=target_def.id, relation="CALLS"))
        # Check base classes
        for base_name in getattr(u_node.symbol, "bases", []):
            base_def = indexer.get_definition(base_name)
            if base_def and base_def.id in nodes:
                edge_key = (u_id, base_def.id, "INHERITS")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(SliceEdge(source=u_id, target=base_def.id, relation="INHERITS"))

    return SlicedGraph(
        nodes=nodes,
        edges=edges,
        seed_ids=seed_ids,
        truncated=graph_truncated,
    )
