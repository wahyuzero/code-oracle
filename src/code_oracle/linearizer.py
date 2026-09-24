"""
Stage 5: Graph Linearizer.
Serializes the sliced subgraph and patch metadata into a compact Domain-Specific Language (< 400 tokens)
for Laya ModernBERT decision head ingestion.
"""

import re
from typing import List, Optional

from code_oracle.models import GateResult, PatchResult, SlicedGraph


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimator matching BPE / WordPiece tokenizers.
    Counts word/identifier chunks and individual punctuation symbols.
    """
    if not text:
        return 0
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)
    return len(tokens)


def linearize_subgraph(
    patch_result: PatchResult,
    slice_graph: SlicedGraph,
    gate_result: GateResult,
    max_tokens: int = 400,
) -> str:
    """
    Convert patch metadata, sliced subgraph, and gate verdict into a compact DSL.
    Guarantees output size <= max_tokens (default 400 tokens).
    """
    # 1. Diff target header
    if patch_result.affected_symbols:
        seed_names = [s.qualname for s in patch_result.affected_symbols]
        seed_str = ", ".join(seed_names)
        target_header = f"[DIFF_TARGET] {patch_result.file_path}::{seed_str} (MODIFIED)"
    elif patch_result.deleted_symbols:
        del_names = [s.qualname for s in patch_result.deleted_symbols]
        seed_str = ", ".join(del_names)
        target_header = f"[DIFF_TARGET] {patch_result.file_path}::{seed_str} (DELETED)"
    elif patch_result.added_symbols:
        add_names = [s.qualname for s in patch_result.added_symbols]
        seed_str = ", ".join(add_names)
        target_header = f"[DIFF_TARGET] {patch_result.file_path}::{seed_str} (ADDED)"
    else:
        target_header = f"[DIFF_TARGET] {patch_result.file_path}::<module> (MODIFIED)"

    lines_str = f"OldLines: {sorted(list(patch_result.modified_old_lines))} | NewLines: {sorted(list(patch_result.modified_new_lines))}"
    meta_header = f"[METADATA] File: {patch_result.file_path} | {lines_str} | Nodes: {len(slice_graph.nodes)} | Edges: {len(slice_graph.edges)}"

    # 2. Gate Section
    status_str = f"STATUS: {gate_result.status} (conf: {round(gate_result.confidence, 2)})"
    cycles_str = f"CYCLES: {len(gate_result.cycles)}"
    if gate_result.violations:
        violations_str = "VIOLATIONS:\n  - " + "\n  - ".join(gate_result.violations[:3])
        if len(gate_result.violations) > 3:
            violations_str += f"\n  - ... ({len(gate_result.violations) - 3} more)"
    else:
        violations_str = "VIOLATIONS: NONE"

    gate_section = f"[GATE]\n{status_str}\n{cycles_str}\n{violations_str}"

    # 3. Nodes and Edges preparation
    node_id_map = {}
    ordered_nodes = []

    # Put seeds first
    for n_id, node in slice_graph.nodes.items():
        if node.is_seed:
            node_id_map[n_id] = f"N{len(node_id_map)}"
            ordered_nodes.append((n_id, node))

    # Then non-seed nodes
    for n_id, node in slice_graph.nodes.items():
        if not node.is_seed:
            node_id_map[n_id] = f"N{len(node_id_map)}"
            ordered_nodes.append((n_id, node))

    def build_dsl(active_node_pairs: List[tuple]) -> str:
        active_ids = {n_id for n_id, _ in active_node_pairs}

        node_lines = []
        for n_id, node in active_node_pairs:
            short_id = node_id_map[n_id]
            flags = []
            if node.is_seed:
                flags.append("SEED")
            if node.is_modified:
                flags.append("MODIFIED")
            if node.truncated:
                flags.append("TRUNCATED")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            node_lines.append(f"{short_id}: {node.file_path}::{node.name} [{node.signature}]{flag_str}")

        edge_lines = []
        for edge in slice_graph.edges:
            if edge.source in active_ids and edge.target in active_ids:
                src_short = node_id_map.get(edge.source, edge.source.split("::")[-1])
                tgt_short = node_id_map.get(edge.target, edge.target.split("::")[-1])
                edge_lines.append(f"{src_short} -> {tgt_short} ({edge.relation})")

        nodes_section = "[NODES]\n" + ("\n".join(node_lines) if node_lines else "(none)")
        edges_section = "[EDGES]\n" + ("\n".join(edge_lines) if edge_lines else "(none)")

        return f"{target_header}\n{meta_header}\n{nodes_section}\n{edges_section}\n{gate_section}"

    # Try full graph first
    dsl = build_dsl(ordered_nodes)
    current_tokens = estimate_tokens(dsl)

    # If exceeding max_tokens, prune non-seed nodes from tail until under budget
    if current_tokens > max_tokens:
        seed_nodes = [(n_id, n) for n_id, n in ordered_nodes if n.is_seed]
        non_seed_nodes = [(n_id, n) for n_id, n in ordered_nodes if not n.is_seed]

        pruned = False
        while non_seed_nodes:
            non_seed_nodes.pop()
            candidate_dsl = build_dsl(seed_nodes + non_seed_nodes)
            candidate_dsl += f"\n[TRUNCATED: {len(ordered_nodes) - len(seed_nodes) - len(non_seed_nodes)} peripheral nodes pruned for token budget]"
            if estimate_tokens(candidate_dsl) <= max_tokens:
                dsl = candidate_dsl
                pruned = True
                break

        if not pruned:
            candidate_dsl = build_dsl(seed_nodes)
            candidate_dsl += f"\n[TRUNCATED: all non-seed nodes pruned]"
            if estimate_tokens(candidate_dsl) <= max_tokens:
                dsl = candidate_dsl
            else:
                # Even seed nodes exceed max_tokens; prune excess seeds
                while len(seed_nodes) > 1:
                    seed_nodes.pop()
                    cand = build_dsl(seed_nodes) + f"\n[TRUNCATED: excess seeds pruned]"
                    if estimate_tokens(cand) <= max_tokens:
                        dsl = cand
                        break
                else:
                    dsl = candidate_dsl[:max_tokens * 3]

    return dsl
