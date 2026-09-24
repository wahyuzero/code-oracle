"""
Deterministic Symbolic Gate: Graph Cycle Detection and Invariant Checks.
"""

from typing import Dict, List, Set

def find_cycles_tarjan(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Find strongly connected components with size > 1 (cycles) using Tarjan's algorithm.
    Deterministic O(V + E) runtime.
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

        for neighbor in graph.get(node, []):
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
                cycles.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return cycles
