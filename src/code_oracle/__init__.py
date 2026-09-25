"""
Code Oracle: Sub-50ms Neuro-Symbolic Verification Oracle for AI Coding Agents.
"""

from code_oracle.dead_code import detect_dead_code
from code_oracle.engine import TopoSliceEngine
from code_oracle.indexer import WorkspaceIndexer
from code_oracle.linearizer import estimate_tokens, linearize_subgraph
from code_oracle.locator import extract_symbols_from_ast, locate_affected_symbols
from code_oracle.server import verify_patch
from code_oracle.slicer import slice_neighborhood
from code_oracle.symbolic import find_cycles_tarjan, verify_symbolic_gate

__version__ = "0.1.0"
__author__ = "Wahyu Febri Tamtomo"

__all__ = [
    "TopoSliceEngine",
    "verify_patch",
    "detect_dead_code",
    "WorkspaceIndexer",
    "find_cycles_tarjan",
    "verify_symbolic_gate",
    "locate_affected_symbols",
    "extract_symbols_from_ast",
    "slice_neighborhood",
    "linearize_subgraph",
    "estimate_tokens",
    "__version__",
]
