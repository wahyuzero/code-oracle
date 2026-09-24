"""
Common Tree-sitter AST utilities and helper functions for language extractors.
"""

from typing import Optional
from tree_sitter import Node


def find_first_error(node: Node) -> Optional[Node]:
    """Recursively locate the first syntax error node in a Tree-sitter AST."""
    if node.is_error or node.type == "ERROR" or node.is_missing:
        return node
    for child in node.children:
        if child.has_error:
            err = find_first_error(child)
            if err:
                return err
    return node if node.has_error else None


def format_syntax_error(root_node: Node, language: str) -> Optional[str]:
    """Format a descriptive syntax error message if the AST contains errors."""
    if not root_node.has_error:
        return None

    err_node = find_first_error(root_node)
    if err_node is None:
        return f"SyntaxError: Parsing failed for {language} source."

    row = err_node.start_point.row + 1
    col = err_node.start_point.column + 1
    snippet = err_node.text.decode("utf-8", errors="ignore").strip().replace("\n", " ")
    if len(snippet) > 40:
        snippet = snippet[:37] + "..."
    if not snippet:
        snippet = err_node.type

    return f"SyntaxError at line {row}:{col}: Unexpected {language} syntax near '{snippet}'"


def get_node_text(node: Node, source_bytes: bytes) -> str:
    """Safely extract decoded text slice for a Tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
