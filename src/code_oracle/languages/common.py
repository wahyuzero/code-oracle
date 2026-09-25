"""
Common Tree-sitter AST utilities and helper functions for language extractors.
"""

from typing import List, Optional
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


def extract_preceding_docstring(node: Node, source_bytes: bytes) -> Optional[str]:
    """
    Extract preceding docstring or doc-comment block for an AST node.
    Inspects previous siblings for comment nodes (e.g. /** ... */, /// ..., or // ...).
    """
    target = node
    # If wrapped in export statement or similar, check outer node
    if target.parent and target.parent.type in (
        "export_statement",
        "export_default_statement",
        "ambient_declaration",
    ):
        target = target.parent

    comments: List[str] = []

    # In tree-sitter, iterate backwards through parent's children to find adjacent comments
    if target.parent:
        children = target.parent.children
        try:
            idx = children.index(target)
            i = idx - 1
            while i >= 0:
                prev_node = children[i]
                if prev_node.type in ("comment", "line_comment", "block_comment"):
                    raw_text = get_node_text(prev_node, source_bytes)
                    comments.insert(0, raw_text)
                    i -= 1
                elif prev_node.type in (",", ";", "\n"):
                    i -= 1
                else:
                    break
        except ValueError:
            pass

    if not comments:
        curr = target.prev_named_sibling or target.prev_sibling
        if curr and "comment" in curr.type:
            comments.append(get_node_text(curr, source_bytes))

    if not comments:
        return None

    cleaned_lines: List[str] = []
    for c in comments:
        c_str = c.strip()
        if c_str.startswith("/*"):
            # Block comment / JSDoc
            c_str = c_str.removeprefix("/*").removesuffix("*/")
            for line in c_str.splitlines():
                line = line.strip()
                if line.startswith("*"):
                    line = line[1:].strip()
                if line:
                    cleaned_lines.append(line)
        elif c_str.startswith("///"):
            # Rust doc comment
            for line in c_str.splitlines():
                line = line.strip()
                if line.startswith("///"):
                    line = line[3:].strip()
                if line:
                    cleaned_lines.append(line)
        elif c_str.startswith("//"):
            # Line comment
            for line in c_str.splitlines():
                line = line.strip()
                if line.startswith("//"):
                    line = line[2:].strip()
                if line:
                    cleaned_lines.append(line)
        elif c_str.startswith("#"):
            # Python/shell comment
            for line in c_str.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    line = line[1:].strip()
                if line:
                    cleaned_lines.append(line)

    return "\n".join(cleaned_lines) if cleaned_lines else None

