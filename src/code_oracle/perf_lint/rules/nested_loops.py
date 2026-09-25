"""
PERF001: Nested Loops Complexity Rule.
Detects nested loop depth >= 2 for O(N^2) warning, >= 3 for O(N^3) error
across Python, TypeScript/JavaScript, Go, and Rust.
"""

from typing import Dict, List, Optional, Set
from tree_sitter import Node

from code_oracle.perf_lint.models import PerfDiagnostic, PerfRule, Severity

COMPREHENSION_NODE_TYPES: Set[str] = {
    "list_comprehension",
    "dictionary_comprehension",
    "set_comprehension",
    "generator_expression",
}

LOOP_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {
        "for_statement",
        "while_statement",
        "list_comprehension",
        "dictionary_comprehension",
        "set_comprehension",
        "generator_expression",
    },
    "typescript": {
        "for_statement",
        "for_in_statement",
        "for_of_statement",
        "while_statement",
        "do_statement",
    },
    "javascript": {
        "for_statement",
        "for_in_statement",
        "for_of_statement",
        "while_statement",
        "do_statement",
    },
    "go": {"for_statement"},
    "rust": {"for_expression", "while_expression", "loop_expression"},
}

FUNCTION_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {"function_definition"},
    "typescript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function_declaration",
    },
    "javascript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function_declaration",
    },
    "go": {"function_declaration", "method_declaration", "func_literal"},
    "rust": {"function_item", "closure_expression"},
}


class NestedLoopsRule:
    """Evaluates PERF001: Loop Complexity Escalation."""

    RULE_ID = PerfRule.PERF001.value

    @staticmethod
    def is_loop_node(node: Node, language: str) -> bool:
        """Check if AST node is a loop in target language."""
        return node.type in LOOP_NODE_TYPES.get(language, set())

    @staticmethod
    def get_loop_clauses(node: Node, language: str) -> List[Node]:
        """Return loop clauses for compound loops like Python comprehensions."""
        if language == "python" and node.type in COMPREHENSION_NODE_TYPES:
            clauses = [c for c in node.children if c.type == "for_in_clause"]
            return clauses if clauses else [node]
        return [node]

    @staticmethod
    def is_function_boundary(node: Node, language: str) -> bool:
        """Check if AST node defines a new function boundary that resets loop depth."""
        return node.type in FUNCTION_NODE_TYPES.get(language, set())

    @classmethod
    def check(
        cls,
        node: Node,
        depth: int,
        max_depth: int,
        file_path: str,
        lines: List[str],
    ) -> Optional[PerfDiagnostic]:
        """
        Evaluate node at given loop depth against max_depth threshold.
        Returns PerfDiagnostic if depth >= 2 and depth >= max_depth.
        """
        if depth < 2 or depth < max_depth:
            return None

        lineno = node.start_point.row + 1
        end_lineno = node.end_point.row + 1
        col = node.start_point.column
        end_col = node.end_point.column

        severity = Severity.WARN if depth == 2 else Severity.ERROR
        complexity = f"O(N^{depth})"
        msg = f"Nested loop complexity {complexity} detected at depth {depth}"
        ctx = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else None

        return PerfDiagnostic(
            rule_id=cls.RULE_ID,
            message=msg,
            severity=severity,
            file_path=file_path,
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=col,
            end_col_offset=end_col,
            context_line=ctx,
        )


def check_nested_loop(
    node: Node,
    depth: int,
    max_depth: int,
    file_path: str,
    lines: List[str],
) -> Optional[PerfDiagnostic]:
    """Convenience helper for PERF001 evaluation."""
    return NestedLoopsRule.check(node, depth, max_depth, file_path, lines)
