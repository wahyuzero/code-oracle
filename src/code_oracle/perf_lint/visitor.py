"""
Multi-Language Tree-sitter AST Visitor for Performance Anti-Patterns & Resource Leaks.
Inspects concrete syntax trees across Python, TypeScript/JavaScript, Go, and Rust for:
- PERF001: Nested Loops Complexity (O(N^2) warning, O(N^3) error)
- PERF002: N+1 I/O & database calls inside loop bodies
- PERF003: Resource Leak / Unclosed Descriptors
- PERF004: Blocking Synchronous Calls in Async Context
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tree_sitter import Language, Node, Parser
import tree_sitter_go
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript

from code_oracle.languages import detect_language
from code_oracle.languages.go import get_go_parser
from code_oracle.languages.rust import get_rust_parser
from code_oracle.languages.typescript import get_ts_parser
from code_oracle.perf_lint.models import PerfDiagnostic, PerfRule, Severity
from code_oracle.perf_lint.rules.async_blocking import (
    AsyncBlockingRule,
    check_async_blocking,
)
from code_oracle.perf_lint.rules.n_plus_one import (
    DB_IO_METHODS,
    DIRECT_IO_FUNCS,
    HTTP_IO_PREFIXES,
    NPlusOneRule,
    check_n_plus_one,
    extract_method_name,
)
from code_oracle.perf_lint.rules.nested_loops import (
    FUNCTION_NODE_TYPES,
    LOOP_NODE_TYPES,
    NestedLoopsRule,
    check_nested_loop,
)
from code_oracle.perf_lint.rules.unclosed_res import (
    UnclosedResourceRule,
    check_unclosed_resource,
)

# Language parsers cache
_PY_LANG = Language(tree_sitter_python.language())
_PY_PARSER = Parser(_PY_LANG)


def get_parser(language: str, file_path: str = "") -> Optional[Parser]:
    """Retrieve Tree-sitter parser for target language."""
    if language == "python":
        return _PY_PARSER
    elif language in ("typescript", "javascript"):
        return get_ts_parser(file_path)
    elif language == "go":
        return get_go_parser()
    elif language == "rust":
        return get_rust_parser()
    return None


# Backward-compatible helper aliases
_extract_method_name = extract_method_name
_is_perf002_io_call = NPlusOneRule.is_io_call
_is_perf003_open_call = UnclosedResourceRule.is_open_call
_is_scoped_by_context_manager = UnclosedResourceRule.is_scoped
_is_perf004_blocking_call = AsyncBlockingRule.is_blocking_call
_is_async_func_node = AsyncBlockingRule.is_async_func_node

# Call node types per language
CALL_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {"call"},
    "typescript": {"call_expression"},
    "javascript": {"call_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression"},
}

# Inline suppression pattern
SUPPRESS_PATTERN = re.compile(
    r"(?:#|//|/\*)\s*code-oracle:\s*ignore-perf(?:\s*[\(\[]([A-Za-z0-9_,\s-]+)[\)\]])?",
    re.IGNORECASE,
)


def _extract_callee_text(node: Node, source_bytes: bytes) -> str:
    """Extract callee name or expression from a call node."""
    fn_node = node.child_by_field_name("function")
    if fn_node is not None:
        return source_bytes[fn_node.start_byte : fn_node.end_byte].decode("utf-8", errors="replace").strip()
    if node.children:
        return source_bytes[node.children[0].start_byte : node.children[0].end_byte].decode("utf-8", errors="replace").strip()
    return ""


def _get_func_name(node: Node, source_bytes: bytes, lang: str) -> str:
    """Extract name of function node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="replace").strip()

    # In TS/JS arrow function assigned to variable
    if lang in ("typescript", "javascript") and node.parent and node.parent.type == "variable_declarator":
        v_name = node.parent.child_by_field_name("name")
        if v_name is not None:
            return source_bytes[v_name.start_byte : v_name.end_byte].decode("utf-8", errors="replace").strip()

    return "<anonymous>"


def _is_suppressed(
    lines: List[str],
    lineno: int,
    rule_id: str,
    extra_lines: Optional[List[int]] = None,
) -> bool:
    """
    Check if a diagnostic is suppressed via inline comments:
    `# code-oracle: ignore-perf` or `// code-oracle: ignore-perf`
    Checks target line, preceding line (if comment-only), and optional extra lines (like root loop).
    """
    check_lines = [lineno]
    if lineno > 1:
        prev_line = lines[lineno - 2].strip()
        if prev_line.startswith(("#", "//", "/*")):
            check_lines.append(lineno - 1)

    if extra_lines:
        for el in extra_lines:
            check_lines.append(el)
            if el > 1:
                prev_el = lines[el - 2].strip()
                if prev_el.startswith(("#", "//", "/*")):
                    check_lines.append(el - 1)

    for l_num in set(check_lines):
        if 1 <= l_num <= len(lines):
            line_text = lines[l_num - 1]
            for match in SUPPRESS_PATTERN.finditer(line_text):
                specified = match.group(1)
                if not specified:
                    return True
                rules = [r.strip().upper() for r in specified.split(",")]
                if rule_id.upper() in rules:
                    return True

    return False


class PerfLintVisitor:
    """
    Single-pass multi-language Tree-sitter AST visitor for performance anti-patterns.
    Evaluates PERF001, PERF002, PERF003, and PERF004 across Python, TS, Go, and Rust.
    """

    def __init__(
        self,
        source: str,
        file_path: str = "",
        language: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        self.source = source
        self.source_bytes = source.encode("utf-8")
        self.file_path = file_path
        self.language = language or detect_language(file_path) or "python"
        self.max_depth = max_depth if max_depth is not None else 2
        self.lines = source.splitlines()
        self.diagnostics: List[PerfDiagnostic] = []

        # Context stacks during traversal
        self.loop_stack: List[Node] = []
        self.func_stack: List[Tuple[str, bool, Node]] = []  # (name, is_async, node)

    def run(self) -> List[PerfDiagnostic]:
        """Execute AST visitor pass and return collected diagnostics."""
        if not self.source.strip():
            return []

        parser = get_parser(self.language, self.file_path)
        if parser is None:
            return []

        tree = parser.parse(self.source_bytes)
        root = tree.root_node

        self._visit(root)
        return self.diagnostics

    def _visit(self, node: Node) -> None:
        """Recursive AST node visitor."""
        lang = self.language
        is_loop = NestedLoopsRule.is_loop_node(node, lang)
        is_func = NestedLoopsRule.is_function_boundary(node, lang)
        is_call = node.type in CALL_NODE_TYPES.get(lang, set())

        # Handle function scope boundaries
        outer_loops: Optional[List[Node]] = None
        if is_func:
            is_async = AsyncBlockingRule.is_async_func_node(node, lang)
            func_name = _get_func_name(node, self.source_bytes, lang)
            self.func_stack.append((func_name, is_async, node))
            # Reset loop depth per function scope
            outer_loops = self.loop_stack
            self.loop_stack = []

        # Handle loop entrance
        loop_clauses: List[Node] = []
        if is_loop:
            loop_clauses = NestedLoopsRule.get_loop_clauses(node, lang)
            for clause in loop_clauses:
                self.loop_stack.append(clause)
                current_depth = len(self.loop_stack)

                # PERF001: Nested Loops Complexity
                diag = NestedLoopsRule.check(
                    node=clause,
                    depth=current_depth,
                    max_depth=self.max_depth,
                    file_path=self.file_path,
                    lines=self.lines,
                )
                if diag:
                    enclosing_lines = [n.start_point.row + 1 for n in self.loop_stack]
                    if not _is_suppressed(self.lines, diag.lineno, PerfRule.PERF001.value, enclosing_lines):
                        self.diagnostics.append(diag)

        # Handle function calls
        if is_call:
            self._check_call(node)

        # Recurse children
        for child in node.children:
            self._visit(child)

        # Cleanup loop exit
        if is_loop:
            for _ in loop_clauses:
                self.loop_stack.pop()

        # Cleanup function exit
        if is_func:
            self.func_stack.pop()
            if outer_loops is not None:
                self.loop_stack = outer_loops

    def _check_call(self, call_node: Node) -> None:
        """Inspect a call node for PERF002, PERF003, and PERF004."""
        callee_text = _extract_callee_text(call_node, self.source_bytes)
        if not callee_text:
            return

        # --- PERF002: N+1 I/O in Loop Bodies ---
        if len(self.loop_stack) > 0:
            diag = NPlusOneRule.check(
                call_node=call_node,
                callee_text=callee_text,
                in_loop=True,
                file_path=self.file_path,
                lines=self.lines,
            )
            if diag:
                enclosing_loop_lines = [n.start_point.row + 1 for n in self.loop_stack]
                if not _is_suppressed(self.lines, diag.lineno, PerfRule.PERF002.value, enclosing_loop_lines):
                    self.diagnostics.append(diag)

        # --- PERF003: Resource Leak / Unclosed Descriptors ---
        diag = UnclosedResourceRule.check(
            call_node=call_node,
            callee_text=callee_text,
            language=self.language,
            source_bytes=self.source_bytes,
            file_path=self.file_path,
            lines=self.lines,
        )
        if diag:
            if not _is_suppressed(self.lines, diag.lineno, PerfRule.PERF003.value):
                self.diagnostics.append(diag)

        # --- PERF004: Blocking Synchronous Calls in Async Context ---
        if self.func_stack:
            current_func_name, is_async, _ = self.func_stack[-1]
            diag = AsyncBlockingRule.check(
                call_node=call_node,
                callee_text=callee_text,
                is_async_context=is_async,
                current_func_name=current_func_name,
                language=self.language,
                file_path=self.file_path,
                lines=self.lines,
            )
            if diag:
                if not _is_suppressed(self.lines, diag.lineno, PerfRule.PERF004.value):
                    self.diagnostics.append(diag)
