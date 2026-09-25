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

# Language parsers cache
_PY_LANG = Language(tree_sitter_python.language())


def get_parser(language: str, file_path: str = "") -> Optional[Parser]:
    """Retrieve Tree-sitter parser for target language."""
    if language == "python":
        return Parser(_PY_LANG)
    elif language in ("typescript", "javascript"):
        return get_ts_parser(file_path)
    elif language == "go":
        return get_go_parser()
    elif language == "rust":
        return get_rust_parser()
    return None


# Loop node types per language
LOOP_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {"for_statement", "while_statement"},
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

# Function/closure node types per language
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

# Call node types per language
CALL_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {"call"},
    "typescript": {"call_expression"},
    "javascript": {"call_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression"},
}

# Database / ORM method names for PERF002 (N+1 query detection)
DB_IO_METHODS: Set[str] = {
    "query",
    "execute",
    "exec",
    "find",
    "find_one",
    "find_many",
    "find_first",
    "find_unique",
    "find_by",
    "select",
    "select_all",
    "fetch",
    "fetchall",
    "fetchone",
    "fetchmany",
    "raw_query",
    "queryrow",
}

# HTTP and network call prefixes for PERF002
HTTP_IO_PREFIXES: Tuple[str, ...] = (
    "requests.",
    "http.get",
    "http.post",
    "http.head",
    "http.postform",
    "http.do",
    "client.get",
    "client.post",
    "client.do",
    "axios.",
    "reqwest::",
    "urllib.request.",
    "aiohttp.",
    "httpx.",
)

DIRECT_IO_FUNCS: Set[str] = {
    "fetch",
    "query",
    "execute",
    "select",
}

# Inline suppression pattern
SUPPRESS_PATTERN = re.compile(
    r"(?:#|//|/\*)\s*code-oracle:\s*ignore-perf(?:\s*[\(\[]([A-Za-z0-9_-]+)[\)\]])?",
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


def _extract_method_name(callee_text: str) -> str:
    """Extract the last identifier (method name) from a callee expression."""
    clean = callee_text.strip().replace("\n", "")
    parts = re.split(r"\.|::", clean)
    if not parts:
        return ""
    last_part = parts[-1].strip()
    match = re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*", last_part)
    return match.group(0).lower() if match else last_part.lower()


def _is_perf002_io_call(callee_text: str) -> bool:
    """Check if callee represents database query or network I/O."""
    clean = callee_text.strip().replace("\n", "")
    clean_lower = clean.lower()
    method = _extract_method_name(clean)

    if method in DB_IO_METHODS:
        return True

    if clean_lower in DIRECT_IO_FUNCS:
        return True

    for prefix in HTTP_IO_PREFIXES:
        if clean_lower.startswith(prefix) or f".{prefix}" in clean_lower:
            return True

    return False


def _is_perf003_open_call(callee_text: str, lang: str) -> bool:
    """Check if callee is a resource open call that requires scoped cleanup."""
    clean = callee_text.strip()
    clean_lower = clean.lower()

    if lang == "python":
        if clean in ("open", "socket.socket", "socket.create_connection"):
            return True
        if clean.endswith((".connect", "connect")) and any(
            db in clean_lower for db in ("sqlite", "psycopg", "mysql", "asyncpg", "db", "conn")
        ):
            return True
        return False

    elif lang in ("typescript", "javascript"):
        return clean.startswith((
            "fs.open",
            "fs.createReadStream",
            "fs.createWriteStream",
            "net.connect",
            "net.createConnection",
            "tls.connect",
        ))

    elif lang == "go":
        return clean.startswith((
            "os.Open",
            "os.OpenFile",
            "os.Create",
            "net.Dial",
            "net.DialTimeout",
            "net.Listen",
            "sql.Open",
            "http.Get",
            "http.Post",
            "http.Head",
        ))

    elif lang == "rust":
        return clean in (
            "Box::leak",
            "std::mem::forget",
            "mem::forget",
            "std::mem::ManuallyDrop::new",
            "ManuallyDrop::new",
        )

    return False


def _is_scoped_by_context_manager(
    call_node: Node,
    callee_text: str,
    lang: str,
    source_bytes: bytes,
) -> bool:
    """
    Check if resource open call is properly scoped:
    - Python: within `with_clause` or `with_item` in `with_statement`
    - TypeScript/JS: within `try_statement` containing `finally_clause`
    - Go: within function containing `defer ...Close()`
    - Rust: Box::leak / mem::forget are always leaks
    """
    if lang == "python":
        curr = call_node.parent
        while curr is not None:
            if curr.type in ("with_clause", "with_item"):
                return True
            if curr.type == "function_definition":
                break
            curr = curr.parent
        return False

    elif lang in ("typescript", "javascript"):
        curr = call_node.parent
        while curr is not None:
            if curr.type == "try_statement":
                # Check if try_statement has finally_clause
                if any(c.type == "finally_clause" for c in curr.children):
                    return True
            if curr.type in ("function_declaration", "arrow_function", "method_definition"):
                break
            curr = curr.parent
        return False

    elif lang == "go":
        # Find enclosing function node
        curr = call_node.parent
        enclosing_func: Optional[Node] = None
        while curr is not None:
            if curr.type in ("function_declaration", "method_declaration", "func_literal"):
                enclosing_func = curr
                break
            curr = curr.parent

        if enclosing_func is None:
            return False

        # Scan for defer statement calling Close
        def has_defer_close(n: Node) -> bool:
            if n.type == "defer_statement":
                txt = source_bytes[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")
                if "Close" in txt:
                    return True
            for ch in n.children:
                # Do not cross inner function boundaries
                if ch.type in ("function_declaration", "method_declaration"):
                    continue
                if has_defer_close(ch):
                    return True
            return False

        return has_defer_close(enclosing_func)

    elif lang == "rust":
        # Rust explicit leaks are never scoped
        return False

    return True


def _is_perf004_blocking_call(callee_text: str, lang: str) -> bool:
    """Check if callee is a blocking synchronous call inside async context."""
    clean = callee_text.strip()
    clean_lower = clean.lower()

    if lang == "python":
        if clean in ("time.sleep", "open", "urllib.request.urlopen", "os.system"):
            return True
        if clean.startswith(("requests.", "subprocess.", "urllib.")):
            return True
        return False

    elif lang in ("typescript", "javascript"):
        if "sync" in clean_lower and (
            clean_lower.startswith("fs.")
            or clean_lower.startswith("child_process.")
            or clean_lower in ("execsync", "readfilesync")
        ):
            return True
        if clean in ("Atomics.wait", "crypto.pbkdf2Sync", "crypto.randomBytesSync"):
            return True
        return False

    elif lang == "rust":
        if clean in ("std::thread::sleep", "thread::sleep"):
            return True
        if clean.startswith(("std::fs::", "fs::")):
            return True
        return False

    return False


def _is_async_func_node(node: Node, lang: str) -> bool:
    """Determine if a function node is asynchronous."""
    if lang == "python":
        return any(c.type == "async" for c in node.children)
    elif lang in ("typescript", "javascript"):
        return any(c.type == "async" for c in node.children)
    elif lang == "rust":
        for c in node.children:
            if c.type == "function_modifiers":
                return any(mc.type == "async" for mc in c.children)
        return False
    return False


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
                specified_rule = match.group(1)
                if not specified_rule or specified_rule.upper() == rule_id.upper():
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
        loop_types = LOOP_NODE_TYPES.get(lang, set())
        func_types = FUNCTION_NODE_TYPES.get(lang, set())
        call_types = CALL_NODE_TYPES.get(lang, set())

        is_loop = node.type in loop_types
        is_func = node.type in func_types
        is_call = node.type in call_types

        # Handle function scope boundaries
        outer_loops: Optional[List[Node]] = None
        if is_func:
            is_async = _is_async_func_node(node, lang)
            func_name = _get_func_name(node, self.source_bytes, lang)
            self.func_stack.append((func_name, is_async, node))
            # Reset loop depth per function scope
            outer_loops = self.loop_stack
            self.loop_stack = []

        # Handle loop entrance
        if is_loop:
            self.loop_stack.append(node)
            current_depth = len(self.loop_stack)

            # PERF001: Nested Loops Complexity
            if current_depth >= 2 and current_depth >= self.max_depth:
                lineno = node.start_point.row + 1
                end_lineno = node.end_point.row + 1
                col = node.start_point.column
                end_col = node.end_point.column

                # Check root loop for whole-nest suppression
                root_line = self.loop_stack[0].start_point.row + 1
                enclosing_lines = [root_line] if root_line != lineno else None

                if not _is_suppressed(self.lines, lineno, PerfRule.PERF001.value, enclosing_lines):
                    severity = Severity.WARN if current_depth == 2 else Severity.ERROR
                    complexity = f"O(N^{current_depth})"
                    msg = f"Nested loop complexity {complexity} detected at depth {current_depth}"
                    ctx = self.lines[lineno - 1].strip() if 1 <= lineno <= len(self.lines) else None

                    self.diagnostics.append(
                        PerfDiagnostic(
                            rule_id=PerfRule.PERF001.value,
                            message=msg,
                            severity=severity,
                            file_path=self.file_path,
                            lineno=lineno,
                            end_lineno=end_lineno,
                            col_offset=col,
                            end_col_offset=end_col,
                            context_line=ctx,
                        )
                    )

        # Handle function calls
        if is_call:
            self._check_call(node)

        # Recurse children
        for child in node.children:
            self._visit(child)

        # Cleanup loop exit
        if is_loop:
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

        lineno = call_node.start_point.row + 1
        end_lineno = call_node.end_point.row + 1
        col = call_node.start_point.column
        end_col = call_node.end_point.column
        ctx = self.lines[lineno - 1].strip() if 1 <= lineno <= len(self.lines) else None

        # --- PERF002: N+1 I/O in Loop Bodies ---
        if len(self.loop_stack) > 0 and _is_perf002_io_call(callee_text):
            enclosing_loop_lines = [n.start_point.row + 1 for n in self.loop_stack]
            if not _is_suppressed(self.lines, lineno, PerfRule.PERF002.value, enclosing_loop_lines):
                msg = f"Possible N+1 query: I/O or database call '{callee_text}' detected inside loop"
                self.diagnostics.append(
                    PerfDiagnostic(
                        rule_id=PerfRule.PERF002.value,
                        message=msg,
                        severity=Severity.WARN,
                        file_path=self.file_path,
                        lineno=lineno,
                        end_lineno=end_lineno,
                        col_offset=col,
                        end_col_offset=end_col,
                        context_line=ctx,
                    )
                )

        # --- PERF003: Resource Leak / Unclosed Descriptors ---
        if _is_perf003_open_call(callee_text, self.language):
            is_scoped = _is_scoped_by_context_manager(
                call_node, callee_text, self.language, self.source_bytes
            )
            if not is_scoped:
                if not _is_suppressed(self.lines, lineno, PerfRule.PERF003.value):
                    scope_mechanism = (
                        "'with' context manager"
                        if self.language == "python"
                        else (
                            "'defer ...Close()'"
                            if self.language == "go"
                            else "'try/finally'"
                        )
                    )
                    if self.language == "rust":
                        msg = f"Potential resource or memory leak via '{callee_text}'"
                    else:
                        msg = f"Resource '{callee_text}' opened without scoped {scope_mechanism}"

                    self.diagnostics.append(
                        PerfDiagnostic(
                            rule_id=PerfRule.PERF003.value,
                            message=msg,
                            severity=Severity.ERROR,
                            file_path=self.file_path,
                            lineno=lineno,
                            end_lineno=end_lineno,
                            col_offset=col,
                            end_col_offset=end_col,
                            context_line=ctx,
                        )
                    )

        # --- PERF004: Blocking Synchronous Calls in Async Context ---
        if self.func_stack:
            current_func_name, is_async, _ = self.func_stack[-1]
            if is_async and _is_perf004_blocking_call(callee_text, self.language):
                if not _is_suppressed(self.lines, lineno, PerfRule.PERF004.value):
                    msg = (
                        f"Blocking synchronous call '{callee_text}' inside async "
                        f"function '{current_func_name}'"
                    )
                    self.diagnostics.append(
                        PerfDiagnostic(
                            rule_id=PerfRule.PERF004.value,
                            message=msg,
                            severity=Severity.ERROR,
                            file_path=self.file_path,
                            lineno=lineno,
                            end_lineno=end_lineno,
                            col_offset=col,
                            end_col_offset=end_col,
                            context_line=ctx,
                        )
                    )
