"""
PERF002: N+1 Database and I/O in Loop Rule.
Detects database queries (query, execute, find, select) and network/HTTP calls
(fetch, get, post) inside loop bodies across Python, TypeScript, Go, and Rust.
"""

import re
from typing import List, Optional, Set, Tuple
from tree_sitter import Node

from code_oracle.perf_lint.models import PerfDiagnostic, PerfRule, Severity

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


def extract_method_name(callee_text: str) -> str:
    """Extract the last identifier (method name) from a callee expression."""
    clean = callee_text.strip().replace("\n", "")
    parts = re.split(r"\.|::", clean)
    if not parts:
        return ""
    last_part = parts[-1].strip()
    match = re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*", last_part)
    return match.group(0).lower() if match else last_part.lower()


class NPlusOneRule:
    """Evaluates PERF002: N+1 I/O in Loop Bodies."""

    RULE_ID = PerfRule.PERF002.value

    @staticmethod
    def is_io_call(callee_text: str) -> bool:
        """Check if callee represents database query or network I/O."""
        clean = callee_text.strip().replace("\n", "")
        clean_lower = clean.lower()
        method = extract_method_name(clean)

        if method in DB_IO_METHODS:
            return True

        if clean_lower in DIRECT_IO_FUNCS:
            return True

        for prefix in HTTP_IO_PREFIXES:
            if clean_lower.startswith(prefix) or f".{prefix}" in clean_lower:
                return True

        return False

    @classmethod
    def check(
        cls,
        call_node: Node,
        callee_text: str,
        in_loop: bool,
        file_path: str,
        lines: List[str],
    ) -> Optional[PerfDiagnostic]:
        """
        Evaluate if call_node represents an N+1 query inside a loop body.
        """
        if not in_loop or not cls.is_io_call(callee_text):
            return None

        lineno = call_node.start_point.row + 1
        end_lineno = call_node.end_point.row + 1
        col = call_node.start_point.column
        end_col = call_node.end_point.column
        ctx = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else None

        msg = f"Possible N+1 query: I/O or database call '{callee_text}' detected inside loop"
        return PerfDiagnostic(
            rule_id=cls.RULE_ID,
            message=msg,
            severity=Severity.WARN,
            file_path=file_path,
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=col,
            end_col_offset=end_col,
            context_line=ctx,
        )


def check_n_plus_one(
    call_node: Node,
    callee_text: str,
    in_loop: bool,
    file_path: str,
    lines: List[str],
) -> Optional[PerfDiagnostic]:
    """Convenience helper for PERF002 evaluation."""
    return NPlusOneRule.check(call_node, callee_text, in_loop, file_path, lines)
