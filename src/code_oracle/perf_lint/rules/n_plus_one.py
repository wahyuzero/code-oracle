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
    "query_row",
    "post",
    "upsert",
    "save",
    # Go context query and exec methods
    "query_context",
    "exec_context",
    "query_row_context",
    "querycontext",
    "execcontext",
    "queryrowcontext",
    # Prisma / MongoDB camelCase variants (normalized and raw)
    "findone",
    "findmany",
    "findunique",
    "findfirst",
    "findby",
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
    "api.",
    "session.",
    "httpclient.",
    "http_client.",
    "service.",
    "c.post",
    "hc.post",
)

DIRECT_IO_FUNCS: Set[str] = {
    "fetch",
    "query",
    "execute",
    "select",
    "urlopen",
}


DB_RECEIVER_KEYWORDS: Set[str] = {
    "db",
    "database",
    "repo",
    "repository",
    "table",
    "tables",
    "model",
    "models",
    "conn",
    "connection",
    "cursor",
    "collection",
    "collections",
    "dao",
    "entity",
    "entities",
    "sql",
}

NETWORK_GET_RECEIVER_KEYWORDS: Set[str] = {
    "http",
    "client",
    "request",
    "requests",
    "api",
    "session",
    "service",
    "db",
    "fetch",
    "conn",
    "connection",
    "rest",
    "remote",
}


def _camel_to_snake(s: str) -> str:
    """Convert camelCase/PascalCase to snake_case."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", s).lower()


def _receiver_has_keyword(receiver: str, keywords: Set[str]) -> bool:
    """Check if receiver identifier contains any target keywords as distinct tokens."""
    snake = _camel_to_snake(receiver)
    tokens = set(re.split(r"[^a-z0-9]+", snake))
    return bool(tokens & keywords)


def extract_method_name(callee_text: str) -> str:
    """Extract the last identifier (method name) from a callee expression with camelCase normalization."""
    clean = callee_text.strip().replace("\n", "")
    parts = re.split(r"\.|::", clean)
    if not parts:
        return ""
    last_part = parts[-1].strip()
    match = re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*", last_part)
    raw = match.group(0) if match else last_part
    return _camel_to_snake(raw)


class NPlusOneRule:
    """Evaluates PERF002: N+1 I/O in Loop Bodies."""

    RULE_ID = PerfRule.PERF002.value

    @staticmethod
    def is_io_call(callee_text: str) -> bool:
        """Check if callee represents database query or network I/O."""
        clean = callee_text.strip().replace("\n", "")
        clean_lower = clean.lower()
        method = extract_method_name(clean)
        method_raw = ""
        parts = re.split(r"\.|::", clean)
        if parts:
            m = re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*", parts[-1].strip())
            if m:
                method_raw = m.group(0).lower()

        if method in DB_IO_METHODS or method_raw in DB_IO_METHODS:
            return True

        if clean_lower in DIRECT_IO_FUNCS or method in DIRECT_IO_FUNCS:
            return True

        for prefix in HTTP_IO_PREFIXES:
            if clean_lower.startswith(prefix) or f".{prefix}" in clean_lower:
                return True

        # Check for database write calls (update/insert/delete) with DB-related receiver
        # to prevent false positives on set.update(), dict.update(), list.insert()
        if method in ("update", "insert", "delete"):
            if len(parts) >= 2:
                receiver = parts[-2]
                if _receiver_has_keyword(receiver, DB_RECEIVER_KEYWORDS):
                    return True

        # Check for network/database .get(...) calls while preventing dict.get() false positives
        if method == "get":
            if len(parts) >= 2:
                receiver = parts[-2]
                if _receiver_has_keyword(receiver, NETWORK_GET_RECEIVER_KEYWORDS):
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
