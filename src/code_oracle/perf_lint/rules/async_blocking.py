"""
PERF004: Blocking Calls in Async Context Rule.
Detects synchronous blocking primitives like time.sleep or sync file I/O
inside async def / async function bodies across Python, TypeScript, and Rust.
"""

from typing import List, Optional
from tree_sitter import Node

from code_oracle.perf_lint.models import PerfDiagnostic, PerfRule, Severity


class AsyncBlockingRule:
    """Evaluates PERF004: Blocking Calls in Async Context."""

    RULE_ID = PerfRule.PERF004.value

    @staticmethod
    def is_async_func_node(node: Node, language: str) -> bool:
        """Determine if a function AST node is asynchronous."""
        if language == "python":
            return any(c.type == "async" for c in node.children)
        elif language in ("typescript", "javascript"):
            return any(c.type == "async" for c in node.children)
        elif language == "rust":
            for c in node.children:
                if c.type == "function_modifiers":
                    return any(mc.type == "async" for mc in c.children)
            return False
        return False

    @staticmethod
    def is_blocking_call(callee_text: str, language: str) -> bool:
        """Check if callee is a blocking synchronous primitive."""
        clean = callee_text.strip()
        clean_lower = clean.lower()

        if language == "python":
            if clean in ("time.sleep", "open", "urllib.request.urlopen", "os.system"):
                return True
            if clean.startswith(("requests.", "subprocess.", "urllib.")):
                return True
            return False

        elif language in ("typescript", "javascript"):
            if "sync" in clean_lower and (
                clean_lower.startswith("fs.")
                or clean_lower.startswith("child_process.")
                or clean_lower in ("execsync", "readfilesync")
            ):
                return True
            if clean in ("Atomics.wait", "crypto.pbkdf2Sync", "crypto.randomBytesSync"):
                return True
            return False

        elif language == "rust":
            if clean in ("std::thread::sleep", "thread::sleep"):
                return True
            if clean.startswith(("std::fs::", "fs::")):
                return True
            return False

        return False

    @classmethod
    def check(
        cls,
        call_node: Node,
        callee_text: str,
        is_async_context: bool,
        current_func_name: str,
        language: str,
        file_path: str,
        lines: List[str],
    ) -> Optional[PerfDiagnostic]:
        """Evaluate if call is a blocking synchronous primitive in an async function."""
        if not is_async_context or not cls.is_blocking_call(callee_text, language):
            return None

        lineno = call_node.start_point.row + 1
        end_lineno = call_node.end_point.row + 1
        col = call_node.start_point.column
        end_col = call_node.end_point.column
        ctx = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else None

        msg = (
            f"Blocking synchronous call '{callee_text}' inside async "
            f"function '{current_func_name}'"
        )
        return PerfDiagnostic(
            rule_id=cls.RULE_ID,
            message=msg,
            severity=Severity.ERROR,
            file_path=file_path,
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=col,
            end_col_offset=end_col,
            context_line=ctx,
        )


def check_async_blocking(
    call_node: Node,
    callee_text: str,
    is_async_context: bool,
    current_func_name: str,
    language: str,
    file_path: str,
    lines: List[str],
) -> Optional[PerfDiagnostic]:
    """Convenience helper for PERF004 evaluation."""
    return AsyncBlockingRule.check(
        call_node=call_node,
        callee_text=callee_text,
        is_async_context=is_async_context,
        current_func_name=current_func_name,
        language=language,
        file_path=file_path,
        lines=lines,
    )
