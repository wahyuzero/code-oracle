"""
PERF003: Resource Leak / Unclosed Descriptors Rule.
Detects unclosed file/socket/db descriptors without scoped context manager:
missing `with` in Python, missing `defer resp.Body.Close()` in Go,
unhandled stream/fd in TS/JS, Box::leak/mem::forget in Rust.
"""

from typing import List, Optional
from tree_sitter import Node

from code_oracle.perf_lint.models import PerfDiagnostic, PerfRule, Severity


class UnclosedResourceRule:
    """Evaluates PERF003: Resource Leak / Unclosed Descriptors."""

    RULE_ID = PerfRule.PERF003.value

    @staticmethod
    def is_open_call(callee_text: str, language: str) -> bool:
        """Check if callee is a resource open call requiring scoped cleanup."""
        clean = callee_text.strip()
        if "(" in clean:
            return False
        clean_lower = clean.lower()

        if language == "python":
            if clean in ("open", "socket.socket", "socket.create_connection", "os.open"):
                return True
            if clean.endswith((".open",)):
                return True
            if "." in clean and clean.endswith(".connect"):
                receiver = clean.rsplit(".", 1)[0].lower()
                if any(
                    db in receiver
                    for db in (
                        "sqlite",
                        "psycopg",
                        "mysql",
                        "asyncpg",
                        "db",
                        "database",
                        "postgres",
                        "engine",
                        "sql",
                    )
                ) or receiver in ("conn", "connection"):
                    return True
            return False

        elif language in ("typescript", "javascript"):
            return clean.startswith((
                "fs.open",
                "fs.createReadStream",
                "fs.createWriteStream",
                "net.connect",
                "net.createConnection",
                "tls.connect",
            ))

        elif language == "go":
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

        elif language == "rust":
            return clean in (
                "Box::leak",
                "std::boxed::Box::leak",
                "Box::into_raw",
                "std::boxed::Box::into_raw",
                "CString::into_raw",
                "std::ffi::CString::into_raw",
                "std::mem::forget",
                "mem::forget",
                "std::mem::ManuallyDrop::new",
                "ManuallyDrop::new",
            )

        return False

    @staticmethod
    def _extract_assigned_var(call_node: Node, source_bytes: bytes, language: str) -> Optional[str]:
        """Extract variable identifier to which the resource is assigned."""
        if language in ("typescript", "javascript"):
            curr = call_node.parent
            while curr is not None and curr.type not in (
                "variable_declarator",
                "assignment_expression",
                "statement_block",
                "program",
            ):
                curr = curr.parent
            if curr and curr.type == "variable_declarator":
                name_node = curr.child_by_field_name("name")
                if name_node:
                    return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore").strip()
            elif curr and curr.type == "assignment_expression":
                left = curr.child_by_field_name("left")
                if left:
                    return source_bytes[left.start_byte : left.end_byte].decode("utf-8", errors="ignore").strip()

        elif language == "go":
            curr = call_node.parent
            while curr is not None and curr.type not in (
                "short_var_declaration",
                "assignment_statement",
                "var_spec",
                "block",
            ):
                curr = curr.parent
            if curr and curr.type in ("short_var_declaration", "assignment_statement", "var_spec"):
                left = curr.child_by_field_name("left")
                if left is None:
                    for ch in curr.children:
                        if ch.type == "expression_list":
                            left = ch
                            break
                if left:
                    for ch in left.children:
                        if ch.type == "identifier":
                            return source_bytes[ch.start_byte : ch.end_byte].decode("utf-8", errors="ignore").strip()

        return None

    @classmethod
    def is_scoped(
        cls,
        call_node: Node,
        callee_text: str,
        language: str,
        source_bytes: bytes,
    ) -> bool:
        """Check if resource open call is properly scoped by language construct."""
        if language == "python":
            curr = call_node.parent
            while curr is not None:
                if curr.type in ("with_clause", "with_item"):
                    return True
                if curr.type == "function_definition":
                    break
                curr = curr.parent
            return False

        elif language in ("typescript", "javascript"):
            # 1. Ancestor check: inside try block with finally
            curr = call_node.parent
            while curr is not None:
                if curr.type == "try_statement":
                    if any(c.type == "finally_clause" for c in curr.children):
                        return True
                if curr.type in ("function_declaration", "arrow_function", "method_definition"):
                    break
                curr = curr.parent

            # 2. Chained with .pipe / .on
            if call_node.parent and call_node.parent.type == "member_expression":
                prop = call_node.parent.child_by_field_name("property")
                if prop:
                    prop_name = source_bytes[prop.start_byte : prop.end_byte].decode("utf-8", errors="ignore").strip()
                    if prop_name in ("pipe", "on", "then"):
                        return True

            # 3. Check enclosing block for try/finally or close/destroy/pipe
            enclosing = call_node.parent
            while enclosing is not None:
                if enclosing.type in ("statement_block", "program", "function_declaration", "arrow_function", "method_definition"):
                    break
                enclosing = enclosing.parent

            if enclosing is not None:
                var_name = cls._extract_assigned_var(call_node, source_bytes, language)
                if var_name:
                    def check_block(n: Node) -> bool:
                        if n.type == "try_statement":
                            if any(c.type == "finally_clause" for c in n.children):
                                txt = source_bytes[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")
                                if var_name in txt:
                                    return True
                        elif n.type == "call_expression":
                            txt = source_bytes[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")
                            if f"{var_name}.pipe" in txt:
                                return True
                        for ch in n.children:
                            if ch.type in ("function_declaration", "arrow_function", "method_definition"):
                                continue
                            if check_block(ch):
                                return True
                        return False

                    if check_block(enclosing):
                        return True

            return False

        elif language == "go":
            curr = call_node.parent
            enclosing_func: Optional[Node] = None
            while curr is not None:
                if curr.type in ("function_declaration", "method_declaration", "func_literal"):
                    enclosing_func = curr
                    break
                curr = curr.parent

            if enclosing_func is None:
                return False

            var_name = cls._extract_assigned_var(call_node, source_bytes, language)
            if not var_name:
                return False

            def has_defer_or_close(n: Node) -> bool:
                if n.type in ("defer_statement", "call_expression"):
                    txt = source_bytes[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")
                    if "Close" in txt and var_name in txt:
                        return True
                for ch in n.children:
                    if ch.type in ("function_declaration", "method_declaration", "func_literal"):
                        continue
                    if has_defer_or_close(ch):
                        return True
                return False

            return has_defer_or_close(enclosing_func)

        elif language == "rust":
            return False

        return True

    @classmethod
    def check(
        cls,
        call_node: Node,
        callee_text: str,
        language: str,
        source_bytes: bytes,
        file_path: str,
        lines: List[str],
    ) -> Optional[PerfDiagnostic]:
        """Evaluate if resource is opened without scoped cleanup."""
        if not cls.is_open_call(callee_text, language):
            return None

        if cls.is_scoped(call_node, callee_text, language, source_bytes):
            return None

        lineno = call_node.start_point.row + 1
        end_lineno = call_node.end_point.row + 1
        col = call_node.start_point.column
        end_col = call_node.end_point.column
        ctx = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else None

        scope_mechanism = (
            "'with' context manager"
            if language == "python"
            else (
                "'defer ...Close()'"
                if language == "go"
                else "'try/finally'"
            )
        )
        if language == "rust":
            msg = f"Potential resource or memory leak via '{callee_text}'"
        else:
            msg = f"Resource '{callee_text}' opened without scoped {scope_mechanism}"

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


def check_unclosed_resource(
    call_node: Node,
    callee_text: str,
    language: str,
    source_bytes: bytes,
    file_path: str,
    lines: List[str],
) -> Optional[PerfDiagnostic]:
    """Convenience helper for PERF003 evaluation."""
    return UnclosedResourceRule.check(
        call_node=call_node,
        callee_text=callee_text,
        language=language,
        source_bytes=source_bytes,
        file_path=file_path,
        lines=lines,
    )
