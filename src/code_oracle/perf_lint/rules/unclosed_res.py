"""
PERF003: Resource Leak / Unclosed Descriptors Rule.
Detects unclosed file/socket/db descriptors without scoped context manager:
missing `with` in Python, missing `defer resp.Body.Close()` in Go,
unhandled stream/fd in TS/JS, Box::leak/mem::forget in Rust.
"""

import re
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
        """Extract variable identifier to which the resource is directly assigned."""
        unwrapped = call_node
        while unwrapped.parent and unwrapped.parent.type == "parenthesized_expression":
            unwrapped = unwrapped.parent

        p = unwrapped.parent
        if p is None:
            return None

        if language in ("typescript", "javascript"):
            if p.type == "variable_declarator":
                val = p.child_by_field_name("value")
                if val == unwrapped:
                    name_node = p.child_by_field_name("name")
                    if name_node and name_node.type == "identifier":
                        return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore").strip()
            elif p.type == "assignment_expression":
                right = p.child_by_field_name("right")
                if right == unwrapped:
                    left = p.child_by_field_name("left")
                    if left and left.type == "identifier":
                        return source_bytes[left.start_byte : left.end_byte].decode("utf-8", errors="ignore").strip()

        elif language == "go":
            stmt = p if p.type in ("short_var_declaration", "assignment_statement", "var_spec") else p.parent
            if stmt and stmt.type in ("short_var_declaration", "assignment_statement", "var_spec"):
                right = stmt.child_by_field_name("right")
                if right is None:
                    for ch in stmt.children:
                        if ch.type == "expression_list" and ch != stmt.child_by_field_name("left"):
                            right = ch
                            break
                is_in_right = (right == unwrapped) or (right and unwrapped in right.named_children)
                if is_in_right:
                    left = stmt.child_by_field_name("left")
                    if left is None:
                        for ch in stmt.children:
                            if ch.type == "expression_list":
                                left = ch
                                break
                    if left:
                        if right and len(right.named_children) == len(left.named_children) and unwrapped in right.named_children:
                            idx = right.named_children.index(unwrapped)
                            if idx < len(left.named_children) and left.named_children[idx].type == "identifier":
                                return source_bytes[left.named_children[idx].start_byte : left.named_children[idx].end_byte].decode("utf-8", errors="ignore").strip()
                        for ch in left.children:
                            if ch.type == "identifier":
                                return source_bytes[ch.start_byte : ch.end_byte].decode("utf-8", errors="ignore").strip()

        elif language == "python":
            if p.type == "assignment":
                right = p.child_by_field_name("right")
                if right == unwrapped:
                    left = p.child_by_field_name("left")
                    if left and left.type == "identifier":
                        return source_bytes[left.start_byte : left.end_byte].decode("utf-8", errors="ignore").strip()
            elif p.type in ("expression_list", "tuple"):
                assign = p.parent
                if assign and assign.type == "assignment" and assign.child_by_field_name("right") == p:
                    named_rhs = p.named_children
                    if unwrapped in named_rhs:
                        idx = named_rhs.index(unwrapped)
                        left = assign.child_by_field_name("left")
                        if left and left.type in ("pattern_list", "tuple_pattern"):
                            named_lhs = left.named_children
                            if idx < len(named_lhs) and named_lhs[idx].type == "identifier":
                                return source_bytes[named_lhs[idx].start_byte : named_lhs[idx].end_byte].decode("utf-8", errors="ignore").strip()

        return None

    @classmethod
    def _is_returned(cls, call_node: Node, source_bytes: bytes, language: str) -> bool:
        """Check if resource or its assigned variable is returned (ownership transferred)."""
        # 1. Direct return: call_node is inside a return statement
        curr = call_node.parent
        while curr is not None:
            if curr.type in (
                "function_definition",
                "function_declaration",
                "method_definition",
                "method_declaration",
                "func_literal",
                "arrow_function",
                "function_expression",
            ):
                break
            if curr.type == "return_statement":
                # Ensure call_node is not simply calling a method (e.g. open().read())
                # or accessing a property (e.g. open().name)
                p = call_node.parent
                while p and p.type == "parenthesized_expression":
                    p = p.parent
                if p and p.type in ("attribute", "member_expression", "selector_expression"):
                    if language in ("typescript", "javascript"):
                        prop = p.child_by_field_name("property")
                        if prop and source_bytes[prop.start_byte : prop.end_byte].strip() == b"pipe":
                            return True
                    return False
                return True
            curr = curr.parent

        # 2. Variable return: resource assigned to var and var is returned
        enclosing_func = None
        curr = call_node.parent
        while curr is not None:
            if curr.type in (
                "function_definition",
                "function_declaration",
                "method_definition",
                "method_declaration",
                "func_literal",
                "arrow_function",
                "function_expression",
            ):
                enclosing_func = curr
                break
            curr = curr.parent

        if enclosing_func is None:
            return False

        var_name = cls._extract_assigned_var(call_node, source_bytes, language)
        if not var_name:
            return False

        def is_var_returned_in_node(ret_node: Node) -> bool:
            def check_node(n: Node) -> bool:
                if n.type in ("identifier", "shorthand_property_identifier"):
                    ident = source_bytes[n.start_byte : n.end_byte].decode("utf-8", errors="ignore").strip()
                    if ident == var_name:
                        p = n.parent
                        while p and p.type == "parenthesized_expression":
                            p = p.parent
                        if p and p.type in ("attribute", "member_expression", "selector_expression"):
                            if language in ("typescript", "javascript"):
                                prop = p.child_by_field_name("property")
                                if prop and source_bytes[prop.start_byte : prop.end_byte].strip() == b"pipe":
                                    return True
                            return False
                        return True
                for ch in n.children:
                    if check_node(ch):
                        return True
                return False

            return check_node(ret_node)

        def search_returns(node: Node) -> bool:
            if node.type == "return_statement":
                if is_var_returned_in_node(node):
                    return True
            for ch in node.children:
                if ch.type in (
                    "function_definition",
                    "function_declaration",
                    "method_definition",
                    "method_declaration",
                    "func_literal",
                    "arrow_function",
                    "function_expression",
                ):
                    continue
                if search_returns(ch):
                    return True
            return False

        return search_returns(enclosing_func)

    @classmethod
    def is_scoped(
        cls,
        call_node: Node,
        callee_text: str,
        language: str,
        source_bytes: bytes,
    ) -> bool:
        """Check if resource open call is properly scoped by language construct."""
        if cls._is_returned(call_node, source_bytes, language):
            return True

        if language == "python":
            curr = call_node.parent
            while curr is not None:
                if curr.type in ("with_clause", "with_item"):
                    return True
                if curr.type == "function_definition":
                    break
                curr = curr.parent

            # Support Python try ... finally: ...close()
            enclosing = call_node.parent
            while enclosing is not None:
                if enclosing.type in ("function_definition", "module"):
                    break
                enclosing = enclosing.parent

            if enclosing is not None:
                var_name = cls._extract_assigned_var(call_node, source_bytes, language)
                if var_name:
                    def has_finally_close(n: Node) -> bool:
                        if n.type == "try_statement":
                            for c in n.children:
                                if c.type == "finally_clause":
                                    txt = source_bytes[c.start_byte : c.end_byte].decode("utf-8", errors="ignore")
                                    if "close" in txt.lower():
                                        if re.search(rf"\b{re.escape(var_name)}\b", txt):
                                            return True
                        for ch in n.children:
                            if ch.type == "function_definition":
                                continue
                            if has_finally_close(ch):
                                return True
                        return False

                    if has_finally_close(enclosing):
                        return True

            return False

        elif language in ("typescript", "javascript"):
            var_name = cls._extract_assigned_var(call_node, source_bytes, language)

            # 1. Ancestor check: inside try block with finally
            curr = call_node.parent
            while curr is not None:
                if curr.type == "try_statement":
                    for c in curr.children:
                        if c.type == "finally_clause":
                            if var_name:
                                txt = source_bytes[c.start_byte : c.end_byte].decode("utf-8", errors="ignore")
                                if re.search(rf"\b{re.escape(var_name)}\b", txt):
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

            # Passed directly to .pipe(...) as argument
            if call_node.parent and call_node.parent.type == "arguments":
                p_call = call_node.parent.parent
                if p_call and p_call.type == "call_expression":
                    fn = p_call.child_by_field_name("function")
                    if fn and fn.type == "member_expression":
                        prop = fn.child_by_field_name("property")
                        if prop:
                            prop_name = source_bytes[prop.start_byte : prop.end_byte].decode("utf-8", errors="ignore").strip()
                            if prop_name == "pipe":
                                return True

            # 3. Check enclosing block for try/finally or close/destroy/pipe
            enclosing = call_node.parent
            while enclosing is not None:
                if enclosing.type in ("statement_block", "program", "function_declaration", "arrow_function", "method_definition"):
                    break
                enclosing = enclosing.parent

            if enclosing is not None and var_name:
                def check_block(n: Node) -> bool:
                    if n.type == "try_statement":
                        for c in n.children:
                            if c.type == "finally_clause":
                                txt = source_bytes[c.start_byte : c.end_byte].decode("utf-8", errors="ignore")
                                if re.search(rf"\b{re.escape(var_name)}\b", txt):
                                    return True
                    elif n.type == "call_expression":
                        fn = n.child_by_field_name("function")
                        if fn and fn.type == "member_expression":
                            prop = fn.child_by_field_name("property")
                            prop_name = source_bytes[prop.start_byte : prop.end_byte].decode("utf-8", errors="ignore").strip() if prop else ""
                            if prop_name == "pipe":
                                obj = fn.child_by_field_name("object")
                                if obj:
                                    obj_name = source_bytes[obj.start_byte : obj.end_byte].decode("utf-8", errors="ignore").strip()
                                    if obj_name == var_name:
                                        return True
                                args = n.child_by_field_name("arguments")
                                if args:
                                    args_txt = source_bytes[args.start_byte : args.end_byte].decode("utf-8", errors="ignore")
                                    if re.search(rf"\b{re.escape(var_name)}\b", args_txt):
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
                    if "Close" in txt and re.search(rf"\b{re.escape(var_name)}\b", txt):
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
