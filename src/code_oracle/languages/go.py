"""
Go AST extractor using Tree-sitter.
Supports Go packages, structs, interfaces, methods, functions, and import declarations.
"""

from typing import List, Optional, Tuple
from tree_sitter import Language, Node, Parser
import tree_sitter_go

from code_oracle.languages.common import extract_preceding_docstring, format_syntax_error, get_node_text
from code_oracle.models import CallReference, ImportReference, Parameter, Symbol

_GO_LANG = Language(tree_sitter_go.language())


def get_go_parser() -> Parser:
    """Return a Tree-sitter parser configured for Go."""
    return Parser(_GO_LANG)


def validate_go_syntax(source: str, file_path: str = "") -> Optional[str]:
    """Validate Go source syntax, returning an error message if invalid."""
    if not source.strip():
        return None
    parser = get_go_parser()
    tree = parser.parse(source.encode("utf-8"))
    return format_syntax_error(tree.root_node, "Go")


def _extract_go_parameters(params_node: Node, source_bytes: bytes) -> List[Parameter]:
    """Extract parameters from a Go parameter_list node."""
    params: List[Parameter] = []
    if not params_node:
        return params

    for child in params_node.children:
        if child.type in ("(", ")", ","):
            continue

        if child.type == "parameter_declaration":
            # Form: 'a, b int' or 'host string'
            type_node = child.child_by_field_name("type") or (
                child.children[-1] if child.children else None
            )
            type_str = get_node_text(type_node, source_bytes) if type_node else None

            # Collect identifier children
            names = []
            for sc in child.children:
                if sc.type == "identifier":
                    names.append(get_node_text(sc, source_bytes))

            if not names:
                # Unnamed parameter e.g. func(int, string)
                params.append(Parameter(name=f"arg{len(params)}", annotation=type_str))
            else:
                for name in names:
                    params.append(Parameter(name=name, annotation=type_str))

        elif child.type == "variadic_parameter_declaration":
            # Form: 'rest ...int'
            name_node = child.child_by_field_name("name") or (
                child.children[0] if child.children and child.children[0].type == "identifier" else None
            )
            param_name = get_node_text(name_node, source_bytes) if name_node else "rest"

            type_node = child.child_by_field_name("type") or (
                child.children[-1] if child.children else None
            )
            type_str = get_node_text(type_node, source_bytes) if type_node else None

            params.append(
                Parameter(
                    name=param_name,
                    annotation=type_str,
                    is_vararg=True,
                )
            )

    return params


def _extract_go_calls(node: Node, source_bytes: bytes, caller_id: Optional[str] = None) -> List[CallReference]:
    """Recursively extract function and method calls inside a Go AST node."""
    calls: List[CallReference] = []

    def walk(n: Node):
        if n.type == "call_expression":
            fn_node = n.child_by_field_name("function")
            callee_name = get_node_text(fn_node, source_bytes).strip() if fn_node else ""

            args_count = 0
            has_vararg = False
            args_node = n.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.children:
                    if arg.type in ("(", ")", ",", "comment", "line_comment", "block_comment") or "comment" in arg.type:
                        continue
                    if arg.type == "...":
                        has_vararg = True
                        continue
                    args_count += 1
                    # Also check if last argument ends with ... (e.g. variadic slice expansion)
                    if arg.text.decode("utf-8", errors="ignore").endswith("..."):
                        has_vararg = True

            if callee_name:
                calls.append(
                    CallReference(
                        callee=callee_name,
                        args_count=args_count,
                        kwargs=[],
                        lineno=n.start_point.row + 1,
                        caller=caller_id,
                        has_vararg=has_vararg,
                    )
                )

        for child in n.children:
            walk(child)

    walk(node)
    return calls


def extract_go_imports(source: str, file_path: str = "") -> List[ImportReference]:
    """Extract all import declarations from Go source."""
    if not source.strip():
        return []

    parser = get_go_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    imports: List[ImportReference] = []

    def walk(node: Node):
        if node.type == "import_declaration":
            for ch in node.children:
                if ch.type == "import_spec":
                    _process_import_spec(ch)
                elif ch.type == "import_spec_list":
                    for spec in ch.children:
                        if spec.type == "import_spec":
                            _process_import_spec(spec)
        for child in node.children:
            walk(child)

    def _process_import_spec(spec_node: Node):
        path_node = spec_node.child_by_field_name("path") or next(
            (c for c in spec_node.children if c.type == "interpreted_string_literal"), None
        )
        if not path_node:
            return

        import_path = get_node_text(path_node, source_bytes).strip('"')
        name_node = spec_node.child_by_field_name("name") or next(
            (c for c in spec_node.children if c.type == "package_identifier"), None
        )
        asname = get_node_text(name_node, source_bytes) if name_node else None

        # Go symbol name is either the alias or the package name (last segment)
        pkg_name = asname if asname else import_path.split("/")[-1]

        imports.append(
            ImportReference(
                module=import_path if "/" in import_path else None,
                name=pkg_name,
                asname=asname,
                lineno=spec_node.start_point.row + 1,
                file_path=file_path,
            )
        )

    walk(tree.root_node)
    return imports


def extract_go_symbols(source: str, file_path: str = "") -> List[Symbol]:
    """Parse Go source into AST and extract symbol entities with detailed metadata."""
    if not source.strip():
        return []

    parser = get_go_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    symbols: List[Symbol] = []

    for child in tree.root_node.children:
        docstring = extract_preceding_docstring(child, source_bytes)
        if child.type == "function_declaration":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            fn_name = get_node_text(name_node, source_bytes)
            sym_id = f"{file_path}::{fn_name}"

            params_node = child.child_by_field_name("parameters")
            params = _extract_go_parameters(params_node, source_bytes) if params_node else []

            res_node = child.child_by_field_name("result")
            ret_type = get_node_text(res_node, source_bytes).strip() if res_node else None

            pos_params = [p for p in params if not p.is_vararg]
            min_args = len(pos_params)
            max_args = None if any(p.is_vararg for p in params) else len(pos_params)

            body_node = child.child_by_field_name("body")
            calls = _extract_go_calls(body_node, source_bytes, caller_id=sym_id) if body_node else []

            param_strs = [p.name + (f" {p.annotation}" if p.annotation else "") for p in params]
            ret_suffix = f" {ret_type}" if ret_type else ""
            signature = f"func {fn_name}({', '.join(param_strs)}){ret_suffix}"

            is_exp = bool(fn_name and fn_name[0].isupper())
            vis = "public" if is_exp else "internal"

            symbols.append(
                Symbol(
                    name=fn_name,
                    qualname=fn_name,
                    file_path=file_path,
                    kind="function",
                    lineno=child.start_point.row + 1,
                    end_lineno=child.end_point.row + 1,
                    signature=signature,
                    params=params,
                    min_args=min_args,
                    max_args=max_args,
                    return_type=ret_type,
                    calls=calls,
                    is_method=False,
                    is_static=False,
                    docstring=docstring,
                    is_exported=is_exp,
                    visibility=vis,
                )
            )

        elif child.type == "method_declaration":
            recv_node = child.child_by_field_name("receiver")
            name_node = child.child_by_field_name("name")
            if not recv_node or not name_node:
                continue

            method_name = get_node_text(name_node, source_bytes)

            # Extract receiver type name
            recv_type = "unknown"
            recv_var = "recv"
            for r_child in recv_node.children:
                if r_child.type == "parameter_declaration":
                    for sc in r_child.children:
                        if sc.type == "identifier":
                            recv_var = get_node_text(sc, source_bytes)
                        elif sc.type == "pointer_type":
                            for ptr_c in sc.children:
                                if ptr_c.type == "type_identifier":
                                    recv_type = get_node_text(ptr_c, source_bytes)
                        elif sc.type == "type_identifier":
                            recv_type = get_node_text(sc, source_bytes)

            qualname = f"{recv_type}.{method_name}"
            sym_id = f"{file_path}::{qualname}"

            params_node = child.child_by_field_name("parameters")
            formal_params = _extract_go_parameters(params_node, source_bytes) if params_node else []

            # Prepend receiver parameter as params[0] for uniform method abstraction
            receiver_param = Parameter(name=recv_var, annotation=recv_type)
            params = [receiver_param] + formal_params

            res_node = child.child_by_field_name("result")
            ret_type = get_node_text(res_node, source_bytes).strip() if res_node else None

            pos_params = [p for p in formal_params if not p.is_vararg]
            min_args = len(pos_params)
            max_args = None if any(p.is_vararg for p in formal_params) else len(pos_params)

            body_node = child.child_by_field_name("body")
            calls = _extract_go_calls(body_node, source_bytes, caller_id=sym_id) if body_node else []

            param_strs = [p.name + (f" {p.annotation}" if p.annotation else "") for p in formal_params]
            ret_suffix = f" {ret_type}" if ret_type else ""
            signature = f"func ({recv_var} *{recv_type}) {method_name}({', '.join(param_strs)}){ret_suffix}"

            is_exp = bool(method_name and method_name[0].isupper() and (not recv_type or recv_type == "unknown" or recv_type[0].isupper()))
            vis = "public" if is_exp else "internal"

            symbols.append(
                Symbol(
                    name=method_name,
                    qualname=qualname,
                    file_path=file_path,
                    kind="method",
                    lineno=child.start_point.row + 1,
                    end_lineno=child.end_point.row + 1,
                    signature=signature,
                    params=params,
                    min_args=min_args,
                    max_args=max_args,
                    return_type=ret_type,
                    calls=calls,
                    is_method=True,
                    is_static=False,
                    docstring=docstring,
                    is_exported=is_exp,
                    visibility=vis,
                )
            )

        elif child.type == "type_declaration":
            for spec in child.children:
                if spec.type == "type_spec":
                    name_node = spec.child_by_field_name("name")
                    type_node = spec.child_by_field_name("type")
                    if name_node:
                        t_name = get_node_text(name_node, source_bytes)
                        if type_node and type_node.type == "struct_type":
                            kind = "struct"
                        elif type_node and type_node.type == "interface_type":
                            kind = "interface"
                        else:
                            kind = "type_alias"

                        is_exp = bool(t_name and t_name[0].isupper())
                        vis = "public" if is_exp else "internal"

                        symbols.append(
                            Symbol(
                                name=t_name,
                                qualname=t_name,
                                file_path=file_path,
                                kind=kind,
                                lineno=child.start_point.row + 1,
                                end_lineno=child.end_point.row + 1,
                                signature=f"type {t_name} {kind}",
                                docstring=docstring,
                                is_exported=is_exp,
                                visibility=vis,
                            )
                        )

        elif child.type in ("const_declaration", "var_declaration"):
            is_const = child.type == "const_declaration"
            kind = "constant" if is_const else "variable"
            spec_type = "const_spec" if is_const else "var_spec"
            for spec in child.children:
                if spec.type == spec_type:
                    for sc in spec.children:
                        if sc.type == "identifier":
                            v_name = get_node_text(sc, source_bytes)
                            is_exp = bool(v_name and v_name[0].isupper())
                            vis = "public" if is_exp else "internal"
                            symbols.append(
                                Symbol(
                                    name=v_name,
                                    qualname=v_name,
                                    file_path=file_path,
                                    kind=kind,
                                    lineno=sc.start_point.row + 1,
                                    end_lineno=sc.end_point.row + 1,
                                    signature=f"{'const' if is_const else 'var'} {v_name}",
                                    min_args=0,
                                    max_args=0,
                                    docstring=docstring,
                                    is_exported=is_exp,
                                    visibility=vis,
                                )
                            )

    # Extract package / module calls
    all_calls = _extract_go_calls(tree.root_node, source_bytes, caller_id=f"{file_path}::<module>")
    module_calls = [
        c for c in all_calls
        if not any(s.lineno <= c.lineno <= s.end_lineno for s in symbols if s.kind != "module")
    ]
    if module_calls:
        line_count = len(source.splitlines()) or 1
        module_sym = Symbol(
            name="<module>",
            qualname="<module>",
            file_path=file_path,
            kind="module",
            lineno=1,
            end_lineno=line_count,
            signature=f"// package {file_path}",
            calls=module_calls,
            is_exported=True,
            visibility="public",
        )
        symbols.append(module_sym)

    return symbols
