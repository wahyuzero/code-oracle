"""
Rust AST extractor using Tree-sitter.
Supports Rust structs, enums, traits, impl blocks, functions, methods, and use/mod declarations.
"""

from typing import List, Optional, Tuple
from tree_sitter import Language, Node, Parser
import tree_sitter_rust

from code_oracle.languages.common import format_syntax_error, get_node_text
from code_oracle.models import CallReference, ImportReference, Parameter, Symbol

_RUST_LANG = Language(tree_sitter_rust.language())


def get_rust_parser() -> Parser:
    """Return a Tree-sitter parser configured for Rust."""
    return Parser(_RUST_LANG)


def validate_rust_syntax(source: str, file_path: str = "") -> Optional[str]:
    """Validate Rust source syntax, returning an error message if invalid."""
    if not source.strip():
        return None
    parser = get_rust_parser()
    tree = parser.parse(source.encode("utf-8"))
    return format_syntax_error(tree.root_node, "Rust")


def _extract_rust_parameters(params_node: Node, source_bytes: bytes) -> Tuple[List[Parameter], bool]:
    """
    Extract parameters from a Rust parameters node.
    Returns: (list_of_parameters, has_self_receiver)
    """
    params: List[Parameter] = []
    has_self = False
    if not params_node:
        return params, has_self

    for child in params_node.children:
        if child.type in ("(", ")", ","):
            continue

        if child.type == "self_parameter":
            has_self = True
            params.append(Parameter(name="self", annotation=get_node_text(child, source_bytes)))

        elif child.type == "parameter":
            name_node = child.child_by_field_name("pattern") or (
                child.children[0] if child.children else None
            )
            param_name = get_node_text(name_node, source_bytes) if name_node else f"arg{len(params)}"

            type_node = child.child_by_field_name("type") or (
                child.children[-1] if len(child.children) > 1 else None
            )
            type_str = get_node_text(type_node, source_bytes).strip() if type_node else None

            params.append(
                Parameter(
                    name=param_name,
                    annotation=type_str,
                )
            )

    return params, has_self


def _extract_rust_calls(node: Node, source_bytes: bytes, caller_id: Optional[str] = None) -> List[CallReference]:
    """Recursively extract function and method calls inside a Rust AST node."""
    calls: List[CallReference] = []

    def walk(n: Node):
        if n.type == "call_expression":
            fn_node = n.child_by_field_name("function")
            callee_name = get_node_text(fn_node, source_bytes).strip() if fn_node else ""

            args_count = 0
            args_node = n.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.children:
                    if arg.type in ("(", ")", ","):
                        continue
                    args_count += 1

            if callee_name:
                calls.append(
                    CallReference(
                        callee=callee_name,
                        args_count=args_count,
                        kwargs=[],
                        lineno=n.start_point.row + 1,
                        caller=caller_id,
                    )
                )

        for child in n.children:
            walk(child)

    walk(node)
    return calls


def extract_rust_imports(source: str, file_path: str = "") -> List[ImportReference]:
    """Extract all use declarations and mod declarations from Rust source."""
    if not source.strip():
        return []

    parser = get_rust_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    imports: List[ImportReference] = []

    def walk(node: Node):
        if node.type == "use_declaration":
            lineno = node.start_point.row + 1
            for ch in node.children:
                if ch.type in ("scoped_identifier", "identifier"):
                    text = get_node_text(ch, source_bytes)
                    if "::" in text:
                        mod_part, name_part = text.rsplit("::", 1)
                        imports.append(
                            ImportReference(
                                module=mod_part,
                                name=name_part,
                                lineno=lineno,
                                file_path=file_path,
                            )
                        )
                    else:
                        imports.append(
                            ImportReference(
                                module=None,
                                name=text,
                                lineno=lineno,
                                file_path=file_path,
                            )
                        )
                elif ch.type == "scoped_use_list":
                    prefix_node = ch.child_by_field_name("path") or (
                        ch.children[0] if ch.children else None
                    )
                    prefix = get_node_text(prefix_node, source_bytes) if prefix_node else ""
                    list_node = ch.child_by_field_name("list") or next(
                        (c for c in ch.children if c.type == "use_list"), None
                    )
                    if list_node:
                        for item in list_node.children:
                            if item.type == "identifier":
                                imports.append(
                                    ImportReference(
                                        module=prefix,
                                        name=get_node_text(item, source_bytes),
                                        lineno=lineno,
                                        file_path=file_path,
                                    )
                                )
                            elif item.type == "use_as_clause":
                                path_child = item.child_by_field_name("path") or item.children[0]
                                alias_child = item.child_by_field_name("alias") or item.children[-1]
                                imports.append(
                                    ImportReference(
                                        module=prefix,
                                        name=get_node_text(path_child, source_bytes),
                                        asname=get_node_text(alias_child, source_bytes),
                                        lineno=lineno,
                                        file_path=file_path,
                                    )
                                )
                elif ch.type == "use_wildcard":
                    text = get_node_text(ch, source_bytes)
                    mod_part = text.replace("::*", "").strip()
                    imports.append(
                        ImportReference(
                            module=mod_part,
                            name="*",
                            lineno=lineno,
                            file_path=file_path,
                        )
                    )

        elif node.type == "mod_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                mod_name = get_node_text(name_node, source_bytes)
                imports.append(
                    ImportReference(
                        module=None,
                        name=mod_name,
                        lineno=node.start_point.row + 1,
                        file_path=file_path,
                    )
                )

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imports


def extract_rust_symbols(source: str, file_path: str = "") -> List[Symbol]:
    """Parse Rust source into AST and extract symbol entities with detailed metadata."""
    if not source.strip():
        return []

    parser = get_rust_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    symbols: List[Symbol] = []

    for child in tree.root_node.children:
        if child.type == "function_item":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            fn_name = get_node_text(name_node, source_bytes)
            sym_id = f"{file_path}::{fn_name}"

            params_node = child.child_by_field_name("parameters")
            params, _ = _extract_rust_parameters(params_node, source_bytes) if params_node else ([], False)

            ret_node = child.child_by_field_name("return_type")
            ret_type = get_node_text(ret_node, source_bytes).lstrip("->").strip() if ret_node else None

            min_args = len(params)
            max_args = len(params)

            body_node = child.child_by_field_name("body")
            calls = _extract_rust_calls(body_node, source_bytes, caller_id=sym_id) if body_node else []

            param_strs = [p.name + (f": {p.annotation}" if p.annotation else "") for p in params]
            ret_suffix = f" -> {ret_type}" if ret_type else ""
            signature = f"fn {fn_name}({', '.join(param_strs)}){ret_suffix}"

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
                )
            )

        elif child.type == "struct_item":
            name_node = child.child_by_field_name("name")
            if name_node:
                s_name = get_node_text(name_node, source_bytes)
                symbols.append(
                    Symbol(
                        name=s_name,
                        qualname=s_name,
                        file_path=file_path,
                        kind="struct",
                        lineno=child.start_point.row + 1,
                        end_lineno=child.end_point.row + 1,
                        signature=f"struct {s_name}",
                    )
                )

        elif child.type == "enum_item":
            name_node = child.child_by_field_name("name")
            if name_node:
                e_name = get_node_text(name_node, source_bytes)
                symbols.append(
                    Symbol(
                        name=e_name,
                        qualname=e_name,
                        file_path=file_path,
                        kind="enum",
                        lineno=child.start_point.row + 1,
                        end_lineno=child.end_point.row + 1,
                        signature=f"enum {e_name}",
                    )
                )

        elif child.type == "trait_item":
            name_node = child.child_by_field_name("name")
            if name_node:
                t_name = get_node_text(name_node, source_bytes)
                symbols.append(
                    Symbol(
                        name=t_name,
                        qualname=t_name,
                        file_path=file_path,
                        kind="interface",
                        lineno=child.start_point.row + 1,
                        end_lineno=child.end_point.row + 1,
                        signature=f"trait {t_name}",
                    )
                )

        elif child.type == "impl_item":
            type_node = child.child_by_field_name("type")
            trait_node = child.child_by_field_name("trait")
            raw_struct = get_node_text(type_node, source_bytes) if type_node else "Unknown"
            raw_trait = get_node_text(trait_node, source_bytes) if trait_node else None
            # Strip generic parameters and lifetime bounds for clean qualname and inheritance resolution
            struct_name = raw_struct.split("<")[0].strip()
            trait_name = raw_trait.split("<")[0].strip() if raw_trait else None

            body_node = child.child_by_field_name("body")
            if body_node:
                for item in body_node.children:
                    if item.type == "function_item":
                        fn_name_node = item.child_by_field_name("name")
                        if not fn_name_node:
                            continue
                        fn_name = get_node_text(fn_name_node, source_bytes)
                        qualname = f"{struct_name}.{fn_name}"
                        sym_id = f"{file_path}::{qualname}"

                        params_node = item.child_by_field_name("parameters")
                        params, has_self = (
                            _extract_rust_parameters(params_node, source_bytes)
                            if params_node
                            else ([], False)
                        )

                        ret_node = item.child_by_field_name("return_type")
                        ret_type = get_node_text(ret_node, source_bytes).lstrip("->").strip() if ret_node else None

                        is_method = has_self
                        is_static = not has_self

                        # For method calls, self is receiver (first param)
                        formal_count = len(params) - 1 if is_method else len(params)
                        min_args = formal_count
                        max_args = formal_count

                        item_body = item.child_by_field_name("body")
                        calls = _extract_rust_calls(item_body, source_bytes, caller_id=sym_id) if item_body else []

                        param_strs = [p.name + (f": {p.annotation}" if p.annotation else "") for p in params]
                        ret_suffix = f" -> {ret_type}" if ret_type else ""
                        signature = f"fn {fn_name}({', '.join(param_strs)}){ret_suffix}"

                        symbols.append(
                            Symbol(
                                name=fn_name,
                                qualname=qualname,
                                file_path=file_path,
                                kind="method" if is_method else "function",
                                lineno=item.start_point.row + 1,
                                end_lineno=item.end_point.row + 1,
                                signature=signature,
                                params=params,
                                min_args=min_args,
                                max_args=max_args,
                                return_type=ret_type,
                                calls=calls,
                                is_method=is_method,
                                is_static=is_static,
                                bases=[trait_name] if trait_name else [],
                            )
                        )

        elif child.type in ("const_item", "static_item"):
            is_const = child.type == "const_item"
            kind = "constant" if is_const else "variable"
            name_node = child.child_by_field_name("name") or next(
                (c for c in child.children if c.type == "identifier"), None
            )
            if name_node:
                v_name = get_node_text(name_node, source_bytes)
                symbols.append(
                    Symbol(
                        name=v_name,
                        qualname=v_name,
                        file_path=file_path,
                        kind=kind,
                        lineno=child.start_point.row + 1,
                        end_lineno=child.end_point.row + 1,
                        signature=f"{'const' if is_const else 'static'} {v_name}",
                        min_args=0,
                        max_args=0,
                    )
                )

        elif child.type == "type_item":
            name_node = child.child_by_field_name("name") or next(
                (c for c in child.children if c.type == "type_identifier"), None
            )
            if name_node:
                t_name = get_node_text(name_node, source_bytes)
                symbols.append(
                    Symbol(
                        name=t_name,
                        qualname=t_name,
                        file_path=file_path,
                        kind="type_alias",
                        lineno=child.start_point.row + 1,
                        end_lineno=child.end_point.row + 1,
                        signature=f"type {t_name}",
                    )
                )

    # Extract module calls
    all_calls = _extract_rust_calls(tree.root_node, source_bytes, caller_id=f"{file_path}::<module>")
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
            signature=f"// module {file_path}",
            calls=module_calls,
        )
        symbols.append(module_sym)

    return symbols
