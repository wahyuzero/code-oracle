"""
TypeScript and JavaScript AST extractor using Tree-sitter.
Supports .ts, .tsx, .js, .jsx, .mjs, and .cjs files.
"""

from typing import List, Optional, Set, Tuple
from tree_sitter import Language, Node, Parser
import tree_sitter_javascript
import tree_sitter_typescript

from code_oracle.languages.common import format_syntax_error, get_node_text
from code_oracle.models import CallReference, ImportReference, Parameter, Symbol

_TS_LANG = Language(tree_sitter_typescript.language_typescript())
_TSX_LANG = Language(tree_sitter_typescript.language_tsx())
_JS_LANG = Language(tree_sitter_javascript.language())


def get_ts_parser(file_path: str = "") -> Parser:
    """Get the appropriate Tree-sitter parser for TypeScript or JavaScript."""
    lower = file_path.lower()
    if lower.endswith(".tsx"):
        return Parser(_TSX_LANG)
    elif lower.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return Parser(_JS_LANG)
    return Parser(_TS_LANG)


def validate_typescript_syntax(source: str, file_path: str = "") -> Optional[str]:
    """Validate syntax of TypeScript/JavaScript source."""
    if not source.strip():
        return None
    parser = get_ts_parser(file_path)
    tree = parser.parse(source.encode("utf-8"))
    lang_name = "TSX" if file_path.endswith(".tsx") else ("TypeScript" if file_path.endswith(".ts") else "JavaScript")
    return format_syntax_error(tree.root_node, lang_name)


def _extract_parameters(params_node: Node, source_bytes: bytes) -> List[Parameter]:
    """Extract parameters from formal_parameters node."""
    params: List[Parameter] = []
    for child in params_node.children:
        if child.type in ("(", ")", ","):
            continue

        if child.type == "required_parameter":
            param_name = ""
            annotation = None
            default_val = None
            has_default = False
            is_vararg = False
            for sc in child.children:
                if sc.type == "rest_pattern":
                    is_vararg = True
                    for ssc in sc.children:
                        if ssc.type in ("identifier", "type_identifier"):
                            param_name = get_node_text(ssc, source_bytes)
                elif sc.type in ("identifier", "type_identifier"):
                    param_name = get_node_text(sc, source_bytes)
                elif sc.type == "type_annotation":
                    annotation = get_node_text(sc, source_bytes).lstrip(": ").strip()
                elif sc.type == "=":
                    has_default = True
                elif has_default and default_val is None and sc.type not in ("=", " "):
                    default_val = get_node_text(sc, source_bytes).strip()

            params.append(
                Parameter(
                    name=param_name,
                    annotation=annotation,
                    default=default_val,
                    has_default=has_default,
                    is_vararg=is_vararg,
                )
            )

        elif child.type == "optional_parameter":
            param_name = ""
            annotation = None
            for sc in child.children:
                if sc.type in ("identifier", "type_identifier"):
                    param_name = get_node_text(sc, source_bytes)
                elif sc.type == "type_annotation":
                    annotation = get_node_text(sc, source_bytes).lstrip(": ").strip()

            params.append(
                Parameter(
                    name=param_name,
                    annotation=annotation,
                    has_default=True,  # Optional parameter doesn't require caller argument
                )
            )

        elif child.type == "rest_pattern":
            param_name = ""
            for sc in child.children:
                if sc.type in ("identifier", "type_identifier"):
                    param_name = get_node_text(sc, source_bytes)
            params.append(
                Parameter(
                    name=param_name or "rest",
                    is_vararg=True,
                )
            )

        elif child.type == "identifier":
            params.append(
                Parameter(
                    name=get_node_text(child, source_bytes),
                )
            )

    return params


def _extract_calls(node: Node, source_bytes: bytes, caller_id: Optional[str] = None) -> List[CallReference]:
    """Recursively extract function and method calls inside a node."""
    calls: List[CallReference] = []

    def walk(n: Node):
        if n.type in ("call_expression", "new_expression"):
            callee_name = ""
            args_count = 0
            has_vararg = False
            kwargs: List[str] = []

            # First child or named child 'function' / 'constructor'
            fn_node = n.child_by_field_name("function") or n.child_by_field_name("constructor")
            if not fn_node and n.children:
                fn_node = n.children[1] if n.type == "new_expression" and len(n.children) > 1 else n.children[0]

            if fn_node:
                callee_name = get_node_text(fn_node, source_bytes).strip()

            args_node = n.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.children:
                    if arg.type in ("(", ")", ","):
                        continue
                    args_count += 1
                    if arg.type == "spread_element":
                        has_vararg = True
                    elif arg.type == "object":
                        # If passing object literal, collect top-level property keys as kwargs
                        for obj_child in arg.children:
                            if obj_child.type in ("pair", "shorthand_property_identifier_pair"):
                                key_node = obj_child.child_by_field_name("key") or (
                                    obj_child.children[0] if obj_child.children else None
                                )
                                if key_node:
                                    kwargs.append(get_node_text(key_node, source_bytes))

            if callee_name and callee_name != "require":
                calls.append(
                    CallReference(
                        callee=callee_name,
                        args_count=args_count,
                        kwargs=kwargs,
                        lineno=n.start_point.row + 1,
                        caller=caller_id,
                        has_vararg=has_vararg,
                    )
                )

        for child in n.children:
            walk(child)

    walk(node)
    return calls


def extract_typescript_imports(source: str, file_path: str = "") -> List[ImportReference]:
    """Extract all import statements and require calls from TypeScript/JavaScript source."""
    if not source.strip():
        return []

    parser = get_ts_parser(file_path)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    imports: List[ImportReference] = []

    def walk(node: Node):
        if node.type == "import_statement":
            source_node = node.child_by_field_name("source")
            mod_name = ""
            if source_node:
                mod_name = get_node_text(source_node, source_bytes).strip("'\"`")
            lineno = node.start_point.row + 1

            # Look for import clauses
            clause = None
            for ch in node.children:
                if ch.type == "import_clause":
                    clause = ch
                    break

            if not clause:
                # e.g. import './styles.css'
                imports.append(
                    ImportReference(
                        module=mod_name,
                        name="",
                        lineno=lineno,
                        file_path=file_path,
                    )
                )
                return

            for ch in clause.children:
                if ch.type == "identifier":
                    # Default import: import Foo from './mod'
                    imports.append(
                        ImportReference(
                            module=mod_name,
                            name=get_node_text(ch, source_bytes),
                            lineno=lineno,
                            file_path=file_path,
                        )
                    )
                elif ch.type == "named_imports":
                    for spec in ch.children:
                        if spec.type == "import_specifier":
                            name_node = spec.child_by_field_name("name")
                            alias_node = spec.child_by_field_name("alias")
                            if name_node:
                                imp_name = get_node_text(name_node, source_bytes)
                                asname = get_node_text(alias_node, source_bytes) if alias_node else None
                                imports.append(
                                    ImportReference(
                                        module=mod_name,
                                        name=imp_name,
                                        asname=asname,
                                        lineno=lineno,
                                        file_path=file_path,
                                    )
                                )
                elif ch.type == "namespace_import":
                    # import * as Foo from './mod'
                    alias_node = None
                    for nch in ch.children:
                        if nch.type == "identifier":
                            alias_node = nch
                    alias_name = get_node_text(alias_node, source_bytes) if alias_node else "all"
                    imports.append(
                        ImportReference(
                            module=mod_name,
                            name="*",
                            asname=alias_name,
                            lineno=lineno,
                            file_path=file_path,
                        )
                    )

        elif node.type == "export_statement" and node.child_by_field_name("source"):
            source_node = node.child_by_field_name("source")
            mod_name = get_node_text(source_node, source_bytes).strip("'\"`") if source_node else ""
            lineno = node.start_point.row + 1
            has_star = any(ch.type == "*" for ch in node.children)
            if has_star:
                imports.append(
                    ImportReference(
                        module=mod_name,
                        name="*",
                        lineno=lineno,
                        file_path=file_path,
                    )
                )
            else:
                for ch in node.children:
                    if ch.type == "export_clause":
                        for spec in ch.children:
                            if spec.type == "export_specifier":
                                name_node = spec.child_by_field_name("name") or (
                                    spec.children[0] if spec.children else None
                                )
                                alias_node = spec.child_by_field_name("alias")
                                if name_node:
                                    imp_name = get_node_text(name_node, source_bytes)
                                    asname = get_node_text(alias_node, source_bytes) if alias_node else None
                                    imports.append(
                                        ImportReference(
                                            module=mod_name,
                                            name=imp_name,
                                            asname=asname,
                                            lineno=lineno,
                                            file_path=file_path,
                                        )
                                    )

        elif node.type == "call_expression":
            # const mod = require('./mod')
            fn = node.child_by_field_name("function")
            if fn and get_node_text(fn, source_bytes) == "require":
                args = node.child_by_field_name("arguments")
                if args and args.children:
                    for arg in args.children:
                        if arg.type == "string":
                            mod_name = get_node_text(arg, source_bytes).strip("'\"`")
                            # Determine identifier if assigned
                            parent = node.parent
                            asname = None
                            if parent and parent.type == "variable_declarator":
                                id_node = parent.child_by_field_name("name")
                                if id_node:
                                    asname = get_node_text(id_node, source_bytes)
                            imports.append(
                                ImportReference(
                                    module=mod_name,
                                    name=asname or mod_name.split("/")[-1],
                                    asname=asname,
                                    lineno=node.start_point.row + 1,
                                    file_path=file_path,
                                )
                            )

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imports


def extract_typescript_symbols(source: str, file_path: str = "") -> List[Symbol]:
    """Parse TypeScript/JavaScript source into AST and extract symbol entities."""
    if not source.strip():
        return []

    parser = get_ts_parser(file_path)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    symbols: List[Symbol] = []

    def process_node(node: Node, parent_qualname: Optional[str] = None):
        # Unwrap export and ambient statements
        target_node = node
        if node.type in ("export_statement", "export_default_statement"):
            for ch in node.children:
                if ch.type in (
                    "function_declaration",
                    "class_declaration",
                    "interface_declaration",
                    "type_alias_declaration",
                    "enum_declaration",
                    "lexical_declaration",
                    "variable_declaration",
                    "internal_module",
                    "module",
                ):
                    target_node = ch
                    break
        elif node.type == "ambient_declaration":
            for ch in node.children:
                if ch.type in (
                    "function_declaration",
                    "class_declaration",
                    "interface_declaration",
                    "type_alias_declaration",
                    "enum_declaration",
                    "lexical_declaration",
                    "variable_declaration",
                    "internal_module",
                    "module",
                ):
                    target_node = ch
                    break

        if target_node.type == "expression_statement":
            for ch in target_node.children:
                if ch.type in ("internal_module", "module"):
                    target_node = ch
                    break

        if target_node.type == "function_declaration":
            name_node = target_node.child_by_field_name("name")
            if not name_node:
                return
            fn_name = get_node_text(name_node, source_bytes)
            qualname = f"{parent_qualname}.{fn_name}" if parent_qualname else fn_name
            sym_id = f"{file_path}::{qualname}"

            is_async = any(ch.type == "async" for ch in target_node.children)
            params_node = target_node.child_by_field_name("parameters")
            params = _extract_parameters(params_node, source_bytes) if params_node else []

            ret_node = target_node.child_by_field_name("return_type")
            ret_type = get_node_text(ret_node, source_bytes).lstrip(": ").strip() if ret_node else None

            # Calculate arity
            pos_params = [p for p in params if not p.is_vararg]
            min_args = len([p for p in pos_params if not p.has_default])
            max_args = None if any(p.is_vararg for p in params) else len(pos_params)

            # Calls inside body
            body_node = target_node.child_by_field_name("body")
            calls = _extract_calls(body_node, source_bytes, caller_id=sym_id) if body_node else []

            prefix = "async function" if is_async else "function"
            param_strs = [p.name + (f": {p.annotation}" if p.annotation else "") for p in params]
            ret_suffix = f": {ret_type}" if ret_type else ""
            signature = f"{prefix} {fn_name}({', '.join(param_strs)}){ret_suffix}"

            symbol = Symbol(
                name=fn_name,
                qualname=qualname,
                file_path=file_path,
                kind="async_function" if is_async else "function",
                lineno=target_node.start_point.row + 1,
                end_lineno=target_node.end_point.row + 1,
                signature=signature,
                params=params,
                min_args=min_args,
                max_args=max_args,
                return_type=ret_type,
                calls=calls,
                is_method=False,
                is_static=False,
            )
            symbols.append(symbol)

        elif target_node.type in ("lexical_declaration", "variable_declaration"):
            # Handle: const fn = (a, b) => ... or let fn = function(...) ...
            is_const = any(ch.type == "const" for ch in target_node.children)
            for decl in target_node.children:
                if decl.type == "variable_declarator":
                    name_node = decl.child_by_field_name("name")
                    val_node = decl.child_by_field_name("value")
                    if name_node and val_node and val_node.type in ("arrow_function", "function_expression"):
                        fn_name = get_node_text(name_node, source_bytes)
                        qualname = f"{parent_qualname}.{fn_name}" if parent_qualname else fn_name
                        sym_id = f"{file_path}::{qualname}"

                        is_async = any(ch.type == "async" for ch in val_node.children)
                        params_node = val_node.child_by_field_name("parameters")
                        params = _extract_parameters(params_node, source_bytes) if params_node else []
                        if not params_node:
                            # Single param arrow function e.g. x => x * 2
                            param_id = val_node.child_by_field_name("parameter")
                            if param_id:
                                params = [Parameter(name=get_node_text(param_id, source_bytes))]

                        ret_node = val_node.child_by_field_name("return_type")
                        ret_type = get_node_text(ret_node, source_bytes).lstrip(": ").strip() if ret_node else None

                        pos_params = [p for p in params if not p.is_vararg]
                        min_args = len([p for p in pos_params if not p.has_default])
                        max_args = None if any(p.is_vararg for p in params) else len(pos_params)

                        body_node = val_node.child_by_field_name("body")
                        calls = _extract_calls(body_node, source_bytes, caller_id=sym_id) if body_node else []

                        param_strs = [p.name + (f": {p.annotation}" if p.annotation else "") for p in params]
                        signature = f"const {fn_name} = ({', '.join(param_strs)}) => ..."

                        symbol = Symbol(
                            name=fn_name,
                            qualname=qualname,
                            file_path=file_path,
                            kind="async_function" if is_async else "function",
                            lineno=target_node.start_point.row + 1,
                            end_lineno=target_node.end_point.row + 1,
                            signature=signature,
                            params=params,
                            min_args=min_args,
                            max_args=max_args,
                            return_type=ret_type,
                            calls=calls,
                            is_method=False,
                            is_static=False,
                        )
                        symbols.append(symbol)
                    elif name_node:
                        # Top-level constant or variable
                        var_name = get_node_text(name_node, source_bytes)
                        if var_name and var_name.isidentifier():
                            qualname = f"{parent_qualname}.{var_name}" if parent_qualname else var_name
                            kind = "constant" if is_const else "variable"
                            sig_prefix = "const" if is_const else "let"
                            symbols.append(
                                Symbol(
                                    name=var_name,
                                    qualname=qualname,
                                    file_path=file_path,
                                    kind=kind,
                                    lineno=target_node.start_point.row + 1,
                                    end_lineno=target_node.end_point.row + 1,
                                    signature=f"{sig_prefix} {var_name}",
                                    min_args=0,
                                    max_args=0,
                                )
                            )

        elif target_node.type == "class_declaration":
            name_node = target_node.child_by_field_name("name")
            if not name_node:
                return
            class_name = get_node_text(name_node, source_bytes)
            qualname = f"{parent_qualname}.{class_name}" if parent_qualname else class_name
            sym_id = f"{file_path}::{qualname}"

            # Base classes & interfaces
            bases: List[str] = []
            heritage_node = None
            for ch in target_node.children:
                if ch.type == "class_heritage":
                    heritage_node = ch
                    break

            if heritage_node:
                for h_child in heritage_node.children:
                    if h_child.type in ("extends_clause", "implements_clause"):
                        for base_ch in h_child.children:
                            if base_ch.type in ("identifier", "type_identifier"):
                                bases.append(get_node_text(base_ch, source_bytes))

            class_body = target_node.child_by_field_name("body")
            class_calls = _extract_calls(class_body, source_bytes, caller_id=sym_id) if class_body else []

            bases_str = f" extends {', '.join(bases)}" if bases else ""
            signature = f"class {class_name}{bases_str}"

            class_sym = Symbol(
                name=class_name,
                qualname=qualname,
                file_path=file_path,
                kind="class",
                lineno=target_node.start_point.row + 1,
                end_lineno=target_node.end_point.row + 1,
                signature=signature,
                calls=class_calls,
                bases=bases,
            )
            symbols.append(class_sym)

            # Process methods inside class_body
            if class_body:
                for member in class_body.children:
                    if member.type == "method_definition":
                        m_name_node = member.child_by_field_name("name")
                        if not m_name_node:
                            continue
                        m_name = get_node_text(m_name_node, source_bytes)
                        m_qualname = f"{qualname}.{m_name}"
                        m_sym_id = f"{file_path}::{m_qualname}"

                        is_static = any(ch.type == "static" for ch in member.children)
                        is_async = any(ch.type == "async" for ch in member.children)
                        params_node = member.child_by_field_name("parameters")
                        raw_params = _extract_parameters(params_node, source_bytes) if params_node else []

                        # Prepend receiver 'this' if non-static method
                        if not is_static:
                            params = [Parameter(name="this", annotation=class_name)] + raw_params
                            is_method = True
                        else:
                            params = raw_params
                            is_method = False

                        ret_node = member.child_by_field_name("return_type")
                        ret_type = get_node_text(ret_node, source_bytes).lstrip(": ").strip() if ret_node else None

                        pos_params = [p for p in raw_params if not p.is_vararg]
                        min_args = len([p for p in pos_params if not p.has_default])
                        max_args = None if any(p.is_vararg for p in raw_params) else len(pos_params)

                        body_node = member.child_by_field_name("body")
                        calls = _extract_calls(body_node, source_bytes, caller_id=m_sym_id) if body_node else []

                        m_prefix = ("static " if is_static else "") + ("async " if is_async else "")
                        param_strs = [p.name + (f": {p.annotation}" if p.annotation else "") for p in raw_params]
                        ret_suffix = f": {ret_type}" if ret_type else ""
                        m_sig = f"{m_prefix}{m_name}({', '.join(param_strs)}){ret_suffix}"

                        method_sym = Symbol(
                            name=m_name,
                            qualname=m_qualname,
                            file_path=file_path,
                            kind="method" if not is_static else "function",
                            lineno=member.start_point.row + 1,
                            end_lineno=member.end_point.row + 1,
                            signature=m_sig,
                            params=params,
                            min_args=min_args,
                            max_args=max_args,
                            return_type=ret_type,
                            calls=calls,
                            is_method=is_method,
                            is_static=is_static,
                        )
                        symbols.append(method_sym)

        elif target_node.type == "interface_declaration":
            name_node = target_node.child_by_field_name("name")
            if name_node:
                if_name = get_node_text(name_node, source_bytes)
                qualname = f"{parent_qualname}.{if_name}" if parent_qualname else if_name
                symbols.append(
                    Symbol(
                        name=if_name,
                        qualname=qualname,
                        file_path=file_path,
                        kind="interface",
                        lineno=target_node.start_point.row + 1,
                        end_lineno=target_node.end_point.row + 1,
                        signature=f"interface {if_name}",
                    )
                )

        elif target_node.type == "type_alias_declaration":
            name_node = target_node.child_by_field_name("name")
            if name_node:
                t_name = get_node_text(name_node, source_bytes)
                qualname = f"{parent_qualname}.{t_name}" if parent_qualname else t_name
                symbols.append(
                    Symbol(
                        name=t_name,
                        qualname=qualname,
                        file_path=file_path,
                        kind="type_alias",
                        lineno=target_node.start_point.row + 1,
                        end_lineno=target_node.end_point.row + 1,
                        signature=f"type {t_name}",
                    )
                )

        elif target_node.type == "enum_declaration":
            name_node = target_node.child_by_field_name("name")
            if name_node:
                enum_name = get_node_text(name_node, source_bytes)
                qualname = f"{parent_qualname}.{enum_name}" if parent_qualname else enum_name
                symbols.append(
                    Symbol(
                        name=enum_name,
                        qualname=qualname,
                        file_path=file_path,
                        kind="enum",
                        lineno=target_node.start_point.row + 1,
                        end_lineno=target_node.end_point.row + 1,
                        signature=f"enum {enum_name}",
                        min_args=0,
                        max_args=0,
                    )
                )

        elif target_node.type in ("internal_module", "module"):
            name_node = target_node.child_by_field_name("name")
            ns_name = get_node_text(name_node, source_bytes).strip("'\"`") if name_node else None
            ns_qualname = f"{parent_qualname}.{ns_name}" if parent_qualname and ns_name else ns_name
            body_node = target_node.child_by_field_name("body") or next(
                (c for c in target_node.children if c.type == "statement_block"), None
            )
            if body_node:
                for b_child in body_node.children:
                    process_node(b_child, parent_qualname=ns_qualname)

    for child in tree.root_node.children:
        process_node(child)

    # Extract module-level calls
    all_calls = _extract_calls(tree.root_node, source_bytes, caller_id=f"{file_path}::<module>")
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
