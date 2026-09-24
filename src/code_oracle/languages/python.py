"""
Python AST extractor using Python's native ast module.
Preserves 100% fidelity and backward compatibility for Python verification.
"""

import ast
from typing import List, Optional

from code_oracle.models import CallReference, ImportReference, Parameter, Symbol


def _extract_calls_from_node(node: ast.AST, caller_id: Optional[str] = None) -> List[CallReference]:
    """Extract all function/method calls inside a Python AST node."""
    calls: List[CallReference] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            try:
                callee_name = ast.unparse(child.func)
            except Exception:
                callee_name = "<unknown>"
            kwargs = [kw.arg for kw in child.keywords if kw.arg is not None]
            has_vararg = any(isinstance(a, ast.Starred) for a in child.args)
            has_kwarg = any(kw.arg is None for kw in child.keywords)
            calls.append(
                CallReference(
                    callee=callee_name,
                    args_count=len(child.args),
                    kwargs=kwargs,
                    lineno=getattr(child, "lineno", 0),
                    caller=caller_id,
                    has_vararg=has_vararg,
                    has_kwarg=has_kwarg,
                )
            )
    return calls


def extract_python_imports(source: str, file_path: str = "") -> List[ImportReference]:
    """Extract all import statements from Python source."""
    if not source.strip():
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imports: List[ImportReference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportReference(
                        module=None,
                        name=alias.name,
                        asname=alias.asname,
                        lineno=getattr(node, "lineno", 0),
                        file_path=file_path,
                        level=0,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    ImportReference(
                        module=node.module,
                        name=alias.name,
                        asname=alias.asname,
                        lineno=getattr(node, "lineno", 0),
                        file_path=file_path,
                        level=node.level,
                    )
                )
    return imports


def extract_python_symbols(source: str, file_path: str = "") -> List[Symbol]:
    """Parse Python source into AST and extract symbol entities with detailed metadata."""
    if not source.strip():
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols: List[Symbol] = []

    def process_body(nodes: List[ast.stmt], parent_qualname: Optional[str] = None, is_parent_class: bool = False):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{parent_qualname}.{node.name}" if parent_qualname else node.name
                is_static = any(
                    (isinstance(d, ast.Name) and d.id == "staticmethod")
                    or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
                    for d in node.decorator_list
                )
                is_method = is_parent_class and not is_static
                kind = "method" if is_parent_class else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")

                # Parameters extraction
                args_node = node.args
                posonly_names = {a.arg for a in args_node.posonlyargs}
                pos_args_nodes = args_node.posonlyargs + args_node.args
                num_defaults = len(args_node.defaults)
                defaults_offset = len(pos_args_nodes) - num_defaults

                params: List[Parameter] = []
                for i, arg in enumerate(pos_args_nodes):
                    default_str = None
                    has_default = False
                    if i >= defaults_offset:
                        default_node = args_node.defaults[i - defaults_offset]
                        default_str = ast.unparse(default_node)
                        has_default = True
                    annotation_str = ast.unparse(arg.annotation) if arg.annotation else None
                    params.append(
                        Parameter(
                            name=arg.arg,
                            annotation=annotation_str,
                            default=default_str,
                            has_default=has_default,
                            is_posonly=(arg.arg in posonly_names),
                        )
                    )

                # Keyword-only parameters
                for i, kwarg in enumerate(args_node.kwonlyargs):
                    kw_default = args_node.kw_defaults[i]
                    has_kw_default = kw_default is not None
                    kw_default_str = ast.unparse(kw_default) if has_kw_default else None
                    annotation_str = ast.unparse(kwarg.annotation) if kwarg.annotation else None
                    params.append(
                        Parameter(
                            name=kwarg.arg,
                            annotation=annotation_str,
                            default=kw_default_str,
                            has_default=has_kw_default,
                            is_kwonly=True,
                        )
                    )

                # Varargs & kwargs
                if args_node.vararg:
                    params.append(Parameter(name=args_node.vararg.arg, is_vararg=True))
                if args_node.kwarg:
                    params.append(Parameter(name=args_node.kwarg.arg, is_kwarg=True))

                min_args = len(pos_args_nodes) - num_defaults
                max_args = None if args_node.vararg is not None else len(pos_args_nodes)

                accepted_kwargs = None if args_node.kwarg is not None else {
                    p.name for p in params if not p.is_vararg and not p.is_kwarg and not p.is_posonly
                }
                required_kwargs = {
                    kwarg.arg for i, kwarg in enumerate(args_node.kwonlyargs)
                    if args_node.kw_defaults[i] is None
                }

                # Return annotation
                return_type = ast.unparse(node.returns) if node.returns else None

                # Signature representation
                sig_params = []
                for p in params:
                    p_str = p.name
                    if p.is_vararg:
                        p_str = f"*{p.name}"
                    elif p.is_kwarg:
                        p_str = f"**{p.name}"
                    if p.annotation:
                        p_str += f": {p.annotation}"
                    if p.has_default and p.default is not None:
                        p_str += f" = {p.default}"
                    sig_params.append(p_str)
                ret_suffix = f" -> {return_type}" if return_type else ""
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                signature = f"{prefix} {node.name}({', '.join(sig_params)}){ret_suffix}"

                sym_id = f"{file_path}::{qualname}"
                calls = _extract_calls_from_node(node, caller_id=sym_id)

                symbol = Symbol(
                    name=node.name,
                    qualname=qualname,
                    file_path=file_path,
                    kind=kind,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                    signature=signature,
                    params=params,
                    min_args=min_args,
                    max_args=max_args,
                    accepted_kwargs=accepted_kwargs,
                    required_kwargs=required_kwargs,
                    return_type=return_type,
                    calls=calls,
                    is_method=is_method,
                    is_static=is_static,
                )
                symbols.append(symbol)
                process_body(node.body, parent_qualname=qualname, is_parent_class=False)

            elif isinstance(node, ast.ClassDef):
                qualname = f"{parent_qualname}.{node.name}" if parent_qualname else node.name
                bases_list = [ast.unparse(b) for b in node.bases]
                bases_str = ", ".join(bases_list)
                signature = f"class {node.name}({bases_str})" if bases_str else f"class {node.name}"
                sym_id = f"{file_path}::{qualname}"
                calls = _extract_calls_from_node(node, caller_id=sym_id)

                symbol = Symbol(
                    name=node.name,
                    qualname=qualname,
                    file_path=file_path,
                    kind="class",
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                    signature=signature,
                    calls=calls,
                    bases=bases_list,
                )
                symbols.append(symbol)
                process_body(node.body, parent_qualname=qualname, is_parent_class=True)

    process_body(tree.body)

    module_calls = [
        c for c in _extract_calls_from_node(tree, caller_id=f"{file_path}::<module>")
        if not any(s.lineno <= c.lineno <= s.end_lineno for s in symbols)
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
            signature=f"# module {file_path}",
            calls=module_calls,
        )
        symbols.append(module_sym)

    return symbols


def validate_python_syntax(source: str, file_path: str = "") -> Optional[str]:
    """Validate Python source syntax, returning an error message if invalid."""
    try:
        ast.parse(source)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}:{e.offset}: {e.msg}"
