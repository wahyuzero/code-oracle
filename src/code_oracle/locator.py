"""
Stage 1: Diff Boundary Locator.
Maps patch diff line changes to AST symbol spans (functions, methods, classes).
"""

import ast
import difflib
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from code_oracle.languages import extract_imports, extract_symbols, validate_syntax
from code_oracle.models import CallReference, DiffHunk, ImportReference, Parameter, PatchResult, Symbol


def parse_unified_diff(diff_text: str) -> List[DiffHunk]:
    """Parse unified diff text into structured DiffHunk objects."""
    hunk_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    hunks: List[DiffHunk] = []
    current_hunk: Optional[DiffHunk] = None

    for line in diff_text.splitlines():
        match = hunk_pattern.match(line)
        if match:
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3)) if match.group(3) else 1
            new_count = int(match.group(4)) if match.group(4) else 1
            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=[],
            )
            hunks.append(current_hunk)
        elif current_hunk is not None:
            if line.startswith(("--- ", "+++ ", "diff --git", "index ")):
                current_hunk = None
            elif line.startswith(("+", "-", " ", "\\")):
                current_hunk.lines.append(line)
            elif line == "":
                current_hunk.lines.append(" ")

    return hunks


def apply_patch(
    original_text: str,
    patch_text: str,
    is_replacement: bool = False,
) -> Tuple[str, Set[int], Set[int]]:
    """
    Applies unified diff or replacement text in memory.
    Returns:
        (patched_text, modified_old_lines, modified_new_lines)
    """
    hunks = [] if is_replacement else parse_unified_diff(patch_text)

    # If no diff hunks found or explicitly replacement, treat patch_text as full replacement content
    if is_replacement or not hunks:
        orig_lines = original_text.splitlines(keepends=True)
        patched_lines = patch_text.splitlines(keepends=True)

        if not original_text.strip():
            # Brand new file
            new_lines = set(range(1, len(patched_lines) + 1))
            return patch_text, set(), new_lines

        # Calculate line differences via difflib
        matcher = difflib.SequenceMatcher(None, orig_lines, patched_lines)
        old_modified = set()
        new_modified = set()
        for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
            if tag != "equal":
                for l in range(alo + 1, ahi + 1):
                    old_modified.add(l)
                for l in range(blo + 1, bhi + 1):
                    new_modified.add(l)
        return patch_text, old_modified, new_modified

    # Apply unified diff hunks line by line
    orig_lines = original_text.splitlines(keepends=True)
    out_lines: List[str] = []
    orig_idx = 0  # 0-indexed cursor into orig_lines
    old_modified = set()
    new_modified = set()
    current_new_line = 1

    for hunk in hunks:
        target_orig_idx = max(0, hunk.old_start - 1) if hunk.old_start > 0 else 0
        while orig_idx < target_orig_idx and orig_idx < len(orig_lines):
            out_lines.append(orig_lines[orig_idx])
            orig_idx += 1
            current_new_line += 1

        cur_old_line = hunk.old_start
        for line in hunk.lines:
            if line.startswith("+"):
                content = line[1:]
                if not content.endswith("\n"):
                    content += "\n"
                out_lines.append(content)
                new_modified.add(current_new_line)
                current_new_line += 1
            elif line.startswith("-"):
                old_modified.add(cur_old_line)
                cur_old_line += 1
                orig_idx += 1
            elif line.startswith(" "):
                content = line[1:]
                if not content.endswith("\n"):
                    content += "\n"
                out_lines.append(content)
                cur_old_line += 1
                orig_idx += 1
                current_new_line += 1
            # Ignore '\ No newline at end of file'

    while orig_idx < len(orig_lines):
        out_lines.append(orig_lines[orig_idx])
        orig_idx += 1
        current_new_line += 1

    return "".join(out_lines), old_modified, new_modified


def _extract_calls_from_node(node: ast.AST, caller_id: Optional[str] = None) -> List[CallReference]:
    """Extract all function/method calls inside an AST node."""
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


def extract_imports_from_ast(source: str, file_path: str = "") -> List[ImportReference]:
    """Extract all import statements from source for any supported language."""
    return extract_imports(source, file_path=file_path)


def extract_symbols_from_ast(source: str, file_path: str = "") -> List[Symbol]:
    """Parse source into AST and extract symbol entities with detailed metadata."""
    if not source.strip():
        return []

    from code_oracle.languages import detect_language, extract_symbols
    lang = detect_language(file_path)
    if lang and lang != "python":
        return extract_symbols(source, file_path=file_path, language=lang)

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
                # Walk inner functions/classes
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

    # Extract module-level calls (calls outside all functions and classes)
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


def locate_affected_symbols(
    file_path: str,
    patch_content: str,
    workspace_root: Optional[Path] = None,
    original_content: Optional[str] = None,
    is_replacement: bool = False,
) -> PatchResult:
    """
    Stage 1 Diff Boundary Locator:
    Maps patch changes to precise AST symbol spans.
    """
    full_path = Path(file_path)
    if workspace_root:
        if not full_path.is_absolute():
            full_path = (workspace_root / file_path).resolve()
        else:
            full_path = full_path.resolve()
        try:
            clean_path = str(full_path.relative_to(workspace_root.resolve())).replace("\\", "/")
        except ValueError:
            clean_path = str(file_path).replace("\\", "/")
    else:
        clean_path = str(file_path).replace("\\", "/")

    # Read original file if not supplied
    if original_content is None:
        if full_path.exists() and full_path.is_file():
            try:
                original_content = full_path.read_text(encoding="utf-8")
            except Exception:
                original_content = ""
        else:
            original_content = ""

    # Apply patch
    patched_content, old_lines, new_lines = apply_patch(
        original_content, patch_content, is_replacement=is_replacement
    )

    # Check syntax of patched content
    syntax_error = validate_syntax(patched_content, file_path=clean_path)
    if syntax_error:
        return PatchResult(
            file_path=clean_path,
            original_content=original_content,
            patched_content=patched_content,
            modified_old_lines=old_lines,
            modified_new_lines=new_lines,
            syntax_error=syntax_error,
        )

    orig_symbols = extract_symbols_from_ast(original_content, file_path=clean_path)
    patched_symbols = extract_symbols_from_ast(patched_content, file_path=clean_path)
    patched_imports = extract_imports_from_ast(patched_content, file_path=clean_path)

    orig_map = {s.qualname: s for s in orig_symbols}
    patched_map = {s.qualname: s for s in patched_symbols}

    # Identify deleted and added symbols
    deleted_symbols = [s for q, s in orig_map.items() if q not in patched_map]
    added_symbols = [s for q, s in patched_map.items() if q not in orig_map]

    # Identify affected symbols (modified or newly added)
    affected: List[Symbol] = []
    seen_qualnames = set()

    for sym in patched_symbols:
        if sym.kind == "module":
            continue
        # Check if any new line overlaps with symbol span
        sym_lines = set(range(sym.lineno, sym.end_lineno + 1))
        if sym_lines.intersection(new_lines):
            affected.append(sym)
            seen_qualnames.add(sym.qualname)

    # Also check if old symbols were modified/removed and not yet caught
    for sym in orig_symbols:
        if sym.kind == "module":
            continue
        sym_lines = set(range(sym.lineno, sym.end_lineno + 1))
        if sym_lines.intersection(old_lines):
            if sym.qualname in patched_map and sym.qualname not in seen_qualnames:
                affected.append(patched_map[sym.qualname])
                seen_qualnames.add(sym.qualname)

    # Check if lines outside any function/class symbol were modified
    non_module_patched = [s for s in patched_symbols if s.kind != "module"]
    non_module_orig = [s for s in orig_symbols if s.kind != "module"]
    has_module_level_change = (
        any(not any(s.lineno <= l <= s.end_lineno for s in non_module_patched) for l in new_lines)
        or any(not any(s.lineno <= l <= s.end_lineno for s in non_module_orig) for l in old_lines)
    )

    if has_module_level_change or (not affected and (old_lines or new_lines)):
        module_sym = next((s for s in patched_symbols if s.qualname == "<module>"), None)
        if not module_sym:
            line_count = len(patched_content.splitlines()) or 1
            module_calls: List[CallReference] = []
            if clean_path.endswith(".py") or not clean_path:
                try:
                    tree = ast.parse(patched_content)
                    module_calls = [
                        c for c in _extract_calls_from_node(tree, caller_id=f"{clean_path}::<module>")
                        if not any(s.lineno <= c.lineno <= s.end_lineno for s in non_module_patched)
                    ]
                except Exception:
                    pass

            module_sym = Symbol(
                name="<module>",
                qualname="<module>",
                file_path=clean_path,
                kind="module",
                lineno=1,
                end_lineno=line_count,
                signature=f"# module {clean_path}",
                calls=module_calls,
            )
            patched_symbols.append(module_sym)

        if module_sym not in affected:
            affected.append(module_sym)

    return PatchResult(
        file_path=clean_path,
        original_content=original_content,
        patched_content=patched_content,
        modified_old_lines=old_lines,
        modified_new_lines=new_lines,
        affected_symbols=affected,
        deleted_symbols=deleted_symbols,
        added_symbols=added_symbols,
        all_patched_symbols=patched_symbols,
        imports=patched_imports,
    )
