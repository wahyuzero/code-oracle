#!/usr/bin/env python3
"""
Surviving Mutants Harvesting Tool for Code Oracle Dataset Curation.

Generates targeted, AST-aware mutations across Python, TypeScript, Go, and Rust
that preserve AST topology (interfaces, callers, dependencies remain intact).
Filters candidates against test runners or deterministic symbolic gates to harvest
genuine 'surviving mutants' representing silent regressions (SilentLogicDrift,
PerformanceRegression, ConcurrencyHazard, etc.). Exports balanced DatasetRecord
entries in JSONL format conforming to ADR-0003 multi-task risk taxonomy.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to sys.path if running directly or from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from code_oracle.dataset import TAXONOMY_CLASSES, DatasetRecord
from code_oracle.engine import TopoSliceEngine
from code_oracle.languages import detect_language, validate_syntax
from code_oracle.linearizer import estimate_tokens


# ============================================================================
# 1. Targeted Mutation Operators (AST Topology Preserving)
# ============================================================================


class MutationOperator(ABC):
    """Abstract base class for AST-aware, topology-preserving mutation operators."""

    name: str = "base_operator"
    target_languages: List[str] = []
    category: str = "silent_logic_drift"
    primary_taxonomy_class: str = "SilentLogicDrift"
    description: str = "Base mutation operator"

    @abstractmethod
    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        """Return True if content has eligible locations for this operator."""
        ...

    @abstractmethod
    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        """
        Generate list of (mutated_content, mutation_description) pairs.
        All mutations MUST preserve AST topology and syntactic validity.
        """
        ...


class TruthinessInversionOperator(MutationOperator):
    """
    Inverts boolean truthiness, conditional checks, and binary comparisons.
    Preserves AST topology (same function signature, same statement structure)
    while introducing a silent semantic logic drift.
    """

    name = "truthiness_inversion"
    target_languages = ["python", "typescript", "javascript", "go", "rust"]
    category = "silent_logic_drift"
    primary_taxonomy_class = "SilentLogicDrift"
    description = "Inverts conditional truthiness and comparison operators"

    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        if language == "python":
            return bool(
                re.search(
                    r"\b(if|elif)\s+[^:\n]+:|\s*(==|!=|<=|>=|\bis None\b|\bis not None\b|\bnot in\b|\bin\b)",
                    content,
                )
            )
        elif language in ("typescript", "javascript"):
            return bool(
                re.search(
                    r"\bif\s*\([^)\n]+\)|\s*(===|!==|==|!=|<=|>=|<|>)",
                    content,
                )
            )
        elif language == "go":
            return bool(
                re.search(
                    r"\bif\s+(err\s*!=\s*nil|err\s*==\s*nil|[^{\n]+)\s*\{|\s*(==|!=)",
                    content,
                )
            )
        elif language == "rust":
            return bool(
                re.search(
                    r"\bif\s+[^{\n]+\{|\s*(==|!=)",
                    content,
                )
            )
        return False

    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        mutants: List[Tuple[str, str]] = []
        lines = content.splitlines(keepends=True)

        for idx, line in enumerate(lines):
            # Skip pure comments or blank lines
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "/*", "*")):
                continue

            # Python-specific truthiness mutations
            if language == "python":
                # Invert 'if not x:' -> 'if x:'
                if re.search(r"^\s*(if|elif)\s+not\s+([a-zA-Z0-9_().'\" ]+):", line):
                    mut_line = re.sub(
                        r"(\b(?:if|elif)\s+)not\s+",
                        r"\1",
                        line,
                        count=1,
                    )
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Truthiness inversion: removed 'not' at line {idx+1}"))

                # Invert 'if x:' -> 'if not (x):'
                elif re.search(r"^\s*(if|elif)\s+([a-zA-Z0-9_.]+):", line):
                    mut_line = re.sub(
                        r"(\b(?:if|elif)\s+)([a-zA-Z0-9_.]+):",
                        r"\1not (\2):",
                        line,
                        count=1,
                    )
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Truthiness inversion: negated condition at line {idx+1}"))

                # Invert comparisons: ' == ' -> ' != '
                if " == " in line and not line.strip().startswith("#"):
                    mut_line = line.replace(" == ", " != ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Comparison inversion: '==' to '!=' at line {idx+1}"))
                elif " != " in line and not line.strip().startswith("#"):
                    mut_line = line.replace(" != ", " == ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Comparison inversion: '!=' to '==' at line {idx+1}"))

                # Invert 'is None' -> 'is not None'
                if " is None" in line:
                    mut_line = line.replace(" is None", " is not None", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Identity inversion: 'is None' to 'is not None' at line {idx+1}"))
                elif " is not None" in line:
                    mut_line = line.replace(" is not None", " is None", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Identity inversion: 'is not None' to 'is None' at line {idx+1}"))

            # TypeScript / JavaScript truthiness mutations
            elif language in ("typescript", "javascript"):
                if " === " in line:
                    mut_line = line.replace(" === ", " !== ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Comparison inversion: '===' to '!==' at line {idx+1}"))
                elif " !== " in line:
                    mut_line = line.replace(" !== ", " === ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Comparison inversion: '!==' to '===' at line {idx+1}"))
                elif re.search(r"\bif\s*\((!\s*([a-zA-Z0-9_.]+))\)", line):
                    mut_line = re.sub(r"\bif\s*\(!\s*([a-zA-Z0-9_.]+)\)", r"if (\1)", line, count=1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Truthiness inversion: stripped '!' at line {idx+1}"))
                elif re.search(r"\bif\s*\(([a-zA-Z0-9_.]+)\)", line):
                    mut_line = re.sub(r"\bif\s*\(([a-zA-Z0-9_.]+)\)", r"if (!\1)", line, count=1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Truthiness inversion: negated condition at line {idx+1}"))

            # Go truthiness mutations
            elif language == "go":
                if "err != nil" in line:
                    mut_line = line.replace("err != nil", "err == nil", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Go error inversion: 'err != nil' to 'err == nil' at line {idx+1}"))
                elif "err == nil" in line:
                    mut_line = line.replace("err == nil", "err != nil", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Go error inversion: 'err == nil' to 'err != nil' at line {idx+1}"))
                elif " == " in line:
                    mut_line = line.replace(" == ", " != ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Go comparison inversion: '==' to '!=' at line {idx+1}"))

            # Rust truthiness mutations
            elif language == "rust":
                if " == " in line:
                    mut_line = line.replace(" == ", " != ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Rust comparison inversion: '==' to '!=' at line {idx+1}"))
                elif " != " in line:
                    mut_line = line.replace(" != ", " == ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Rust comparison inversion: '!=' to '==' at line {idx+1}"))

            if len(mutants) >= 5:
                break

        return mutants


class MutableDefaultsOperator(MutationOperator):
    """
    Transforms safe default parameter handling into mutable default arguments
    (e.g., opts: dict = {} or items: list = []) or injects cross-invocation state leaks.
    Preserves AST topology (callers remain valid) while causing memory/state leaks.
    """

    name = "mutable_defaults"
    target_languages = ["python"]
    category = "performance_regression"
    primary_taxonomy_class = "PerformanceRegression"
    description = "Introduces mutable default argument or cross-invocation state leakage"

    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        if language != "python":
            return False
        return bool(
            re.search(
                r"def\s+\w+\([^)]*:\s*(?:Optional\[(?:dict|list|set)\]|(?:dict|list|set)\s*\|\s*None|dict|list|set|None)\s*=\s*None",
                content,
            )
            or re.search(r"def\s+\w+\([^)]*\)", content)
        )

    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        if language != "python":
            return []

        mutants: List[Tuple[str, str]] = []
        lines = content.splitlines(keepends=True)

        for idx, line in enumerate(lines):
            # Target 1: Replace safe '= None' parameter with mutable default
            if re.search(r"def\s+\w+\(", line):
                # Pattern: opts: Optional[dict] | dict | None = None -> opts: dict = {}
                if re.search(r"(\w+)\s*:\s*(?:Optional\[dict\]|dict\s*\|\s*None|dict)\s*=\s*None", line):
                    mut_line = re.sub(
                        r"(\w+)\s*:\s*(?:Optional\[dict\]|dict\s*\|\s*None|dict)\s*=\s*None",
                        r"\1: dict = {}",
                        line,
                        count=1,
                    )
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Mutable default dict argument at line {idx+1}"))

                # Pattern: items: Optional[list] | list | None = None -> items: list = []
                elif re.search(r"(\w+)\s*:\s*(?:Optional\[list\]|list\s*\|\s*None|list)\s*=\s*None", line):
                    mut_line = re.sub(
                        r"(\w+)\s*:\s*(?:Optional\[list\]|list\s*\|\s*None|list)\s*=\s*None",
                        r"\1: list = []",
                        line,
                        count=1,
                    )
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Mutable default list argument at line {idx+1}"))

                # Target 2: Append optional mutable memo parameter preserving caller compatibility
                elif len(mutants) == 0:
                    func_match = re.match(r"^(\s*def\s+\w+\((.*?)\))(\s*(?:->\s*[^:]+)?\s*:)", line.rstrip())
                    if func_match:
                        prefix = func_match.group(1)
                        params = func_match.group(2).strip()
                        suffix = func_match.group(3)
                        if "_memo" not in params:
                            if "**" in params:
                                p_pre, p_post = params.rsplit("**", 1)
                                mut_params = f"{p_pre}_memo: dict = {{}}, **{p_post}"
                            else:
                                sep = ", " if params else ""
                                mut_params = f"{params}{sep}_memo: dict = {{}}"
                            indent = " " * (len(line) - len(line.lstrip()))
                            body_indent = " " * (len(line) - len(line.lstrip()) + 4)
                            fn_name_match = re.match(r"^\s*def\s+(\w+)\(", prefix)
                            if fn_name_match:
                                fn_name = fn_name_match.group(1)
                                mut_header = f"{indent}def {fn_name}({mut_params}){suffix}\n"
                                leak_stmt = f"{body_indent}_memo[str(len(_memo))] = True\n"
                                new_code = (
                                    "".join(lines[:idx])
                                    + mut_header
                                    + leak_stmt
                                    + "".join(lines[idx + 1 :])
                                )
                                if validate_syntax(new_code, file_path) is None:
                                    mutants.append((new_code, f"Injected mutable default _memo at line {idx+1}"))

            if len(mutants) >= 3:
                break

        return mutants


class FloatingPromisesOperator(MutationOperator):
    """
    Strips 'await' keywords in TypeScript / JavaScript async functions.
    Preserves AST topology (call expression remains intact) while causing
    unhandled floating promises, race conditions, and concurrency hazards.
    """

    name = "floating_promises"
    target_languages = ["typescript", "javascript"]
    category = "concurrency_hazard"
    primary_taxonomy_class = "ConcurrencyHazard"
    description = "Removes 'await' from asynchronous operations leading to floating promises"

    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        if language not in ("typescript", "javascript"):
            return False
        return bool(re.search(r"\bawait\s+([a-zA-Z0-9_.]+\()", content))

    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        if language not in ("typescript", "javascript"):
            return []

        mutants: List[Tuple[str, str]] = []
        lines = content.splitlines(keepends=True)

        for idx, line in enumerate(lines):
            if "await " in line and not line.strip().startswith("//"):
                mut_line = line.replace("await ", "", 1)
                new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                if validate_syntax(new_code, file_path) is None:
                    mutants.append((new_code, f"Floating promise: stripped 'await' at line {idx+1}"))

            if len(mutants) >= 4:
                break

        return mutants


class OptionalChainingDriftOperator(MutationOperator):
    """
    Replaces safe optional chaining (a?.b, a?.[b], a?.()) with direct member access (a.b),
    or replaces Python dict.get(k, default) with direct subscript access d[k].
    Preserves AST topology while causing silent runtime exceptions on nullish inputs.
    """

    name = "optional_chaining_drift"
    target_languages = ["typescript", "javascript", "python"]
    category = "silent_logic_drift"
    primary_taxonomy_class = "SilentLogicDrift"
    description = "Removes safe optional chaining or fallback lookup"

    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        if language in ("typescript", "javascript"):
            return "?." in content
        elif language == "python":
            return bool(re.search(r"\.get\([^)]+\)", content))
        return False

    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        mutants: List[Tuple[str, str]] = []
        lines = content.splitlines(keepends=True)

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//")):
                continue

            if language in ("typescript", "javascript"):
                if "?." in line:
                    if "?.(" in line:
                        mut_line = line.replace("?.(", "(", 1)
                        desc = f"Optional invocation drift: '?.(' replaced with '(' at line {idx+1}"
                    else:
                        mut_line = line.replace("?.", ".", 1)
                        desc = f"Optional chaining drift: '?.' replaced with '.' at line {idx+1}"
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, desc))

            elif language == "python":
                # Replace dict.get(key, default) or dict.get(key) -> dict[key]
                match = re.search(r"(\w+)\.get\((['\"][^'\"]+['\"])(?:,\s*[^)]+)?\)", line)
                if match:
                    dict_name = match.group(1)
                    key_expr = match.group(2)
                    mut_line = line.replace(match.group(0), f"{dict_name}[{key_expr}]", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Fallback lookup drift: dict.get() replaced with subscript at line {idx+1}"))

            if len(mutants) >= 4:
                break

        return mutants


class UnhandledChannelReadOperator(MutationOperator):
    """
    In Go, removes the two-value receive check 'val, ok := <-ch', dropping 'ok'
    and leading to silent zero-value reads from closed channels or concurrency bugs.
    In Python, replaces blocking queue.get() with unsafe get_nowait().
    """

    name = "unhandled_channel_read"
    target_languages = ["go", "python"]
    category = "concurrency_hazard"
    primary_taxonomy_class = "ConcurrencyHazard"
    description = "Drops channel closure check or unhandled queue receive"

    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        if language == "go":
            return bool(re.search(r"\b(\w+),\s*(\w+)\s*:=\s*<-\s*(\w+)", content))
        elif language == "python":
            return bool(re.search(r"\.get\(timeout=", content) or re.search(r"\b\w+\.get\(\)", content))
        return False

    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        mutants: List[Tuple[str, str]] = []
        lines = content.splitlines(keepends=True)

        for idx, line in enumerate(lines):
            if language == "go":
                # Go: val, ok := <-ch or val, more := <-ch -> val := <-ch
                match = re.search(r"\b(\w+),\s*(\w+)\s*:=\s*<-\s*(\w+)", line)
                if match:
                    val_var = match.group(1)
                    bool_var = match.group(2)
                    ch_var = match.group(3)
                    mut_line = line.replace(match.group(0), f"{val_var} := <-{ch_var}", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Unhandled channel read: dropped '{bool_var}' at line {idx+1}"))

            elif language == "python":
                if ".get(timeout=" in line:
                    mut_line = re.sub(r"\.get\(timeout=[^)]*\)", ".get_nowait()", line, count=1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Queue timeout dropped to get_nowait() at line {idx+1}"))
                elif re.search(r"\b(\w+)\.get\(\)", line):
                    mut_line = re.sub(r"\b(\w+)\.get\(\)", r"\1.get_nowait()", line, count=1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Queue get() dropped to get_nowait() at line {idx+1}"))

            if len(mutants) >= 3:
                break

        return mutants


class UnclosedResourceOperator(MutationOperator):
    """
    Removes resource cleanup calls (defer f.Close(), with open(...) as f, conn.close()).
    Preserves AST topology while introducing genuine resource and connection leaks.
    """

    name = "unclosed_resource"
    target_languages = ["python", "go", "typescript", "rust"]
    category = "performance_regression"
    primary_taxonomy_class = "PerformanceRegression"
    description = "Removes resource cleanup statements causing leaks"

    def can_mutate(self, file_path: str, content: str, language: str) -> bool:
        if language == "python":
            return bool(
                re.search(r"with\s+open\([^)]+\)\s+as\s+\w+:", content)
                or re.search(r"\b\w+\.close\(\)", content)
            )
        elif language == "go":
            return bool(re.search(r"defer\s+[\w.]+\.(Close|Unlock)\(\)", content))
        elif language in ("typescript", "javascript"):
            return bool(re.search(r"\b\w+\.(destroy|close|end)\(\)", content))
        elif language == "rust":
            return bool(re.search(r"drop\(\w+\)", content))
        return False

    def generate_mutants(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Tuple[str, str]]:
        mutants: List[Tuple[str, str]] = []
        lines = content.splitlines(keepends=True)

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//")):
                continue

            if language == "python":
                # Convert 'with open(...) as f:' into unclosed 'f = open(...)'
                match = re.search(r"^(\s*)with\s+open\(([^)]+)\)\s+as\s+(\w+):", line)
                if match:
                    indent = match.group(1)
                    open_args = match.group(2)
                    var_name = match.group(3)
                    mut_line = f"{indent}{var_name} = open({open_args})  # unclosed descriptor leak\n{indent}if True:\n"
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Unclosed file descriptor leak at line {idx+1}"))
                elif re.search(r"\b(\w+\.close\(\))", line):
                    mut_line = line.replace(".close()", ".flush()", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Omitted close() at line {idx+1}"))

            elif language == "go":
                if re.search(r"defer\s+[\w.]+\.(?:Close|Unlock)\(\)", line):
                    mut_line = line.replace("defer ", "// defer ", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Go omitted defer cleanup at line {idx+1}"))

            elif language in ("typescript", "javascript"):
                match = re.search(r"(\b\w+\.(?:destroy|close|end)\(\))", line)
                if match:
                    mut_line = line.replace(match.group(1), f"// {match.group(1)}", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Omitted resource cleanup at line {idx+1}"))

            elif language == "rust":
                match = re.search(r"drop\((\w+)\)", line)
                if match:
                    res_var = match.group(1)
                    mut_line = line.replace(f"drop({res_var})", f"std::mem::forget({res_var})", 1)
                    new_code = "".join(lines[:idx]) + mut_line + "".join(lines[idx + 1 :])
                    if validate_syntax(new_code, file_path) is None:
                        mutants.append((new_code, f"Rust leaked resource via std::mem::forget at line {idx+1}"))

            if len(mutants) >= 3:
                break

        return mutants


# Register all available targeted mutation operators
ALL_OPERATORS: List[MutationOperator] = [
    TruthinessInversionOperator(),
    MutableDefaultsOperator(),
    FloatingPromisesOperator(),
    OptionalChainingDriftOperator(),
    UnhandledChannelReadOperator(),
    UnclosedResourceOperator(),
]


# ============================================================================
# 2. Mutant Harvester & Tester
# ============================================================================


class MutantHarvester:
    """
    Discovers candidate files, applies AST-aware mutations, verifies via
    TopoSliceEngine (symbolic gate), and optionally executes test runners to
    isolate genuine surviving mutants.
    """

    def __init__(
        self,
        workspace_root: Path,
        languages: Optional[List[str]] = None,
        operator_names: Optional[List[str]] = None,
        filter_symbolic_gate: bool = True,
        seed: int = 42,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.languages = languages or ["python", "typescript", "go", "rust"]
        self.filter_symbolic_gate = filter_symbolic_gate
        self.rng = random.Random(seed)

        # Select operators
        if operator_names:
            op_set = set(operator_names)
            self.operators = [op for op in ALL_OPERATORS if op.name in op_set]
        else:
            self.operators = list(ALL_OPERATORS)

        self.engine = TopoSliceEngine(workspace_root=self.workspace_root)
        self.engine.indexer.scan_workspace()

    def harvest_file(
        self,
        file_path: Path,
        test_cmd: Optional[str] = None,
        timeout: float = 15.0,
    ) -> List[Tuple[DatasetRecord, DatasetRecord]]:
        """
        Harvest surviving mutants from a single file.
        Returns a list of (negative_record, positive_record) pairs.
        """
        file_path = file_path.resolve()
        if not file_path.exists():
            return []

        lang = detect_language(str(file_path))
        if not lang or lang not in self.languages:
            return []

        try:
            orig_content = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        try:
            rel_path = str(file_path.relative_to(self.workspace_root)).replace("\\", "/")
        except ValueError:
            rel_path = str(file_path).replace("\\", "/")

        harvested_pairs: List[Tuple[DatasetRecord, DatasetRecord]] = []

        # Find applicable operators
        applicable_ops = [
            op
            for op in self.operators
            if lang in op.target_languages and op.can_mutate(str(file_path), orig_content, lang)
        ]

        for op in applicable_ops:
            candidates = op.generate_mutants(str(file_path), orig_content, lang)

            for mutated_code, desc in candidates:
                try:
                    rep_mut = self.engine.verify(
                        rel_path,
                        mutated_code,
                        original_content=orig_content,
                        is_replacement=True,
                    )
                except Exception:
                    continue

                # Strictly filter via symbolic gate: must pass AST & Tarjan cycle checks
                if self.filter_symbolic_gate and rep_mut.status != "APPROVED":
                    continue

                # 2. Test Suite Execution (Mutant Survival Filtering)
                survived = True
                if test_cmd:
                    try:
                        # Write mutated content temporarily to disk
                        file_path.write_text(mutated_code, encoding="utf-8")
                        run_res = subprocess.run(
                            test_cmd,
                            shell=True,
                            cwd=str(self.workspace_root),
                            capture_output=True,
                            timeout=timeout,
                        )
                        # If test passes, the mutant SURVIVED the test suite!
                        if run_res.returncode != 0:
                            survived = False
                    except Exception:
                        survived = False
                    finally:
                        # Rollback Resilience: always restore original file
                        file_path.write_text(orig_content, encoding="utf-8")

                if not survived:
                    continue

                # 3. Construct ADR-0003 taxonomy labels for the surviving mutant
                tax_labels = {c: 0.0 for c in TAXONOMY_CLASSES}
                primary_score = round(self.rng.uniform(0.88, 0.95), 4)
                tax_labels[op.primary_taxonomy_class] = primary_score

                # Secondary taxonomy activations where applicable
                if op.name == "mutable_defaults":
                    tax_labels["SilentLogicDrift"] = round(primary_score * 0.85, 4)
                elif op.name == "floating_promises":
                    tax_labels["SilentLogicDrift"] = round(primary_score * 0.75, 4)
                elif op.name == "optional_chaining_drift":
                    tax_labels["BreakingPublicAPI"] = round(primary_score * 0.65, 4)
                elif op.name == "unhandled_channel_read":
                    tax_labels["SilentLogicDrift"] = round(primary_score * 0.80, 4)

                neg_record = DatasetRecord(
                    input_dsl=rep_mut.linearized_subgraph,
                    label=0,
                    risk_score=primary_score,
                    category=op.category,
                    language=lang,
                    taxonomy_labels=tax_labels,
                    symbolic_gate_passed=True,
                    source_type="surviving_mutant",
                )

                # 4. Generate paired clean positive record
                # Add harmless invariant comment to verify clean pass
                comment_token = "#" if lang == "python" else "//"
                clean_pass_code = f"{comment_token} topocache-clean-pass\n" + orig_content
                rep_clean = self.engine.verify(
                    rel_path,
                    clean_pass_code,
                    original_content=orig_content,
                    is_replacement=True,
                )

                pos_record = DatasetRecord(
                    input_dsl=rep_clean.linearized_subgraph,
                    label=1,
                    risk_score=round(self.rng.uniform(0.02, 0.12), 4),
                    category="clean_pass",
                    language=lang,
                    taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                    symbolic_gate_passed=True,
                    source_type="surviving_mutant_base",
                )

                harvested_pairs.append((neg_record, pos_record))
                if len(harvested_pairs) >= 6:
                    return harvested_pairs

        return harvested_pairs

    def harvest_directory(
        self,
        target_dir: Path,
        max_samples: int = 50,
        test_cmd: Optional[str] = None,
        balance: bool = True,
    ) -> List[DatasetRecord]:
        """
        Recursively discover source files in target_dir and harvest surviving mutants.
        """
        target_dir = Path(target_dir).resolve()
        matching_files: List[Path] = []

        ext_map = {
            "python": [".py"],
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "go": [".go"],
            "rust": [".rs"],
        }
        allowed_exts = set()
        for lang in self.languages:
            allowed_exts.update(ext_map.get(lang, []))

        ignored_dirs = {
            "test",
            "tests",
            "__tests__",
            "testing",
            ".git",
            ".code_oracle",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            "node_modules",
            "target",
            "vendor",
        }
        test_suffixes = (
            "_test.py",
            "_test.go",
            "_test.rs",
            ".test.ts",
            ".test.tsx",
            ".test.js",
            ".test.jsx",
            ".spec.ts",
            ".spec.tsx",
            ".spec.js",
            ".spec.jsx",
        )

        if target_dir.is_file():
            matching_files = [target_dir]
        else:
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d.lower() not in ignored_dirs]
                for f in files:
                    if f.startswith("test_") or f.lower().endswith(test_suffixes):
                        continue
                    ext = Path(f).suffix.lower()
                    if ext in allowed_exts:
                        matching_files.append(Path(root) / f)

        all_negatives: List[DatasetRecord] = []
        all_positives: List[DatasetRecord] = []

        for fpath in matching_files:
            pairs = self.harvest_file(fpath, test_cmd=test_cmd)
            for neg, pos in pairs:
                all_negatives.append(neg)
                all_positives.append(pos)
                if len(all_negatives) >= max_samples // 2:
                    break
            if len(all_negatives) >= max_samples // 2:
                break

        if balance:
            # Pair count
            pair_count = min(len(all_negatives), len(all_positives))
            records = []
            for i in range(pair_count):
                records.append(all_positives[i])
                records.append(all_negatives[i])
            self.rng.shuffle(records)
            return records[:max_samples]

        combined = all_negatives + all_positives
        self.rng.shuffle(combined)
        return combined[:max_samples]


def resolve_workspace_root(target_path: Path) -> Path:
    """Find the enclosing workspace or repository root for target_path."""
    p = target_path.resolve()
    if p.is_file():
        p = p.parent
    curr = p
    while curr != curr.parent:
        if (
            (curr / ".git").exists()
            or (curr / "go.mod").exists()
            or (curr / "Cargo.toml").exists()
            or (curr / "package.json").exists()
            or (curr / "pyproject.toml").exists()
        ):
            return curr
        curr = curr.parent
    return p


def export_mutants_to_jsonl(
    records: List[DatasetRecord],
    output_path: Path,
) -> int:
    """Export dataset records to JSONL enforcing < 400 token ceiling."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compliant = [r for r in records if estimate_tokens(r.input_dsl) <= 400]
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in compliant:
            f.write(json.dumps(rec.to_dict()) + "\n")

    return len(compliant)


def main():
    parser = argparse.ArgumentParser(
        description="Surviving Mutants Harvesting Tool for Code Oracle Dataset Curation."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=REPO_ROOT / "src",
        help="Target directory or file to mutate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "dataset_surviving_mutants.jsonl",
        help="Path for output JSONL dataset.",
    )
    parser.add_argument(
        "--test-cmd",
        type=str,
        default=None,
        help="Optional test command to run (e.g. 'pytest tests/test_calc.py').",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,typescript,go,rust",
        help="Comma-separated target languages.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Optional workspace root directory (auto-detected from target-dir if omitted).",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default=None,
        help="Comma-separated operator names (default: all operators).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum total records to harvest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balancing.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_false",
        dest="balance",
        help="Do not balance positive and negative samples.",
    )

    args = parser.parse_args()
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    ops = [o.strip() for o in args.operators.split(",")] if args.operators else None
    ws_root = args.workspace_root or resolve_workspace_root(args.target_dir)

    harvester = MutantHarvester(
        workspace_root=ws_root,
        languages=langs,
        operator_names=ops,
        filter_symbolic_gate=True,
        seed=args.seed,
    )

    records = harvester.harvest_directory(
        target_dir=args.target_dir,
        max_samples=args.max_samples,
        test_cmd=args.test_cmd,
        balance=args.balance,
    )

    written = export_mutants_to_jsonl(records, args.output)
    print(f"[+] Harvested {written} compliant surviving mutant records to {args.output}")


if __name__ == "__main__":
    main()
