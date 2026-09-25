"""
Orchestration Engine for Static Performance Anti-Patterns & Resource Leak Detector.
Coordinates workspace and file-level scans, applies diff patches in-memory,
filters diagnostics by severity thresholds, and benchmarks analysis latency.
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Union

from code_oracle.indexer import IGNORE_DIRS
from code_oracle.languages import SUPPORTED_EXTENSIONS, detect_language
from code_oracle.locator import apply_patch
from code_oracle.perf_lint.models import PerfDiagnostic, PerfReport, Severity
from code_oracle.perf_lint.visitor import PerfLintVisitor


class PerfLintEngine:
    """
    Sub-50ms Static Performance Anti-Pattern & Resource Leak Engine.
    Executes Tree-sitter AST visitor passes over Python, TypeScript, Go, and Rust files.
    """

    def __init__(self, workspace_root: Optional[Union[str, Path]] = None) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()

    def lint_source(
        self,
        source: str,
        file_path: str = "",
        language: Optional[str] = None,
        severity: str = "warn",
        max_depth: Optional[int] = None,
    ) -> List[PerfDiagnostic]:
        """
        Analyze in-memory source code string and return active diagnostics.
        Filters findings by minimum severity threshold.
        """
        visitor = PerfLintVisitor(
            source=source,
            file_path=file_path,
            language=language,
            max_depth=max_depth,
        )
        diagnostics = visitor.run()

        # Apply minimum severity filter
        min_sev = Severity.from_str(severity)
        if min_sev == Severity.ERROR:
            diagnostics = [d for d in diagnostics if d.severity == Severity.ERROR]

        return diagnostics

    def lint_file(
        self,
        file_path: Union[str, Path],
        severity: str = "warn",
        max_depth: Optional[int] = None,
    ) -> List[PerfDiagnostic]:
        """Read and analyze a single source file from disk."""
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = (self.workspace_root / target_path).resolve()

        if not target_path.is_file():
            return []

        ext = target_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return []

        try:
            content = target_path.read_text(encoding="utf-8")
        except Exception:
            return []

        try:
            rel_path = str(target_path.relative_to(self.workspace_root)).replace("\\", "/")
        except ValueError:
            rel_path = str(target_path).replace("\\", "/")

        return self.lint_source(
            source=content,
            file_path=rel_path,
            severity=severity,
            max_depth=max_depth,
        )

    def lint_workspace(
        self,
        paths: Optional[List[str]] = None,
        severity: str = "warn",
        max_depth: Optional[int] = None,
    ) -> PerfReport:
        """
        Scan workspace files or specific file/directory targets for performance anti-patterns.
        """
        start_time = time.perf_counter()
        target_files: Set[Path] = set()

        if paths:
            for p in paths:
                raw_path = Path(p)
                abs_path = raw_path if raw_path.is_absolute() else (self.workspace_root / raw_path).resolve()

                if abs_path.is_file():
                    if abs_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        target_files.add(abs_path)
                elif abs_path.is_dir():
                    for root, dirs, files in os.walk(abs_path):
                        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
                        for file in files:
                            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                                target_files.add(Path(root) / file)
        else:
            for root, dirs, files in os.walk(self.workspace_root):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
                for file in files:
                    if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                        target_files.add(Path(root) / file)

        all_diagnostics: List[PerfDiagnostic] = []
        scanned_count = len(target_files)

        for f_path in target_files:
            try:
                rel_path = str(f_path.relative_to(self.workspace_root)).replace("\\", "/")
            except ValueError:
                rel_path = str(f_path).replace("\\", "/")

            try:
                content = f_path.read_text(encoding="utf-8")
            except Exception:
                continue

            diags = self.lint_source(
                source=content,
                file_path=rel_path,
                severity=severity,
                max_depth=max_depth,
            )
            all_diagnostics.extend(diags)

        # Sort diagnostics by file path and line number
        all_diagnostics.sort(key=lambda d: (d.file_path, d.lineno, d.col_offset))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PerfReport(
            workspace_root=str(self.workspace_root),
            diagnostics=all_diagnostics,
            scanned_files_count=scanned_count,
            latency_ms=elapsed_ms,
        )


def lint_performance(
    workspace_root: Optional[Union[str, Path]] = None,
    paths: Optional[List[str]] = None,
    severity: str = "warn",
    max_depth: Optional[int] = None,
) -> PerfReport:
    """Convenience function to scan a workspace or path list for performance anti-patterns."""
    engine = PerfLintEngine(workspace_root=workspace_root)
    return engine.lint_workspace(paths=paths, severity=severity, max_depth=max_depth)


def lint_performance_patterns(
    file_path: str,
    patch_content: Optional[str] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    severity: str = "warn",
    max_depth: Optional[int] = None,
) -> PerfReport:
    """
    FastMCP and patch-level performance evaluation endpoint.
    Applies unified diff or replacement patch in-memory with zero side-effects.
    """
    start_time = time.perf_counter()
    engine = PerfLintEngine(workspace_root=workspace_root)

    target_path = Path(file_path)
    if not target_path.is_absolute():
        target_path = (engine.workspace_root / target_path).resolve()

    try:
        norm_path = str(target_path.relative_to(engine.workspace_root)).replace("\\", "/")
    except ValueError:
        norm_path = str(file_path).replace("\\", "/")

    orig_content = ""
    if target_path.is_file():
        try:
            orig_content = target_path.read_text(encoding="utf-8")
        except Exception:
            orig_content = ""

    if patch_content is not None:
        patched_content, _, _ = apply_patch(orig_content, patch_content)
        content_to_lint = patched_content
    else:
        content_to_lint = orig_content

    diags = engine.lint_source(
        source=content_to_lint,
        file_path=norm_path,
        severity=severity,
        max_depth=max_depth,
    )
    diags.sort(key=lambda d: (d.file_path, d.lineno, d.col_offset))
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return PerfReport(
        workspace_root=str(engine.workspace_root),
        diagnostics=diags,
        scanned_files_count=1,
        latency_ms=elapsed_ms,
    )


__all__ = [
    "PerfLintEngine",
    "PerfLintVisitor",
    "lint_performance",
    "lint_performance_patterns",
]
