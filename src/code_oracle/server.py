"""
Code Oracle: Lean FastMCP Server Interface.
Exposes a single minimal verification endpoint to prevent agent context bloat.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from code_oracle.engine import TopoSliceEngine

_global_engine: Optional[TopoSliceEngine] = None


def get_engine(workspace_dir: Optional[str] = None) -> TopoSliceEngine:
    """Get or create singleton TopoSlice engine instance for workspace."""
    global _global_engine
    target_root = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
    if _global_engine is None or _global_engine.workspace_root != target_root:
        _global_engine = TopoSliceEngine(workspace_root=target_root)
    return _global_engine


def verify_patch(
    file_path: str,
    patch_content: str,
    workspace_dir: Optional[str] = None,
    enable_neural: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Lean verification endpoint for AI coding agents.
    Evaluates AST topology and neuro-symbolic invariants in sub-50ms.
    """
    engine = get_engine(workspace_dir)
    if enable_neural is True:
        engine.decision_head.enable_neural_head()
        engine.enable_neural = True
    elif enable_neural is False:
        engine.enable_neural = False
        engine.decision_head.enabled = False

    report = engine.verify(file_path=file_path, patch_content=patch_content)
    return report.to_dict()


def run_dead_code_detection(
    workspace_dir: Optional[str] = None,
    paths: Optional[List[str]] = None,
    min_lines: int = 0,
    include_unexported: bool = False,
) -> Dict[str, Any]:
    """
    Dead code detection endpoint for AI coding agents.
    Evaluates symbol reachability graph in sub-50ms across Python, TS, Go, and Rust.
    """
    from code_oracle.dead_code import detect_dead_code as _detect

    engine = get_engine(workspace_dir)
    report = _detect(
        workspace_root=engine.workspace_root,
        indexer=engine.indexer,
        paths=paths,
        min_lines=min_lines,
        include_unexported=include_unexported,
    )
    return report.to_dict()


detect_dead_code = run_dead_code_detection


def run_perf_lint(
    file_path: str,
    patch_content: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    severity: str = "warn",
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Performance anti-pattern and resource leak detection endpoint for AI coding agents.
    Evaluates AST subtrees in sub-50ms across Python, TypeScript, Go, and Rust.
    """
    from code_oracle.perf_lint import lint_performance_patterns as _lint_patterns

    engine = get_engine(workspace_dir)
    report = _lint_patterns(
        file_path=file_path,
        patch_content=patch_content,
        workspace_root=engine.workspace_root,
        severity=severity,
        max_depth=max_depth,
    )
    return report.to_dict()


lint_performance_patterns = run_perf_lint


def run_server():
    """Start the FastMCP server."""
    try:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Code-Oracle")

        @mcp.tool()
        def verify_code_patch(
            file_path: str,
            patch_content: str,
            neural: bool = False,
        ) -> Dict[str, Any]:
            """
            Verify code modification topology and contract invariants in sub-50ms.
            Set neural=True to activate deep Laya ModernBERT risk scoring.
            """
            return verify_patch(file_path, patch_content, enable_neural=neural)

        @mcp.tool()
        def detect_dead_code(
            workspace_dir: Optional[str] = None,
            paths: Optional[List[str]] = None,
            min_lines: int = 0,
            include_unexported: bool = False,
        ) -> Dict[str, Any]:
            """
            Detect unreachable, orphan, and transitively dead symbols in sub-50ms.
            Multi-language support across Python, TypeScript, Go, and Rust.
            """
            return run_dead_code_detection(
                workspace_dir=workspace_dir,
                paths=paths,
                min_lines=min_lines,
                include_unexported=include_unexported,
            )

        @mcp.tool()
        def lint_performance_patterns(
            file_path: str,
            patch_content: Optional[str] = None,
            workspace_dir: Optional[str] = None,
            severity: str = "warn",
            max_depth: Optional[int] = None,
        ) -> Dict[str, Any]:
            """
            Detect performance anti-patterns (nested loops, N+1 queries, resource leaks, blocking async calls) in sub-50ms.
            Multi-language support across Python, TypeScript, Go, and Rust.
            """
            return run_perf_lint(
                file_path=file_path,
                patch_content=patch_content,
                workspace_dir=workspace_dir,
                severity=severity,
                max_depth=max_depth,
            )

        mcp.run()
    except ImportError:
        print("MCP library not found. Running in standalone CLI mode.")


if __name__ == "__main__":
    run_server()
