"""
Code Oracle: Lean FastMCP Server Interface.
Exposes a single minimal verification endpoint to prevent agent context bloat.
"""

from pathlib import Path
from typing import Any, Dict, Optional

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
    file_path: str, patch_content: str, workspace_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lean verification endpoint for AI coding agents.
    Evaluates AST topology and neuro-symbolic invariants in sub-50ms.
    """
    engine = get_engine(workspace_dir)
    report = engine.verify(file_path=file_path, patch_content=patch_content)
    return report.to_dict()


def run_server():
    """Start the FastMCP server."""
    try:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Code-Oracle")

        @mcp.tool()
        def verify_code_patch(file_path: str, patch_content: str) -> Dict[str, Any]:
            """Verify code modification topology and contract invariants in sub-50ms."""
            return verify_patch(file_path, patch_content)

        mcp.run()
    except ImportError:
        print("MCP library not found. Running in standalone CLI mode.")


if __name__ == "__main__":
    run_server()
