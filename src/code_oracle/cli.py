"""
Command Line Interface for Code Oracle (TopoSlice).
Entrypoint for the `code-oracle` command.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from code_oracle import __version__
from code_oracle.engine import TopoSliceEngine
from code_oracle.indexer import WorkspaceIndexer
from code_oracle.linearizer import linearize_subgraph
from code_oracle.locator import locate_affected_symbols
from code_oracle.models import GateResult, PatchResult
from code_oracle.server import run_server
from code_oracle.slicer import slice_neighborhood


def format_report_pretty(report_dict: dict) -> str:
    """Format verification report into clean, readable terminal output."""
    status = report_dict["status"]
    conf = report_dict["confidence"]
    latency = report_dict["latency_ms"]

    color_prefix = "\033[92m" if status == "APPROVED" else "\033[91m"
    color_reset = "\033[0m"

    lines = [
        f"{color_prefix}===================================================={color_reset}",
        f"{color_prefix} VERDICT: {status} (Confidence: {conf}) in {latency} ms{color_reset}",
        f"{color_prefix}===================================================={color_reset}",
    ]

    violations = report_dict.get("invariant_violations", [])
    if violations:
        lines.append("\nViolations:")
        for v in violations:
            lines.append(f"  \033[91m✖\033[0m {v}")

    cycles = report_dict.get("cycles_detected", [])
    if cycles:
        lines.append("\nCycles Detected:")
        for c in cycles:
            lines.append(f"  \033[93m↺\033[0m {' -> '.join(c)} -> {c[0] if c else ''}")

    dsl = report_dict.get("linearized_subgraph", "")
    if dsl:
        lines.append("\nLinearized Subgraph (< 400 tokens):")
        lines.append("----------------------------------------------------")
        lines.append(dsl)
        lines.append("----------------------------------------------------")

    return "\n".join(lines)


def cmd_verify(args: argparse.Namespace) -> int:
    """Execute patch verification."""
    file_path = args.file
    patch_content = ""

    if args.patch == "-":
        patch_content = sys.stdin.read()
    elif args.patch:
        # Check if patch is a file path or direct string
        p_path = Path(args.patch)
        if p_path.is_file():
            patch_content = p_path.read_text(encoding="utf-8")
        elif args.workspace and (Path(args.workspace) / args.patch).is_file():
            patch_content = (Path(args.workspace) / args.patch).read_text(encoding="utf-8")
        else:
            patch_content = args.patch
    elif not sys.stdin.isatty():
        # Read from stdin
        patch_content = sys.stdin.read()
    else:
        print("Error: No patch content provided. Use --patch <file_or_diff> or pipe via stdin.", file=sys.stderr)
        return 2

    engine = TopoSliceEngine(workspace_root=Path(args.workspace) if args.workspace else None)
    report = engine.verify(file_path=file_path, patch_content=patch_content, k=args.k)
    report_dict = report.to_dict()

    if args.json:
        print(json.dumps(report_dict, indent=2))
    else:
        print(format_report_pretty(report_dict))

    return 0 if report.status == "APPROVED" else 1


def cmd_index(args: argparse.Namespace) -> int:
    """Index workspace symbols into .code_oracle/index.json."""
    ws = Path(args.workspace) if args.workspace else Path.cwd()
    indexer = WorkspaceIndexer(workspace_root=ws)
    stats = indexer.scan_workspace(force=args.force)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Workspace indexed successfully: {ws}")
        print(f"  Scanned files:    {stats['scanned']}")
        print(f"  Reindexed files:  {stats['reindexed']}")
        print(f"  Symbols indexed:  {stats['symbols_indexed']}")
        print(f"  Latency:          {stats['latency_ms']:.2f} ms")
        print(f"  Index location:   {indexer.index_file}")

    return 0


def cmd_slice(args: argparse.Namespace) -> int:
    """Extract k-hop subgraph slice for a symbol."""
    ws = Path(args.workspace) if args.workspace else Path.cwd()
    indexer = WorkspaceIndexer(workspace_root=ws)
    indexer.scan_workspace()

    symbol = indexer.get_definition(args.symbol)
    if not symbol:
        # Search by file and name
        matches = [
            s for s in indexer.get_file_symbols(args.file)
            if s.name == args.symbol or s.qualname == args.symbol
        ]
        if matches:
            symbol = matches[0]

    if not symbol:
        err_msg = f"Error: Symbol '{args.symbol}' not found in {args.file} or workspace."
        if getattr(args, "json", False):
            print(json.dumps({"error": err_msg}, indent=2))
        else:
            print(err_msg, file=sys.stderr)
        return 1

    graph = slice_neighborhood(seeds=[symbol], indexer=indexer, k=args.k)
    dummy_patch = PatchResult(
        file_path=args.file,
        original_content="",
        patched_content="",
        affected_symbols=[symbol],
    )
    dummy_gate = GateResult(status="APPROVED", confidence=1.0)
    dsl = linearize_subgraph(dummy_patch, graph, dummy_gate)

    if getattr(args, "json", False):
        slice_data = {
            "symbol": symbol.qualname,
            "file": symbol.file_path,
            "k": args.k,
            "truncated": graph.truncated,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "file_path": n.file_path,
                    "kind": n.kind,
                    "signature": n.signature,
                    "is_seed": n.is_seed,
                    "is_modified": n.is_modified,
                }
                for n in graph.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation}
                for e in graph.edges
            ],
            "linearized_subgraph": dsl,
        }
        print(json.dumps(slice_data, indent=2))
    else:
        print(dsl)

    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Safely remove .code_oracle/ cache directory (Rollback Resilience)."""
    ws = Path(args.workspace) if args.workspace else Path.cwd()
    indexer = WorkspaceIndexer(workspace_root=ws)
    cleaned = indexer.clean()
    if getattr(args, "json", False):
        print(json.dumps({"cleaned": cleaned, "workspace": str(ws)}, indent=2))
    else:
        if cleaned:
            print(f"Code Oracle cache cleaned successfully from {ws}")
        else:
            print(f"Notice: Cache directory not found or already clean in {ws}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the FastMCP server."""
    run_server()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="code-oracle",
        description="Sub-50ms Neuro-Symbolic Verification Oracle for AI Coding Agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify a code patch proposal")
    p_verify.add_argument("file", help="Target source file being patched")
    p_verify.add_argument("--patch", "-p", help="Patch diff string or path to diff file (use '-' for stdin)")
    p_verify.add_argument("--k", type=int, default=1, help="k-hop neighborhood radius (default: 1)")
    p_verify.add_argument("--workspace", "-w", help="Workspace root directory")
    p_verify.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_verify.set_defaults(func=cmd_verify)

    # index
    p_index = subparsers.add_parser("index", help="Index workspace symbols into .code_oracle/index.json")
    p_index.add_argument("workspace", nargs="?", default=".", help="Workspace root directory")
    p_index.add_argument("--force", "-f", action="store_true", help="Force re-indexing of all files")
    p_index.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_index.set_defaults(func=cmd_index)

    # slice
    p_slice = subparsers.add_parser("slice", help="Extract k-hop neighborhood slice for a symbol")
    p_slice.add_argument("file", help="Source file containing the symbol")
    p_slice.add_argument("--symbol", "-s", required=True, help="Target symbol name")
    p_slice.add_argument("--k", type=int, default=1, help="k-hop depth (1 or 2)")
    p_slice.add_argument("--workspace", "-w", help="Workspace root directory")
    p_slice.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_slice.set_defaults(func=cmd_slice)

    # clean
    p_clean = subparsers.add_parser("clean", help="Clean .code_oracle/ cache (Rollback Resilience)")
    p_clean.add_argument("workspace", nargs="?", default=".", help="Workspace root directory")
    p_clean.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_clean.set_defaults(func=cmd_clean)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run the Lean FastMCP server")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        sys.exit(2)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
