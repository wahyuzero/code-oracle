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
from code_oracle.config import load_config, resolve_workspace_root, set_enabled, set_mode
from code_oracle.engine import TopoSliceEngine
from code_oracle.hook import (
    format_hook_output,
    get_hook_status,
    install_hook,
    run_hook_verification,
    uninstall_hook,
)
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

    risk = report_dict.get("risk_score")
    risk_str = f" | Risk Score: {risk:.4f}" if risk is not None else ""

    lines = [
        f"{color_prefix}===================================================={color_reset}",
        f"{color_prefix} VERDICT: {status} (Confidence: {conf}{risk_str}) in {latency} ms{color_reset}",
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

    engine = TopoSliceEngine(
        workspace_root=Path(args.workspace) if args.workspace else None,
        enable_neural=getattr(args, "neural", None),
    )
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


def cmd_dead_code(args: argparse.Namespace) -> int:
    """Execute dead code and orphan symbol detection."""
    from code_oracle.dead_code import detect_dead_code

    ws = Path(args.workspace) if args.workspace else Path.cwd()
    report = detect_dead_code(
        workspace_root=ws,
        paths=args.paths if args.paths else None,
        min_lines=args.min_lines,
        include_unexported=args.include_unexported,
    )

    fmt = "json" if getattr(args, "json", False) else args.format

    if fmt == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif fmt == "text":
        print(report.format_text())
    else:  # "table"
        print(report.format_table())

    return 1 if report.dead_symbols_count > 0 else 0


def cmd_perf_lint(args: argparse.Namespace) -> int:
    """Execute performance anti-pattern and resource leak scan."""
    from code_oracle.perf_lint import lint_performance

    ws = Path(args.workspace) if args.workspace else Path.cwd()
    max_depth = getattr(args, "max_loop_depth", None)
    if max_depth is None:
        max_depth = getattr(args, "max_depth", 2)

    report = lint_performance(
        workspace_root=ws,
        paths=args.paths if args.paths else None,
        severity=args.severity,
        max_depth=max_depth,
    )

    fmt = "json" if getattr(args, "json", False) else args.format

    if fmt == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif fmt == "text":
        print(report.format_text())
    else:  # "table"
        print(report.format_table())

    fail_on = args.fail_on.lower() if args.fail_on else "error"
    if fail_on == "none":
        return 0
    elif fail_on == "warn":
        return 1 if (report.warnings_count > 0 or report.errors_count > 0) else 0
    else:  # "error"
        return 1 if report.errors_count > 0 else 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the FastMCP server."""
    run_server()
    return 0


def cmd_hook_install(args: argparse.Namespace) -> int:
    """Safely install git pre-commit hook non-destructively."""
    ws = Path(args.workspace) if args.workspace else None
    success, msg = install_hook(
        workspace_root=ws,
        hook_name=args.hook,
        mode=args.mode,
    )
    if getattr(args, "json", False):
        status_info = get_hook_status(ws)
        status_info["success"] = success
        status_info["message"] = msg
        print(json.dumps(status_info, indent=2))
    else:
        print(msg)
    return 0 if success else 1


def cmd_hook_uninstall(args: argparse.Namespace) -> int:
    """Safely uninstall git hook (Rollback Resilience)."""
    ws = Path(args.workspace) if args.workspace else None
    success, msg = uninstall_hook(
        workspace_root=ws,
        hook_name=args.hook,
    )
    if getattr(args, "json", False):
        status_info = get_hook_status(ws)
        status_info["success"] = success
        status_info["message"] = msg
        print(json.dumps(status_info, indent=2))
    else:
        print(msg)
    return 0


def cmd_hook_on(args: argparse.Namespace) -> int:
    """Enable Code Oracle pre-commit hook."""
    ws = Path(args.workspace) if args.workspace else None
    cfg = set_enabled(ws, True)
    target_ws = resolve_workspace_root(ws)
    if getattr(args, "json", False):
        print(json.dumps({"enabled": True, "mode": cfg.get("mode", "block"), "workspace": str(target_ws)}, indent=2))
    else:
        print("Code Oracle hook enabled.")
    return 0


def cmd_hook_off(args: argparse.Namespace) -> int:
    """Disable Code Oracle pre-commit hook."""
    ws = Path(args.workspace) if args.workspace else None
    cfg = set_enabled(ws, False)
    target_ws = resolve_workspace_root(ws)
    if getattr(args, "json", False):
        print(json.dumps({"enabled": False, "mode": cfg.get("mode", "block"), "workspace": str(target_ws)}, indent=2))
    else:
        print("Code Oracle hook disabled.")
    return 0


def cmd_hook_mode(args: argparse.Namespace) -> int:
    """Switch hook mode between 'block' and 'warn'."""
    ws = Path(args.workspace) if args.workspace else None
    cfg = set_mode(ws, args.mode)
    target_ws = resolve_workspace_root(ws)
    if getattr(args, "json", False):
        print(json.dumps({"enabled": cfg.get("enabled", True), "mode": args.mode, "workspace": str(target_ws)}, indent=2))
    else:
        print(f"Code Oracle hook mode set to '{args.mode}'.")
    return 0


def cmd_hook_status(args: argparse.Namespace) -> int:
    """Show hook installation and configuration status."""
    ws = Path(args.workspace) if args.workspace else None
    status_info = get_hook_status(ws)
    if getattr(args, "json", False):
        print(json.dumps(status_info, indent=2))
    else:
        git_str = f"Yes ({status_info['workspace']})" if status_info["is_git_repo"] else "No"
        inst_list = []
        if status_info["pre_commit_installed"]:
            inst_list.append("pre-commit: installed")
        else:
            inst_list.append("pre-commit: not installed")
        if status_info["pre_push_installed"]:
            inst_list.append("pre-push: installed")
        else:
            inst_list.append("pre-push: not installed")
        inst_str = f"Yes ({', '.join(inst_list)})" if status_info["installed"] else f"No ({', '.join(inst_list)})"
        state_str = "ENABLED" if status_info["enabled"] else "DISABLED"
        mode_desc = (
            f"{status_info['mode']} (strict exit code 1)"
            if status_info["mode"] == "block"
            else f"{status_info['mode']} (advisory exit code 0)"
        )

        print("Code Oracle Hook Status:")
        print(f"  Git Repository:  {git_str}")
        print(f"  Hook Installed:  {inst_str}")
        print(f"  Hook State:      {state_str}")
        print(f"  Hook Mode:       {mode_desc}")
        print(f"  Config File:     {status_info['config_file']}")
    return 0


def cmd_hook_run(args: argparse.Namespace) -> int:
    """Execute pre-commit verification on staged files."""
    ws = Path(args.workspace) if args.workspace else None
    result = run_hook_verification(
        workspace_root=ws,
        mode_override=args.mode,
        k=args.k,
        files=args.files,
    )
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        print(format_hook_output(result))
    return result["exit_code"]


def cmd_dataset(args: argparse.Namespace) -> int:
    """Execute dataset mining and synthetic generation."""
    from code_oracle.dataset import DatasetGenerator

    lang_list = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    generator = DatasetGenerator(
        languages=lang_list,
        seed=args.seed,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    out_dir = Path(args.output_dir)
    train_count, val_count = generator.generate_and_export(
        output_dir=out_dir,
        num_samples=args.num_samples,
        val_ratio=args.val_ratio,
        repo_path=Path(args.repo) if args.repo else None,
    )

    if args.json:
        print(json.dumps({
            "status": "SUCCESS",
            "output_dir": str(out_dir),
            "train_samples": train_count,
            "val_samples": val_count,
            "total_samples": train_count + val_count,
            "languages": lang_list,
        }, indent=2))
    else:
        print(f"Generated {train_count} train samples, {val_count} val samples into {out_dir}")

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
    p_verify.add_argument(
        "--neural",
        dest="neural",
        action="store_true",
        default=None,
        help="Enable Laya ModernBERT neural decision head and risk calibration",
    )
    p_verify.add_argument(
        "--no-neural",
        dest="neural",
        action="store_false",
        help="Disable Laya ModernBERT neural decision head (pure symbolic mode)",
    )
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

    # dead-code
    p_dead = subparsers.add_parser(
        "dead-code",
        help="Detect unreachable and orphan symbols across workspace",
    )
    p_dead.add_argument(
        "paths",
        nargs="*",
        default=[],
        help="Optional target files or directories to filter report (default: workspace root)",
    )
    p_dead.add_argument("--workspace", "-w", help="Workspace root directory")
    p_dead.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "text"],
        default="table",
        help="Output format (table, json, text)",
    )
    p_dead.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Minimum line count threshold for reporting dead code (default: 0)",
    )
    p_dead.add_argument(
        "--include-unexported",
        action="store_true",
        default=False,
        help="Include unexported (private) symbols in dead code detection",
    )
    p_dead.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON (alias for --format json)",
    )
    p_dead.set_defaults(func=cmd_dead_code)

    # perf-lint
    p_perf = subparsers.add_parser(
        "perf-lint",
        help="Detect performance anti-patterns and resource leaks across workspace",
    )
    p_perf.add_argument(
        "paths",
        nargs="*",
        default=[],
        help="Optional target files or directories to lint (default: workspace root)",
    )
    p_perf.add_argument("--workspace", "-w", help="Workspace root directory")
    p_perf.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "text"],
        default="table",
        help="Output format (table, json, text)",
    )
    p_perf.add_argument(
        "--severity",
        choices=["warn", "error"],
        default="warn",
        help="Minimum severity threshold to report (warn or error, default: warn)",
    )
    p_perf.add_argument(
        "--max-loop-depth",
        "--max-depth",
        dest="max_loop_depth",
        type=int,
        default=2,
        help="Loop depth threshold for PERF001 reporting (default: 2)",
    )
    p_perf.add_argument(
        "--fail-on",
        choices=["warn", "error", "none"],
        default="error",
        help="Exit with non-zero exit code if findings meet threshold (default: error)",
    )
    p_perf.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON (alias for --format json)",
    )
    p_perf.set_defaults(func=cmd_perf_lint)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run the Lean FastMCP server")
    p_serve.set_defaults(func=cmd_serve)

    # dataset
    p_dataset = subparsers.add_parser("dataset", help="Mine and generate multi-language training datasets")
    p_dataset.add_argument("--repo", "-r", help="Path to existing repository to mine")
    p_dataset.add_argument("--output-dir", "-o", default="./dataset_output", help="Output directory for JSONL datasets")
    p_dataset.add_argument("--num-samples", "-n", type=int, default=100, help="Target total samples")
    p_dataset.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    p_dataset.add_argument("--languages", "-l", default="python,typescript,go,rust", help="Comma-separated languages")
    p_dataset.add_argument("--seed", type=int, default=42, help="Random seed")
    p_dataset.add_argument("--positive-label", type=int, default=1, help="Positive label (default: 1)")
    p_dataset.add_argument("--negative-label", type=int, default=0, help="Negative label (default: 0)")
    p_dataset.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_dataset.set_defaults(func=cmd_dataset)

    # hook
    p_hook = subparsers.add_parser("hook", help="Git pre-commit hook and toggle system")
    p_hook_sub = p_hook.add_subparsers(dest="hook_command", help="Hook subcommands")

    # hook install
    p_h_install = p_hook_sub.add_parser("install", help="Safely install git pre-commit hook")
    p_h_install.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_install.add_argument("--mode", choices=["block", "warn"], default=None, help="Hook mode (block or warn)")
    p_h_install.add_argument("--hook", choices=["pre-commit", "pre-push"], default="pre-commit", help="Target hook (default: pre-commit)")
    p_h_install.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_install.set_defaults(func=cmd_hook_install)

    # hook uninstall
    p_h_uninstall = p_hook_sub.add_parser("uninstall", help="Safely uninstall git hook (Rollback Resilience)")
    p_h_uninstall.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_uninstall.add_argument("--hook", choices=["pre-commit", "pre-push"], default=None, help="Target hook (default: all installed)")
    p_h_uninstall.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_uninstall.set_defaults(func=cmd_hook_uninstall)

    # hook on / enable
    p_h_on = p_hook_sub.add_parser("on", aliases=["enable"], help="Enable pre-commit verification")
    p_h_on.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_on.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_on.set_defaults(func=cmd_hook_on)

    # hook off / disable
    p_h_off = p_hook_sub.add_parser("off", aliases=["disable"], help="Disable pre-commit verification")
    p_h_off.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_off.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_off.set_defaults(func=cmd_hook_off)

    # hook mode
    p_h_mode = p_hook_sub.add_parser("mode", help="Switch mode between block and warn")
    p_h_mode.add_argument("mode", choices=["block", "warn"], help="Hook mode: 'block' (strict exit 1) or 'warn' (advisory exit 0)")
    p_h_mode.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_mode.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_mode.set_defaults(func=cmd_hook_mode)

    # hook status
    p_h_status = p_hook_sub.add_parser("status", help="Show hook installation and configuration status")
    p_h_status.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_status.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_status.set_defaults(func=cmd_hook_status)

    # hook run
    p_h_run = p_hook_sub.add_parser("run", help="Run pre-commit hook verification")
    p_h_run.add_argument("files", nargs="*", default=[], help="Optional files to verify")
    p_h_run.add_argument("--workspace", "-w", help="Workspace root directory")
    p_h_run.add_argument("--mode", choices=["block", "warn"], default=None, help="Override mode (block or warn)")
    p_h_run.add_argument("--k", type=int, default=1, help="k-hop neighborhood radius (default: 1)")
    p_h_run.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_h_run.set_defaults(func=cmd_hook_run)

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
