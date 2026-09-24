"""
Git Pre-Commit Hook & Toggle System for Code Oracle.
Enforces invariant verification on staged files with atomic multi-file evaluation,
staged diff isolation, rollback resilience, and sub-50ms latency.
"""

import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from code_oracle.config import find_git_root, get_config_path, load_config, resolve_workspace_root, save_config

HOOK_MARKER_BEGIN = "### BEGIN CODE ORACLE HOOK ###"
HOOK_MARKER_END = "### END CODE ORACLE HOOK ###"

HOOK_SCRIPT_TEMPLATE = """{begin_marker}
# Code Oracle Git Hook - Sub-50ms Neuro-Symbolic Verification Gate
# Fast bypass if skipped via environment (< 1ms)
case "${{CODE_ORACLE_SKIP:-0}}" in
    1|[tT][rR][uU][eE]|[yY][eE][sS])
        exit 0
        ;;
esac

# Fast shell bypass if disabled in config (< 1ms)
GIT_DIR_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
for CFG in "$GIT_DIR_ROOT/.code_oracle/config.json" "$GIT_DIR_ROOT/.code_oracle/hook_config.json"; do
    if [ -f "$CFG" ]; then
        if grep -q -i '"enabled"[[:space:]]*:[[:space:]]*false' "$CFG" 2>/dev/null; then
            exit 0
        fi
    fi
done

# Execute Code Oracle hook verification
if [ -x "$GIT_DIR_ROOT/.venv/bin/code-oracle" ]; then
    exec "$GIT_DIR_ROOT/.venv/bin/code-oracle" hook run "$@"
elif [ -x "$GIT_DIR_ROOT/venv/bin/code-oracle" ]; then
    exec "$GIT_DIR_ROOT/venv/bin/code-oracle" hook run "$@"
elif command -v code-oracle >/dev/null 2>&1; then
    exec code-oracle hook run "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 -m code_oracle.cli hook run "$@"
elif command -v python >/dev/null 2>&1; then
    exec python -m code_oracle.cli hook run "$@"
else
    echo "Code Oracle: neither code-oracle nor python found in PATH." >&2
    exit 0
fi
{end_marker}"""


def find_git_root(workspace_root: Optional[Path] = None) -> Optional[Path]:
    """Find root directory of the git repository."""
    ws = Path(workspace_root or Path.cwd()).resolve()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip()).resolve()
    except Exception:
        pass

    if (ws / ".git").exists():
        return ws
    for parent in ws.parents:
        if (parent / ".git").exists():
            return parent
    return None


def get_git_hooks_dir(workspace_root: Optional[Path] = None) -> Optional[Path]:
    """Locate the git hooks directory, respecting core.hooksPath if set."""
    ws = Path(workspace_root or Path.cwd()).resolve()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            p = Path(res.stdout.strip())
            return p if p.is_absolute() else (ws / p).resolve()
    except Exception:
        pass

    git_root = find_git_root(ws)
    if git_root:
        return git_root / ".git" / "hooks"
    return None


def get_staged_files(git_root: Path) -> List[str]:
    """Extract staged files via git diff --cached --name-only --diff-filter=ACMR."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=str(git_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return [line.strip().replace("\\", "/") for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def get_git_show_content(git_root: Path, ref_path: str) -> Optional[str]:
    """Retrieve content from git object database (e.g. ':file' or 'HEAD:file')."""
    try:
        res = subprocess.run(
            ["git", "show", ref_path],
            cwd=str(git_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return None


def get_unstaged_dirty_files(git_root: Path) -> List[str]:
    """List files modified in the working tree relative to the git index."""
    try:
        res = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(git_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return [line.strip().replace("\\", "/") for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def install_hook(
    workspace_root: Optional[Path] = None,
    hook_name: str = "pre-commit",
    mode: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Safely installs git hook non-destructively, preserving existing hook logic
    by wrapping code within BEGIN/END markers.
    """
    git_root = find_git_root(workspace_root)
    if not git_root:
        return False, "Not a git repository (no .git directory found)."

    hooks_dir = get_git_hooks_dir(git_root)
    if not hooks_dir:
        return False, "Could not locate git hooks directory."

    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_file = hooks_dir / hook_name

    block = HOOK_SCRIPT_TEMPLATE.format(
        begin_marker=HOOK_MARKER_BEGIN,
        end_marker=HOOK_MARKER_END,
    )

    if hook_file.exists():
        content = hook_file.read_text(encoding="utf-8")
        if HOOK_MARKER_BEGIN in content and HOOK_MARKER_END in content:
            # Replace existing block
            pattern = re.compile(
                re.escape(HOOK_MARKER_BEGIN) + r".*?" + re.escape(HOOK_MARKER_END),
                re.DOTALL,
            )
            new_content = pattern.sub(block, content)
        else:
            prefix = content.rstrip()
            if not prefix.startswith("#!"):
                prefix = "#!/bin/sh\n\n" + prefix
            new_content = prefix + "\n\n" + block + "\n"
    else:
        new_content = "#!/bin/sh\n\n" + block + "\n"

    hook_file.write_text(new_content, encoding="utf-8")

    # Make executable
    current_mode = hook_file.stat().st_mode
    hook_file.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Update config: ensure enabled=True and set mode if specified
    cfg = load_config(git_root)
    cfg["enabled"] = True
    if mode in ("block", "warn"):
        cfg["mode"] = mode
    save_config(git_root, cfg)

    rel_hook = str(hook_file.relative_to(git_root))
    return True, f"Code Oracle hook installed successfully to {rel_hook} (mode: {cfg['mode']})."


def uninstall_hook(
    workspace_root: Optional[Path] = None,
    hook_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Cleanly removes only the Code Oracle hook block from .git/hooks/
    (Rollback Resilience).
    """
    git_root = find_git_root(workspace_root)
    if not git_root:
        return False, "Not a git repository."

    hooks_dir = get_git_hooks_dir(git_root)
    if not hooks_dir or not hooks_dir.exists():
        return False, "Git hooks directory not found."

    target_hooks = [hook_name] if hook_name else ["pre-commit", "pre-push"]
    uninstalled_from = []

    for name in target_hooks:
        h_file = hooks_dir / name
        if not h_file.exists():
            continue
        content = h_file.read_text(encoding="utf-8")
        if HOOK_MARKER_BEGIN in content:
            pattern = re.compile(
                r"\n?" + re.escape(HOOK_MARKER_BEGIN) + r".*?(?:" + re.escape(HOOK_MARKER_END) + r"|$)\n?",
                re.DOTALL,
            )
            cleaned = pattern.sub("", content).strip()
            non_comment_lines = [
                line for line in cleaned.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if not non_comment_lines:
                h_file.unlink()
            else:
                h_file.write_text(cleaned + "\n", encoding="utf-8")
            uninstalled_from.append(name)

    if uninstalled_from:
        return True, f"Code Oracle hook uninstalled successfully from: {', '.join(uninstalled_from)}"
    return False, f"Notice: Code Oracle hook block not found in {', '.join(target_hooks)}."


def get_hook_status(workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve current hook installation and toggle status."""
    git_root = find_git_root(workspace_root)
    cfg = load_config(git_root or workspace_root)

    pre_commit_installed = False
    pre_push_installed = False

    if git_root:
        hooks_dir = get_git_hooks_dir(git_root)
        if hooks_dir and hooks_dir.exists():
            pc = hooks_dir / "pre-commit"
            if pc.exists() and HOOK_MARKER_BEGIN in pc.read_text(encoding="utf-8", errors="ignore"):
                pre_commit_installed = True
            pp = hooks_dir / "pre-push"
            if pp.exists() and HOOK_MARKER_BEGIN in pp.read_text(encoding="utf-8", errors="ignore"):
                pre_push_installed = True

    installed = pre_commit_installed or pre_push_installed
    ws_path = str(git_root or Path(workspace_root or Path.cwd()).resolve())
    cfg_file = str(get_config_path(git_root or workspace_root))

    return {
        "is_git_repo": git_root is not None,
        "workspace": ws_path,
        "installed": installed,
        "pre_commit_installed": pre_commit_installed,
        "pre_push_installed": pre_push_installed,
        "enabled": cfg.get("enabled", True),
        "mode": cfg.get("mode", "block"),
        "config_file": cfg_file,
    }


def format_hook_output(result: Dict[str, Any]) -> str:
    """Format hook result into clear terminal output with actionable unblock instructions."""
    status = result.get("status", "APPROVED")
    mode = result.get("mode", "block")
    latency = result.get("latency_ms", 0.0)
    staged_files = result.get("staged_files", [])
    violations = result.get("violations", [])
    cycles = result.get("cycles", [])

    color_red = "\033[91m"
    color_green = "\033[92m"
    color_yellow = "\033[93m"
    color_reset = "\033[0m"

    border = "=" * 70
    lines = []

    if status == "SKIPPED":
        reason = result.get("reason", "")
        lines.append(f"{color_yellow}Code Oracle Pre-Commit Gate: Skipped ({reason}){color_reset}")
        return "\n".join(lines)

    if status == "APPROVED":
        file_count = len(staged_files)
        file_str = f"{file_count} staged {'file' if file_count == 1 else 'files'}"
        lines.append(f"{color_green}{border}{color_reset}")
        lines.append(f"{color_green}✔ CODE ORACLE PRE-COMMIT GATE: APPROVED ({file_str} verified in {latency:.1f} ms){color_reset}")
        lines.append(f"{color_green}{border}{color_reset}")
        return "\n".join(lines)

    if status == "WARNING" or (violations and mode == "warn"):
        v_count = len(violations)
        v_str = f"{v_count} {'violation' if v_count == 1 else 'violations'}"
        lines.append(f"{color_yellow}{border}{color_reset}")
        lines.append(f"{color_yellow}⚠ CODE ORACLE PRE-COMMIT GATE: WARNING ({v_str}, {latency:.1f} ms){color_reset}")
        lines.append(f"{color_yellow}{border}{color_reset}")

        if violations:
            lines.append("\nViolations:")
            for v in violations:
                lines.append(f"  {color_yellow}⚠{color_reset} {v}")

        if cycles:
            lines.append("\nCycles Detected:")
            for c in cycles:
                lines.append(f"  {color_yellow}↺{color_reset} {' -> '.join(c)} -> {c[0] if c else ''}")

        lines.append("\nNotice: Commit proceeding because hook mode is set to 'warn'.")
        lines.append("To switch to strict blocking mode:")
        lines.append("  code-oracle hook mode block")
        lines.append(f"{color_yellow}{border}{color_reset}")
        return "\n".join(lines)

    # REJECTED (mode == 'block')
    v_count = len(violations)
    v_str = f"{v_count} {'violation' if v_count == 1 else 'violations'}"
    lines.append(f"{color_red}{border}{color_reset}")
    lines.append(f"{color_red}✖ CODE ORACLE PRE-COMMIT GATE: REJECTED ({v_str}, {latency:.1f} ms){color_reset}")
    lines.append(f"{color_red}{border}{color_reset}")

    if violations:
        lines.append("\nViolations:")
        for v in violations:
            lines.append(f"  {color_red}✖{color_reset} {v}")

    if cycles:
        lines.append("\nCycles Detected:")
        for c in cycles:
            lines.append(f"  {color_yellow}↺{color_reset} {' -> '.join(c)} -> {c[0] if c else ''}")

    lines.append("\nTo unblock / bypass:")
    lines.append("  • Bypass for current commit:")
    lines.append("      git commit -n (or git commit --no-verify)")
    lines.append("      CODE_ORACLE_SKIP=1 git commit")
    lines.append("  • Toggle hook off:")
    lines.append("      code-oracle hook off")
    lines.append("  • Switch to warn mode:")
    lines.append("      code-oracle hook mode warn")
    lines.append(f"{color_red}{border}{color_reset}")

    return "\n".join(lines)


def run_hook_verification(
    workspace_root: Optional[Path] = None,
    mode_override: Optional[str] = None,
    k: int = 1,
    files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Core runner called by pre-commit hooks.
    Extracts staged diffs, isolates working tree dirty changes,
    and runs atomic batch evaluation.
    """
    # Fast bypass check (< 1ms)
    skip = os.environ.get("CODE_ORACLE_SKIP", "").strip().lower()
    if skip in ("1", "true", "yes"):
        return {
            "status": "SKIPPED",
            "mode": mode_override or "block",
            "enabled": True,
            "bypassed": True,
            "reason": "CODE_ORACLE_SKIP",
            "staged_files": [],
            "violations": [],
            "cycles": [],
            "affected_symbols": [],
            "latency_ms": 0.0,
            "exit_code": 0,
        }

    git_root = find_git_root(workspace_root)
    cfg = load_config(git_root or workspace_root)

    if not cfg.get("enabled", True):
        return {
            "status": "SKIPPED",
            "mode": cfg.get("mode", "block"),
            "enabled": False,
            "bypassed": True,
            "reason": "hook disabled in configuration",
            "staged_files": [],
            "violations": [],
            "cycles": [],
            "affected_symbols": [],
            "latency_ms": 0.0,
            "exit_code": 0,
        }

    mode = mode_override or cfg.get("mode", "block")
    if mode not in ("block", "warn"):
        mode = "block"

    if not git_root:
        return {
            "status": "APPROVED",
            "mode": mode,
            "enabled": True,
            "bypassed": False,
            "reason": "not a git repository",
            "staged_files": [],
            "violations": [],
            "cycles": [],
            "affected_symbols": [],
            "latency_ms": 0.0,
            "exit_code": 0,
        }

    staged_all = get_staged_files(git_root)
    staged_py = [f for f in staged_all if f.endswith(".py")]

    if files:
        # Only filter if files contains actual Python file targets
        # Non-Python arguments (such as git pre-push remote name/url) are ignored
        py_candidates = [f for f in files if f.endswith(".py")]
        if py_candidates:
            resolved_files = set()
            ws = Path(workspace_root or Path.cwd()).resolve()
            for f in py_candidates:
                p = Path(f)
                if not p.is_absolute():
                    p = (ws / f).resolve()
                try:
                    resolved_files.add(str(p.relative_to(git_root)).replace("\\", "/"))
                except ValueError:
                    resolved_files.add(f.replace("\\", "/"))
            staged_py = [f for f in staged_py if f in resolved_files]

    if not staged_py:
        return {
            "status": "APPROVED",
            "mode": mode,
            "enabled": True,
            "bypassed": False,
            "staged_files": [],
            "violations": [],
            "cycles": [],
            "affected_symbols": [],
            "latency_ms": 0.0,
            "exit_code": 0,
        }

    # Extract staged contents and HEAD contents for diff isolation
    file_patches = []
    for f in staged_py:
        staged_content = get_git_show_content(git_root, f":{f}")
        if staged_content is None:
            continue
        head_content = get_git_show_content(git_root, f"HEAD:{f}")
        file_patches.append({
            "file_path": f,
            "patch_content": staged_content,
            "original_content": head_content if head_content is not None else "",
            "is_replacement": True,
        })

    # Unstaged dirty isolation: extract staged index contents for unstaged dirty files
    unstaged_dirty = get_unstaged_dirty_files(git_root)
    dirty_overlays: Dict[str, str] = {}
    for uf in unstaged_dirty:
        if uf.endswith(".py") and uf not in staged_py:
            idx_content = get_git_show_content(git_root, f":{uf}")
            if idx_content is not None:
                dirty_overlays[uf] = idx_content

    from code_oracle.engine import TopoSliceEngine

    engine = TopoSliceEngine(workspace_root=git_root)
    report = engine.verify_batch(
        file_patches=file_patches,
        dirty_overlays=dirty_overlays,
        k=k,
    )

    has_violations = bool(report.invariant_violations or report.cycles_detected)

    if has_violations:
        if mode == "block":
            status = "REJECTED"
            exit_code = 1
        else:
            status = "WARNING"
            exit_code = 0
    else:
        status = "APPROVED"
        exit_code = 0

    return {
        "status": status,
        "mode": mode,
        "enabled": True,
        "bypassed": False,
        "exit_code": exit_code,
        "staged_files": staged_py,
        "violations": report.invariant_violations,
        "cycles": report.cycles_detected,
        "affected_symbols": report.affected_symbols,
        "latency_ms": report.latency_ms,
    }
