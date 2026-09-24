"""
Configuration and state management for Code Oracle.
Maintains persistent settings in .code_oracle/config.json.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "mode": "block",
}


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


def resolve_workspace_root(workspace_root: Optional[Path] = None) -> Path:
    """
    Resolve the canonical workspace directory for configuration.
    If workspace_root or cwd contains .git or .code_oracle, returns it.
    Otherwise checks parent directories for .git or .code_oracle.
    Falls back to workspace_root or Path.cwd().
    """
    ws = Path(workspace_root or Path.cwd()).resolve()
    if (ws / ".git").exists() or (ws / ".code_oracle").exists():
        return ws

    for parent in ws.parents:
        if (parent / ".git").exists() or (parent / ".code_oracle").exists():
            return parent

    git_root = find_git_root(ws)
    if git_root:
        return git_root

    return ws


def get_config_path(workspace_root: Optional[Path] = None) -> Path:
    """Return path to .code_oracle/config.json (or hook_config.json if existing)."""
    ws = resolve_workspace_root(workspace_root)
    cfg_file = ws / ".code_oracle" / "config.json"
    legacy_file = ws / ".code_oracle" / "hook_config.json"
    if not cfg_file.exists() and legacy_file.exists():
        return legacy_file
    return cfg_file


def load_config(workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from .code_oracle/config.json or hook_config.json.
    Returns default config if file does not exist or is corrupted.
    """
    target_file = get_config_path(workspace_root)
    config = dict(DEFAULT_CONFIG)
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config.update(data)
        except Exception:
            # Corrupted config, return defaults
            pass

    # Ensure mode is valid
    if config.get("mode") not in ("block", "warn"):
        config["mode"] = "block"
    config["enabled"] = bool(config.get("enabled", True))

    return config


def save_config(workspace_root: Optional[Path], config: Dict[str, Any]) -> None:
    """Atomically write configuration to .code_oracle/config.json (or hook_config.json)."""
    cfg_file = get_config_path(workspace_root)
    cfg_dir = cfg_file.parent
    cfg_dir.mkdir(parents=True, exist_ok=True)
    temp_file = cfg_dir / f"{cfg_file.name}.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        os.replace(temp_file, cfg_file)
    except Exception:
        # Fallback direct write if atomic replace fails
        try:
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass


def set_enabled(workspace_root: Optional[Path], enabled: bool) -> Dict[str, Any]:
    """Update enabled flag in config."""
    cfg = load_config(workspace_root)
    cfg["enabled"] = bool(enabled)
    save_config(workspace_root, cfg)
    return cfg


def set_mode(workspace_root: Optional[Path], mode: str) -> Dict[str, Any]:
    """Update mode in config ('block' or 'warn')."""
    if mode not in ("block", "warn"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'block' or 'warn'.")
    cfg = load_config(workspace_root)
    cfg["mode"] = mode
    save_config(workspace_root, cfg)
    return cfg


def is_bypassed(workspace_root: Optional[Path] = None) -> bool:
    """
    Fast check if verification is bypassed via CODE_ORACLE_SKIP=1 or enabled=False.
    """
    skip = os.environ.get("CODE_ORACLE_SKIP", "").strip().lower()
    if skip in ("1", "true", "yes"):
        return True
    cfg = load_config(workspace_root)
    return not cfg.get("enabled", True)
