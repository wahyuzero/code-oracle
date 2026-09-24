"""
Comprehensive unit and integration tests for Code Oracle Git Pre-Commit Hook
and Toggle System.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from code_oracle.cli import build_parser
from code_oracle.config import (
    DEFAULT_CONFIG,
    get_config_path,
    is_bypassed,
    load_config,
    save_config,
    set_enabled,
    set_mode,
)
from code_oracle.hook import (
    HOOK_MARKER_BEGIN,
    HOOK_MARKER_END,
    find_git_root,
    get_git_hooks_dir,
    get_hook_status,
    install_hook,
    run_hook_verification,
    uninstall_hook,
)


def init_git_repo(path: Path) -> Path:
    """Helper to initialize a git repository with mock user credentials."""
    subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Agent"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agent@test.local"], cwd=str(path), check=True, capture_output=True)
    return path


def test_config_lifecycle(tmp_path: Path):
    """Test config loading, saving, mutation, recovery, and bypass checks."""
    # 1. Default config when no file exists
    cfg = load_config(tmp_path)
    assert cfg["enabled"] is True
    assert cfg["mode"] == "block"

    # 2. Mutate enabled
    set_enabled(tmp_path, False)
    cfg2 = load_config(tmp_path)
    assert cfg2["enabled"] is False

    # 3. Mutate mode
    set_mode(tmp_path, "warn")
    cfg3 = load_config(tmp_path)
    assert cfg3["mode"] == "warn"

    with pytest.raises(ValueError):
        set_mode(tmp_path, "invalid_mode")

    # 4. Corrupted file recovery
    cfg_file = get_config_path(tmp_path)
    cfg_file.write_text("{invalid json", encoding="utf-8")
    cfg4 = load_config(tmp_path)
    assert cfg4["enabled"] is True
    assert cfg4["mode"] == "block"

    # 5. Bypass check
    set_enabled(tmp_path, False)
    assert is_bypassed(tmp_path) is True
    set_enabled(tmp_path, True)
    assert is_bypassed(tmp_path) is False

    # Environment variable bypass
    os.environ["CODE_ORACLE_SKIP"] = "1"
    try:
        assert is_bypassed(tmp_path) is True
    finally:
        os.environ.pop("CODE_ORACLE_SKIP", None)


def test_hook_install_fresh_repo(tmp_path: Path):
    """Test hook installation into a fresh git repo."""
    repo = init_git_repo(tmp_path)
    success, msg = install_hook(workspace_root=repo, mode="block")
    assert success is True
    assert "installed successfully" in msg

    hook_file = repo / ".git" / "hooks" / "pre-commit"
    assert hook_file.exists()
    content = hook_file.read_text(encoding="utf-8")
    assert HOOK_MARKER_BEGIN in content
    assert HOOK_MARKER_END in content
    assert "code-oracle hook run" in content

    # Check executable permission
    mode = hook_file.stat().st_mode
    assert bool(mode & stat.S_IXUSR)


def test_hook_install_preserves_existing_hook(tmp_path: Path):
    """Test that installing Code Oracle preserves existing user hooks."""
    repo = init_git_repo(tmp_path)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_file = hooks_dir / "pre-commit"

    existing_content = "#!/bin/sh\n# Existing user hook\necho 'running linting'\n"
    hook_file.write_text(existing_content, encoding="utf-8")

    # Install code-oracle
    success, msg = install_hook(workspace_root=repo)
    assert success is True

    content = hook_file.read_text(encoding="utf-8")
    assert "echo 'running linting'" in content
    assert HOOK_MARKER_BEGIN in content
    assert HOOK_MARKER_END in content

    # Reinstall does not duplicate block
    install_hook(workspace_root=repo)
    content2 = hook_file.read_text(encoding="utf-8")
    assert content2.count(HOOK_MARKER_BEGIN) == 1
    assert content2.count(HOOK_MARKER_END) == 1
    assert "echo 'running linting'" in content2


def test_hook_uninstall_lifecycle(tmp_path: Path):
    """Test clean uninstallation and rollback resilience."""
    repo = init_git_repo(tmp_path)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_file = hooks_dir / "pre-commit"

    # Case A: Only code oracle was installed
    install_hook(workspace_root=repo)
    assert hook_file.exists()
    success, msg = uninstall_hook(workspace_root=repo)
    assert success is True
    # Hook file should be removed cleanly when it contained only code-oracle
    assert not hook_file.exists()

    # Case B: Other hook logic was present
    hook_file.write_text("#!/bin/sh\npytest -q\n", encoding="utf-8")
    install_hook(workspace_root=repo)
    success, msg = uninstall_hook(workspace_root=repo)
    assert success is True
    assert hook_file.exists()
    remaining = hook_file.read_text(encoding="utf-8")
    assert "pytest -q" in remaining
    assert HOOK_MARKER_BEGIN not in remaining


def test_hook_pre_push_install_and_uninstall(tmp_path: Path):
    """Test installing and uninstalling pre-push hook."""
    repo = init_git_repo(tmp_path)
    success, msg = install_hook(workspace_root=repo, hook_name="pre-push")
    assert success is True
    pp_file = repo / ".git" / "hooks" / "pre-push"
    assert pp_file.exists()
    assert HOOK_MARKER_BEGIN in pp_file.read_text(encoding="utf-8")

    status = get_hook_status(workspace_root=repo)
    assert status["pre_push_installed"] is True

    uninstall_hook(workspace_root=repo, hook_name="pre-push")
    assert not pp_file.exists()


def test_cli_hook_toggles_and_status(tmp_path: Path, capsys):
    """Test CLI commands: on, off, enable, disable, mode, status."""
    repo = init_git_repo(tmp_path)
    parser = build_parser()

    # 1. status initial
    args = parser.parse_args(["hook", "status", "--workspace", str(repo), "--json"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["is_git_repo"] is True
    assert data["installed"] is False
    assert data["enabled"] is True
    assert data["mode"] == "block"

    # 2. install via hook install
    args = parser.parse_args(["hook", "install", "--workspace", str(repo), "--mode", "warn"])
    assert args.func(args) == 0
    assert (repo / ".git" / "hooks" / "pre-commit").exists()

    # 3. off (and alias disable)
    args = parser.parse_args(["hook", "off", "--workspace", str(repo)])
    assert args.func(args) == 0
    assert load_config(repo)["enabled"] is False

    args = parser.parse_args(["hook", "enable", "--workspace", str(repo)])
    assert args.func(args) == 0
    assert load_config(repo)["enabled"] is True

    args = parser.parse_args(["hook", "disable", "--workspace", str(repo)])
    assert args.func(args) == 0
    assert load_config(repo)["enabled"] is False

    args = parser.parse_args(["hook", "on", "--workspace", str(repo)])
    assert args.func(args) == 0
    assert load_config(repo)["enabled"] is True

    # 4. mode switch
    args = parser.parse_args(["hook", "mode", "warn", "--workspace", str(repo)])
    assert args.func(args) == 0
    assert load_config(repo)["mode"] == "warn"

    args = parser.parse_args(["hook", "mode", "block", "--workspace", str(repo)])
    assert args.func(args) == 0
    assert load_config(repo)["mode"] == "block"


def test_hook_run_fast_bypasses(tmp_path: Path):
    """Test that hook run exits 0 immediately (< 1ms) when skipped or disabled."""
    repo = init_git_repo(tmp_path)

    # 1. Bypassed via CODE_ORACLE_SKIP=1
    os.environ["CODE_ORACLE_SKIP"] = "1"
    try:
        res = run_hook_verification(workspace_root=repo)
        assert res["status"] == "SKIPPED"
        assert res["exit_code"] == 0
        assert res["latency_ms"] < 5.0
    finally:
        os.environ.pop("CODE_ORACLE_SKIP", None)

    # 2. Bypassed via enabled=False
    set_enabled(repo, False)
    res2 = run_hook_verification(workspace_root=repo)
    assert res2["status"] == "SKIPPED"
    assert res2["exit_code"] == 0
    assert res2["latency_ms"] < 5.0

    # 3. No staged files
    set_enabled(repo, True)
    res3 = run_hook_verification(workspace_root=repo)
    assert res3["status"] == "APPROVED"
    assert res3["exit_code"] == 0
    assert len(res3["staged_files"]) == 0


def test_hook_run_staged_vs_unstaged_isolation(tmp_path: Path):
    """
    Test staged diff isolation:
    Staged content is valid, but working tree has dirty unstaged broken changes.
    The hook MUST evaluate against the staged version and pass.
    """
    repo = init_git_repo(tmp_path)

    # 1. Base commit
    lib_file = repo / "lib.py"
    lib_file.write_text("def multiply(x: int, y: int = 1) -> int:\n    return x * y\n", encoding="utf-8")
    app_file = repo / "app.py"
    app_file.write_text("from lib import multiply\n\ndef main():\n    return multiply(5)\n", encoding="utf-8")

    subprocess.run(["git", "add", "lib.py", "app.py"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo), check=True, capture_output=True)

    # 2. Stage a valid change to lib.py
    lib_file.write_text(
        "def multiply(x: int, y: int = 1, scale: int = 1) -> int:\n    return x * y * scale\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "lib.py"], cwd=str(repo), check=True, capture_output=True)

    # 3. Introduce dirty UNSTAGED corruption into working tree lib.py
    lib_file.write_text("def multiply(x: int, y: int = 1): SYNTAX ERROR !!!", encoding="utf-8")

    # 4. Introduce dirty UNSTAGED arity mismatch into app.py
    app_file.write_text("from lib import multiply\ndef main(): return multiply() # broken\n", encoding="utf-8")

    # 5. Run hook verification
    res = run_hook_verification(workspace_root=repo)

    # Since the staged content of lib.py is completely valid, and app.py staged/HEAD is valid,
    # the dirty unstaged modifications on disk must be ignored!
    assert res["status"] == "APPROVED"
    assert res["exit_code"] == 0
    assert "lib.py" in res["staged_files"]


def test_hook_run_atomic_multi_file_refactor(tmp_path: Path):
    """
    Test atomic multi-file refactor evaluation:
    Changing signature in lib.py and call site in app.py in the same staged commit
    must pass verification together.
    """
    repo = init_git_repo(tmp_path)

    # Base commit
    lib_file = repo / "calc.py"
    lib_file.write_text("def compute(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
    client_file = repo / "client.py"
    client_file.write_text("from calc import compute\n\ndef run():\n    return compute(10)\n", encoding="utf-8")

    subprocess.run(["git", "add", "calc.py", "client.py"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Base"], cwd=str(repo), check=True, capture_output=True)

    # Coordinated refactor: signature requires 2 arguments now
    lib_file.write_text("def compute(x: int, multiplier: int) -> int:\n    return x * multiplier\n", encoding="utf-8")
    client_file.write_text("from calc import compute\n\ndef run():\n    return compute(10, 3)\n", encoding="utf-8")

    # Scenario A: BOTH files staged together
    subprocess.run(["git", "add", "calc.py", "client.py"], cwd=str(repo), check=True, capture_output=True)
    res_both = run_hook_verification(workspace_root=repo)
    assert res_both["status"] == "APPROVED"
    assert res_both["exit_code"] == 0
    assert len(res_both["violations"]) == 0

    # Scenario B: ONLY calc.py is staged, client.py is UNSTAGED
    subprocess.run(["git", "reset", "HEAD", "client.py"], cwd=str(repo), check=True, capture_output=True)
    res_single = run_hook_verification(workspace_root=repo)
    assert res_single["status"] == "REJECTED"
    assert res_single["exit_code"] == 1
    assert any("ARITY_MISMATCH" in v for v in res_single["violations"])


def test_hook_block_vs_warn_exit_codes_and_ux(tmp_path: Path, capsys):
    """
    Test exit codes and UX output for block vs warn modes on contract violations:
    - block: exit 1, visual box, explicit unblock commands
    - warn: exit 0, warning box, notice of advisory mode
    """
    repo = init_git_repo(tmp_path)
    parser = build_parser()

    # Base commit
    svc = repo / "service.py"
    svc.write_text("def get_data(uid: int) -> str:\n    return str(uid)\n", encoding="utf-8")
    subprocess.run(["git", "add", "service.py"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Init"], cwd=str(repo), check=True, capture_output=True)

    # Stage breaking syntax error
    svc.write_text("def get_data(uid: int): invalid syntax :(", encoding="utf-8")
    subprocess.run(["git", "add", "service.py"], cwd=str(repo), check=True, capture_output=True)

    # 1. Mode: block
    set_mode(repo, "block")
    args_block = parser.parse_args(["hook", "run", "--workspace", str(repo)])
    code_block = args_block.func(args_block)
    out_block = capsys.readouterr().out

    assert code_block == 1
    assert "REJECTED" in out_block
    assert "SYNTAX_ERROR" in out_block
    # Check unblock instructions
    assert "git commit -n (or git commit --no-verify)" in out_block
    assert "CODE_ORACLE_SKIP=1 git commit" in out_block
    assert "code-oracle hook off" in out_block
    assert "code-oracle hook mode warn" in out_block

    # 2. Mode: warn
    set_mode(repo, "warn")
    args_warn = parser.parse_args(["hook", "run", "--workspace", str(repo)])
    code_warn = args_warn.func(args_warn)
    out_warn = capsys.readouterr().out

    assert code_warn == 0  # Allowed to proceed in warn mode
    assert "WARNING" in out_warn
    assert "Commit proceeding because hook mode is set to 'warn'." in out_warn
    assert "code-oracle hook mode block" in out_warn


def test_hook_run_json_output(tmp_path: Path, capsys):
    """Test machine-readable JSON output of hook run."""
    repo = init_git_repo(tmp_path)
    parser = build_parser()

    f = repo / "test_mod.py"
    f.write_text("def hello(): return 'world'\n", encoding="utf-8")
    subprocess.run(["git", "add", "test_mod.py"], cwd=str(repo), check=True, capture_output=True)

    args = parser.parse_args(["hook", "run", "--workspace", str(repo), "--json"])
    ret = args.func(args)
    assert ret == 0

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] == "APPROVED"
    assert data["exit_code"] == 0
    assert "test_mod.py" in data["staged_files"]
    assert isinstance(data["latency_ms"], (int, float))


def test_pre_commit_hooks_yaml_exists():
    """Verify that .pre-commit-hooks.yaml exists in repo root with proper configuration."""
    yaml_path = Path("/home/wxsys/code-oracle/.pre-commit-hooks.yaml")
    assert yaml_path.exists()
    content = yaml_path.read_text(encoding="utf-8")
    assert "id: code-oracle" in content
    assert "entry: code-oracle hook run" in content
    assert "types: [python]" in content
    assert "pass_filenames: false" in content


def test_hook_run_file_containing_diff_markers(tmp_path: Path):
    """
    Test that staging a Python file that contains diff hunk strings
    (e.g. @@ -1,2 +1,2 @@ in docstrings or test fixtures) is NOT misinterpreted
    as a unified diff, and evaluates cleanly without syntax errors.
    """
    repo = init_git_repo(tmp_path)
    f = repo / "test_diff_fixture.py"
    f.write_text(
        'def get_example_diff() -> str:\n'
        '    """Returns sample diff."""\n'
        '    return """@@ -1,2 +1,2 @@\n'
        '-foo\n'
        '+bar\n'
        '"""\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "test_diff_fixture.py"], cwd=str(repo), check=True, capture_output=True)

    res = run_hook_verification(workspace_root=repo)
    assert res["status"] == "APPROVED"
    assert res["exit_code"] == 0
    assert "test_diff_fixture.py" in res["staged_files"]
    assert len(res["violations"]) == 0


def test_hook_run_with_pre_push_remote_arguments(tmp_path: Path):
    """
    Test that when git pre-push passes remote name and url as arguments,
    hook run ignores non-python arguments and verifies staged files properly.
    """
    repo = init_git_repo(tmp_path)
    f = repo / "calc.py"
    f.write_text("def add(x: int, y: int) -> int:\n    return x + y\n", encoding="utf-8")
    subprocess.run(["git", "add", "calc.py"], cwd=str(repo), check=True, capture_output=True)

    # Git pre-push passes: <remote-name> <remote-url>
    res = run_hook_verification(
        workspace_root=repo,
        files=["origin", "https://github.com/org/repo.git"],
    )
    assert res["status"] == "APPROVED"
    assert res["exit_code"] == 0
    assert "calc.py" in res["staged_files"]


def test_hook_subdirectory_config_resolution(tmp_path: Path, monkeypatch):
    """
    Test that toggling hook state from a subdirectory correctly mutates
    the repository root config file, rather than creating a detached config.
    """
    repo = init_git_repo(tmp_path)
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    parser = build_parser()

    # 1. Turn hook off from sub
    args_off = parser.parse_args(["hook", "off"])
    assert args_off.func(args_off) == 0

    root_cfg = load_config(repo)
    assert root_cfg["enabled"] is False
    assert not (sub / ".code_oracle").exists()

    # 2. Switch mode from sub
    args_mode = parser.parse_args(["hook", "mode", "warn"])
    assert args_mode.func(args_mode) == 0

    root_cfg2 = load_config(repo)
    assert root_cfg2["mode"] == "warn"

    # 3. Turn hook back on from sub
    args_on = parser.parse_args(["hook", "on"])
    assert args_on.func(args_on) == 0

    root_cfg3 = load_config(repo)
    assert root_cfg3["enabled"] is True


def test_git_commit_end_to_end(tmp_path: Path):
    """
    End-to-end integration test: execute real `git commit` commands with the hook installed.
    Tests pass, block, unblock instructions, git commit -n bypass, CODE_ORACLE_SKIP bypass,
    warn mode, and hook off.
    """
    repo = init_git_repo(tmp_path)
    install_hook(workspace_root=repo)

    # 1. Valid commit succeeds
    f = repo / "math_util.py"
    f.write_text("def square(x: int) -> int:\n    return x * x\n", encoding="utf-8")
    subprocess.run(["git", "add", "math_util.py"], cwd=str(repo), check=True, capture_output=True)

    res_valid = subprocess.run(["git", "commit", "-m", "Valid commit"], cwd=str(repo), capture_output=True, text=True)
    assert res_valid.returncode == 0
    assert "APPROVED" in res_valid.stderr or "APPROVED" in res_valid.stdout

    # 2. Broken commit fails with code 1 and actionable unblock instructions
    f.write_text("def square(x: int): BROKEN SYNTAX !!!", encoding="utf-8")
    subprocess.run(["git", "add", "math_util.py"], cwd=str(repo), check=True, capture_output=True)

    res_invalid = subprocess.run(["git", "commit", "-m", "Broken commit"], cwd=str(repo), capture_output=True, text=True)
    assert res_invalid.returncode == 1
    combined_err = res_invalid.stdout + res_invalid.stderr
    assert "REJECTED" in combined_err
    assert "git commit -n (or git commit --no-verify)" in combined_err
    assert "CODE_ORACLE_SKIP=1 git commit" in combined_err
    assert "code-oracle hook off" in combined_err
    assert "code-oracle hook mode warn" in combined_err

    # 3. Bypass via git commit -n
    res_no_verify = subprocess.run(["git", "commit", "-n", "-m", "Bypassed commit"], cwd=str(repo), capture_output=True, text=True)
    assert res_no_verify.returncode == 0

    # 4. Bypass via CODE_ORACLE_SKIP=1
    f.write_text("def square(x: int): ANOTHER SYNTAX ERROR !!!", encoding="utf-8")
    subprocess.run(["git", "add", "math_util.py"], cwd=str(repo), check=True, capture_output=True)

    env_skip = os.environ.copy()
    env_skip["CODE_ORACLE_SKIP"] = "1"
    res_env_skip = subprocess.run(["git", "commit", "-m", "Skip commit"], cwd=str(repo), capture_output=True, text=True, env=env_skip)
    assert res_env_skip.returncode == 0

    # 5. Warn mode allows commit to proceed with code 0
    set_mode(repo, "warn")
    f.write_text("def square(x: int): WARN SYNTAX ERROR !!!", encoding="utf-8")
    subprocess.run(["git", "add", "math_util.py"], cwd=str(repo), check=True, capture_output=True)

    res_warn = subprocess.run(["git", "commit", "-m", "Warn commit"], cwd=str(repo), capture_output=True, text=True)
    assert res_warn.returncode == 0
    assert "WARNING" in (res_warn.stdout + res_warn.stderr)

    # 6. Hook off allows commit to proceed with code 0
    set_mode(repo, "block")
    set_enabled(repo, False)
    f.write_text("def square(x: int): DISABLED SYNTAX ERROR !!!", encoding="utf-8")
    subprocess.run(["git", "add", "math_util.py"], cwd=str(repo), check=True, capture_output=True)

    res_disabled = subprocess.run(["git", "commit", "-m", "Disabled commit"], cwd=str(repo), capture_output=True, text=True)
    assert res_disabled.returncode == 0

