"""
Tests for Stage 2: Workspace Indexer & Symbol Cache.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import Symbol


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory with sample files."""
    tmpdir = tempfile.mkdtemp(prefix="code_oracle_test_ws_")
    ws_path = Path(tmpdir)

    # File 1: utils.py
    (ws_path / "utils.py").write_text(
        """def helper(a: int) -> int:
    return a * 2

def logger(msg: str):
    print(msg)
""",
        encoding="utf-8",
    )

    # File 2: service.py
    (ws_path / "service.py").write_text(
        """from utils import helper, logger

def process_order(order_id: int):
    val = helper(order_id)
    logger("Order processed")
    return val
""",
        encoding="utf-8",
    )

    yield ws_path
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_indexer_initial_scan(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    stats = indexer.scan_workspace()

    assert stats["scanned"] == 2
    assert stats["reindexed"] == 2
    assert stats["symbols_indexed"] >= 3

    # Check definitions
    helper_sym = indexer.get_definition("helper")
    assert helper_sym is not None
    assert helper_sym.name == "helper"
    assert helper_sym.file_path == "utils.py"

    # Check callers inverted index
    callers = indexer.get_callers("helper")
    assert len(callers) >= 1
    assert any(c.caller == "service.py::process_order" for c in callers)


def test_indexer_incremental_cache_hit(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    # First scan
    stats1 = indexer.scan_workspace()
    assert stats1["reindexed"] == 2

    # Second scan immediately - files have not changed
    stats2 = indexer.scan_workspace()
    assert stats2["scanned"] == 2
    assert stats2["reindexed"] == 0  # 0 files reindexed due to mtime cache


def test_indexer_cache_invalidation_on_change(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    indexer.scan_workspace()

    # Modify service.py
    time_shift = os.path.getmtime(temp_workspace / "service.py") + 2
    (temp_workspace / "service.py").write_text(
        """def new_service():
    return 42
""",
        encoding="utf-8",
    )
    os.utime(temp_workspace / "service.py", (time_shift, time_shift))

    stats = indexer.scan_workspace()
    assert stats["reindexed"] == 1
    assert indexer.get_definition("new_service") is not None
    assert indexer.get_definition("process_order") is None


def test_indexer_prunes_deleted_files(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    indexer.scan_workspace()

    # Delete utils.py
    (temp_workspace / "utils.py").unlink()
    stats = indexer.scan_workspace()

    assert "utils.py" not in indexer._file_cache
    assert indexer.get_definition("helper") is None


def test_indexer_corrupted_cache_recovery(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    indexer.scan_workspace()

    # Corrupt index.json
    index_file = temp_workspace / ".code_oracle" / "index.json"
    index_file.write_text("CORRUPTED NOT JSON", encoding="utf-8")

    # New indexer should recover cleanly
    indexer2 = WorkspaceIndexer(workspace_root=temp_workspace)
    assert indexer2.load_cache() is False
    stats = indexer2.scan_workspace()
    assert stats["reindexed"] == 2
    assert indexer2.get_definition("helper") is not None


def test_indexer_clean_rollback(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    indexer.scan_workspace()
    cache_dir = temp_workspace / ".code_oracle"
    assert cache_dir.exists()
    assert (cache_dir / "index.json").exists()

    # Clean cache
    assert indexer.clean() is True
    assert not cache_dir.exists()
    assert len(indexer._definitions) == 0


def test_indexer_overlay_transient_symbols(temp_workspace):
    indexer = WorkspaceIndexer(workspace_root=temp_workspace)
    indexer.scan_workspace()

    assert indexer.get_definition("transient_func") is None

    transient = Symbol(
        name="transient_func",
        qualname="transient_func",
        file_path="service.py",
        kind="function",
        lineno=1,
        end_lineno=3,
        signature="def transient_func()",
    )
    indexer.overlay_transient_symbols("service.py", [transient])

    assert indexer.get_definition("transient_func") is not None
    # Disk should not be modified
    content = (temp_workspace / "service.py").read_text(encoding="utf-8")
    assert "transient_func" not in content
