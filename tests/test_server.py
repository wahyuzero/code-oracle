"""
Tests for server.py FastMCP interface and verify_patch endpoint.
"""

from pathlib import Path
import pytest

from code_oracle.server import verify_patch


@pytest.fixture
def server_workspace(tmp_path):
    (tmp_path / "handler.py").write_text(
        """def execute():
    return 100
""",
        encoding="utf-8",
    )
    return tmp_path


def test_verify_patch_endpoint(server_workspace):
    patch = """@@ -1,2 +1,2 @@
-def execute():
+def execute(mode: str = "fast"):
"""
    result = verify_patch(
        file_path="handler.py",
        patch_content=patch,
        workspace_dir=str(server_workspace),
    )

    assert isinstance(result, dict)
    assert result["status"] == "APPROVED"
    assert result["confidence"] >= 0.95
    assert result["cycles_detected"] == []
    assert result["invariant_violations"] == []
    assert "linearized_subgraph" in result
    assert result["latency_ms"] < 50.0


def test_verify_patch_endpoint_rejected(server_workspace):
    # Breaking patch: syntax error
    patch = "def execute(:\n"
    result = verify_patch(
        file_path="handler.py",
        patch_content=patch,
        workspace_dir=str(server_workspace),
    )

    assert result["status"] == "REJECTED"
    assert len(result["invariant_violations"]) > 0
    assert result["latency_ms"] < 50.0


def test_verify_patch_endpoint_neural_flag(server_workspace):
    patch = """@@ -1,2 +1,2 @@
-def execute():
+def execute(mode: str = "fast"):
"""
    result = verify_patch(
        file_path="handler.py",
        patch_content=patch,
        workspace_dir=str(server_workspace),
        enable_neural=True,
    )

    assert result["status"] == "APPROVED"
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0

