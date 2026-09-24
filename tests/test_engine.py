"""
Integration tests for TopoSliceEngine (End-to-End Pipeline).
"""

import time
import pytest
from code_oracle.engine import TopoSliceEngine


@pytest.fixture
def workspace_with_services(tmp_path):
    # Service A: billing
    (tmp_path / "billing.py").write_text(
        """def charge(amount: float) -> bool:
    return process_payment(amount)

def process_payment(amount: float) -> bool:
    return amount > 0
""",
        encoding="utf-8",
    )

    # Service B: checkout
    (tmp_path / "checkout.py").write_text(
        """from billing import charge

def checkout_cart(user_id: int, total: float):
    success = charge(total)
    return success
""",
        encoding="utf-8",
    )

    return tmp_path


def test_engine_clean_patch_approved(workspace_with_services):
    engine = TopoSliceEngine(workspace_root=workspace_with_services)

    # Patch: charge adds an optional currency parameter with default
    patch = """@@ -1,2 +1,2 @@
-def charge(amount: float) -> bool:
+def charge(amount: float, currency: str = "USD") -> bool:
"""
    report = engine.verify(file_path="billing.py", patch_content=patch)

    assert report.status == "APPROVED"
    assert report.confidence >= 0.95
    assert report.invariant_violations == []
    assert report.cycles_detected == []
    assert "billing.py::charge" in report.linearized_subgraph
    assert report.latency_ms < 50.0  # Must be sub-50ms!


def test_engine_breaking_contract_rejected(workspace_with_services):
    engine = TopoSliceEngine(workspace_root=workspace_with_services)

    # Patch: charge adds a REQUIRED currency parameter without default
    patch = """@@ -1,2 +1,2 @@
-def charge(amount: float) -> bool:
+def charge(amount: float, currency: str) -> bool:
"""
    report = engine.verify(file_path="billing.py", patch_content=patch)

    assert report.status == "REJECTED"
    assert any("ARITY_MISMATCH" in v for v in report.invariant_violations)
    assert any("requires at least 2 arguments" in v for v in report.invariant_violations)
    assert report.latency_ms < 50.0


def test_engine_circular_dependency_rejected(workspace_with_services):
    engine = TopoSliceEngine(workspace_root=workspace_with_services)

    # Currently: process_payment is called by charge
    # Patch: process_payment calls charge(amount) -> cycle: charge -> process_payment -> charge
    patch = """@@ -4,2 +4,3 @@
 def process_payment(amount: float) -> bool:
-    return amount > 0
+    charge(amount)
+    return amount > 0
"""
    report = engine.verify(file_path="billing.py", patch_content=patch)

    assert report.status == "REJECTED"
    assert any("CIRCULAR_DEPENDENCY" in v for v in report.invariant_violations)
    assert len(report.cycles_detected) >= 1
    assert report.latency_ms < 50.0


def test_engine_syntax_error_rejected(workspace_with_services):
    engine = TopoSliceEngine(workspace_root=workspace_with_services)

    patch = """@@ -1,2 +1,2 @@
-def charge(amount: float) -> bool:
+def charge(amount: float:
"""
    report = engine.verify(file_path="billing.py", patch_content=patch)

    assert report.status == "REJECTED"
    assert report.confidence == 1.0
    assert any("SYNTAX_ERROR" in v for v in report.invariant_violations)
    assert report.latency_ms < 50.0


def test_engine_deleted_symbol_with_caller_rejected(workspace_with_services):
    engine = TopoSliceEngine(workspace_root=workspace_with_services)

    # Full replacement of billing.py deleting charge() completely
    patch = """def process_payment(amount: float) -> bool:
    return amount > 0
"""
    report = engine.verify(file_path="billing.py", patch_content=patch)

    assert report.status == "REJECTED"
    assert any("BROKEN_REFERENCE" in v for v in report.invariant_violations)
    assert any("charge" in v for v in report.invariant_violations)
    assert report.latency_ms < 50.0


def test_engine_sub_50ms_latency_benchmark(workspace_with_services):
    engine = TopoSliceEngine(workspace_root=workspace_with_services)
    patch = """@@ -1,2 +1,2 @@
-def charge(amount: float) -> bool:
+def charge(amount: float, note: str = "") -> bool:
"""
    latencies = []
    # Warmup
    engine.verify(file_path="billing.py", patch_content=patch)

    # 10 runs
    for _ in range(10):
        t0 = time.perf_counter()
        rep = engine.verify(file_path="billing.py", patch_content=patch)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert rep.status == "APPROVED"

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nEmpirical Benchmark: Average Verification Latency = {avg_latency:.2f} ms")
    # Assert strictly well under 50ms budget
    assert avg_latency < 25.0
