"""
Static Performance Anti-Patterns & Resource Leak Detector.
Detects nested loop complexity, N+1 queries, resource leaks, and blocking async calls.
"""

from code_oracle.perf_lint.engine import (
    PerfLintEngine,
    lint_performance,
    lint_performance_patterns,
)
from code_oracle.perf_lint.models import (
    PerfDiagnostic,
    PerfReport,
    PerfRule,
    Severity,
)
from code_oracle.perf_lint.rules import (
    AsyncBlockingRule,
    NPlusOneRule,
    NestedLoopsRule,
    UnclosedResourceRule,
)
from code_oracle.perf_lint.visitor import PerfLintVisitor

__all__ = [
    "AsyncBlockingRule",
    "NPlusOneRule",
    "NestedLoopsRule",
    "PerfDiagnostic",
    "PerfLintEngine",
    "PerfLintVisitor",
    "PerfReport",
    "PerfRule",
    "Severity",
    "UnclosedResourceRule",
    "lint_performance",
    "lint_performance_patterns",
]
