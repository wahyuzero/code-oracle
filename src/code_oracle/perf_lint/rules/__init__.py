"""
Performance and resource leak lint rules.
"""

from code_oracle.perf_lint.rules.async_blocking import (
    AsyncBlockingRule,
    check_async_blocking,
)
from code_oracle.perf_lint.rules.n_plus_one import (
    NPlusOneRule,
    check_n_plus_one,
)
from code_oracle.perf_lint.rules.nested_loops import (
    NestedLoopsRule,
    check_nested_loop,
)
from code_oracle.perf_lint.rules.unclosed_res import (
    UnclosedResourceRule,
    check_unclosed_resource,
)

__all__ = [
    "AsyncBlockingRule",
    "NPlusOneRule",
    "NestedLoopsRule",
    "UnclosedResourceRule",
    "check_async_blocking",
    "check_n_plus_one",
    "check_nested_loop",
    "check_unclosed_resource",
]
