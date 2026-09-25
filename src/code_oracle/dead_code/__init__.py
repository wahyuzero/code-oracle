"""
Dead Code and Orphan Symbol Detection Engine.
"""

from code_oracle.dead_code.detector import DeadCodeDetector, detect_dead_code
from code_oracle.dead_code.entrypoints import EntrypointDetector, is_entrypoint
from code_oracle.dead_code.models import (
    DeadCodeReport,
    DeadSymbol,
    SemanticClassification,
    SemanticDeadSymbol,
)
from code_oracle.dead_code.semantics import DeadCodeSemanticsClassifier

__all__ = [
    "DeadCodeDetector",
    "detect_dead_code",
    "DeadCodeReport",
    "DeadSymbol",
    "SemanticClassification",
    "SemanticDeadSymbol",
    "DeadCodeSemanticsClassifier",
    "EntrypointDetector",
    "is_entrypoint",
]

