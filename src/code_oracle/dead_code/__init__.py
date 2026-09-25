"""
Dead Code and Orphan Symbol Detection Engine.
"""

from code_oracle.dead_code.detector import DeadCodeDetector, detect_dead_code
from code_oracle.dead_code.entrypoints import EntrypointDetector, is_entrypoint
from code_oracle.dead_code.models import DeadCodeReport, DeadSymbol

__all__ = [
    "DeadCodeDetector",
    "detect_dead_code",
    "DeadCodeReport",
    "DeadSymbol",
    "EntrypointDetector",
    "is_entrypoint",
]
