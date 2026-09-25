"""
Data models for the Graph Reachability Dead Code Engine.
Defines representation for dead/orphan symbols and scan reports.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SemanticClassification(str, Enum):
    """Semantic classification for unreachable symbols."""
    PUBLIC_API_SURFACE = "PUBLIC_API_SURFACE"
    INTERNAL_ORPHAN = "INTERNAL_ORPHAN"
    GENUINE_CRUFT = "GENUINE_CRUFT"


@dataclass
class SemanticDeadSymbol:
    """Rich semantic dead code symbol representation."""
    id: str
    name: str
    qualname: str
    file_path: str
    kind: str
    lineno: int
    end_lineno: int
    is_orphan: bool
    is_transitive: bool
    cluster_id: Optional[str]
    raw_reachability_confidence: float
    semantic_classification: SemanticClassification
    calibrated_confidence: float
    semantic_probabilities: Dict[str, float]
    suppressed: bool
    reason: str

    @property
    def lines_count(self) -> int:
        """Count of lines occupied by this symbol."""
        return max(1, self.end_lineno - self.lineno + 1)

    @property
    def confidence(self) -> float:
        """Alias for backward compatibility with DeadSymbol."""
        return self.calibrated_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize semantic dead symbol representation to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "qualname": self.qualname,
            "file_path": self.file_path,
            "kind": self.kind,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "lines_count": max(1, self.end_lineno - self.lineno + 1),
            "is_orphan": self.is_orphan,
            "is_transitive": self.is_transitive,
            "cluster_id": self.cluster_id,
            "raw_reachability_confidence": round(self.raw_reachability_confidence, 4),
            "semantic_classification": (
                self.semantic_classification.value
                if isinstance(self.semantic_classification, Enum)
                else str(self.semantic_classification)
            ),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "semantic_probabilities": {
                k: round(v, 4) for k, v in self.semantic_probabilities.items()
            },
            "suppressed": self.suppressed,
            "reason": self.reason,
            "confidence": round(self.calibrated_confidence, 4),
        }


@dataclass
class DeadSymbol:
    """Represents an unreachable, orphan, or transitively dead symbol."""
    id: str
    name: str
    qualname: str
    file_path: str
    kind: str
    lineno: int
    end_lineno: int
    is_orphan: bool = True
    is_transitive: bool = False
    cluster_id: Optional[str] = None
    confidence: float = 1.0
    reason: str = "Unreferenced symbol with 0 incoming calls"

    @property
    def lines_count(self) -> int:
        """Count of lines occupied by this symbol."""
        return max(1, self.end_lineno - self.lineno + 1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dead symbol representation to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "qualname": self.qualname,
            "file_path": self.file_path,
            "kind": self.kind,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "lines_count": self.lines_count,
            "is_orphan": self.is_orphan,
            "is_transitive": self.is_transitive,
            "cluster_id": self.cluster_id,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass
class DeadCodeReport:
    """Complete dead code and orphan symbol detection report."""
    workspace_root: str
    total_symbols_scanned: int
    dead_symbols: List[Union[DeadSymbol, SemanticDeadSymbol]] = field(default_factory=list)
    suppressed_symbols: List[SemanticDeadSymbol] = field(default_factory=list)
    roots_count: int = 0
    scanned_files_count: int = 0
    latency_ms: float = 0.0

    @property
    def dead_symbols_count(self) -> int:
        """Total number of dead symbols found."""
        return len(self.dead_symbols)

    @property
    def dead_lines_count(self) -> int:
        """Sum of lines across all detected dead symbols."""
        return sum(s.lines_count for s in self.dead_symbols)

    @property
    def orphan_symbols(self) -> List[Union[DeadSymbol, SemanticDeadSymbol]]:
        """Direct orphan symbols (in-degree == 0)."""
        return [s for s in self.dead_symbols if s.is_orphan]

    @property
    def transitive_symbols(self) -> List[Union[DeadSymbol, SemanticDeadSymbol]]:
        """Transitive dead cluster symbols."""
        return [s for s in self.dead_symbols if s.is_transitive]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "workspace_root": self.workspace_root,
            "total_symbols_scanned": self.total_symbols_scanned,
            "dead_symbols_count": self.dead_symbols_count,
            "dead_lines_count": self.dead_lines_count,
            "orphan_symbols_count": len(self.orphan_symbols),
            "transitive_symbols_count": len(self.transitive_symbols),
            "roots_count": self.roots_count,
            "scanned_files_count": self.scanned_files_count,
            "latency_ms": round(self.latency_ms, 2),
            "dead_symbols": [s.to_dict() for s in self.dead_symbols],
            "suppressed_symbols_count": len(self.suppressed_symbols),
            "suppressed_symbols": [s.to_dict() for s in self.suppressed_symbols],
        }

    def format_table(self) -> str:
        """Format report into clean ASCII table."""
        if not self.dead_symbols:
            suppressed_note = f" ({len(self.suppressed_symbols)} public API symbols suppressed)" if self.suppressed_symbols else ""
            return (
                f"\033[92m✔ No dead code detected\033[0m{suppressed_note} across "
                f"{self.total_symbols_scanned} symbols in {self.scanned_files_count} files "
                f"({self.latency_ms:.2f} ms)."
            )

        headers = ["Symbol", "Kind", "Location", "Lines", "Classification", "Confidence"]
        rows: List[List[str]] = []
        for s in self.dead_symbols:
            if hasattr(s, "semantic_classification"):
                sem = s.semantic_classification
                classification = sem.value if isinstance(sem, Enum) else str(sem)
            else:
                classification = "ORPHAN" if s.is_orphan else "TRANSITIVE"
            loc = f"{s.file_path}:{s.lineno}"
            rows.append([
                s.qualname,
                s.kind,
                loc,
                str(s.lines_count),
                classification,
                f"{s.confidence:.2f}",
            ])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        def make_separator(char: str = "-") -> str:
            parts = [char * (w + 2) for w in col_widths]
            return f"+{'+'.join(parts)}+"

        lines = [
            make_separator("-"),
            "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |",
            make_separator("="),
        ]
        for row in rows:
            color = "\033[91m" if row[4] in ("ORPHAN", "GENUINE_CRUFT") else "\033[93m"
            color_reset = "\033[0m"
            formatted_cells = [row[i].ljust(col_widths[i]) for i in range(len(row))]
            lines.append(f"| {color}{' | '.join(formatted_cells)}{color_reset} |")
        lines.append(make_separator("-"))

        suppressed_str = f" ({len(self.suppressed_symbols)} public API symbols suppressed)" if self.suppressed_symbols else ""
        summary = (
            f"\033[91m✖ Found {self.dead_symbols_count} dead symbols\033[0m "
            f"({self.dead_lines_count} lines){suppressed_str} across {self.scanned_files_count} files "
            f"in {self.latency_ms:.2f} ms."
        )
        lines.append(summary)
        return "\n".join(lines)

    def format_text(self) -> str:
        """Format report into concise text lines."""
        if not self.dead_symbols:
            suppressed_note = f" ({len(self.suppressed_symbols)} public API symbols suppressed)" if self.suppressed_symbols else ""
            return (
                f"✔ No dead code detected{suppressed_note} across {self.total_symbols_scanned} symbols "
                f"in {self.scanned_files_count} files ({self.latency_ms:.2f} ms)."
            )

        suppressed_header = f", {len(self.suppressed_symbols)} suppressed" if self.suppressed_symbols else ""
        lines = [
            f"Dead Code Report ({self.dead_symbols_count} dead symbols, {self.dead_lines_count} lines{suppressed_header}):",
            "--------------------------------------------------------------------------------",
        ]
        for s in self.dead_symbols:
            if hasattr(s, "semantic_classification"):
                sem = s.semantic_classification
                tag = sem.value if isinstance(sem, Enum) else str(sem)
            else:
                tag = "ORPHAN" if s.is_orphan else "TRANSITIVE"
            lines.append(
                f"  • {s.file_path}:{s.lineno} {s.qualname} ({s.kind}, {s.lines_count} lines) [{tag}]"
            )
            lines.append(f"    Reason: {s.reason}")
        if self.suppressed_symbols:
            lines.append(f"  Note: {len(self.suppressed_symbols)} public API symbols were suppressed from this report.")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(
            f"Scan completed in {self.latency_ms:.2f} ms (Roots: {self.roots_count}, "
            f"Total Symbols: {self.total_symbols_scanned})."
        )
        return "\n".join(lines)
