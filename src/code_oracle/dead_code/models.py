"""
Data models for the Graph Reachability Dead Code Engine.
Defines representation for dead/orphan symbols and scan reports.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    dead_symbols: List[DeadSymbol] = field(default_factory=list)
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
    def orphan_symbols(self) -> List[DeadSymbol]:
        """Direct orphan symbols (in-degree == 0)."""
        return [s for s in self.dead_symbols if s.is_orphan]

    @property
    def transitive_symbols(self) -> List[DeadSymbol]:
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
        }

    def format_table(self) -> str:
        """Format report into clean ASCII table."""
        if not self.dead_symbols:
            return (
                f"\033[92m✔ No dead code detected\033[0m across "
                f"{self.total_symbols_scanned} symbols in {self.scanned_files_count} files "
                f"({self.latency_ms:.2f} ms)."
            )

        headers = ["Symbol", "Kind", "Location", "Lines", "Classification", "Confidence"]
        rows: List[List[str]] = []
        for s in self.dead_symbols:
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
            color = "\033[91m" if row[4] == "ORPHAN" else "\033[93m"
            color_reset = "\033[0m"
            formatted_cells = [row[i].ljust(col_widths[i]) for i in range(len(row))]
            lines.append(f"| {color}{' | '.join(formatted_cells)}{color_reset} |")
        lines.append(make_separator("-"))

        summary = (
            f"\033[91m✖ Found {self.dead_symbols_count} dead symbols\033[0m "
            f"({self.dead_lines_count} lines) across {self.scanned_files_count} files "
            f"in {self.latency_ms:.2f} ms."
        )
        lines.append(summary)
        return "\n".join(lines)

    def format_text(self) -> str:
        """Format report into concise text lines."""
        if not self.dead_symbols:
            return (
                f"✔ No dead code detected across {self.total_symbols_scanned} symbols "
                f"in {self.scanned_files_count} files ({self.latency_ms:.2f} ms)."
            )

        lines = [
            f"Dead Code Report ({self.dead_symbols_count} dead symbols, {self.dead_lines_count} lines):",
            "--------------------------------------------------------------------------------",
        ]
        for s in self.dead_symbols:
            tag = "ORPHAN" if s.is_orphan else "TRANSITIVE"
            lines.append(
                f"  • {s.file_path}:{s.lineno} {s.qualname} ({s.kind}, {s.lines_count} lines) [{tag}]"
            )
            lines.append(f"    Reason: {s.reason}")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(
            f"Scan completed in {self.latency_ms:.2f} ms (Roots: {self.roots_count}, "
            f"Total Symbols: {self.total_symbols_scanned})."
        )
        return "\n".join(lines)
