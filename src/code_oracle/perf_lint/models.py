"""
Data models for the Static Performance Anti-Patterns & Resource Leak Detector.
Defines representations for diagnostics, severity levels, rules, and reports.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Diagnostic severity levels."""
    WARN = "warn"
    ERROR = "error"

    @classmethod
    def from_str(cls, value: str) -> "Severity":
        """Convert string to Severity enum value safely."""
        val = value.strip().lower()
        if val in ("error", "err"):
            return cls.ERROR
        return cls.WARN


class PerfRule(str, Enum):
    """Identifier codes for performance and leak linting rules."""
    PERF001 = "PERF001"
    PERF002 = "PERF002"
    PERF003 = "PERF003"
    PERF004 = "PERF004"


RULE_METADATA: Dict[str, Dict[str, Any]] = {
    PerfRule.PERF001.value: {
        "name": "Nested Loops Complexity",
        "description": "Nested loop complexity escalation (O(N^2) warning, O(N^3) error)",
        "default_severity": Severity.WARN,
    },
    PerfRule.PERF002.value: {
        "name": "N+1 I/O in Loop",
        "description": "N+1 database queries or network I/O calls inside iteration loop bodies",
        "default_severity": Severity.WARN,
    },
    PerfRule.PERF003.value: {
        "name": "Resource Leak / Unclosed Descriptor",
        "description": "File or socket descriptor opened without scoped context manager or defer Close",
        "default_severity": Severity.ERROR,
    },
    PerfRule.PERF004.value: {
        "name": "Blocking Call in Async Context",
        "description": "Blocking synchronous call inside asynchronous function",
        "default_severity": Severity.ERROR,
    },
}


@dataclass
class PerfDiagnostic:
    """Represents a single performance anti-pattern or resource leak finding."""
    rule_id: str
    message: str
    severity: Severity
    file_path: str
    lineno: int
    end_lineno: int = 0
    col_offset: int = 0
    end_col_offset: int = 0
    context_line: Optional[str] = None
    rule_name: Optional[str] = None
    suppressed: bool = False

    def __post_init__(self) -> None:
        if hasattr(self.rule_id, "value"):
            self.rule_id = self.rule_id.value
        if isinstance(self.severity, str):
            self.severity = Severity.from_str(self.severity)
        if not self.rule_name and self.rule_id in RULE_METADATA:
            self.rule_name = RULE_METADATA[self.rule_id]["name"]
        if self.end_lineno <= 0:
            self.end_lineno = self.lineno

    def to_dict(self) -> Dict[str, Any]:
        """Serialize diagnostic to dictionary."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name or self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "col_offset": self.col_offset,
            "end_col_offset": self.end_col_offset,
            "context_line": self.context_line,
            "suppressed": self.suppressed,
        }


@dataclass
class PerfReport:
    """Complete static performance and resource leak evaluation report."""
    workspace_root: str
    diagnostics: List[PerfDiagnostic] = field(default_factory=list)
    scanned_files_count: int = 0
    latency_ms: float = 0.0
    rules_checked: List[str] = field(
        default_factory=lambda: [r.value for r in PerfRule]
    )

    @property
    def total_diagnostics_count(self) -> int:
        """Total number of active (non-suppressed) diagnostics."""
        return len([d for d in self.diagnostics if not d.suppressed])

    @property
    def errors_count(self) -> int:
        """Number of error-severity diagnostics."""
        return len([
            d for d in self.diagnostics
            if not d.suppressed and d.severity == Severity.ERROR
        ])

    @property
    def warnings_count(self) -> int:
        """Number of warn-severity diagnostics."""
        return len([
            d for d in self.diagnostics
            if not d.suppressed and d.severity == Severity.WARN
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        active = [d for d in self.diagnostics if not d.suppressed]
        return {
            "workspace_root": self.workspace_root,
            "total_diagnostics_count": self.total_diagnostics_count,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "scanned_files_count": self.scanned_files_count,
            "latency_ms": round(self.latency_ms, 2),
            "rules_checked": self.rules_checked,
            "diagnostics": [d.to_dict() for d in active],
        }

    def format_table(self) -> str:
        """Format report into clean, colored ASCII terminal table."""
        active = [d for d in self.diagnostics if not d.suppressed]
        if not active:
            return (
                f"\033[92m✔ No performance anti-patterns or resource leaks detected\033[0m "
                f"across {self.scanned_files_count} files ({self.latency_ms:.2f} ms)."
            )

        headers = ["Rule", "Severity", "Location", "Message"]
        rows: List[List[str]] = []
        for d in active:
            loc = f"{d.file_path}:{d.lineno}"
            sev = d.severity.value.upper()
            rows.append([d.rule_id, sev, loc, d.message])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        # Enforce maximum column width for message to prevent terminal overflow
        if col_widths[3] > 80:
            col_widths[3] = 80

        def make_separator(char: str = "-") -> str:
            parts = [char * (w + 2) for w in col_widths]
            return f"+{'+'.join(parts)}+"

        lines = [
            make_separator("-"),
            "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |",
            make_separator("="),
        ]

        for row in rows:
            color = "\033[91m" if row[1] == "ERROR" else "\033[93m"
            color_reset = "\033[0m"
            msg = row[3]
            if len(msg) > col_widths[3]:
                msg = msg[: col_widths[3] - 3] + "..."
            formatted_cells = [
                row[0].ljust(col_widths[0]),
                row[1].ljust(col_widths[1]),
                row[2].ljust(col_widths[2]),
                msg.ljust(col_widths[3]),
            ]
            lines.append(f"| {color}{' | '.join(formatted_cells)}{color_reset} |")

        lines.append(make_separator("-"))

        summary = (
            f"\033[91m✖ Found {self.total_diagnostics_count} performance anti-patterns\033[0m "
            f"({self.errors_count} errors, {self.warnings_count} warnings) "
            f"across {self.scanned_files_count} files in {self.latency_ms:.2f} ms."
        )
        lines.append(summary)
        return "\n".join(lines)

    def format_text(self) -> str:
        """Format report into concise text lines."""
        active = [d for d in self.diagnostics if not d.suppressed]
        if not active:
            return (
                f"✔ No performance anti-patterns or resource leaks detected "
                f"across {self.scanned_files_count} files ({self.latency_ms:.2f} ms)."
            )

        lines = [
            f"Performance Anti-Patterns & Resource Leaks ({self.total_diagnostics_count} issues: "
            f"{self.errors_count} errors, {self.warnings_count} warnings):",
            "--------------------------------------------------------------------------------",
        ]
        for d in active:
            sev = d.severity.value.upper()
            lines.append(
                f"  • {d.file_path}:{d.lineno} [{d.rule_id}] ({sev}): {d.message}"
            )
            if d.context_line:
                lines.append(f"    Line: {d.context_line}")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(
            f"Scan completed in {self.latency_ms:.2f} ms across {self.scanned_files_count} files."
        )
        return "\n".join(lines)
