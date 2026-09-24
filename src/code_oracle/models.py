"""
Core data structures for the TopoSlice neuro-symbolic verification engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any


@dataclass
class CallReference:
    """Represents a function or method call site."""
    callee: str
    args_count: int
    kwargs: List[str] = field(default_factory=list)
    lineno: int = 0
    caller: Optional[str] = None


@dataclass
class ImportReference:
    """Represents an imported module or symbol reference."""
    module: Optional[str]
    name: str
    asname: Optional[str] = None
    lineno: int = 0
    file_path: str = ""
    level: int = 0


@dataclass
class Parameter:
    """Represents a function parameter in an AST."""
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    has_default: bool = False
    is_vararg: bool = False
    is_kwarg: bool = False
    is_kwonly: bool = False
    is_posonly: bool = False


@dataclass
class Symbol:
    """Represents a code symbol (function, method, class, module)."""
    name: str
    qualname: str
    file_path: str
    kind: str  # 'function', 'async_function', 'class', 'method', 'module'
    lineno: int
    end_lineno: int
    signature: str = ""
    params: List[Parameter] = field(default_factory=list)
    min_args: int = 0
    max_args: Optional[int] = None  # None indicates *args allowed
    accepted_kwargs: Optional[Set[str]] = None  # None indicates **kwargs allowed
    required_kwargs: Set[str] = field(default_factory=set)
    return_type: Optional[str] = None
    calls: List[CallReference] = field(default_factory=list)
    is_method: bool = False
    is_static: bool = False
    bases: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.file_path}::{self.qualname}"


@dataclass
class DiffHunk:
    """Represents a unified diff hunk."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)


@dataclass
class PatchResult:
    """Represents the output of the Diff Boundary Locator."""
    file_path: str
    original_content: str
    patched_content: str
    modified_old_lines: Set[int] = field(default_factory=set)
    modified_new_lines: Set[int] = field(default_factory=set)
    affected_symbols: List[Symbol] = field(default_factory=list)
    deleted_symbols: List[Symbol] = field(default_factory=list)
    added_symbols: List[Symbol] = field(default_factory=list)
    all_patched_symbols: List[Symbol] = field(default_factory=list)
    imports: List[ImportReference] = field(default_factory=list)
    syntax_error: Optional[str] = None


@dataclass
class SliceNode:
    """Node in the k-hop sliced subgraph."""
    id: str
    name: str
    file_path: str
    kind: str
    signature: str
    is_seed: bool = False
    is_modified: bool = False
    truncated: bool = False
    symbol: Optional[Symbol] = None


@dataclass
class SliceEdge:
    """Directed edge in the k-hop sliced subgraph."""
    source: str
    target: str
    relation: str  # 'CALLS', 'IMPORTS', 'INHERITS'


@dataclass
class SlicedGraph:
    """Compact directed subgraph representing the affected neighborhood."""
    nodes: Dict[str, SliceNode] = field(default_factory=dict)
    edges: List[SliceEdge] = field(default_factory=list)
    seed_ids: Set[str] = field(default_factory=set)
    truncated: bool = False


@dataclass
class GateResult:
    """Outcome of the Deterministic Symbolic Gate."""
    status: str  # 'APPROVED', 'REJECTED'
    confidence: float
    cycles: List[List[str]] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Complete verification response for agent consumption."""
    status: str
    confidence: float
    cycles_detected: List[List[str]]
    invariant_violations: List[str]
    linearized_subgraph: str
    affected_symbols: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "cycles_detected": self.cycles_detected,
            "invariant_violations": self.invariant_violations,
            "linearized_subgraph": self.linearized_subgraph,
            "affected_symbols": self.affected_symbols,
            "latency_ms": round(self.latency_ms, 2),
        }
