"""
Graph Reachability Dead Code Engine.
Traverses the workspace symbol reference graph from entrypoint roots,
isolating direct orphans (in-degree == 0) and transitive dead clusters.
"""

from collections import deque
from pathlib import Path
import time
from typing import Dict, List, Optional, Set

from code_oracle.dead_code.entrypoints import EntrypointDetector
from code_oracle.dead_code.models import DeadCodeReport, DeadSymbol
from code_oracle.indexer import WorkspaceIndexer
from code_oracle.models import Symbol


def is_symbol_exported(symbol: Symbol, detector: EntrypointDetector) -> bool:
    """
    Determine if a symbol is exported / public in its host language.
    - Python: names without leading underscore
    - Go: uppercase first letter
    - Rust: pub visibility
    - TypeScript: export keyword
    """
    ext = Path(symbol.file_path).suffix.lower()

    if ext == ".py":
        return not symbol.name.startswith("_")

    if ext == ".go":
        return bool(symbol.name and symbol.name[0].isupper())

    # For Rust and TypeScript, inspect source code lines
    lines = detector.get_source_lines(symbol.file_path)
    if lines and 1 <= symbol.lineno <= len(lines):
        line_content = lines[symbol.lineno - 1].strip()
        if ext == ".rs":
            return line_content.startswith("pub ") or "pub fn " in line_content or "pub struct " in line_content
        if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
            return line_content.startswith("export ") or "export default" in line_content

    # Fallback to signature inspection
    sig = symbol.signature.strip()
    if ext == ".rs":
        return sig.startswith("pub ")
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
        return sig.startswith("export ")

    return True


class DeadCodeDetector:
    """
    Reachability engine that identifies unreachable and orphan symbols.
    Operates over the WorkspaceIndexer symbol topology.
    """

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        indexer: Optional[WorkspaceIndexer] = None,
    ):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.indexer = indexer or WorkspaceIndexer(workspace_root=self.workspace_root)
        self.entrypoint_detector = EntrypointDetector(workspace_root=self.workspace_root)

    def _matches_paths(self, file_path: str, filter_paths: List[str]) -> bool:
        """Check if symbol file path matches any requested filter path."""
        norm_file = file_path.replace("\\", "/").lstrip("./")

        for p in filter_paths:
            # Normalize candidate filter path
            p_obj = Path(p)
            if p_obj.is_absolute():
                try:
                    norm_p = str(p_obj.resolve().relative_to(self.workspace_root)).replace("\\", "/")
                except ValueError:
                    norm_p = str(p).replace("\\", "/")
            else:
                norm_p = str(p).replace("\\", "/").lstrip("./")

            if norm_file == norm_p:
                return True
            if norm_file.startswith(norm_p.rstrip("/") + "/"):
                return True

        return False

    def _find_cluster_root(
        self,
        symbol_id: str,
        reverse_graph: Dict[str, Set[str]],
        dead_ids: Set[str],
    ) -> str:
        """
        Trace back callers within dead candidates to locate the root orphan or cycle anchor.
        """
        visited = set()
        queue = deque([symbol_id])
        cluster_orphans = []

        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)

            dead_callers = [c for c in reverse_graph.get(curr, set()) if c in dead_ids and c != curr]
            if not dead_callers:
                cluster_orphans.append(curr)
            else:
                for c in dead_callers:
                    if c not in visited:
                        queue.append(c)

        if cluster_orphans:
            # Pick first/lowest orphan as root
            return sorted(cluster_orphans)[0]
        # In a closed mutual cycle, pick the lexicographically lowest symbol ID
        return sorted(visited)[0] if visited else symbol_id

    def detect(
        self,
        paths: Optional[List[str]] = None,
        min_lines: int = 0,
        include_unexported: bool = False,
    ) -> DeadCodeReport:
        """
        Execute full workspace reachability analysis and return dead code report.
        """
        start_time = time.perf_counter()

        # Ensure index is updated
        self.indexer.scan_workspace()

        all_symbols = list(self.indexer._definitions.values())
        forward_graph: Dict[str, Set[str]] = {s.id: set() for s in all_symbols}
        reverse_graph: Dict[str, Set[str]] = {s.id: set() for s in all_symbols}

        roots: Set[str] = set()

        # 1. Identify entrypoint roots
        for sym in all_symbols:
            if self.entrypoint_detector.is_entrypoint(sym):
                roots.add(sym.id)

        # 2. Expand roots from public export files (__init__.py, index.ts, mod.rs)
        for rel_file, imports in self.indexer._file_imports.items():
            if self.entrypoint_detector.is_root_export(
                Symbol(name="", qualname="", file_path=rel_file, kind="module", lineno=1, end_lineno=1)
            ):
                for imp in imports:
                    target_file = self.indexer.resolve_import_to_file(imp, rel_file)
                    if target_file:
                        if imp.name == "*":
                            for file_sym in self.indexer.get_file_symbols(target_file):
                                roots.add(file_sym.id)
                        else:
                            target_id = f"{target_file}::{imp.name}"
                            if target_id in self.indexer._definitions:
                                roots.add(target_id)
                            else:
                                for file_sym in self.indexer.get_file_symbols(target_file):
                                    if file_sym.name == imp.name:
                                        roots.add(file_sym.id)

        # 3. Build directed reference edges
        for sym in all_symbols:
            # Call edges
            for call in sym.calls:
                callee_sym = self.indexer.resolve_callee(call, sym)
                if callee_sym and callee_sym.id in self.indexer._definitions:
                    forward_graph[sym.id].add(callee_sym.id)
                    reverse_graph[callee_sym.id].add(sym.id)

            # Class inheritance and constructor edges
            if sym.kind == "class":
                init_sym = self.indexer.resolve_class_init(sym)
                if init_sym and init_sym.id in self.indexer._definitions:
                    forward_graph[sym.id].add(init_sym.id)
                    reverse_graph[init_sym.id].add(sym.id)

                for base_name in getattr(sym, "bases", []):
                    base_sym = self.indexer.get_definition(base_name)
                    if base_sym and base_sym.id in self.indexer._definitions:
                        forward_graph[sym.id].add(base_sym.id)
                        reverse_graph[base_sym.id].add(sym.id)

        # 4. Forward reachability traversal (BFS from roots)
        reachable: Set[str] = set(roots)
        queue = deque(roots)

        while queue:
            curr_id = queue.popleft()
            for callee_id in forward_graph.get(curr_id, ()):
                if callee_id not in reachable:
                    reachable.add(callee_id)
                    queue.append(callee_id)

        # 5. Extract unreachable symbols (ignoring module blocks)
        dead_candidates: List[Symbol] = []
        dead_candidate_ids: Set[str] = set()

        for sym in all_symbols:
            if sym.kind == "module" or sym.name == "<module>":
                continue

            if sym.id in reachable:
                continue

            # Exemption: dunder/magic methods on an ALIVE class are not dead
            if self.entrypoint_detector.is_magic_method(sym) and sym.is_method:
                parent_qualname = sym.qualname.rsplit(".", 1)[0] if "." in sym.qualname else ""
                class_id = f"{sym.file_path}::{parent_qualname}"
                if class_id in reachable:
                    continue

            dead_candidates.append(sym)
            dead_candidate_ids.add(sym.id)

        # 6. Classify direct orphans and transitive dead clusters
        dead_symbols: List[DeadSymbol] = []

        for sym in dead_candidates:
            # Check visibility
            is_exported = is_symbol_exported(sym, self.entrypoint_detector)
            if not include_unexported and not is_exported:
                continue

            # Check min lines
            lines_count = max(1, sym.end_lineno - sym.lineno + 1)
            if min_lines > 0 and lines_count < min_lines:
                continue

            # Check paths filter
            if paths and not self._matches_paths(sym.file_path, paths):
                continue

            # Classify orphan vs transitive
            live_callers = [c for c in reverse_graph.get(sym.id, set()) if c in reachable]
            dead_callers = [c for c in reverse_graph.get(sym.id, set()) if c in dead_candidate_ids and c != sym.id]

            if not dead_callers and not live_callers:
                is_orphan = True
                is_transitive = False
                cluster_id = None
                reason = "Unreferenced symbol with 0 incoming calls"
            else:
                is_orphan = False
                is_transitive = True
                cluster_id = self._find_cluster_root(sym.id, reverse_graph, dead_candidate_ids)
                if cluster_id == sym.id:
                    reason = "Dead cycle: mutual calls with no external entrypoint"
                else:
                    reason = f"Transitive dead symbol: only called by unreachable symbols (cluster: {cluster_id})"

            dead_symbols.append(
                DeadSymbol(
                    id=sym.id,
                    name=sym.name,
                    qualname=sym.qualname,
                    file_path=sym.file_path,
                    kind=sym.kind,
                    lineno=sym.lineno,
                    end_lineno=sym.end_lineno,
                    is_orphan=is_orphan,
                    is_transitive=is_transitive,
                    cluster_id=cluster_id,
                    confidence=1.0,
                    reason=reason,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Sort dead symbols by file path and line number
        dead_symbols.sort(key=lambda s: (s.file_path, s.lineno))

        return DeadCodeReport(
            workspace_root=str(self.workspace_root),
            total_symbols_scanned=len(all_symbols),
            dead_symbols=dead_symbols,
            roots_count=len(roots),
            scanned_files_count=len(self.indexer._file_cache),
            latency_ms=elapsed_ms,
        )


def detect_dead_code(
    workspace_root: Optional[Path] = None,
    indexer: Optional[WorkspaceIndexer] = None,
    paths: Optional[List[str]] = None,
    min_lines: int = 0,
    include_unexported: bool = False,
) -> DeadCodeReport:
    """Top-level convenience function to detect dead code."""
    detector = DeadCodeDetector(workspace_root=workspace_root, indexer=indexer)
    return detector.detect(
        paths=paths,
        min_lines=min_lines,
        include_unexported=include_unexported,
    )
