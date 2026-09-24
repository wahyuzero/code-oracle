"""
TopoSlice Verification Engine.
Coordinates the 5 stages of the lean neuro-symbolic verification pipeline.
"""

import copy
import time
from pathlib import Path
from typing import Optional

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.linearizer import linearize_subgraph
from code_oracle.locator import locate_affected_symbols
from code_oracle.models import VerificationReport
from code_oracle.slicer import slice_neighborhood
from code_oracle.symbolic import verify_symbolic_gate


class TopoSliceEngine:
    """
    Sub-50ms Neuro-Symbolic Verification Engine.
    Executes AST Diff Boundary Locating, Inverted Workspace Indexing,
    k-Hop Slicing, Tarjan SCC Cycle Detection & Contract Checks, and Graph Linearization.
    """

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.indexer = WorkspaceIndexer(workspace_root=self.workspace_root)

    def verify(
        self,
        file_path: str,
        patch_content: str,
        k: int = 1,
        max_fanout: int = 20,
    ) -> VerificationReport:
        """
        Verify a code patch proposal against AST topology and contract invariants.
        Returns a VerificationReport with structured verdict and sub-400 token DSL.
        """
        start_time = time.perf_counter()

        # Path normalization relative to workspace root
        p = Path(file_path)
        if p.is_absolute():
            try:
                norm_path = str(p.resolve().relative_to(self.workspace_root.resolve())).replace("\\", "/")
            except ValueError:
                norm_path = str(file_path).replace("\\", "/")
        else:
            full_p = (self.workspace_root / file_path).resolve()
            try:
                norm_path = str(full_p.relative_to(self.workspace_root.resolve())).replace("\\", "/")
            except ValueError:
                norm_path = str(file_path).replace("\\", "/")

        # Stage 1: Diff Boundary Locator
        patch_result = locate_affected_symbols(
            file_path=norm_path,
            patch_content=patch_content,
            workspace_root=self.workspace_root,
        )

        # Immediate exit on syntax error
        if patch_result.syntax_error:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            violation_msg = f"SYNTAX_ERROR: {patch_result.syntax_error}"
            return VerificationReport(
                status="REJECTED",
                confidence=1.0,
                cycles_detected=[],
                invariant_violations=[violation_msg],
                linearized_subgraph=f"[DIFF_TARGET] {norm_path} (SYNTAX_ERROR)\n[GATE]\nSTATUS: REJECTED\nVIOLATIONS:\n  - {violation_msg}",
                affected_symbols=[],
                latency_ms=elapsed_ms,
            )

        # Stage 2: Workspace Indexer (incremental update)
        self.indexer.scan_workspace()

        # Save existing file cache entry for zero-side-effect transient evaluation
        cached_backup = copy.deepcopy(self.indexer._file_cache.get(norm_path))

        try:
            # In-memory overlay of transient patched symbols and imports
            if patch_result.all_patched_symbols or patch_result.deleted_symbols or patch_result.imports:
                self.indexer.overlay_transient_symbols(
                    norm_path,
                    patch_result.all_patched_symbols,
                    imports=patch_result.imports,
                )

            # Stage 3: k-Hop Neighborhood Slicer
            seed_symbols = (
                patch_result.affected_symbols
                or patch_result.added_symbols
                or patch_result.deleted_symbols
            )
            slice_graph = slice_neighborhood(
                seeds=seed_symbols,
                indexer=self.indexer,
                k=k,
                max_fanout=max_fanout,
            )

            # Stage 4: Deterministic Symbolic Gate
            gate_result = verify_symbolic_gate(
                patch_result=patch_result,
                slice_graph=slice_graph,
                indexer=self.indexer,
            )

            # Stage 5: Graph Linearizer (< 400 tokens)
            linearized_dsl = linearize_subgraph(
                patch_result=patch_result,
                slice_graph=slice_graph,
                gate_result=gate_result,
                max_tokens=400,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return VerificationReport(
                status=gate_result.status,
                confidence=gate_result.confidence,
                cycles_detected=gate_result.cycles,
                invariant_violations=gate_result.violations,
                linearized_subgraph=linearized_dsl,
                affected_symbols=[s.qualname for s in seed_symbols],
                latency_ms=elapsed_ms,
            )
        finally:
            # Restore indexer to default disk state (Rollback Resilience)
            if cached_backup is not None:
                self.indexer._file_cache[norm_path] = cached_backup
            else:
                self.indexer._file_cache.pop(norm_path, None)
            self.indexer._rebuild_indices()
