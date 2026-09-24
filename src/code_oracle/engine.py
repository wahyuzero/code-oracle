"""
TopoSlice Verification Engine.
Coordinates the 5 stages of the lean neuro-symbolic verification pipeline.
"""

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from code_oracle.indexer import WorkspaceIndexer
from code_oracle.linearizer import linearize_subgraph
from code_oracle.locator import extract_imports_from_ast, extract_symbols_from_ast, locate_affected_symbols
from code_oracle.models import PatchResult, VerificationReport
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
            all_seeds = (
                patch_result.affected_symbols
                + patch_result.added_symbols
                + patch_result.deleted_symbols
            )
            seen_ids = set()
            seed_symbols = []
            for s in all_seeds:
                if s.id not in seen_ids:
                    seen_ids.add(s.id)
                    seed_symbols.append(s)
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

    def verify_batch(
        self,
        file_patches: List[Dict[str, Any]],
        dirty_overlays: Optional[Dict[str, str]] = None,
        k: int = 1,
        max_fanout: int = 20,
    ) -> VerificationReport:
        """
        Atomically verify a batch of file patches (e.g. staged git files)
        against AST topology and contract invariants with zero-side-effect isolation.
        """
        start_time = time.perf_counter()

        if not file_patches:
            return VerificationReport(
                status="APPROVED",
                confidence=1.0,
                cycles_detected=[],
                invariant_violations=[],
                linearized_subgraph="[BATCH] No files to verify.\n[GATE]\nSTATUS: APPROVED",
                affected_symbols=[],
                latency_ms=0.0,
            )

        # Stage 1: Diff Boundary Locator for each target file
        patch_results: Dict[str, PatchResult] = {}
        syntax_errors: List[str] = []

        for item in file_patches:
            raw_path = item["file_path"]
            patch_content = item["patch_content"]
            orig_content = item.get("original_content")

            p = Path(raw_path)
            if p.is_absolute():
                try:
                    norm_path = str(p.resolve().relative_to(self.workspace_root.resolve())).replace("\\", "/")
                except ValueError:
                    norm_path = str(raw_path).replace("\\", "/")
            else:
                full_p = (self.workspace_root / raw_path).resolve()
                try:
                    norm_path = str(full_p.relative_to(self.workspace_root.resolve())).replace("\\", "/")
                except ValueError:
                    norm_path = str(raw_path).replace("\\", "/")

            pr = locate_affected_symbols(
                file_path=norm_path,
                patch_content=patch_content,
                workspace_root=self.workspace_root,
                original_content=orig_content,
                is_replacement=item.get("is_replacement", True),
            )
            if pr.syntax_error:
                syntax_errors.append(f"SYNTAX_ERROR in '{norm_path}': {pr.syntax_error}")
            patch_results[norm_path] = pr

        # Immediate exit on syntax error in any file
        if syntax_errors:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return VerificationReport(
                status="REJECTED",
                confidence=1.0,
                cycles_detected=[],
                invariant_violations=syntax_errors,
                linearized_subgraph="[BATCH_DIFF] SYNTAX_ERROR\n[GATE]\nSTATUS: REJECTED\nVIOLATIONS:\n"
                + "\n".join(f"  - {err}" for err in syntax_errors),
                affected_symbols=[],
                latency_ms=elapsed_ms,
            )

        # Stage 2: Workspace Indexer scan
        self.indexer.scan_workspace()

        # Normalize dirty_overlays keys
        norm_dirty: Dict[str, str] = {}
        if dirty_overlays:
            for uf, ucontent in dirty_overlays.items():
                p = Path(uf)
                if p.is_absolute():
                    try:
                        n_uf = str(p.resolve().relative_to(self.workspace_root.resolve())).replace("\\", "/")
                    except ValueError:
                        n_uf = str(uf).replace("\\", "/")
                else:
                    full_p = (self.workspace_root / uf).resolve()
                    try:
                        n_uf = str(full_p.relative_to(self.workspace_root.resolve())).replace("\\", "/")
                    except ValueError:
                        n_uf = str(uf).replace("\\", "/")
                norm_dirty[n_uf] = ucontent

        # Save existing file cache entries for clean rollback
        all_touched_paths = set(patch_results.keys()) | set(norm_dirty.keys())
        cached_backups = {
            path: copy.deepcopy(self.indexer._file_cache.get(path))
            for path in all_touched_paths
        }

        try:
            # Overlay unstaged dirty files (with their git index state)
            for uf_path, uf_content in norm_dirty.items():
                if uf_path in patch_results:
                    continue
                syms = extract_symbols_from_ast(uf_content, file_path=uf_path)
                imps = extract_imports_from_ast(uf_content, file_path=uf_path)
                self.indexer.overlay_transient_symbols(uf_path, syms, imports=imps)

            # Atomic Batch Overlay: overlay all staged symbols into indexer
            for norm_path, pr in patch_results.items():
                if pr.all_patched_symbols or pr.deleted_symbols or pr.imports:
                    self.indexer.overlay_transient_symbols(
                        norm_path,
                        pr.all_patched_symbols,
                        imports=pr.imports,
                    )

            # Slicing & Symbolic Gate across all staged files
            all_violations: List[str] = []
            all_cycles: List[List[str]] = []
            all_affected_symbols: List[str] = []
            seen_violations: Set[str] = set()

            for norm_path, pr in patch_results.items():
                all_seeds = pr.affected_symbols + pr.added_symbols + pr.deleted_symbols
                seen_ids = set()
                seed_symbols = []
                for s in all_seeds:
                    if s.id not in seen_ids:
                        seen_ids.add(s.id)
                        seed_symbols.append(s)

                all_affected_symbols.extend([s.qualname for s in seed_symbols])

                slice_graph = slice_neighborhood(
                    seeds=seed_symbols,
                    indexer=self.indexer,
                    k=k,
                    max_fanout=max_fanout,
                )

                gate_result = verify_symbolic_gate(
                    patch_result=pr,
                    slice_graph=slice_graph,
                    indexer=self.indexer,
                )

                for v in gate_result.violations:
                    if v not in seen_violations:
                        seen_violations.add(v)
                        all_violations.append(v)

                for c in gate_result.cycles:
                    if c not in all_cycles:
                        all_cycles.append(c)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if all_violations or all_cycles:
                status = "REJECTED"
                confidence = 0.95
            else:
                status = "APPROVED"
                confidence = 0.98

            dsl = f"[BATCH_VERIFIED] {len(patch_results)} staged files.\n[GATE]\nSTATUS: {status}"
            if all_violations:
                dsl += "\nVIOLATIONS:\n" + "\n".join(f"  - {v}" for v in all_violations)

            return VerificationReport(
                status=status,
                confidence=confidence,
                cycles_detected=all_cycles,
                invariant_violations=all_violations,
                linearized_subgraph=dsl,
                affected_symbols=list(dict.fromkeys(all_affected_symbols)),
                latency_ms=elapsed_ms,
            )
        finally:
            # Restore indexer to default disk state (Rollback Resilience)
            for path, backup in cached_backups.items():
                if backup is not None:
                    self.indexer._file_cache[path] = backup
                else:
                    self.indexer._file_cache.pop(path, None)
            self.indexer._rebuild_indices()

