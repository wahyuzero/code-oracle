# ADR-0001: TopoSlice Lean Topological Slicer Architecture

**Date**: 2026-09-24  
**Status**: accepted  
**Deciders**: Wahyu Febri Tamtomo (@wahyuzero), Antigravity

## Context

AI coding agents (such as Claude Code, Cursor, OpenCode, and Codex) require rapid validation when proposing code patches. Existing verification approaches rely either on full cloud LLM reviews (which suffer from 2,500 to 6,500 ms round-trip latency, hallucination, and high token costs) or full-repository graph analyzers (which build global knowledge graphs, community clusters, and visualizations over 2 to 15 seconds).

For Code Oracle to meet its target service-level agreement of sub-50ms local verification, rebuilding a full repository graph on every patch proposal is unviable. We require an isolated, deterministic, and frugal topological slicer that extracts only the impacted neighborhood around modified code entities in under 5 ms.

## Decision

We adopt **TopoSlice**: a localized, incremental AST topological slicer and deterministic symbolic gate designed specifically for the Code Oracle pipeline.

TopoSlice operates in five stages:
1. **Diff Boundary Locator**: Maps patch diffs to precise AST symbol spans (functions, methods, classes).
2. **Workspace Indexer**: Maintains an incremental, mtime-hashed inverted symbol index (`.code_oracle/index.json`) mapping exported symbols to caller files.
3. **k-Hop Neighborhood Slicer**: Isolates a compact directed subgraph (10 to 50 nodes) representing immediate callers and callees ($k=1$ or $k=2$), capped by a fan-out threshold to prevent graph explosion on utility nodes.
4. **Deterministic Symbolic Gate**: Executes Tarjan's Strongly Connected Components (SCC) algorithm for cycle detection and verifies parameter arity/contract invariants in under 2 ms.
5. **Graph Linearizer**: Serializes the sliced subgraph and patch metadata into a compact Domain-Specific Language (< 400 tokens) that fits comfortably within the 512-token context window of the Laya ModernBERT decision head.

## Alternatives Considered

### Alternative 1: Full-Repository Macro Knowledge Graph Extraction
- **Pros**: Complete repository visibility, deep multi-hop relationship extraction, and rich community metadata.
- **Cons**: Requires 2,000 to 15,000 ms per run; high CPU and memory footprint.
- **Why not**: Directly violates the sub-50ms latency objective.

### Alternative 2: Pure Compiler / Syntax Checking Without Neural Layer
- **Pros**: Zero neural inference overhead; deterministic.
- **Cons**: Rigid rule set; cannot evaluate semantic risk, drift probability, or subtle architectural intention.
- **Why not**: Fails to detect non-syntactic semantic invariant violations that a calibrated decision model can capture.

### Alternative 3: Frontier LLM-as-a-Judge Review
- **Pros**: High-level reasoning across complex prose and logic.
- **Cons**: Incurs 2.5 to 6.5 seconds of network latency, monetary costs per call, self-review confirmation bias, and massive context bloat.
- **Why not**: Incompatible with offline execution and radical frugality principles.

## Empirical Grounding & Measurements

Measurements executed locally on this workstation validate the latency budget:
* **Python AST Parsing (`ast.parse`)**: 0.094 ms average per file (< 100 microseconds).
* **Tarjan's SCC Cycle Detection (`find_cycles_tarjan`)**: 0.066 ms average on a 50-node circular graph.
* **Combined Symbolic Budget**: $\sim 1.4\text{ ms}$, leaving over 45 ms for in-memory Laya neural inference.

## Consequences

### Positive
- Achieves end-to-end patch verification in under 50 ms.
- Constrains subgraph representations to under 400 tokens, fitting the 512-token limit of ModernBERT backbones.
- Runs 100% locally with zero cloud API dependencies and zero token generation tax.
- Operates through a single FastMCP endpoint (`verify_patch`), preventing agent context bloat.

### Negative
- Requires maintaining a lightweight workspace index cache on disk.
- Highly dynamic language constructs (e.g., Python `getattr` or dynamic string imports) cannot be statically resolved without heuristic tagging.

### Risks & Mitigations
- **Risk**: Fan-out explosion when modifying central utility functions (e.g., loggers or helpers).  
  **Mitigation**: Enforce a strict degree cutoff (maximum 20 edges per node) and mark unexpanded edges with an explicit `TRUNCATED` attribute.
- **Risk**: Cache drift between disk state and index.  
  **Mitigation**: File modification times (`mtime`) and content hashes invalidate stale index entries automatically upon read.

## Rollback Resilience

All cache data and temporary indices are isolated inside `.code_oracle/` and excluded via `.gitignore`. Running `code-oracle clean` or deleting `.code_oracle/` safely restores the workspace to default state with zero destructive impact on user source files.
