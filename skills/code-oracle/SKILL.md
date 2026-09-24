---
name: code-oracle
description: Sub-50ms neuro-symbolic verification oracle for AI coding agents. Evaluates proposed code modifications in-memory against AST topology, contract arity, circular dependencies (calls/imports), and symbol lifetimes before committing changes to disk. Activate automatically before writing Python edits to disk.
---

# Code Oracle: Neuro-Symbolic Pre-Flight Verification Gate

Code Oracle is a local, offline verification oracle designed for AI coding agents (Antigravity, Claude Code, Cursor, OpenCode). It decouples code generation from verification by testing proposed patches against workspace AST topology and invariants in under 50ms with zero token cost.

## Mental Model: The Pre-Flight Gate

```
  [ AI Agent ] ──(1. Propose Patch)──► [ Code Oracle In-Memory Gate ]
                                                  │
                                          ┌───────┴───────┐
                                          ▼               ▼
                                     [ APPROVED ]    [ REJECTED ]
                                          │               │
                                   Apply to disk    Self-Correct via
                                                   Violations & DSL
```

1. **Evaluator, Not Author:** Code Oracle does not write or modify your code; it evaluates proposed modifications against the workspace AST graph.
2. **Zero-Side-Effect Transient Overlay:** Patches are evaluated in-memory against transient symbols. Workspace disk files are untouched during verification.
3. **Deterministic AST Symbolic Engine:** Validates syntax trees, runs Tarjan's SCC for circular calls and import cycles, and verifies arity, keyword, and symbol lifetime contracts.
4. **Sub-50ms Latency & Zero Output Token Tax:** Replaces slow (2,000–6,000ms) LLM self-critique loops with deterministic local checks returning machine-readable JSON.

---

## When to Activate (Proactive Triggers)

Activate Code Oracle automatically before writing changes to disk whenever:
- Modifying existing Python functions, methods, or class signatures / constructors (`__init__`).
- Adding new parameters to existing functions or classes (detecting missing arguments in existing callers / instantiations).
- Adding or modifying call sites to existing functions or class constructors (verifying arity and parameter contracts).
- Refactoring modules, renaming files, or reorganizing package imports (detecting circular imports).
- Deleting or deprecating functions, methods, or classes (detecting broken references across the workspace).
- Generating multi-file edits where dependencies must remain acyclic.

---

## CLI Commands

### 1. `verify` (Pre-Flight Verification Gate)

Verify a patch proposal before applying it to disk.

```bash
# Pipe diff via stdin (standard unified diff)
git diff path/to/file.py | code-oracle verify path/to/file.py --workspace . --json

# Pipe unified diff via stdin using '-'
echo "$DIFF_CONTENT" | code-oracle verify path/to/file.py -p - --workspace . --json

# Pass diff file path
code-oracle verify path/to/file.py --patch patch.diff --workspace . --json

# Pass full replacement content directly
code-oracle verify path/to/file.py --patch "def target(): return 42" --workspace . --json

# Set k-hop neighborhood search radius (default: 1, max: 2)
code-oracle verify path/to/file.py --patch patch.diff -k 2 --workspace . --json
```

**Exit Codes:**
- `0`: Patch **APPROVED** (all invariants hold).
- `1`: Patch **REJECTED** (invariant violations or cycles detected).
- `2`: CLI argument or invocation error.

### 2. `slice` (Compact Neighborhood Extraction)

Inspect the immediate topological neighborhood of a symbol (< 400 tokens DSL) without reading entire multi-KB files into context.

```bash
# Extract 1-hop slice as compact DSL
code-oracle slice path/to/file.py --symbol MyClass --workspace .

# Extract 2-hop slice as machine-readable JSON
code-oracle slice path/to/file.py --symbol process_data --k 2 --workspace . --json
```

### 3. `index` (Workspace Symbol Ingestion)

Incrementally index all Python files into `.code_oracle/index.json`.

```bash
# Incremental scan (mtime and content hash cached)
code-oracle index .

# Force full re-indexing and output JSON statistics
code-oracle index . --force --json
```

### 4. `clean` (Rollback Resilience)

Safely remove `.code_oracle/` cache directory to restore the workspace to default state with zero destructive impact.

```bash
code-oracle clean .
code-oracle clean . --json
```

### 5. `serve` (FastMCP Server)

Start the lightweight FastMCP server exposing `verify_code_patch` for MCP-enabled agents.

```bash
code-oracle serve
```

---

## Parsing Verification Output

Always pass `--json` to `code-oracle verify` for structured programmatic parsing.

### Output JSON Schema

```json
{
  "status": "APPROVED | REJECTED",
  "confidence": 0.98,
  "cycles_detected": [
    ["caller_a", "caller_b"]
  ],
  "invariant_violations": [
    "ARITY_MISMATCH: Caller 'service.py::run' (line 12) calls 'target' with 3 positional arguments, but 'target' accepts at most 2 positional arguments."
  ],
  "linearized_subgraph": "[DIFF_TARGET] ...\n[NODES] ...\n[EDGES] ...\n[GATE] ...",
  "affected_symbols": ["target"],
  "latency_ms": 14.2
}
```

### Violation Taxonomy & Meanings

| Violation Prefix | Cause |
|---|---|
| `SYNTAX_ERROR` | Patched content fails AST parsing (syntax error at line/offset). |
| `CIRCULAR_DEPENDENCY` | Direct or multi-hop call cycle (`f -> g -> f`) or import cycle (`a.py -> b.py -> a.py`) detected via Tarjan SCC. |
| `ARITY_MISMATCH` | Missing required positional argument, or caller supplies more positional arguments than accepted. |
| `KEYWORD_MISMATCH` | Unexpected keyword argument, missing required keyword-only argument, or positional-only argument called as keyword. |
| `DUPLICATE_ARGUMENT` | Argument supplied both positionally and by keyword name. |
| `BROKEN_REFERENCE` | A deleted symbol is still called or imported elsewhere, or an imported symbol does not exist in the target workspace file. |

---

## Agent Self-Correction Playbook

When `code-oracle verify --json` returns `"status": "REJECTED"`, follow this decision tree before asking the user or applying changes:

```
                  ┌──────────────────────────────┐
                  │ code-oracle verify --json    │
                  └──────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
            [APPROVED]                      [REJECTED]
                 │                               │
        Proceed with write             Inspect violations list
                                                 │
            ┌────────────────────┬───────────────┴────────────────────┬────────────────────┐
            ▼                    ▼                                    ▼                    ▼
     [SYNTAX_ERROR]      [ARITY_MISMATCH]                    [CIRCULAR_DEPENDENCY]  [BROKEN_REFERENCE]
            │                    │                                    │                    │
     Fix syntax at        Check params:                        Invert dependency,   Update callers/
     reported line        - Add default value to callee        extract shared code  importers before
                          - OR update caller arguments         to new module        deleting symbol
```

### 1. Handling `ARITY_MISMATCH`
- **Cause A (Modified callee):** You added a new parameter `param: int` to `process()`, but existing caller `caller.py::run` calls `process()` without it.
  - **Correction:** Provide a sensible default value: `process(..., param: int = 0)`.
- **Cause B (Modified caller):** Your patch in `app.py` calls `calculate(x, y, z)` but `calculate` only accepts `(x, y)`.
  - **Correction:** Remove extra argument or check `code-oracle slice target.py --symbol calculate` to inspect the exact signature.
- **Cause C (Class Instantiation):** Your patch calls `Worker("Alice")` without supplying required argument `email`, or changes `Worker.__init__` adding required parameters that break existing instantiations.
  - **Correction:** Supply required constructor arguments or provide default parameter values in `__init__`.

### 2. Handling `CIRCULAR_DEPENDENCY`
- **Call Cycle:** Function `A` calls `B`, and `B` calls `A`.
  - **Correction:** Break the mutual recursion by passing a callback, decomposing responsibilities, or merging into a single sequential workflow.
- **Import Cycle:** File `auth.py` imports from `users.py`, and `users.py` imports from `auth.py`.
  - **Correction:** Move shared models or types into a neutral leaf module (e.g., `models.py` or `types.py`), or use deferred function-level imports.

### 3. Handling `BROKEN_REFERENCE`
- **Deleted Symbol with Callers:** You deleted `deprecated_helper()`, but `service.py::step` still calls it.
  - **Correction:** Either refactor `service.py` to use the replacement before deleting `deprecated_helper()`, or keep `deprecated_helper()` as an alias.
- **Nonexistent Imported Symbol:** You wrote `from utils import parse_date`, but `utils.py` only defines `parse_timestamp`.
  - **Correction:** Use the actual symbol name defined in the workspace file.

### 4. Handling `KEYWORD_MISMATCH`
- **Positional-Only Violation:** Callee has `def config(port: int, /, host: str)`. Caller passed `config(port=8080)`.
  - **Correction:** Pass `port` positionally: `config(8080, host="localhost")`.
- **Missing Keyword-Only:** Callee has `def run(*, timeout: int)`. Caller called `run()`.
  - **Correction:** Supply required keyword argument: `run(timeout=30)`.

---

## Agent Guardrails & Anti-Patterns

### ❌ Anti-Patterns to Avoid

- **DO NOT write unverified patches directly to disk:** Always test with `code-oracle verify` first.
- **DO NOT bypass the oracle when it rejects:** If `status == "REJECTED"`, the topological invariant failed. Do not dismiss it as a tool glitch; fix the underlying contract.
- **DO NOT read whole multi-thousand line files for topological context:** Use `code-oracle slice <file> --symbol <name>` to get the precise neighborhood in under 400 tokens.
- **DO NOT delete symbols without verifying workspace callers:** Run verification to ensure zero callers or importers remain broken.
- **DO NOT leave dirty transient files:** If testing scratch modifications, use `code-oracle clean` to restore cache state cleanly.

### ✅ Recommended Agent Workflow Loop

```bash
# Step 1: Check existing symbol neighborhood before designing changes
code-oracle slice src/service.py --symbol process_order --json

# Step 2: Formulate proposed patch in memory

# Step 3: Run pre-flight verification gate via stdin
git diff src/service.py | code-oracle verify src/service.py --workspace . --json
# Or:
echo "$PROPOSED_DIFF" | code-oracle verify src/service.py -p - --workspace . --json

# Step 4: Check verdict
# If APPROVED -> Apply patch to disk with confidence
# If REJECTED -> Read invariant_violations, self-correct, and re-verify
```
