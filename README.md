# Code Oracle

> Sub-50ms neuro-symbolic verification for AI coding agents.  
> Structural AST topology validated by Tarjan SCC and a non-autoregressive decision model (Laya).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Inference Latency](https://img.shields.io/badge/Verification-<50ms_local-brightgreen.svg)](#key-characteristics)
[![Token Waste](https://img.shields.io/badge/Output_Tokens-0_tokens-success.svg)](#key-characteristics)
[![Paradigm](https://img.shields.io/badge/Architecture-Neuro--Symbolic-orange.svg)](#architecture)

---

## The Problem: Autoregressive Verification Overhead

When coding agents (Claude Code, Cursor, OpenCode, Codex) inspect code modifications, they face three operational bottlenecks:

1. **Confirmation Bias:** Generative models reviewing their own diffs frequently rationalize their own logic errors.
2. **Latency and Token Overhead:** Re-evaluating complete files through a frontier model introduces 2,000 to 6,000 ms of round-trip latency and consumes output tokens on conversational explanations.
3. **Tool Schema Bloat:** Standard MCP servers inject sprawling multi-tool schemas into the prompt context on every turn, reducing effective agent window capacity.

---

## Architecture

Code Oracle decouples code generation from verification. It runs locally as an independent evaluation layer.

Instead of generating conversational critiques, Code Oracle constructs a localized AST graph using Tree-sitter, checks topological invariants symbolically, and evaluates residual drift using Laya (a 421M parameter ModernBERT decision model).

### Verification Pipeline

```
    [ AI Coding Agent / LLM ]
                │
                │  (1) Proposes Patch / Refactor
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                         CODE ORACLE                         │
  │                                                             │
  │  [Stage 1: Tree-sitter Ingestion & k-Hop Subgraph Slicing]  │
  │  - Parses structural syntax trees in < 8ms                  │
  │  - Extracts callers, callees, and imported interfaces       │
  │  - Isolates affected neighborhood (prevents graph explosion)│
  │                             │                               │
  │  [Stage 2: Deterministic Symbolic Gate]                     │
  │  - Algorithmic cycle detection (Tarjan's SCC)               │
  │  - Syntax validity & direct signature checks                │
  │                             │                               │
  │                             ▼ (Compact Subgraph State)      │
  │  [Stage 3: Laya In-Memory Decision Head (421M)]             │
  │  - Calibrated non-autoregressive probability evaluation     │
  │  - 35ms latency | 0 output tokens generated                 │
  └─────────────────────────────────────────────────────────────┘
                │
                │  (2) Structured Verdict (< 50ms)
                ▼
  "VERDICT: REJECT (Confidence: 0.94)
   - Invariant Breach: Breaks caller contract in [billing_service.py]
   - Topology Drift: Unhandled Optional[T] propagation"
```

---

## Key Characteristics

* **Sub-50ms Target Latency:** In-memory execution using ONNX Runtime or MLX on local CPU and Apple Silicon.
* **Zero Output Token Tax:** Emits structured status codes and calibrated probability vectors (`Pass`, `Fail`, `Risk Score`) without text generation.
* **Lean Tool Surface:** Exposes a single endpoint (`verify_patch`), avoiding multi-tool schema overhead in agent context.
* **Offline Execution:** Runs without external API calls or network egress.

---

## Architectural Targets vs. Frontier LLM Review

| Metric | Autoregressive LLM Code Review | Code Oracle (Neuro-Symbolic) |
| :--- | :--- | :--- |
| **Response Latency** | 2,500 ms to 6,500 ms | **< 50 ms (Local In-Memory)** |
| **Output Token Cost** | 150 to 500 tokens / check | **0 tokens** |
| **Monetary Cost** | $0.003 to $0.02 / call | **$0.00 (Local / Offline)** |
| **Verification Method** | Probabilistic text generation | **Deterministic AST + Calibrated Score** |
| **Context Consumption** | Multi-KB schema injection | **Single-tool lean schema (< 100 tokens)** |

---

## Operational Scope and Boundaries

Code Oracle operates within explicit technical boundaries:

1. **Evaluator, Not Author:** Code Oracle does not generate, autocomplete, or refactor code. It evaluates proposed patches against existing syntax and topology.
2. **Syntax Requirement:** Patches must produce a valid Tree-sitter AST. Syntactically invalid inputs fail at Stage 1 before invoking the decision model.
3. **Static Topology Bounds:** Focuses on structural invariants, dependency cycles, and interface compatibility. It does not replace dynamic test suites, integration environments, or runtime race condition detectors.
4. **Memory Footprint:** Requires approximately 1.2 GB of RAM to hold the 421M parameter model in memory for single-pass inference.

---

## Roadmap

- [x] Architecture Specification & Subgraph Slicing Design
- [x] TopoSlice AST Slicer & Incremental Workspace Indexer
- [x] Tarjan's SCC Cycle Detector & Deterministic Symbolic Gate
- [x] Persistent In-Memory Laya Decision Head & Fine-Tuned Weights (`weights/`)
- [x] Lean FastMCP Server interface (`verify_patch`)
- [x] Agentic `SKILL.md` distribution for Claude Code, Cursor, and Antigravity
- [x] Git pre-commit & pre-push verification hook with unblock toggle (`code-oracle hook`)
- [x] Multi-language AST extractors for Tier 1 languages (Python, TypeScript, Go, Rust)
- [x] Synthetic mutation and training dataset mining engine (`tools/mine_top_repos.py`)
- [x] Google Colab Multi-Language Fine-Tuning Pipeline (`notebooks/Laya_Code_Oracle_Finetune.ipynb`)
- [x] CPU Thread Auto-Tuning & Hybrid Neuro-Symbolic Latency Optimization

---

## License & Attribution

Distributed under the **Apache-2.0 License**. See `LICENSE` for details.

**Architect & Maintainer:**  
Wahyu Febri Tamtomo ([@wahyuzero](https://github.com/wahyuzero))  
Founder of [frugaldev.biz.id](https://frugaldev.biz.id) (Radical AI Efficiency & Frugal Computing).
