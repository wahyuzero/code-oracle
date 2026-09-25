"""
Laya ModernBERT Decision Head and Risk Calibration for Code Oracle.

Evaluates linearized Micro-DSL subgraphs using fine-tuned Laya weights
(or falls back deterministically to symbolic gate verdicts when weights are absent).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Verification Questions Definition for Laya Typed-Decisions Head
VERIFICATION_QUESTIONS = {
    "status": {
        "type": "choice",
        "instructions": "Determine if this code patch proposal should be APPROVED or REJECTED based on AST topology, cycles, and contract invariants:",
        "criteria": {
            "APPROVED": "Clean invariant, acyclic call/import topology, parameter contracts valid",
            "REJECTED": "Contains circular dependencies, arity mismatches, unexpected keywords, or deleted symbol references",
        },
    },
    "risk": {
        "type": "score",
        "instructions": "Calibrate the semantic risk of this patch proposal from 0 (completely safe) to 4 (critical breaking change):",
        "criteria": [
            "Level 0: Safe / Invariants Preserved",
            "Level 1: Low Risk / Harmless Additions",
            "Level 2: Moderate Risk / Signature Drift",
            "Level 3: High Risk / Broken Callers",
            "Level 4: Critical / Topological Cycle",
        ],
    },
}


class LayaDecisionHead:
    """
    Lean neural decision head interfacing with Laya ModernBERT 421M.
    Provides sub-50ms local verification with calibrated risk scores.
    """

    def __init__(self, weights_path: Optional[Path] = None, enabled: bool = False):
        self.enabled = enabled
        self.weights_path = self._resolve_weights_path(weights_path) if enabled else None
        self.agent = None
        self._loaded = False
        if self.enabled and self.weights_path and self.weights_path.exists():
            self._try_load_model()

    def _resolve_weights_path(self, explicit_path: Optional[Path]) -> Optional[Path]:
        if explicit_path:
            return Path(explicit_path).resolve()

        # Candidate paths
        candidates = [
            Path.cwd() / ".code_oracle" / "weights",
            Path(__file__).resolve().parent / "weights",
            Path.cwd() / "weights",
            Path("/content/code_oracle_laya_model"),
        ]
        for c in candidates:
            if c.exists() and (c / "model.safetensors").exists():
                return c.resolve()
        return None

    @staticmethod
    def _tune_cpu_threads() -> int:
        """
        Auto-tune PyTorch thread settings for CPU inference.
        Restricts threads to physical cores to avoid hyperthread cache contention
        and memory bus saturation on DDR RAM.
        """
        try:
            import os
            import torch

            # If GPU is available, do not constrain CPU threads
            if torch.cuda.is_available():
                return torch.get_num_threads()

            physical_cores = None
            # Try reading /proc/cpuinfo on Linux for exact physical core count
            if os.path.exists("/proc/cpuinfo"):
                try:
                    with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                        cores = set()
                        phys_id = "0"
                        for line in f:
                            if line.startswith("physical id"):
                                phys_id = line.split(":")[1].strip()
                            elif line.startswith("core id"):
                                core_id = line.split(":")[1].strip()
                                cores.add(f"{phys_id}:{core_id}")
                        if cores:
                            physical_cores = len(cores)
                except Exception:
                    pass

            if not physical_cores:
                total = os.cpu_count() or 1
                # Standard hyperthreading heuristic
                physical_cores = max(1, total // 2 if total > 2 else total)

            tuned_threads = max(1, min(physical_cores, 8))
            torch.set_num_threads(tuned_threads)
            return tuned_threads
        except Exception:
            return 1

    def _try_load_model(self) -> None:
        try:
            import contextlib
            import io
            import warnings
            import laya

            # Auto-tune CPU threads before model initialization
            self._tune_cpu_threads()

            logger.info(f"Loading fine-tuned Laya weights from {self.weights_path}")
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                self.agent = laya.load(str(self.weights_path))
            self._loaded = True
        except Exception as e:
            logger.warning(f"Could not load Laya model from {self.weights_path}: {e}")
            self.agent = None
            self._loaded = False

    def enable_neural_head(self) -> bool:
        """Dynamically enable and load neural head if weights exist and not loaded."""
        if not self._loaded:
            self.enabled = True
            if not self.weights_path:
                self.weights_path = self._resolve_weights_path(None)
            if self.weights_path and self.weights_path.exists():
                self._try_load_model()
        return self.is_neural_enabled

    @property
    def is_neural_enabled(self) -> bool:
        """Returns True if fine-tuned neural weights are loaded and active."""
        return self._loaded and self.agent is not None

    def predict(
        self,
        linearized_dsl: str,
        symbolic_status: str,
        symbolic_confidence: float,
        has_violations: bool,
    ) -> Tuple[str, float, float]:
        """
        Evaluate linearized DSL subgraph.
        Returns: (verdict_status, confidence, risk_score)
        """
        # Hard rule: If deterministic symbolic gate caught a definite violation (cycle or arity),
        # symbolic gate has absolute veto power (REJECTED).
        if has_violations or symbolic_status == "REJECTED":
            return "REJECTED", 1.0, 0.95

        # If neural weights are available, run inference
        if self.is_neural_enabled:
            try:
                import contextlib
                import io
                import warnings
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    res = self.agent.predict(linearized_dsl, VERIFICATION_QUESTIONS)
                status_ans = res["answers"]["status"]
                risk_ans = res["answers"]["risk"]

                pred_status = status_ans["choice"]
                pred_confidence = max(symbolic_confidence, float(status_ans["confidence"]))
                pred_risk = float(risk_ans["score"]) / 4.0  # Normalize 0..4 to 0.0..1.0

                return pred_status, pred_confidence, pred_risk
            except Exception as e:
                logger.warning(f"Laya neural inference error: {e}. Falling back to symbolic gate.")

        # Fallback to deterministic symbolic gate result
        risk = 0.05 if symbolic_status == "APPROVED" else 0.95
        return symbolic_status, symbolic_confidence, risk
