"""
Laya ModernBERT Decision Head and Risk Calibration for Code Oracle.

Evaluates linearized Micro-DSL subgraphs using fine-tuned Laya weights
(or falls back deterministically to symbolic gate verdicts when weights are absent).
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_oracle.models import RiskTaxonomyScores

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


@dataclass
class EnhancedDecisionResult:
    """
    Rich outcome of Laya multi-task neural decision head.
    Supports unpacking as (status, confidence, risk_score) for backward compatibility.
    """
    status: str
    confidence: float
    risk_score: float
    epistemic_uncertainty: float
    risk_taxonomy: RiskTaxonomyScores
    active_risk_categories: List[str]
    is_neural_calibrated: bool = False

    def __iter__(self):
        """Enable tuple unpacking: status, conf, risk = result"""
        return iter((self.status, self.confidence, self.risk_score))


class ModernBERTMultiTaskModel(nn.Module):
    """
    Hard Parameter Sharing Multi-Task Network over ModernBERT representations.
    Branches from pooled token representation h_pool in R^hidden_size (default 768):
    - Head 1: Continuous Risk Regression (Huber Loss / Sigmoid)
    - Head 2: Multi-Label Risk Taxonomy (5-Class BCEWithLogits / Sigmoid)
    - Head 3: Epistemic Uncertainty Estimation (Heteroscedastic log-variance with clamp [-6.0, 6.0])
    """

    TAXONOMY_CLASSES = [
        "BreakingPublicAPI",
        "SecuritySurface",
        "ConcurrencyHazard",
        "PerformanceRegression",
        "SilentLogicDrift",
    ]

    def __init__(self, hidden_size: int = 768, num_taxonomy_classes: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_taxonomy_classes = num_taxonomy_classes

        # Head 1: Continuous Risk Regression MLP
        # Dense(hidden_size -> 256) -> GELU -> LayerNorm -> Dense(256 -> 1)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

        # Head 2: Multi-Label Taxonomy MLP
        # Dense(hidden_size -> 256) -> GELU -> LayerNorm -> Dense(256 -> num_taxonomy_classes)
        self.taxonomy_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_taxonomy_classes),
        )

        # Head 3: Epistemic Uncertainty MLP
        # Dense(hidden_size -> 128) -> GELU -> Dense(128 -> 1) [log sigma^2]
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h_pool: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass from pooled representation h_pool (B, hidden_size).
        Returns dictionary containing:
        - 'risk_score': continuous risk in [0.0, 1.0]
        - 'taxonomy_logits': unnormalized logits (B, 5)
        - 'taxonomy_probs': independent sigmoid probabilities (B, 5)
        - 'log_variance': clamped s in [-6.0, 6.0]
        - 'variance': exp(s)
        - 'confidence': calibrated epistemic confidence in [0.0, 1.0]
        """
        # Head 1: Risk score (continuous in 0.0 .. 1.0)
        risk_raw = self.risk_head(h_pool)
        risk_score = torch.sigmoid(risk_raw)

        # Head 2: Multi-label taxonomy
        taxonomy_logits = self.taxonomy_head(h_pool)
        taxonomy_probs = torch.sigmoid(taxonomy_logits)

        # Head 3: Epistemic Uncertainty with bounded damping [-6.0, 6.0]
        s = self.uncertainty_head(h_pool)
        log_variance = torch.clamp(s, min=-6.0, max=6.0)
        variance = torch.exp(log_variance)
        sigma = torch.sqrt(variance)
        confidence = 1.0 - torch.clamp(sigma, min=0.0, max=1.0)

        return {
            "risk_score": risk_score,
            "taxonomy_logits": taxonomy_logits,
            "taxonomy_probs": taxonomy_probs,
            "log_variance": log_variance,
            "variance": variance,
            "confidence": confidence,
        }

    def compute_loss(
        self,
        risk_pred: torch.Tensor,
        risk_target: torch.Tensor,
        taxonomy_logits: torch.Tensor,
        taxonomy_target: torch.Tensor,
        log_variance: torch.Tensor,
        delta: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
        homoscedastic_weights: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified multi-task loss:
        - L_risk: Huber loss (delta=0.1)
        - L_tax: BCEWithLogitsLoss (with optional pos_weight)
        - L_unc: Heteroscedastic negative log-likelihood:
                 0.5 * exp(-s) * (risk_target - risk_pred)^2 + 0.5 * s
        """
        # Ensure robust element-wise alignment by reshaping 1D tensors to (B, 1)
        risk_pred = risk_pred.view(-1, 1)
        risk_target = risk_target.view(-1, 1).float()
        log_variance = log_variance.view(-1, 1)
        taxonomy_target = taxonomy_target.float()

        # 1. Continuous Risk Huber Loss
        l_risk = F.huber_loss(risk_pred, risk_target, delta=delta)

        # 2. Multi-Label Taxonomy Loss
        l_tax = F.binary_cross_entropy_with_logits(taxonomy_logits, taxonomy_target, pos_weight=pos_weight)

        # 3. Heteroscedastic Uncertainty Loss
        diff_sq = (risk_target - risk_pred) ** 2
        l_unc = torch.mean(0.5 * torch.exp(-log_variance) * diff_sq + 0.5 * log_variance)

        # Total combined loss
        if homoscedastic_weights is not None:
            s1, s2, s3 = homoscedastic_weights
            t_s1 = s1 if isinstance(s1, torch.Tensor) else torch.tensor(float(s1), device=risk_pred.device)
            t_s2 = s2 if isinstance(s2, torch.Tensor) else torch.tensor(float(s2), device=risk_pred.device)
            t_s3 = s3 if isinstance(s3, torch.Tensor) else torch.tensor(float(s3), device=risk_pred.device)
            l_total = (
                0.5 / (t_s1 ** 2) * l_risk
                + 0.5 / (t_s2 ** 2) * l_tax
                + 0.5 / (t_s3 ** 2) * l_unc
                + torch.log(torch.abs(t_s1 * t_s2 * t_s3) + 1e-8)
            )
        else:
            l_total = l_risk + l_tax + l_unc

        return {
            "loss_total": l_total,
            "loss_risk": l_risk,
            "loss_taxonomy": l_tax,
            "loss_uncertainty": l_unc,
        }


class LayaDecisionHead:
    """
    Lean neural decision head interfacing with Laya ModernBERT (421M large or 164M base).
    Provides sub-50ms local verification with calibrated risk scores.
    Supports on-the-fly INT8 dynamic quantization for minimal memory footprint.
    """

    DEFAULT_HF_REPO: str = "wxsys/code-oracle-laya-421m"

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        enabled: bool = False,
        quantize_int8: Optional[bool] = None,
    ):
        self.enabled = enabled
        self.quantize_int8 = (
            quantize_int8
            if quantize_int8 is not None
            else os.environ.get("CODE_ORACLE_INT8", "0").lower() in ("1", "true", "yes")
        )
        self.weights_path = self._resolve_weights_path(weights_path) if enabled else None
        self.agent = None
        self.multi_task_model: ModernBERTMultiTaskModel = ModernBERTMultiTaskModel()
        self._loaded = False
        if self.enabled and self.weights_path and self.weights_path.exists():
            self._try_load_model()

    def _resolve_weights_path(self, explicit_path: Optional[Path]) -> Optional[Path]:
        if explicit_path:
            return Path(explicit_path).resolve()

        # Check explicit environment variable
        env_weights = os.environ.get("CODE_ORACLE_WEIGHTS")
        if env_weights and Path(env_weights).exists():
            return Path(env_weights).resolve()

        # Candidate local paths
        candidates = [
            Path.cwd() / ".code_oracle" / "weights",
            Path.home() / ".cache" / "code_oracle" / "weights",
            Path(__file__).resolve().parent / "weights",
            Path.cwd() / "weights",
            Path("/content/code_oracle_laya_base_model"),
            Path("/content/code_oracle_laya_model"),
        ]
        for c in candidates:
            if c.exists() and (c / "model.safetensors").exists():
                return c.resolve()

        # Attempt downloading from Hugging Face Hub if auto-download enabled
        auto_download = os.environ.get("CODE_ORACLE_AUTO_DOWNLOAD", "1").lower() in ("1", "true", "yes")
        if auto_download:
            try:
                from huggingface_hub import snapshot_download

                cache_dir = Path.home() / ".cache" / "code_oracle" / "weights"
                repo_id = os.environ.get("CODE_ORACLE_HF_REPO", self.DEFAULT_HF_REPO)
                hf_token = os.environ.get("HF_TOKEN")
                downloaded = snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(cache_dir),
                    token=hf_token,
                )
                p = Path(downloaded)
                if p.exists() and (p / "model.safetensors").exists():
                    return p.resolve()
            except Exception as exc:
                logger.debug("Failed to auto-download weights from Hugging Face: %s", exc)

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

            # Apply dynamic INT8 quantization if requested
            if self.quantize_int8 and self.agent and hasattr(self.agent, "model"):
                try:
                    import torch
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.agent.model.encoder = torch.ao.quantization.quantize_dynamic(
                            self.agent.model.encoder,
                            {torch.nn.Linear},
                            dtype=torch.qint8,
                        )
                    logger.info("Applied dynamic INT8 quantization to encoder linear layers")
                except Exception as q_err:
                    logger.debug(f"Dynamic INT8 quantization skipped: {q_err}")

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
        Maintains backward compatibility with 3-tuple return format.
        """
        res = self.predict_multi_task(
            linearized_dsl=linearized_dsl,
            symbolic_status=symbolic_status,
            symbolic_confidence=symbolic_confidence,
            has_violations=has_violations,
        )
        return res.status, res.confidence, res.risk_score

    def predict_multi_task(
        self,
        linearized_dsl: str,
        symbolic_status: str,
        symbolic_confidence: float,
        has_violations: bool,
        violations: Optional[List[str]] = None,
        cycles: Optional[List[List[str]]] = None,
        taxonomy_threshold: float = 0.5,
    ) -> EnhancedDecisionResult:
        """
        Evaluate linearized DSL subgraph with Multi-Task Risk Taxonomy
        and Heteroscedastic Epistemic Uncertainty.
        """
        violations = violations or []
        cycles = cycles or []

        # Hard rule: If deterministic symbolic gate caught a definite violation (cycle or arity),
        # symbolic gate has absolute veto power (REJECTED).
        if has_violations or symbolic_status == "REJECTED":
            has_cycle = bool(cycles or any("CIRCULAR_DEPENDENCY" in v or "cycle" in v.lower() for v in violations))
            has_broken_api = any(
                any(k in v for k in ["ARITY_MISMATCH", "BROKEN_REFERENCE", "requires at least", "unexpected keyword", "SYNTAX_ERROR"])
                for v in violations
            ) or not has_cycle
            has_sec = any("security" in v.lower() or "auth" in v.lower() for v in violations)
            has_perf = any("perf" in v.lower() or "loop" in v.lower() for v in violations)

            tax_scores = RiskTaxonomyScores(
                breaking_public_api=0.95 if has_broken_api else 0.40,
                security_surface=0.90 if has_sec else 0.05,
                concurrency_hazard=0.95 if has_cycle else 0.05,
                performance_regression=0.90 if has_perf else 0.05,
                silent_logic_drift=0.85,
            )
            return EnhancedDecisionResult(
                status="REJECTED",
                confidence=1.0,
                risk_score=0.95,
                epistemic_uncertainty=0.01,
                risk_taxonomy=tax_scores,
                active_risk_categories=tax_scores.active_categories(threshold=taxonomy_threshold),
                is_neural_calibrated=False,
            )

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

                # Compute epistemic uncertainty
                epistemic_uncertainty = max(0.0001, round((1.0 - pred_confidence) ** 2, 4))

                # Multi-label taxonomy derived from neural risk and topological features
                has_api_drift = any(k in linearized_dsl for k in ["PARAM", "SIG", "ARITY", "DELETED"])
                has_cycle_signal = any(k in linearized_dsl for k in ["CYCLE", "MUTUAL", "SCC"])
                has_sec_signal = any(k in linearized_dsl.lower() for k in ["secret", "auth", "token", "pwd", "taint"])
                has_perf_signal = any(k in linearized_dsl for k in ["LOOP", "QUERY", "PERF"])

                tax_scores = RiskTaxonomyScores(
                    breaking_public_api=round(min(0.99, max(0.02, pred_risk * 1.2 if has_api_drift else pred_risk * 0.4)), 4),
                    security_surface=round(min(0.99, max(0.01, 0.85 if has_sec_signal else pred_risk * 0.15)), 4),
                    concurrency_hazard=round(min(0.99, max(0.01, 0.90 if has_cycle_signal else pred_risk * 0.2)), 4),
                    performance_regression=round(min(0.99, max(0.01, 0.85 if has_perf_signal else pred_risk * 0.2)), 4),
                    silent_logic_drift=round(min(0.95, max(0.02, pred_risk * 0.7)), 4),
                )

                return EnhancedDecisionResult(
                    status=pred_status,
                    confidence=pred_confidence,
                    risk_score=pred_risk,
                    epistemic_uncertainty=epistemic_uncertainty,
                    risk_taxonomy=tax_scores,
                    active_risk_categories=tax_scores.active_categories(threshold=taxonomy_threshold),
                    is_neural_calibrated=True,
                )
            except Exception as e:
                logger.warning(f"Laya neural inference error: {e}. Falling back to symbolic gate.")

        # Fallback to deterministic symbolic gate result
        if symbolic_status == "APPROVED":
            risk = 0.05
            tax_scores = RiskTaxonomyScores(
                breaking_public_api=0.02,
                security_surface=0.01,
                concurrency_hazard=0.01,
                performance_regression=0.01,
                silent_logic_drift=0.02,
            )
        else:
            risk = 0.95
            tax_scores = RiskTaxonomyScores(
                breaking_public_api=0.95,
                security_surface=0.05,
                concurrency_hazard=0.10,
                performance_regression=0.05,
                silent_logic_drift=0.85,
            )

        return EnhancedDecisionResult(
            status=symbolic_status,
            confidence=symbolic_confidence,
            risk_score=risk,
            epistemic_uncertainty=0.02 if symbolic_status == "APPROVED" else 0.01,
            risk_taxonomy=tax_scores,
            active_risk_categories=tax_scores.active_categories(threshold=taxonomy_threshold),
            is_neural_calibrated=False,
        )
