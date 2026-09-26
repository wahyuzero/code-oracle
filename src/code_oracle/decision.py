"""
Laya ModernBERT Decision Head and Risk Calibration for Code Oracle.

Evaluates linearized Micro-DSL subgraphs using fine-tuned Laya weights via ONNX Runtime
(or PyTorch / legacy fallback) or falls back deterministically to symbolic gate verdicts.
Supports pure ONNX inference with zero hard dependency on PyTorch at runtime.
"""

import contextlib
import io
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

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

_ModuleBase = nn.Module if (HAS_TORCH and nn is not None) else object


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
    engine_mode: str = "heuristic"

    def __iter__(self):
        """Enable tuple unpacking: status, conf, risk = result"""
        return iter((self.status, self.confidence, self.risk_score))


class ModernBERTMultiTaskModel(_ModuleBase):
    """
    Hard Parameter Sharing Multi-Task Network over ModernBERT representations.
    Branches from pooled token representation h_pool in R^hidden_size (default 768)
    or optionally couples end-to-end with a ModernBERT encoder:
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

    def __init__(
        self,
        hidden_size_or_encoder: Any = 768,
        num_taxonomy_classes: int = 5,
        **kwargs,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to instantiate ModernBERTMultiTaskModel")
        super().__init__()

        encoder_name = kwargs.get("encoder_name")
        if isinstance(hidden_size_or_encoder, str):
            encoder_name = hidden_size_or_encoder
            hidden_size = kwargs.get("hidden_size", 768)
        else:
            hidden_size = int(hidden_size_or_encoder)

        self.encoder_name = encoder_name
        self.encoder = None
        if encoder_name:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(encoder_name)
            hidden_size = self.encoder.config.hidden_size

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

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Forward pass. Accepts either:
        - `h_pool: torch.Tensor` (B, hidden_size) directly.
        - `(input_ids, attention_mask)` tensors when initialized with an encoder.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model forward pass")

        if "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        elif len(args) >= 2 and hasattr(args[0], "dtype") and args[0].dtype in (torch.int32, torch.int64):
            input_ids = args[0]
            attention_mask = args[1]
        elif len(args) == 1 and hasattr(args[0], "dtype") and args[0].dtype in (torch.int32, torch.int64):
            input_ids = args[0]
            attention_mask = torch.ones_like(input_ids)
        elif len(args) == 1:
            h_pool = args[0]
            input_ids = None
        elif "h_pool" in kwargs:
            h_pool = kwargs["h_pool"]
            input_ids = None
        else:
            raise ValueError("Expected either (input_ids, attention_mask) or (h_pool)")

        if input_ids is not None:
            if self.encoder is None:
                raise RuntimeError("Encoder not loaded. Initialize with encoder_name to accept input_ids.")

            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            h_pool = sum_embeddings / sum_mask

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
            "risk_logits": risk_raw,
            "taxonomy_logits": taxonomy_logits,
            "taxonomy_probs": taxonomy_probs,
            "log_variance": log_variance,
            "variance": variance,
            "confidence": confidence,
        }

    def compute_loss(
        self,
        risk_pred: Any,
        risk_target: Any,
        taxonomy_logits: Any,
        taxonomy_target: Any,
        log_variance: Any,
        delta: float = 0.1,
        pos_weight: Optional[Any] = None,
        risk_pos_weight: float = 1.0,
        homoscedastic_weights: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute unified multi-task loss:
        - L_risk: Huber loss (delta=0.1) with optional asymmetric risk weighting (risk_pos_weight)
        - L_tax: BCEWithLogitsLoss (with optional pos_weight)
        - L_unc: Heteroscedastic negative log-likelihood:
                 0.5 * exp(-s) * (risk_target - risk_pred)^2 + 0.5 * s
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for compute_loss")

        risk_pred = risk_pred.view(-1, 1)
        risk_target = risk_target.view(-1, 1).float()
        log_variance = log_variance.view(-1, 1)
        taxonomy_target = taxonomy_target.float()

        # 1. Continuous Risk Huber Loss (with optional asymmetric risk weighting)
        if risk_pos_weight > 1.0:
            risk_weight = torch.where(risk_target >= 0.5, risk_pos_weight, 1.0)
            l_risk = torch.mean(risk_weight * F.huber_loss(risk_pred, risk_target, delta=delta, reduction="none"))
        else:
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


class ModernBERTWithMultiTaskHead(ModernBERTMultiTaskModel):
    """
    End-to-end ModernBERT encoder coupled with 3 multi-task evaluation heads:
    1. Continuous calibrated risk regression (0.0 to 1.0)
    2. Multi-label risk taxonomy (5 classes)
    3. Epistemic uncertainty estimation
    Matches the weights stored in fine-tuned model.safetensors.
    """

    def __init__(self, encoder_name: str = "answerdotai/ModernBERT-base"):
        super().__init__(hidden_size_or_encoder=encoder_name)


class LayaDecisionHead:
    """
    Lean neural decision head interfacing with Laya ModernBERT (421M large or 164M base).
    Provides sub-50ms local verification with calibrated risk scores.
    Prioritizes ONNX Runtime (dynamic INT8 or FP32) with graceful fallback to PyTorch
    or deterministic heuristic gate verdicts.
    """

    DEFAULT_HF_REPO: str = "wxsys/code-oracle-laya-421m"

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        enabled: bool = False,
        quantize_int8: Optional[bool] = None,
        risk_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        prefer_onnx: bool = True,
    ):
        self.enabled = enabled
        self.prefer_onnx = prefer_onnx
        self._risk_threshold_explicit = risk_threshold is not None
        if risk_threshold is not None:
            self.risk_threshold = float(risk_threshold)
        elif "CODE_ORACLE_RISK_THRESHOLD" in os.environ:
            try:
                self.risk_threshold = float(os.environ["CODE_ORACLE_RISK_THRESHOLD"])
            except ValueError:
                self.risk_threshold = 0.50
        else:
            self.risk_threshold = 0.50

        self._temperature_explicit = temperature is not None
        if temperature is not None:
            self.temperature = float(temperature)
        elif "CODE_ORACLE_TEMPERATURE" in os.environ:
            try:
                self.temperature = float(os.environ["CODE_ORACLE_TEMPERATURE"])
            except ValueError:
                self.temperature = 1.0
        else:
            self.temperature = 1.0

        self.quantize_int8 = (
            quantize_int8
            if quantize_int8 is not None
            else os.environ.get("CODE_ORACLE_INT8", "1").lower() in ("1", "true", "yes")
        )
        self.weights_path = self._resolve_weights_path(weights_path) if enabled else None
        self.agent = None
        self.pytorch_multitask_model: Optional[Any] = None
        self.onnx_session: Optional[Any] = None
        self.onnx_model_path: Optional[Path] = None
        self.tokenizer = None
        self.multi_task_model: Optional[ModernBERTMultiTaskModel] = (
            ModernBERTMultiTaskModel() if HAS_TORCH else None
        )
        self._engine_mode: str = "heuristic"
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
            Path.cwd() / "weights_base",
            Path(__file__).resolve().parent.parent.parent / "weights_base",
            Path.cwd() / "weights",
            Path("/content/code_oracle_laya_base_model"),
            Path("/content/code_oracle_laya_model"),
        ]
        target_files = ["model_int8.onnx", "model.onnx", "model.safetensors"]
        for c in candidates:
            if c.exists() and any((c / f).exists() for f in target_files):
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
                if p.exists() and any((p / f).exists() for f in target_files):
                    return p.resolve()
            except Exception as exc:
                logger.debug("Failed to auto-download weights from Hugging Face: %s", exc)

        return None

    @staticmethod
    def _tune_cpu_threads() -> int:
        """
        Auto-tune thread settings for CPU inference.
        Restricts threads to physical cores to avoid hyperthread cache contention.
        """
        try:
            if HAS_TORCH and torch is not None and torch.cuda.is_available():
                return torch.get_num_threads()

            physical_cores = None
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
                physical_cores = max(1, total // 2 if total > 2 else total)

            tuned_threads = max(1, min(physical_cores, 8))
            if HAS_TORCH and torch is not None:
                torch.set_num_threads(tuned_threads)
            return tuned_threads
        except Exception:
            return 1

    def _load_tokenizer(self) -> None:
        """Load tokenizer using transformers or standalone tokenizers library."""
        # 1. Try AutoTokenizer from transformers
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.weights_path))
            return
        except Exception as e_tf:
            logger.debug(f"AutoTokenizer loading skipped: {e_tf}")

        # 2. Try pure tokenizers library from tokenizer.json
        if self.weights_path is not None:
            tok_json = self.weights_path / "tokenizer.json"
            if tok_json.exists():
                try:
                    from tokenizers import Tokenizer
                    self.tokenizer = Tokenizer.from_file(str(tok_json))
                    return
                except Exception as e_tk:
                    logger.debug(f"tokenizers.Tokenizer loading skipped: {e_tk}")

    def _tokenize(self, text: str, max_length: int = 512) -> Tuple[Any, Any]:
        """Tokenize text into numpy input_ids and attention_mask."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        if hasattr(self.tokenizer, "batch_encode_plus") or callable(self.tokenizer):
            try:
                tokens = self.tokenizer(
                    text,
                    return_tensors="np",
                    truncation=True,
                    max_length=max_length,
                )
                return tokens["input_ids"].astype(np.int64), tokens["attention_mask"].astype(np.int64)
            except Exception:
                pass

        if hasattr(self.tokenizer, "encode"):
            enc = self.tokenizer.encode(text)
            ids = enc.ids[:max_length]
            mask = enc.attention_mask[:max_length]
            return np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64)

        raise RuntimeError("Unsupported tokenizer instance")

    def _try_load_model(self) -> None:
        try:
            tuned_threads = self._tune_cpu_threads()

            # Read config.json if available to extract threshold and calibrated temperature
            config_file = self.weights_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f_cfg:
                        cfg_data = json.load(f_cfg)
                    if not self._risk_threshold_explicit and "CODE_ORACLE_RISK_THRESHOLD" not in os.environ:
                        if "default_decision_threshold" in cfg_data:
                            self.risk_threshold = float(cfg_data["default_decision_threshold"])
                    if not self._temperature_explicit and "CODE_ORACLE_TEMPERATURE" not in os.environ:
                        if "calibrated_temperature" in cfg_data:
                            self.temperature = float(cfg_data["calibrated_temperature"])
                except Exception as e_cfg:
                    logger.debug(f"Could not load config.json: {e_cfg}")

            env_engine = os.environ.get("CODE_ORACLE_ENGINE", "").lower().strip()
            if env_engine == "heuristic":
                self._loaded = False
                self._engine_mode = "heuristic"
                return

            # 1. Prioritize ONNX Runtime inference
            if self.prefer_onnx and HAS_ORT and env_engine != "pytorch":
                if env_engine == "onnx_fp32":
                    candidate_onnx_files = [self.weights_path / "model.onnx"]
                elif env_engine == "onnx_int8":
                    candidate_onnx_files = [self.weights_path / "model_int8.onnx"]
                elif self.quantize_int8:
                    candidate_onnx_files = [
                        self.weights_path / "model_int8.onnx",
                        self.weights_path / "model.onnx",
                    ]
                else:
                    candidate_onnx_files = [
                        self.weights_path / "model.onnx",
                        self.weights_path / "model_int8.onnx",
                    ]

                for onnx_file in candidate_onnx_files:
                    if onnx_file.exists():
                        try:
                            logger.info(f"Loading ONNX decision session from {onnx_file}")
                            sess_opts = ort.SessionOptions()
                            sess_opts.intra_op_num_threads = tuned_threads
                            sess_opts.inter_op_num_threads = 1
                            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            self.onnx_session = ort.InferenceSession(
                                str(onnx_file),
                                sess_options=sess_opts,
                                providers=["CPUExecutionProvider"],
                            )
                            self.onnx_model_path = onnx_file
                            self._engine_mode = "onnx_int8" if "int8" in onnx_file.name else "onnx_fp32"
                            self._load_tokenizer()
                            self._loaded = True
                            logger.info(f"Successfully loaded ONNX decision head ({self._engine_mode})")
                            return
                        except Exception as e_onnx:
                            logger.warning(f"Failed to load ONNX model {onnx_file}: {e_onnx}. Trying fallback...")

            # 2. Check for native Multi-Task PyTorch safetensors model (fallback)
            safetensors_file = self.weights_path / "model.safetensors"
            if HAS_TORCH and safetensors_file.exists() and env_engine not in ("onnx_int8", "onnx_fp32"):
                try:
                    from safetensors.torch import load_file
                    logger.info(f"Checking native Multi-Task ModernBERT weights from {self.weights_path}")
                    sd = load_file(str(safetensors_file))
                    if any("risk_head" in k for k in sd.keys()):
                        mt_model = ModernBERTWithMultiTaskHead()
                        mt_model.load_state_dict(sd)
                        mt_model.eval()
                        self.pytorch_multitask_model = mt_model
                        self._load_tokenizer()
                        self._engine_mode = "pytorch"
                        self._loaded = True
                        logger.info("Successfully loaded native Multi-Task ModernBERT PyTorch model")
                        return
                except Exception as e_native:
                    logger.debug(f"Native Multi-Task loading skipped: {e_native}")

            # 3. Legacy laya.load loader (fallback)
            if env_engine not in ("onnx_int8", "onnx_fp32"):
                try:
                    import laya
                    logger.info(f"Loading fine-tuned Laya weights from {self.weights_path}")
                    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        warnings.simplefilter("ignore")
                        self.agent = laya.load(str(self.weights_path))

                    if self.quantize_int8 and self.agent and hasattr(self.agent, "model") and HAS_TORCH:
                        try:
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

                    self._engine_mode = "pytorch"
                    self._loaded = True
                    return
                except Exception as e_laya:
                    logger.debug(f"Legacy laya loading skipped: {e_laya}")

            self._loaded = False
            self._engine_mode = "heuristic"
        except Exception as e:
            logger.warning(f"Could not load decision model from {self.weights_path}: {e}")
            self.agent = None
            self.pytorch_multitask_model = None
            self.onnx_session = None
            self._loaded = False
            self._engine_mode = "heuristic"

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
        """Returns True if fine-tuned neural weights (ONNX, PyTorch, or legacy) are loaded and active."""
        return self._loaded and (
            self.onnx_session is not None
            or self.pytorch_multitask_model is not None
            or self.agent is not None
        )

    @property
    def engine_mode(self) -> str:
        """Active engine status: 'onnx_int8', 'onnx_fp32', 'pytorch', or 'heuristic'."""
        if not self.is_neural_enabled:
            return "heuristic"
        return self._engine_mode

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
        risk_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> EnhancedDecisionResult:
        """
        Evaluate linearized DSL subgraph with Multi-Task Risk Taxonomy
        and Heteroscedastic Epistemic Uncertainty.
        """
        violations = violations or []
        cycles = cycles or []
        eff_risk_threshold = self.risk_threshold if risk_threshold is None else risk_threshold
        eff_temperature = self.temperature if temperature is None else temperature

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
                engine_mode=self.engine_mode,
            )

        # If neural weights are available, run inference
        if self.is_neural_enabled:
            # 1. Prioritized ONNX Runtime inference
            if self.onnx_session is not None and self.tokenizer is not None and HAS_NUMPY:
                try:
                    input_ids_np, attention_mask_np = self._tokenize(linearized_dsl, max_length=512)
                    ort_inputs = {
                        "input_ids": input_ids_np,
                        "attention_mask": attention_mask_np,
                    }
                    ort_outputs = self.onnx_session.run(None, ort_inputs)
                    out_map = dict(zip([
                        "risk_score",
                        "risk_logits",
                        "taxonomy_logits",
                        "taxonomy_probs",
                        "log_variance",
                        "variance",
                        "confidence",
                    ], ort_outputs))

                    raw_risk = float(out_map["risk_score"][0][0])
                    raw_logits = float(out_map["risk_logits"][0][0]) if "risk_logits" in out_map else None

                    if eff_temperature is not None and eff_temperature > 0 and abs(eff_temperature - 1.0) > 1e-4:
                        if raw_logits is not None:
                            scaled_risk = 1.0 / (1.0 + math.exp(-raw_logits / eff_temperature))
                        else:
                            eps = 1e-6
                            clamped_r = min(max(raw_risk, eps), 1.0 - eps)
                            z = math.log(clamped_r / (1.0 - clamped_r))
                            scaled_risk = 1.0 / (1.0 + math.exp(-z / eff_temperature))
                        pred_risk = round(scaled_risk, 4)
                    else:
                        pred_risk = round(raw_risk, 4)

                    tax_probs = [float(x) for x in out_map["taxonomy_probs"][0]]
                    pred_conf = round(float(out_map["confidence"][0][0]), 4)
                    log_var = float(out_map["log_variance"][0][0])
                    epistemic_uncertainty = round(float(math.sqrt(math.exp(log_var))), 4)

                    pred_status = "APPROVED" if pred_risk < eff_risk_threshold else "REJECTED"

                    tax_scores = RiskTaxonomyScores(
                        breaking_public_api=round(tax_probs[0], 4),
                        security_surface=round(tax_probs[1], 4),
                        concurrency_hazard=round(tax_probs[2], 4),
                        performance_regression=round(tax_probs[3], 4),
                        silent_logic_drift=round(tax_probs[4], 4),
                    )
                    return EnhancedDecisionResult(
                        status=pred_status,
                        confidence=pred_conf,
                        risk_score=pred_risk,
                        epistemic_uncertainty=epistemic_uncertainty,
                        risk_taxonomy=tax_scores,
                        active_risk_categories=tax_scores.active_categories(threshold=taxonomy_threshold),
                        is_neural_calibrated=True,
                        engine_mode=self.engine_mode,
                    )
                except Exception as e_onnx_inf:
                    logger.warning(f"ONNX inference error: {e_onnx_inf}. Falling back to PyTorch / heuristics.")

            # 2. Native Multi-Task PyTorch Model
            if self.pytorch_multitask_model is not None and self.tokenizer is not None and HAS_TORCH:
                try:
                    tokens = self.tokenizer(
                        linearized_dsl,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    with torch.no_grad():
                        out = self.pytorch_multitask_model(tokens["input_ids"], tokens["attention_mask"])

                    raw_risk = out["risk_score"].item()
                    raw_logits = out.get("risk_logits")

                    if eff_temperature is not None and eff_temperature > 0 and abs(eff_temperature - 1.0) > 1e-4:
                        if raw_logits is not None:
                            scaled_risk = torch.sigmoid(raw_logits / eff_temperature).item()
                        else:
                            eps = 1e-6
                            clamped_r = min(max(raw_risk, eps), 1.0 - eps)
                            z = math.log(clamped_r / (1.0 - clamped_r))
                            scaled_risk = 1.0 / (1.0 + math.exp(-z / eff_temperature))
                        pred_risk = round(scaled_risk, 4)
                    else:
                        pred_risk = round(raw_risk, 4)

                    tax_probs = out["taxonomy_probs"].squeeze(0).tolist()
                    pred_conf = round(out["confidence"].item(), 4)
                    s = out["log_variance"]
                    epistemic_uncertainty = round(float(torch.exp(s).sqrt().item()), 4)

                    pred_status = "APPROVED" if pred_risk < eff_risk_threshold else "REJECTED"

                    tax_scores = RiskTaxonomyScores(
                        breaking_public_api=round(float(tax_probs[0]), 4),
                        security_surface=round(float(tax_probs[1]), 4),
                        concurrency_hazard=round(float(tax_probs[2]), 4),
                        performance_regression=round(float(tax_probs[3]), 4),
                        silent_logic_drift=round(float(tax_probs[4]), 4),
                    )
                    return EnhancedDecisionResult(
                        status=pred_status,
                        confidence=pred_conf,
                        risk_score=pred_risk,
                        epistemic_uncertainty=epistemic_uncertainty,
                        risk_taxonomy=tax_scores,
                        active_risk_categories=tax_scores.active_categories(threshold=taxonomy_threshold),
                        is_neural_calibrated=True,
                        engine_mode="pytorch",
                    )
                except Exception as e_nt:
                    logger.warning(f"Native Multi-Task inference error: {e_nt}. Falling back.")

            # 3. Legacy laya agent
            elif self.agent is not None:
                try:
                    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                        warnings.simplefilter("ignore")
                        res = self.agent.predict(linearized_dsl, VERIFICATION_QUESTIONS)
                    status_ans = res["answers"]["status"]
                    risk_ans = res["answers"]["risk"]

                    pred_status = status_ans["choice"]
                    pred_confidence = max(symbolic_confidence, float(status_ans["confidence"]))
                    pred_risk = float(risk_ans["score"]) / 4.0  # Normalize 0..4 to 0.0..1.0
                    if pred_risk >= eff_risk_threshold:
                        pred_status = "REJECTED"

                    epistemic_uncertainty = max(0.0001, round((1.0 - pred_confidence) ** 2, 4))

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
                        engine_mode="pytorch",
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

        fallback_status = symbolic_status
        if fallback_status == "APPROVED" and risk >= eff_risk_threshold:
            fallback_status = "REJECTED"

        return EnhancedDecisionResult(
            status=fallback_status,
            confidence=symbolic_confidence,
            risk_score=risk,
            epistemic_uncertainty=0.02 if symbolic_status == "APPROVED" else 0.01,
            risk_taxonomy=tax_scores,
            active_risk_categories=tax_scores.active_categories(threshold=taxonomy_threshold),
            is_neural_calibrated=False,
            engine_mode="heuristic",
        )
