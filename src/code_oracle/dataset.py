"""
Multi-Language Dataset Mining & Synthetic Mutation Engine.
Generates balanced positive (PASS) and negative (REJECT) pairs across Tier 1 languages
(Python, TypeScript, Go, Rust) for Laya ModernBERT fine-tuning.
"""

import json
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from code_oracle.engine import TopoSliceEngine
from code_oracle.languages import SUPPORTED_EXTENSIONS, detect_language

TAXONOMY_CLASSES = [
    "BreakingPublicAPI",
    "SecuritySurface",
    "ConcurrencyHazard",
    "PerformanceRegression",
    "SilentLogicDrift",
]


@dataclass
class DatasetRecord:
    """Represents a single training or validation instance for Laya ModernBERT."""
    input_dsl: str
    label: int  # 1 for PASS (positive), 0 for REJECT (negative)
    risk_score: float  # 0.0 - 0.2 for PASS, 0.8 - 1.0 for REJECT
    category: str  # "clean_pass", "silent_logic_drift", "security_surface", etc.
    language: str  # "python", "typescript", "go", "rust"
    taxonomy_labels: Optional[Dict[str, float]] = None
    symbolic_gate_passed: bool = True
    source_type: str = "mutation_subtle"

    def __post_init__(self):
        if self.taxonomy_labels is None:
            self.taxonomy_labels = self._default_taxonomy_for_category(
                self.category, self.label, self.risk_score
            )
        else:
            # Ensure all 5 taxonomy classes exist with float values
            for c in TAXONOMY_CLASSES:
                if c not in self.taxonomy_labels:
                    self.taxonomy_labels[c] = 0.0

    @classmethod
    def assign_taxonomy_labels(
        cls,
        category: str,
        label: int,
        risk_score: float,
        context_text: str = "",
        invariant_violations: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Assign ADR-0003 multi-task risk taxonomy scores:
        BreakingPublicAPI, SecuritySurface, ConcurrencyHazard, PerformanceRegression, SilentLogicDrift.
        """
        base = {c: 0.0 for c in TAXONOMY_CLASSES}
        if label == 1:
            return base

        score = max(0.65, min(1.0, risk_score))
        cat_lower = category.lower()

        # 1. Exact Category-Based Mapping (ADR-0003 specification)
        if (
            "concurrency_leak" in cat_lower
            or "concurrency_hazard" in cat_lower
            or "goroutine" in cat_lower
            or "thread" in cat_lower
            or "deadlock" in cat_lower
            or "race" in cat_lower
            or "mutex" in cat_lower
            or "concurr" in cat_lower
        ):
            base["ConcurrencyHazard"] = score
        elif (
            "resource_leak" in cat_lower
            or "memory_leak" in cat_lower
            or "performance_regression" in cat_lower
            or "resource" in cat_lower
            or "memory" in cat_lower
            or "perf" in cat_lower
            or "quadratic" in cat_lower
            or "loop" in cat_lower
            or "timeout" in cat_lower
            or "slow" in cat_lower
            or "leak" in cat_lower
        ):
            base["PerformanceRegression"] = score
        elif (
            "api" in cat_lower
            or "arity" in cat_lower
            or "keyword" in cat_lower
            or "type_drift" in cat_lower
            or "deleted" in cat_lower
            or "param" in cat_lower
            or "signature" in cat_lower
            or "breaking" in cat_lower
        ):
            base["BreakingPublicAPI"] = score
        elif (
            "security" in cat_lower
            or "surface" in cat_lower
            or "taint" in cat_lower
            or "side_effect" in cat_lower
            or "cve" in cat_lower
            or "vuln" in cat_lower
            or "auth" in cat_lower
            or "sanitize" in cat_lower
        ):
            base["SecuritySurface"] = score
        elif "logic" in cat_lower or "drift" in cat_lower:
            base["SilentLogicDrift"] = score
        elif "circular" in cat_lower:
            base["BreakingPublicAPI"] = round(score * 0.7, 4)
            base["SilentLogicDrift"] = round(score * 0.8, 4)

        # 2. Context-Based Keyword Matching (for real_revert and real_hotfix commits)
        if context_text:
            normalized_ctx = context_text.lower().replace("_", " ").replace("-", " ")
            if re.search(r"\b(race|race condition|data race|deadlock|mutex|rwlock|atomic|goroutine|channel|hazard|concurr\w*|promise|async|await|floating)\b", normalized_ctx):
                base["ConcurrencyHazard"] = max(base["ConcurrencyHazard"], score)
            if re.search(r"\b(perf\w*|performance|memory leak|resource leak|slow\w*|speed|alloc\w*|latency|cpu|timeout|hang)\b", normalized_ctx):
                base["PerformanceRegression"] = max(base["PerformanceRegression"], score)
            if re.search(r"\b(breaking|deprecat\w*|signature|param\w*|argument\w*|interface|proto|abi|export|arity|widening|destructur\w*|kwargs)\b", normalized_ctx):
                base["BreakingPublicAPI"] = max(base["BreakingPublicAPI"], score)
            if re.search(r"\b(security|cve|vuln\w*|vulnerability|vulnerabilities|xss|csrf|injection|auth\w*|token|sanitize|escape|permission|privilege|secret|credential|overflow|ssrf)\b", normalized_ctx):
                base["SecuritySurface"] = max(base["SecuritySurface"], score)
            if re.search(r"\b(logic|off by one|wrong|incorrect|regression|edge case|boundary|nil|null|unhandled|condition|optional chaining|truth\w*)\b", normalized_ctx):
                base["SilentLogicDrift"] = max(base["SilentLogicDrift"], score)

        if invariant_violations:
            for v in invariant_violations:
                v_lower = v.lower()
                if "cycle" in v_lower:
                    base["BreakingPublicAPI"] = max(base["BreakingPublicAPI"], round(score * 0.7, 4))
                    base["SilentLogicDrift"] = max(base["SilentLogicDrift"], round(score * 0.8, 4))
                if "arity" in v_lower or "keyword" in v_lower:
                    base["BreakingPublicAPI"] = max(base["BreakingPublicAPI"], score)
                if "security" in v_lower:
                    base["SecuritySurface"] = max(base["SecuritySurface"], score)
                if "perf" in v_lower or "leak" in v_lower:
                    base["PerformanceRegression"] = max(base["PerformanceRegression"], score)

        # Fallback to SilentLogicDrift if no specific category was activated
        if not any(v > 0.0 for v in base.values()):
            base["SilentLogicDrift"] = score

        return {k: round(v, 4) for k, v in base.items()}


    @staticmethod
    def _default_taxonomy_for_category(category: str, label: int, risk_score: float) -> Dict[str, float]:
        return DatasetRecord.assign_taxonomy_labels(category=category, label=label, risk_score=risk_score)


    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dsl": self.input_dsl,
            "label": self.label,
            "risk_score": round(self.risk_score, 4),
            "category": self.category,
            "language": self.language,
            "taxonomy_labels": {k: round(v, 4) for k, v in self.taxonomy_labels.items()},
            "symbolic_gate_passed": self.symbolic_gate_passed,
            "source_type": self.source_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        return cls(
            input_dsl=data["input_dsl"],
            label=data["label"],
            risk_score=data["risk_score"],
            category=data["category"],
            language=data["language"],
            taxonomy_labels=data.get("taxonomy_labels"),
            symbolic_gate_passed=data.get("symbolic_gate_passed", True),
            source_type=data.get("source_type", "mutation_subtle"),
        )
# Default Top-Tier Target Repositories (Python, TypeScript, Go, Rust)
DEFAULT_TOP_REPOS: List[Dict[str, str]] = [
    # Python
    {"language": "python", "name": "fastapi", "url": "https://github.com/tiangolo/fastapi.git"},
    {"language": "python", "name": "pydantic", "url": "https://github.com/pydantic/pydantic.git"},
    {"language": "python", "name": "requests", "url": "https://github.com/psf/requests.git"},
    {"language": "python", "name": "werkzeug", "url": "https://github.com/pallets/werkzeug.git"},
    {"language": "python", "name": "starlette", "url": "https://github.com/encode/starlette.git"},
    # TypeScript
    {"language": "typescript", "name": "hono", "url": "https://github.com/honojs/hono.git"},
    {"language": "typescript", "name": "zod", "url": "https://github.com/colinhacks/zod.git"},
    {"language": "typescript", "name": "trpc", "url": "https://github.com/trpc/trpc.git"},
    {"language": "typescript", "name": "nest", "url": "https://github.com/nestjs/nest.git"},
    # Go
    {"language": "go", "name": "gin", "url": "https://github.com/gin-gonic/gin.git"},
    {"language": "go", "name": "cobra", "url": "https://github.com/spf13/cobra.git"},
    {"language": "go", "name": "fiber", "url": "https://github.com/gofiber/fiber.git"},
    {"language": "go", "name": "client-go", "url": "https://github.com/kubernetes/client-go.git"},
    # Rust
    {"language": "rust", "name": "tokio", "url": "https://github.com/tokio-rs/tokio.git"},
    {"language": "rust", "name": "axum", "url": "https://github.com/tokio-rs/axum.git"},
    {"language": "rust", "name": "ripgrep", "url": "https://github.com/BurntSushi/ripgrep.git"},
    {"language": "rust", "name": "clap", "url": "https://github.com/clap-rs/clap.git"},
]

# Designated Held-Out Evaluation Repositories for Independent Verification
DEFAULT_HELDOUT_REPOS: List[Dict[str, str]] = [
    # Python
    {"language": "python", "name": "flask", "url": "https://github.com/pallets/flask.git"},
    {"language": "python", "name": "httpx", "url": "https://github.com/encode/httpx.git"},
    # TypeScript
    {"language": "typescript", "name": "fastify", "url": "https://github.com/fastify/fastify.git"},
    # Go
    {"language": "go", "name": "chi", "url": "https://github.com/go-chi/chi.git"},
    # Rust
    {"language": "rust", "name": "serde", "url": "https://github.com/serde-rs/serde.git"},
]

KNOWN_REPOS: Dict[str, Dict[str, str]] = {
    # Python - Target
    "fastapi": {"language": "python", "name": "fastapi", "url": "https://github.com/tiangolo/fastapi.git"},
    "tiangolo/fastapi": {"language": "python", "name": "fastapi", "url": "https://github.com/tiangolo/fastapi.git"},
    "pydantic": {"language": "python", "name": "pydantic", "url": "https://github.com/pydantic/pydantic.git"},
    "pydantic/pydantic": {"language": "python", "name": "pydantic", "url": "https://github.com/pydantic/pydantic.git"},
    "requests": {"language": "python", "name": "requests", "url": "https://github.com/psf/requests.git"},
    "psf/requests": {"language": "python", "name": "requests", "url": "https://github.com/psf/requests.git"},
    "starlette": {"language": "python", "name": "starlette", "url": "https://github.com/encode/starlette.git"},
    "encode/starlette": {"language": "python", "name": "starlette", "url": "https://github.com/encode/starlette.git"},
    "werkzeug": {"language": "python", "name": "werkzeug", "url": "https://github.com/pallets/werkzeug.git"},
    "pallets/werkzeug": {"language": "python", "name": "werkzeug", "url": "https://github.com/pallets/werkzeug.git"},
    # Python - Held-out
    "flask": {"language": "python", "name": "flask", "url": "https://github.com/pallets/flask.git"},
    "pallets/flask": {"language": "python", "name": "flask", "url": "https://github.com/pallets/flask.git"},
    "httpx": {"language": "python", "name": "httpx", "url": "https://github.com/encode/httpx.git"},
    "encode/httpx": {"language": "python", "name": "httpx", "url": "https://github.com/encode/httpx.git"},
    # Python - Additional
    "rich": {"language": "python", "name": "rich", "url": "https://github.com/Textualize/rich.git"},
    "textualize/rich": {"language": "python", "name": "rich", "url": "https://github.com/Textualize/rich.git"},

    # TypeScript - Target
    "hono": {"language": "typescript", "name": "hono", "url": "https://github.com/honojs/hono.git"},
    "honojs/hono": {"language": "typescript", "name": "hono", "url": "https://github.com/honojs/hono.git"},
    "zod": {"language": "typescript", "name": "zod", "url": "https://github.com/colinhacks/zod.git"},
    "colinhacks/zod": {"language": "typescript", "name": "zod", "url": "https://github.com/colinhacks/zod.git"},
    "trpc": {"language": "typescript", "name": "trpc", "url": "https://github.com/trpc/trpc.git"},
    "trpc/trpc": {"language": "typescript", "name": "trpc", "url": "https://github.com/trpc/trpc.git"},
    "nest": {"language": "typescript", "name": "nest", "url": "https://github.com/nestjs/nest.git"},
    "nestjs/nest": {"language": "typescript", "name": "nest", "url": "https://github.com/nestjs/nest.git"},
    # TypeScript - Held-out
    "fastify": {"language": "typescript", "name": "fastify", "url": "https://github.com/fastify/fastify.git"},
    "fastify/fastify": {"language": "typescript", "name": "fastify", "url": "https://github.com/fastify/fastify.git"},
    # TypeScript - Additional
    "express": {"language": "typescript", "name": "express", "url": "https://github.com/expressjs/express.git"},
    "expressjs/express": {"language": "typescript", "name": "express", "url": "https://github.com/expressjs/express.git"},

    # Go - Target
    "gin": {"language": "go", "name": "gin", "url": "https://github.com/gin-gonic/gin.git"},
    "gin-gonic/gin": {"language": "go", "name": "gin", "url": "https://github.com/gin-gonic/gin.git"},
    "cobra": {"language": "go", "name": "cobra", "url": "https://github.com/spf13/cobra.git"},
    "spf13/cobra": {"language": "go", "name": "cobra", "url": "https://github.com/spf13/cobra.git"},
    "fiber": {"language": "go", "name": "fiber", "url": "https://github.com/gofiber/fiber.git"},
    "gofiber/fiber": {"language": "go", "name": "fiber", "url": "https://github.com/gofiber/fiber.git"},
    "client-go": {"language": "go", "name": "client-go", "url": "https://github.com/kubernetes/client-go.git"},
    "kubernetes/client-go": {"language": "go", "name": "client-go", "url": "https://github.com/kubernetes/client-go.git"},
    # Go - Held-out
    "chi": {"language": "go", "name": "chi", "url": "https://github.com/go-chi/chi.git"},
    "go-chi/chi": {"language": "go", "name": "chi", "url": "https://github.com/go-chi/chi.git"},

    # Rust - Target
    "tokio": {"language": "rust", "name": "tokio", "url": "https://github.com/tokio-rs/tokio.git"},
    "tokio-rs/tokio": {"language": "rust", "name": "tokio", "url": "https://github.com/tokio-rs/tokio.git"},
    "axum": {"language": "rust", "name": "axum", "url": "https://github.com/tokio-rs/axum.git"},
    "tokio-rs/axum": {"language": "rust", "name": "axum", "url": "https://github.com/tokio-rs/axum.git"},
    "ripgrep": {"language": "rust", "name": "ripgrep", "url": "https://github.com/BurntSushi/ripgrep.git"},
    "burntsushi/ripgrep": {"language": "rust", "name": "ripgrep", "url": "https://github.com/BurntSushi/ripgrep.git"},
    "BurntSushi/ripgrep": {"language": "rust", "name": "ripgrep", "url": "https://github.com/BurntSushi/ripgrep.git"},
    "clap": {"language": "rust", "name": "clap", "url": "https://github.com/clap-rs/clap.git"},
    "clap-rs/clap": {"language": "rust", "name": "clap", "url": "https://github.com/clap-rs/clap.git"},
    # Rust - Held-out
    "serde": {"language": "rust", "name": "serde", "url": "https://github.com/serde-rs/serde.git"},
    "serde-rs/serde": {"language": "rust", "name": "serde", "url": "https://github.com/serde-rs/serde.git"},
}

FALLBACK_REPOS: Dict[str, str] = {
    "python": "https://github.com/tiangolo/fastapi.git",
    "typescript": "https://github.com/trpc/trpc.git",
    "go": "https://github.com/spf13/cobra.git",
    "rust": "https://github.com/clap-rs/clap.git",
}


# Multi-Language Templates for Synthetic Generation
TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "python": [
        {
            "name": "math_service",
            "files": {
                "math_ops.py": (
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n\n"
                    "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                    "    return x * y * factor\n"
                ),
                "main.py": (
                    "from math_ops import add, multiply\n\n"
                    "def calculate_total(a: int, b: int) -> int:\n"
                    "    sum_val = add(a, b)\n"
                    "    return multiply(sum_val, 2)\n"
                ),
            },
            "target_file": "math_ops.py",
            "target_symbol": "add",
            "caller_file": "main.py",
            "caller_symbol": "calculate_total",
            "pass_patch": (
                "def add(a: int, b: int) -> int:\n"
                "    # Optimized addition\n"
                "    return a + b\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "arity_patch": (
                "def add(a: int, b: int, c: int) -> int:\n"
                "    return a + b + c\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "keyword_patch": (
                "from math_ops import add, multiply\n\n"
                "def calculate_total(a: int, b: int) -> int:\n"
                "    sum_val = add(a, b, unexpected_kw=True)\n"
                "    return multiply(sum_val, 2)\n"
            ),
            "circular_patch": (
                "from main import calculate_total\n\n"
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "deleted_patch": (
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "silent_logic_drift_patch": (
                "def add(a: int, b: int) -> int:\n"
                "    return a + b if a >= 0 else a - b\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "security_surface_patch": (
                "import os\n\n"
                "def add(a: int, b: int) -> int:\n"
                "    os.system(f'echo {a} > /dev/null')\n"
                "    return a + b\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "concurrency_hazard_patch": (
                "_SHARED_STATE = []\n\n"
                "def add(a: int, b: int) -> int:\n"
                "    _SHARED_STATE.append(a + b)\n"
                "    return a + b\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "performance_regression_patch": (
                "def add(a: int, b: int) -> int:\n"
                "    for _ in range(200):\n"
                "        for _ in range(10):\n"
                "            a += 0\n"
                "    return a + b\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
            "breaking_public_api_patch": (
                "def add(a: int, b: int) -> int:\n"
                "    return 0\n\n"
                "def multiply(x: int, y: int, factor: int = 1) -> int:\n"
                "    return x * y * factor\n"
            ),
        },
        {
            "name": "user_service",
            "files": {
                "user_repo.py": (
                    "def get_user_by_id(user_id: str) -> dict:\n"
                    "    return {'id': user_id, 'name': 'Alice'}\n\n"
                    "def delete_user(user_id: str) -> bool:\n"
                    "    return True\n"
                ),
                "user_handler.py": (
                    "from user_repo import get_user_by_id\n\n"
                    "def handle_get(req_id: str) -> dict:\n"
                    "    return get_user_by_id(req_id)\n"
                ),
            },
            "target_file": "user_repo.py",
            "target_symbol": "get_user_by_id",
            "caller_file": "user_handler.py",
            "caller_symbol": "handle_get",
            "pass_patch": (
                "def get_user_by_id(user_id: str, fetch_details: bool = False) -> dict:\n"
                "    return {'id': user_id, 'name': 'Alice'}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "arity_patch": (
                "def get_user_by_id(user_id: str, required_auth_token: str) -> dict:\n"
                "    return {'id': user_id, 'name': 'Alice'}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "keyword_patch": (
                "from user_repo import get_user_by_id\n\n"
                "def handle_get(req_id: str) -> dict:\n"
                "    return get_user_by_id(req_id, wrong_flag=False)\n"
            ),
            "circular_patch": (
                "from user_handler import handle_get\n\n"
                "def get_user_by_id(user_id: str) -> dict:\n"
                "    return {'id': user_id, 'name': 'Alice'}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "deleted_patch": (
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "silent_logic_drift_patch": (
                "def get_user_by_id(user_id: str) -> dict:\n"
                "    if user_id == 'admin':\n"
                "        return {}\n"
                "    return {'id': user_id, 'name': 'Alice'}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "security_surface_patch": (
                "def get_user_by_id(user_id: str) -> dict:\n"
                "    query = f\"SELECT * FROM users WHERE id = '{user_id}'\"\n"
                "    return {'id': user_id, 'name': 'Alice', 'query': query}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "concurrency_hazard_patch": (
                "_SESSIONS = set()\n\n"
                "def get_user_by_id(user_id: str) -> dict:\n"
                "    _SESSIONS.add(user_id)\n"
                "    return {'id': user_id, 'name': 'Alice'}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "performance_regression_patch": (
                "def get_user_by_id(user_id: str) -> dict:\n"
                "    dummy = [str(i) for i in range(200) for _ in range(10)]\n"
                "    return {'id': user_id, 'name': 'Alice'}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
            "breaking_public_api_patch": (
                "def get_user_by_id(user_id: str) -> dict:\n"
                "    return {'uid': user_id}\n\n"
                "def delete_user(user_id: str) -> bool:\n"
                "    return True\n"
            ),
        },
        {
            "name": "py_kwargs_service",
            "files": {
                "options.py": (
                    "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                    "    pool_size = kwargs.get('pool_size', 10)\n"
                    "    max_overflow = kwargs.get('max_overflow', 5)\n"
                    "    return {\n"
                    "        'driver': driver,\n"
                    "        'timeout': timeout,\n"
                    "        'pool_size': pool_size,\n"
                    "        'max_overflow': max_overflow,\n"
                    "        'options': kwargs,\n"
                    "    }\n"
                ),
                "engine.py": (
                    "from options import configure_engine\n\n"
                    "def create_pool(name: str) -> dict:\n"
                    "    return configure_engine('postgres', timeout=60, pool_size=50, max_overflow=20)\n"
                ),
            },
            "target_file": "options.py",
            "target_symbol": "configure_engine",
            "caller_file": "engine.py",
            "caller_symbol": "create_pool",
            "pass_patch": (
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    # Optimized engine configuration\n"
                "    pool_size = int(kwargs.get('pool_size', 10))\n"
                "    max_overflow = int(kwargs.get('max_overflow', 5))\n"
                "    return {\n"
                "        'driver': driver.strip(),\n"
                "        'timeout': timeout,\n"
                "        'pool_size': pool_size,\n"
                "        'max_overflow': max_overflow,\n"
                "        'options': kwargs,\n"
                "    }\n"
            ),
            "py_kwargs_drift_patch": (
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    # Keyword argument omission drift: kwargs ignored, default pool_size hardcoded\n"
                "    return {\n"
                "        'driver': driver,\n"
                "        'timeout': timeout,\n"
                "        'pool_size': 10,\n"
                "        'max_overflow': 5,\n"
                "        'options': {},\n"
                "    }\n"
            ),
            "breaking_public_api_patch": (
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    return {\n"
                "        'driver': driver,\n"
                "        'timeout': timeout,\n"
                "        'pool_size': 10,\n"
                "        'max_overflow': 5,\n"
                "        'options': {},\n"
                "    }\n"
            ),
            "arity_patch": (
                "def configure_engine(driver: str, timeout: int = 30, extra_pos: int = 1) -> dict:\n"
                "    return {'driver': driver}\n"
            ),
            "keyword_patch": (
                "from options import configure_engine\n\n"
                "def create_pool(name: str) -> dict:\n"
                "    return configure_engine('postgres', 60, 50, 20, 10, 5)\n"
            ),
            "circular_patch": (
                "from engine import create_pool\n\n"
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    create_pool('circ')\n"
                "    return {'driver': driver}\n"
            ),
            "deleted_patch": (
                "def other_helper():\n"
                "    return True\n"
            ),
            "silent_logic_drift_patch": (
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    return {'driver': driver, 'timeout': timeout if timeout > 0 else 30}\n"
            ),
            "security_surface_patch": (
                "import os\n\n"
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    os.system(f'echo {driver} > /dev/null')\n"
                "    return {'driver': driver}\n"
            ),
            "concurrency_hazard_patch": (
                "_SHARED_POOLS = []\n\n"
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    _SHARED_POOLS.append(driver)\n"
                "    return {'driver': driver}\n"
            ),
            "performance_regression_patch": (
                "def configure_engine(driver: str, timeout: int = 30, **kwargs) -> dict:\n"
                "    for _ in range(200):\n"
                "        for _ in range(10): pass\n"
                "    return {'driver': driver}\n"
            ),
        },
        {
            "name": "py_mutable_default_service",
            "files": {
                "cache.py": (
                    "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                    "    val = cache_dict.get(key)\n"
                    "    if val is not None:\n"
                    "        return val\n"
                    "    return default_val\n"
                ),
                "service.py": (
                    "from cache import lookup_cache_entry\n\n"
                    "def get_setting(settings: dict) -> str:\n"
                    "    return lookup_cache_entry(settings, 'env', 'production')\n"
                ),
            },
            "target_file": "cache.py",
            "target_symbol": "lookup_cache_entry",
            "caller_file": "service.py",
            "caller_symbol": "get_setting",
            "pass_patch": (
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                "    # Invariant preserved: safe dictionary lookup\n"
                "    if key in cache_dict:\n"
                "        return str(cache_dict[key])\n"
                "    return default_val\n"
            ),
            "py_mutable_default_patch": (
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default', call_history: list = []) -> str:\n"
                "    # Mutable default argument leak + dict.get fallback mutation\n"
                "    call_history.append(key)\n"
                "    return cache_dict.get(key)\n"
            ),
            "performance_regression_patch": (
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default', call_history: list = []) -> str:\n"
                "    call_history.append(key)\n"
                "    return cache_dict.get(key)\n"
            ),
            "arity_patch": (
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str, extra_required: int) -> str:\n"
                "    return default_val\n"
            ),
            "keyword_patch": (
                "from cache import lookup_cache_entry\n\n"
                "def get_setting(settings: dict) -> str:\n"
                "    return lookup_cache_entry(settings, 'env', unexpected_param=True)\n"
            ),
            "circular_patch": (
                "from service import get_setting\n\n"
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                "    get_setting({})\n"
                "    return default_val\n"
            ),
            "deleted_patch": (
                "def dummy_cache():\n"
                "    return None\n"
            ),
            "breaking_public_api_patch": (
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                "    return 'fixed'\n"
            ),
            "silent_logic_drift_patch": (
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                "    return cache_dict.get(key, 'wrong')\n"
            ),
            "security_surface_patch": (
                "import os\n\n"
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                "    os.system(f'echo {key} > /dev/null')\n"
                "    return default_val\n"
            ),
            "concurrency_hazard_patch": (
                "_CACHE_LOCK_LEAK = {}\n\n"
                "def lookup_cache_entry(cache_dict: dict, key: str, default_val: str = 'default') -> str:\n"
                "    _CACHE_LOCK_LEAK[key] = 1\n"
                "    return default_val\n"
            ),
        },
        {
            "name": "py_truthiness_service",
            "files": {
                "validator.py": (
                    "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                    "    if threshold is not None:\n"
                    "        return threshold >= 0.0\n"
                    "    return True\n"
                ),
                "monitor.py": (
                    "from validator import check_threshold\n\n"
                    "def should_alert(metric: str) -> bool:\n"
                    "    return check_threshold(metric, 0.0)\n"
                ),
            },
            "target_file": "validator.py",
            "target_symbol": "check_threshold",
            "caller_file": "monitor.py",
            "caller_symbol": "should_alert",
            "pass_patch": (
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    # Explicit None check preserved\n"
                "    if threshold is not None:\n"
                "        return float(threshold) >= 0.0\n"
                "    return True\n"
            ),
            "py_truthiness_drift_patch": (
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    # Truthiness drift: 0.0 evaluates to False, silently defaulting to True!\n"
                "    if threshold:\n"
                "        return threshold > 0.0\n"
                "    return True\n"
            ),
            "silent_logic_drift_patch": (
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    if threshold:\n"
                "        return threshold > 0.0\n"
                "    return True\n"
            ),
            "arity_patch": (
                "def check_threshold(metric_name: str, threshold: float, extra_token: str) -> bool:\n"
                "    return True\n"
            ),
            "keyword_patch": (
                "from validator import check_threshold\n\n"
                "def should_alert(metric: str) -> bool:\n"
                "    return check_threshold(metric, 0.0, invalid_flag=True)\n"
            ),
            "circular_patch": (
                "from monitor import should_alert\n\n"
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    should_alert('circ')\n"
                "    return True\n"
            ),
            "deleted_patch": (
                "def dummy_val():\n"
                "    return False\n"
            ),
            "breaking_public_api_patch": (
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    return False\n"
            ),
            "security_surface_patch": (
                "import os\n\n"
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    os.system(f'echo {metric_name} > /dev/null')\n"
                "    return True\n"
            ),
            "concurrency_hazard_patch": (
                "_THRESHOLD_MUTEX = []\n\n"
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    _THRESHOLD_MUTEX.append(metric_name)\n"
                "    return True\n"
            ),
            "performance_regression_patch": (
                "def check_threshold(metric_name: str, threshold: float | None = None) -> bool:\n"
                "    for _ in range(200):\n"
                "        for _ in range(10): pass\n"
                "    return True\n"
            ),
        },
        {
            "name": "py_revert_mimic_service",
            "files": {
                "sanitizer.py": (
                    "import html\n\n"
                    "def sanitize_user_input(raw_input: str) -> str:\n"
                    "    # Security hotfix: escaped HTML rendering to prevent XSS\n"
                    "    return html.escape(raw_input.strip())\n"
                ),
                "render.py": (
                    "from sanitizer import sanitize_user_input\n\n"
                    "def render_badge(username: str) -> str:\n"
                    "    clean = sanitize_user_input(username)\n"
                    "    return f'<span>{clean}</span>'\n"
                ),
            },
            "target_file": "sanitizer.py",
            "target_symbol": "sanitize_user_input",
            "caller_file": "render.py",
            "caller_symbol": "render_badge",
            "pass_patch": (
                "import html\n\n"
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    # Multi-layer sanitization\n"
                "    stripped = raw_input.strip()\n"
                "    return html.escape(stripped, quote=True)\n"
            ),
            "py_revert_mimic_patch": (
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    # Revert-mimicking patch: inverts XSS sanitization hotfix\n"
                "    return str(raw_input).strip()\n"
            ),
            "security_surface_patch": (
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    return str(raw_input).strip()\n"
            ),
            "silent_logic_drift_patch": (
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    return str(raw_input).strip()\n"
            ),
            "arity_patch": (
                "def sanitize_user_input(raw_input: str, escape_table: dict) -> str:\n"
                "    return raw_input\n"
            ),
            "keyword_patch": (
                "from sanitizer import sanitize_user_input\n\n"
                "def render_badge(username: str) -> str:\n"
                "    return sanitize_user_input(username, invalid_mode='strict')\n"
            ),
            "circular_patch": (
                "from render import render_badge\n\n"
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    render_badge('circ')\n"
                "    return raw_input\n"
            ),
            "deleted_patch": (
                "def dummy_sanitizer():\n"
                "    return ''\n"
            ),
            "breaking_public_api_patch": (
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    return ''\n"
            ),
            "concurrency_hazard_patch": (
                "_SAN_LOG = []\n\n"
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    _SAN_LOG.append(raw_input)\n"
                "    return raw_input\n"
            ),
            "performance_regression_patch": (
                "def sanitize_user_input(raw_input: str) -> str:\n"
                "    for _ in range(200):\n"
                "        for _ in range(10): pass\n"
                "    return raw_input\n"
            ),
        },
    ],
    "typescript": [
        {
            "name": "math_service",
            "files": {
                "mathOps.ts": (
                    "export function add(a: number, b: number): number {\n"
                    "    return a + b;\n"
                    "}\n\n"
                    "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                    "    return x * y * factor;\n"
                    "}\n"
                ),
                "main.ts": (
                    "import { add, multiply } from './mathOps';\n\n"
                    "export function calculateTotal(a: number, b: number): number {\n"
                    "    const sumVal = add(a, b);\n"
                    "    return multiply(sumVal, 2);\n"
                    "}\n"
                ),
            },
            "target_file": "mathOps.ts",
            "target_symbol": "add",
            "caller_file": "main.ts",
            "caller_symbol": "calculateTotal",
            "pass_patch": (
                "export function add(a: number, b: number): number {\n"
                "    // Optimized sum\n"
                "    return a + b;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "arity_patch": (
                "export function add(a: number, b: number, c: number): number {\n"
                "    return a + b + c;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "keyword_patch": (
                "import { add, multiply } from './mathOps';\n\n"
                "export function calculateTotal(a: number, b: number): number {\n"
                "    const sumVal = add(a, b);\n"
                "    return multiply(sumVal, 2, 1, 99);\n"
                "}\n"
            ),
            "circular_patch": (
                "import { calculateTotal } from './main';\n\n"
                "export function add(a: number, b: number): number {\n"
                "    return a + b;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "deleted_patch": (
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "export function add(a: number, b: number): number {\n"
                "    return a >= 0 ? a + b : a - b;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "security_surface_patch": (
                "export function add(a: number, b: number): number {\n"
                "    eval('var _sec_check = ' + a + ' + ' + b);\n"
                "    return a + b;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "let _sharedCounter = 0;\n\n"
                "export function add(a: number, b: number): number {\n"
                "    _sharedCounter += a + b;\n"
                "    return a + b;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "export function add(a: number, b: number): number {\n"
                "    for (let i = 0; i < 200; i++) {\n"
                "        for (let j = 0; j < 10; j++) {\n"
                "            a += 0;\n"
                "        }\n"
                "    }\n"
                "    return a + b;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "export function add(a: number, b: number): number {\n"
                "    return 0;\n"
                "}\n\n"
                "export function multiply(x: number, y: number, factor: number = 1): number {\n"
                "    return x * y * factor;\n"
                "}\n"
            ),
        },
        {
            "name": "user_service",
            "files": {
                "userRepo.ts": (
                    "export class UserRepo {\n"
                    "    getUserById(userId: string): object {\n"
                    "        return { id: userId, name: 'Alice' };\n"
                    "    }\n"
                    "}\n"
                ),
                "userService.ts": (
                    "import { UserRepo } from './userRepo';\n\n"
                    "export function fetchUser(repo: UserRepo, id: string): object {\n"
                    "    return repo.getUserById(id);\n"
                    "}\n"
                ),
            },
            "target_file": "userRepo.ts",
            "target_symbol": "getUserById",
            "caller_file": "userService.ts",
            "caller_symbol": "fetchUser",
            "pass_patch": (
                "export class UserRepo {\n"
                "    getUserById(userId: string, activeOnly: boolean = true): object {\n"
                "        return { id: userId, name: 'Alice', active: activeOnly };\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "export class UserRepo {\n"
                "    getUserById(userId: string, token: string): object {\n"
                "        return { id: userId, name: 'Alice' };\n"
                "    }\n"
                "}\n"
            ),
            "keyword_patch": (
                "import { UserRepo } from './userRepo';\n\n"
                "export function fetchUser(repo: UserRepo, id: string): object {\n"
                "    return repo.getUserById(id, 'extra', 'unused');\n"
                "}\n"
            ),
            "circular_patch": (
                "import { fetchUser } from './userService';\n\n"
                "export class UserRepo {\n"
                "    getUserById(userId: string): object {\n"
                "        return { id: userId, name: 'Alice' };\n"
                "    }\n"
                "}\n"
            ),
            "deleted_patch": (
                "export class UserRepo {\n"
                "    deleteUser(userId: string): boolean {\n"
                "        return true;\n"
                "    }\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "export class UserRepo {\n"
                "    getUserById(userId: string): object {\n"
                "        if (userId === 'admin') return {};\n"
                "        return { id: userId, name: 'Alice' };\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "export class UserRepo {\n"
                "    getUserById(userId: string): object {\n"
                "        eval(\"const _u = '\" + userId + \"'\");\n"
                "        return { id: userId, name: 'Alice' };\n"
                "    }\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "const _cache: any = {};\n\n"
                "export class UserRepo {\n"
                "    getUserById(userId: string): object {\n"
                "        _cache[userId] = (_cache[userId] || 0) + 1;\n"
                "        return { id: userId, name: 'Alice' };\n"
                "    }\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "export class UserRepo {\n"
                "    getUserById(userId: string): object {\n"
                "        const dummy: string[] = [];\n"
                "        for (let i = 0; i < 200; i++) dummy.push(userId);\n"
                "        return { id: userId, name: 'Alice' };\n"
                "    }\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "export class UserRepo {\n"
                "    getUserById(userId: string): object {\n"
                "        return { uid: userId };\n"
                "    }\n"
                "}\n"
            ),
        },
        {
            "name": "ts_type_widening_service",
            "files": {
                "types.ts": (
                    "export interface ApiResponse<T> {\n"
                    "    status: number;\n"
                    "    data: T;\n"
                    "}\n"
                    "export interface UserProfile {\n"
                    "    id: string;\n"
                    "    username: string;\n"
                    "    permissions: string[];\n"
                    "}\n"
                ),
                "auth.ts": (
                    "import { ApiResponse, UserProfile } from './types';\n\n"
                    "export function parseUserProfile(rawJson: string): ApiResponse<UserProfile> {\n"
                    "    const p = JSON.parse(rawJson);\n"
                    "    return {\n"
                    "        status: 200,\n"
                    "        data: {\n"
                    "            id: String(p.id),\n"
                    "            username: String(p.username),\n"
                    "            permissions: Array.isArray(p.permissions) ? p.permissions : [],\n"
                    "        },\n"
                    "    };\n"
                    "}\n"
                ),
                "router.ts": (
                    "import { parseUserProfile } from './auth';\n\n"
                    "export function handleRequest(raw: string): string {\n"
                    "    const res = parseUserProfile(raw);\n"
                    "    return res.data.permissions.join(',');\n"
                    "}\n"
                ),
            },
            "target_file": "auth.ts",
            "target_symbol": "parseUserProfile",
            "caller_file": "router.ts",
            "caller_symbol": "handleRequest",
            "pass_patch": (
                "import { ApiResponse, UserProfile } from './types';\n\n"
                "export function parseUserProfile(rawJson: string): ApiResponse<UserProfile> {\n"
                "    const p = JSON.parse(rawJson);\n"
                "    return {\n"
                "        status: 200,\n"
                "        data: {\n"
                "            id: String(p.id).trim(),\n"
                "            username: String(p.username).trim(),\n"
                "            permissions: Array.isArray(p.permissions) ? p.permissions : [],\n"
                "        },\n"
                "    };\n"
                "}\n"
            ),
            "ts_type_widening_patch": (
                "export function parseUserProfile(rawJson: any): any {\n"
                "    const p = JSON.parse(rawJson);\n"
                "    return {\n"
                "        status: 200,\n"
                "        data: { id: p.id, username: p.username },\n"
                "    };\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "export function parseUserProfile(rawJson: any): any {\n"
                "    const p = JSON.parse(rawJson);\n"
                "    return {\n"
                "        status: 200,\n"
                "        data: { id: p.id, username: p.username },\n"
                "    };\n"
                "}\n"
            ),
            "arity_patch": (
                "import { ApiResponse, UserProfile } from './types';\n\n"
                "export function parseUserProfile(rawJson: string, requiredToken: string): ApiResponse<UserProfile> {\n"
                "    return { status: 200, data: { id: '1', username: 'u', permissions: [] } };\n"
                "}\n"
            ),
            "keyword_patch": (
                "import { parseUserProfile } from './auth';\n\n"
                "export function handleRequest(raw: string): string {\n"
                "    const res = parseUserProfile(raw, 'extra', 'unused');\n"
                "    return 'ok';\n"
                "}\n"
            ),
            "circular_patch": (
                "import { handleRequest } from './router';\n\n"
                "export function parseUserProfile(rawJson: string): any {\n"
                "    handleRequest(rawJson);\n"
                "    return { status: 200, data: {} };\n"
                "}\n"
            ),
            "deleted_patch": (
                "export function dummyAuth(): boolean {\n"
                "    return true;\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "export function parseUserProfile(rawJson: any): any {\n"
                "    return { status: 200, data: { id: '0', username: 'anon' } };\n"
                "}\n"
            ),
            "security_surface_patch": (
                "export function parseUserProfile(rawJson: any): any {\n"
                "    eval('var _raw = ' + rawJson);\n"
                "    return { status: 200, data: { id: '0', username: 'anon' } };\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "let _cachedProfile: any = null;\n\n"
                "export function parseUserProfile(rawJson: any): any {\n"
                "    _cachedProfile = JSON.parse(rawJson);\n"
                "    return { status: 200, data: { id: '0', username: 'anon' } };\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "export function parseUserProfile(rawJson: any): any {\n"
                "    for (let i = 0; i < 200; i++) { for (let j = 0; j < 10; j++) {} }\n"
                "    return { status: 200, data: { id: '0', username: 'anon' } };\n"
                "}\n"
            ),
        },
        {
            "name": "ts_optional_chaining_service",
            "files": {
                "session.ts": (
                    "export interface Session {\n"
                    "    id: string;\n"
                    "    user: { token: string; active: boolean };\n"
                    "}\n\n"
                    "export function validateSessionToken(session: Session): string {\n"
                    "    if (!session || !session.user || !session.user.token) {\n"
                    "        throw new Error('Invalid session token');\n"
                    "    }\n"
                    "    return session.user.token;\n"
                    "}\n"
                ),
                "guard.ts": (
                    "import { validateSessionToken, Session } from './session';\n\n"
                    "export function authenticate(sess: Session): boolean {\n"
                    "    const tok = validateSessionToken(sess);\n"
                    "    return tok.length > 0;\n"
                    "}\n"
                ),
            },
            "target_file": "session.ts",
            "target_symbol": "validateSessionToken",
            "caller_file": "guard.ts",
            "caller_symbol": "authenticate",
            "pass_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: Session): string {\n"
                "    if (!session || !session.user || !session.user.token || session.user.token.length < 1) {\n"
                "        throw new Error('Invalid session token');\n"
                "    }\n"
                "    return session.user.token;\n"
                "}\n"
            ),
            "ts_optional_chaining_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: any): string {\n"
                "    return session?.user?.token;\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: any): string {\n"
                "    return session?.user?.token;\n"
                "}\n"
            ),
            "arity_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: Session, strictMode: boolean): string {\n"
                "    return 'valid';\n"
                "}\n"
            ),
            "keyword_patch": (
                "import { validateSessionToken, Session } from './session';\n\n"
                "export function authenticate(sess: Session): boolean {\n"
                "    const tok = validateSessionToken(sess, 'invalid_arg', 9999);\n"
                "    return true;\n"
                "}\n"
            ),
            "circular_patch": (
                "import { authenticate } from './guard';\n\n"
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: Session): string {\n"
                "    authenticate(session);\n"
                "    return 'tok';\n"
                "}\n"
            ),
            "deleted_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function dummySession(): boolean {\n"
                "    return true;\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: Session): string {\n"
                "    return '';\n"
                "}\n"
            ),
            "security_surface_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: Session): string {\n"
                "    eval('var _sec = true');\n"
                "    return 'tok';\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\nlet _sessions: any = {};\n\n"
                "export function validateSessionToken(session: Session): string {\n"
                "    _sessions[session.id] = 1;\n"
                "    return 'tok';\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "export interface Session {\n"
                "    id: string;\n"
                "    user: { token: string; active: boolean };\n"
                "}\n\n"
                "export function validateSessionToken(session: Session): string {\n"
                "    for (let i = 0; i < 200; i++) { for (let j = 0; j < 10; j++) {} }\n"
                "    return 'tok';\n"
                "}\n"
            ),
        },
        {
            "name": "ts_floating_promise_service",
            "files": {
                "audit.ts": (
                    "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                    "    return eventId.length > 0 && details.length > 0;\n"
                    "}\n\n"
                    "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                    "    const success = await persistAuditRecord(eventId, details);\n"
                    "    return success;\n"
                    "}\n"
                ),
                "controller.ts": (
                    "import { logAuditEvent } from './audit';\n\n"
                    "export async function processEvent(id: string, payload: string): Promise<boolean> {\n"
                    "    return await logAuditEvent(id, payload);\n"
                    "}\n"
                ),
            },
            "target_file": "audit.ts",
            "target_symbol": "logAuditEvent",
            "caller_file": "controller.ts",
            "caller_symbol": "processEvent",
            "pass_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return eventId.length > 0 && details.length > 0;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    const success = await persistAuditRecord(eventId, details);\n"
                "    return success === true;\n"
                "}\n"
            ),
            "ts_floating_promise_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return eventId.length > 0 && details.length > 0;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    persistAuditRecord(eventId, details);\n"
                "    return true;\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return eventId.length > 0 && details.length > 0;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    persistAuditRecord(eventId, details);\n"
                "    return true;\n"
                "}\n"
            ),
            "arity_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return true;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string, mandatoryFlag: boolean): Promise<boolean> {\n"
                "    return true;\n"
                "}\n"
            ),
            "keyword_patch": (
                "import { logAuditEvent } from './audit';\n\n"
                "export async function processEvent(id: string, payload: string): Promise<boolean> {\n"
                "    return await logAuditEvent(id, payload, 'extra_arg', 9999);\n"
                "}\n"
            ),
            "circular_patch": (
                "import { processEvent } from './controller';\n\n"
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return true;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    await processEvent(eventId, details);\n"
                "    return true;\n"
                "}\n"
            ),
            "deleted_patch": (
                "export async function otherAudit(): Promise<boolean> {\n"
                "    return true;\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return true;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    return false;\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return true;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    return eventId === 'admin';\n"
                "}\n"
            ),
            "security_surface_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return true;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    eval('var _audit = true');\n"
                "    return true;\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "export async function persistAuditRecord(eventId: string, details: string): Promise<boolean> {\n"
                "    return true;\n"
                "}\n\n"
                "export async function logAuditEvent(eventId: string, details: string): Promise<boolean> {\n"
                "    for (let i = 0; i < 200; i++) { for (let j = 0; j < 10; j++) {} }\n"
                "    return true;\n"
                "}\n"
            ),
        },
        {
            "name": "ts_destructuring_service",
            "files": {
                "formatter.ts": (
                    "export interface CustomerAccount {\n"
                    "    id: string;\n"
                    "    email: string;\n"
                    "    tier: string;\n"
                    "}\n\n"
                    "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                    "    return {\n"
                    "        id: acc.id,\n"
                    "        email: acc.email,\n"
                    "        tier: acc.tier,\n"
                    "    };\n"
                    "}\n"
                ),
                "api.ts": (
                    "import { sanitizeCustomerAccount, CustomerAccount } from './formatter';\n\n"
                    "export function exportAccount(acc: CustomerAccount): object {\n"
                    "    const clean = sanitizeCustomerAccount(acc);\n"
                    "    return clean;\n"
                    "}\n"
                ),
            },
            "target_file": "formatter.ts",
            "target_symbol": "sanitizeCustomerAccount",
            "caller_file": "api.ts",
            "caller_symbol": "exportAccount",
            "pass_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                "    return {\n"
                "        id: String(acc.id).toLowerCase(),\n"
                "        email: String(acc.email).trim(),\n"
                "        tier: acc.tier || 'standard',\n"
                "    };\n"
                "}\n"
            ),
            "ts_destructuring_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: any): object {\n"
                "    const { email, ...rest } = acc;\n"
                "    return { id: rest.id, tier: rest.tier };\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: any): object {\n"
                "    const { email, ...rest } = acc;\n"
                "    return { id: rest.id, tier: rest.tier };\n"
                "}\n"
            ),
            "arity_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount, prefix: string): object {\n"
                "    return {};\n"
                "}\n"
            ),
            "keyword_patch": (
                "import { sanitizeCustomerAccount, CustomerAccount } from './formatter';\n\n"
                "export function exportAccount(acc: CustomerAccount): object {\n"
                "    return sanitizeCustomerAccount(acc, 'extra', 'unused');\n"
                "}\n"
            ),
            "circular_patch": (
                "import { exportAccount } from './api';\n\n"
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                "    exportAccount(acc);\n"
                "    return {};\n"
                "}\n"
            ),
            "deleted_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function dummyFormatter(): boolean {\n"
                "    return true;\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                "    return { id: acc.id };\n"
                "}\n"
            ),
            "security_surface_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                "    eval('var _acc = true');\n"
                "    return { id: acc.id };\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\nlet _accPool: any = [];\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                "    _accPool.push(acc);\n"
                "    return { id: acc.id };\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "export interface CustomerAccount {\n"
                "    id: string;\n"
                "    email: string;\n"
                "    tier: string;\n"
                "}\n\n"
                "export function sanitizeCustomerAccount(acc: CustomerAccount): object {\n"
                "    for (let i = 0; i < 200; i++) { for (let j = 0; j < 10; j++) {} }\n"
                "    return { id: acc.id };\n"
                "}\n"
            ),
        },
    ],
    "go": [
        {
            "name": "math_service",
            "files": {
                "calc.go": (
                    "package main\n\n"
                    "func Add(a, b int) int {\n"
                    "    return a + b\n"
                    "}\n\n"
                    "func Multiply(x, y int) int {\n"
                    "    return x * y\n"
                    "}\n"
                ),
                "main.go": (
                    "package main\n\n"
                    "func Run() int {\n"
                    "    return Add(1, 2)\n"
                    "}\n"
                ),
            },
            "target_file": "calc.go",
            "target_symbol": "Add",
            "caller_file": "main.go",
            "caller_symbol": "Run",
            "pass_patch": (
                "package main\n\n"
                "func Add(a, b int) int {\n"
                "    // Optimized sum\n"
                "    return a + b\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "func Add(a, b, c int) int {\n"
                "    return a + b + c\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func Run() int {\n"
                "    return Add(1, 2, 3)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "func Add(a, b int) int {\n"
                "    Run()\n"
                "    return a + b\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "func Add(a, b int) int {\n"
                "    if a < 0 {\n"
                "        return a - b\n"
                "    }\n"
                "    return a + b\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "func Add(a, b int) int {\n"
                "    _ = exec.Command(\"echo\", \"add\").Start()\n"
                "    return a + b\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var sharedCounter int\n\n"
                "func Add(a, b int) int {\n"
                "    go func() {\n"
                "        sharedCounter += a + b\n"
                "    }()\n"
                "    return a + b\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "func Add(a, b int) int {\n"
                "    for i := 0; i < 200; i++ {\n"
                "        for j := 0; j < 10; j++ {\n"
                "            a += 0\n"
                "        }\n"
                "    }\n"
                "    return a + b\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "func Add(a, b int) int {\n"
                "    return 0\n"
                "}\n\n"
                "func Multiply(x, y int) int {\n"
                "    return x * y\n"
                "}\n"
            ),
        },
        {
            "name": "server_service",
            "files": {
                "server.go": (
                    "package main\n\n"
                    "type Server struct {\n"
                    "    Port int\n"
                    "}\n\n"
                    "func (s *Server) Start(host string) bool {\n"
                    "    return true\n"
                    "}\n"
                ),
                "app.go": (
                    "package main\n\n"
                    "func Boot(s *Server) bool {\n"
                    "    return s.Start(\"localhost\")\n"
                    "}\n"
                ),
            },
            "target_file": "server.go",
            "target_symbol": "Start",
            "caller_file": "app.go",
            "caller_symbol": "Boot",
            "pass_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    // Init server\n"
                "    return true\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string, timeout int) bool {\n"
                "    return true\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func Boot(s *Server) bool {\n"
                "    return s.Start(\"localhost\", 50, 100)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    Boot(s)\n"
                "    return true\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    if host == \"localhost\" {\n"
                "        return false\n"
                "    }\n"
                "    return true\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    exec.Command(\"sh\", \"-c\", \"echo \"+host).Run()\n"
                "    return true\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    go func() {\n"
                "        s.Port++\n"
                "    }()\n"
                "    return true\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    for i := 0; i < 200; i++ {\n"
                "        for j := 0; j < 10; j++ {\n"
                "            _ = i + j\n"
                "        }\n"
                "    }\n"
                "    return true\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type Server struct {\n"
                "    Port int\n"
                "}\n\n"
                "func (s *Server) Start(host string) bool {\n"
                "    return false\n"
                "}\n"
            ),
        },
        {
            "name": "go_channel_service",
            "files": {
                "chan_mgr.go": (
                    "package main\n\n"
                    "type ChannelManager struct {\n"
                    "    Jobs chan int\n"
                    "}\n\n"
                    "func NewManager(size int) *ChannelManager {\n"
                    "    return &ChannelManager{Jobs: make(chan int, size)}\n"
                    "}\n\n"
                    "func (m *ChannelManager) Enqueue(item int) bool {\n"
                    "    select {\n"
                    "    case m.Jobs <- item:\n"
                    "        return true\n"
                    "    default:\n"
                    "        return false\n"
                    "    }\n"
                    "}\n"
                ),
                "worker.go": (
                    "package main\n\n"
                    "func Dispatch(m *ChannelManager, val int) bool {\n"
                    "    return m.Enqueue(val)\n"
                    "}\n"
                ),
            },
            "target_file": "chan_mgr.go",
            "target_symbol": "Enqueue",
            "caller_file": "worker.go",
            "caller_symbol": "Dispatch",
            "pass_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func NewManager(size int) *ChannelManager {\n"
                "    return &ChannelManager{Jobs: make(chan int, size)}\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    // Non-blocking safe enqueue\n"
                "    select {\n"
                "    case m.Jobs <- item:\n"
                "        return true\n"
                "    default:\n"
                "        return false\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int, priority bool) bool {\n"
                "    return true\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func Dispatch(m *ChannelManager, val int) bool {\n"
                "    return m.Enqueue(val, true, false)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    Dispatch(m, item)\n"
                "    return true\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func NewManager(size int) *ChannelManager {\n"
                "    return &ChannelManager{Jobs: make(chan int, size)}\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    if item < 0 {\n"
                "        return false\n"
                "    }\n"
                "    select {\n"
                "    case m.Jobs <- item:\n"
                "        return true\n"
                "    default:\n"
                "        return true // silent drop logic drift\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    exec.Command(\"sh\", \"-c\", \"echo task\").Run()\n"
                "    return true\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func NewManager(size int) *ChannelManager {\n"
                "    return &ChannelManager{Jobs: make(chan int, size)}\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    go func() {\n"
                "        m.Jobs <- item\n"
                "    }()\n"
                "    return true\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "import \"time\"\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    time.Sleep(5 * time.Millisecond)\n"
                "    return true\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    return false\n"
                "}\n"
            ),
            "go_channel_leak_patch": (
                "package main\n\n"
                "type ChannelManager struct {\n"
                "    Jobs chan int\n"
                "}\n\n"
                "func NewManager(size int) *ChannelManager {\n"
                "    return &ChannelManager{Jobs: make(chan int, size)}\n"
                "}\n\n"
                "func (m *ChannelManager) Enqueue(item int) bool {\n"
                "    unbuf := make(chan int)\n"
                "    go func() {\n"
                "        unbuf <- item\n"
                "    }()\n"
                "    return true\n"
                "}\n"
            ),
        },
        {
            "name": "go_resource_service",
            "files": {
                "resource.go": (
                    "package main\n\n"
                    "import \"os\"\n\n"
                    "type FileResource struct {\n"
                    "    Path string\n"
                    "}\n\n"
                    "func (r *FileResource) ReadHeader() (int, error) {\n"
                    "    f, err := os.Open(r.Path)\n"
                    "    if err != nil {\n"
                    "        return 0, err\n"
                    "    }\n"
                    "    defer f.Close()\n"
                    "    return 100, nil\n"
                    "}\n"
                ),
                "client.go": (
                    "package main\n\n"
                    "func AccessResource(r *FileResource) (int, error) {\n"
                    "    return r.ReadHeader()\n"
                    "}\n"
                ),
            },
            "target_file": "resource.go",
            "target_symbol": "ReadHeader",
            "caller_file": "client.go",
            "caller_symbol": "AccessResource",
            "pass_patch": (
                "package main\n\n"
                "import \"os\"\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    // Safely reads file and ensures cleanup\n"
                "    f, err := os.Open(r.Path)\n"
                "    if err != nil {\n"
                "        return 0, err\n"
                "    }\n"
                "    defer f.Close()\n"
                "    return 100, nil\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader(bufSize int) (int, error) {\n"
                "    return bufSize, nil\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func AccessResource(r *FileResource) (int, error) {\n"
                "    return r.ReadHeader(1024, 2048)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    return AccessResource(r)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "import \"os\"\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    f, err := os.Open(r.Path)\n"
                "    if err != nil {\n"
                "        return -1, nil // swallowed error logic drift\n"
                "    }\n"
                "    defer f.Close()\n"
                "    return 100, nil\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    exec.Command(\"cat\", r.Path).Run()\n"
                "    return 100, nil\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var activeHandles int\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    go func() {\n"
                "        activeHandles++\n"
                "    }()\n"
                "    return 100, nil\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "import \"os\"\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    f, err := os.Open(r.Path)\n"
                "    if err != nil {\n"
                "        return 0, err\n"
                "    }\n"
                "    // Descriptor leak via omitted defer Close\n"
                "    _ = f\n"
                "    return 100, nil\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    return 0, nil\n"
                "}\n"
            ),
            "go_resource_leak_patch": (
                "package main\n\n"
                "import \"os\"\n\n"
                "type FileResource struct {\n"
                "    Path string\n"
                "}\n\n"
                "func (r *FileResource) ReadHeader() (int, error) {\n"
                "    f, err := os.Open(r.Path)\n"
                "    if err != nil {\n"
                "        return 0, err\n"
                "    }\n"
                "    _ = f\n"
                "    return 100, nil\n"
                "}\n"
            ),
        },
        {
            "name": "go_truthiness_service",
            "files": {
                "auth.go": (
                    "package main\n\n"
                    "type AuthSession struct {\n"
                    "    UID string\n"
                    "    Role string\n"
                    "}\n\n"
                    "func (a *AuthSession) IsAuthorized() bool {\n"
                    "    if a == nil || a.UID == \"\" {\n"
                    "        return false\n"
                    "    }\n"
                    "    return a.Role == \"admin\"\n"
                    "}\n"
                ),
                "guard.go": (
                    "package main\n\n"
                    "func CheckAccess(a *AuthSession) bool {\n"
                    "    return a.IsAuthorized()\n"
                    "}\n"
                ),
            },
            "target_file": "auth.go",
            "target_symbol": "IsAuthorized",
            "caller_file": "guard.go",
            "caller_symbol": "CheckAccess",
            "pass_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "    Role string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    // Validates authentication session\n"
                "    if a == nil || a.UID == \"\" {\n"
                "        return false\n"
                "    }\n"
                "    return a.Role == \"admin\"\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized(scope string) bool {\n"
                "    return true\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func CheckAccess(a *AuthSession) bool {\n"
                "    return a.IsAuthorized(\"admin\", \"read\")\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    return CheckAccess(a)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "    Role string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    if a == nil {\n"
                "        return true // Inverted null check logic drift\n"
                "    }\n"
                "    return a.Role == \"admin\"\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "    Role string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    if a != nil {\n"
                "        return true\n"
                "    }\n"
                "    return false\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var accessCount int\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "    Role string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    go func() {\n"
                "        accessCount++\n"
                "    }()\n"
                "    return a.Role == \"admin\"\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "    Role string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    for i := 0; i < 500; i++ {\n"
                "        _ = i\n"
                "    }\n"
                "    return a.Role == \"admin\"\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    return false\n"
                "}\n"
            ),
            "go_truthiness_drift_patch": (
                "package main\n\n"
                "type AuthSession struct {\n"
                "    UID string\n"
                "    Role string\n"
                "}\n\n"
                "func (a *AuthSession) IsAuthorized() bool {\n"
                "    if a == nil || a.UID == \"\" {\n"
                "        return true\n"
                "    }\n"
                "    return a.Role != \"admin\"\n"
                "}\n"
            ),
        },
        {
            "name": "go_api_service",
            "files": {
                "api.go": (
                    "package main\n\n"
                    "type ApiResponse struct {\n"
                    "    Code int\n"
                    "    Data string\n"
                    "}\n\n"
                    "func NewResponse(code int, msg string) *ApiResponse {\n"
                    "    return &ApiResponse{Code: code, Data: msg}\n"
                    "}\n"
                ),
                "handler.go": (
                    "package main\n\n"
                    "func Handle() *ApiResponse {\n"
                    "    return NewResponse(200, \"OK\")\n"
                    "}\n"
                ),
            },
            "target_file": "api.go",
            "target_symbol": "NewResponse",
            "caller_file": "handler.go",
            "caller_symbol": "Handle",
            "pass_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    // Encapsulates standard HTTP response\n"
                "    return &ApiResponse{Code: code, Data: msg}\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string, err string) *ApiResponse {\n"
                "    return &ApiResponse{Code: code, Data: msg}\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func Handle() *ApiResponse {\n"
                "    return NewResponse(200, \"OK\", \"extra\", \"err\")\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    return Handle()\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    if code == 200 {\n"
                "        return &ApiResponse{Code: 204, Data: \"\"} // silent payload strip\n"
                "    }\n"
                "    return &ApiResponse{Code: code, Data: msg}\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    return &ApiResponse{Code: code, Data: msg + \" DEBUG_SECRET=123\"}\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var reqCounter int\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    go func() {\n"
                "        reqCounter++\n"
                "    }()\n"
                "    return &ApiResponse{Code: code, Data: msg}\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    for i := 0; i < 300; i++ {\n"
                "        _ = i\n"
                "    }\n"
                "    return &ApiResponse{Code: code, Data: msg}\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    return &ApiResponse{Code: 500, Data: \"Error\"}\n"
                "}\n"
            ),
            "go_api_drift_patch": (
                "package main\n\n"
                "type ApiResponse struct {\n"
                "    Code int\n"
                "    Data string\n"
                "}\n\n"
                "func NewResponse(code int, msg string) *ApiResponse {\n"
                "    return &ApiResponse{Code: code, Data: \"DEPRECATED\"}\n"
                "}\n"
            ),
        },
        {
            "name": "go_ignored_error_service",
            "files": {
                "account.go": (
                    "package main\n\n"
                    "import \"errors\"\n\n"
                    "type Account struct {\n"
                    "    ID      string\n"
                    "    Balance int\n"
                    "}\n\n"
                    "func Deduct(acc *Account, amount int) error {\n"
                    "    if acc.Balance < amount {\n"
                    "        return errors.New(\"insufficient balance\")\n"
                    "    }\n"
                    "    acc.Balance -= amount\n"
                    "    return nil\n"
                    "}\n\n"
                    "func TransferFunds(acc *Account, amount int) error {\n"
                    "    err := Deduct(acc, amount)\n"
                    "    if err != nil {\n"
                    "        return err\n"
                    "    }\n"
                    "    return nil\n"
                    "}\n"
                ),
                "service.go": (
                    "package main\n\n"
                    "func ProcessTransaction(acc *Account, amount int) error {\n"
                    "    return TransferFunds(acc, amount)\n"
                    "}\n"
                ),
            },
            "target_file": "account.go",
            "target_symbol": "TransferFunds",
            "caller_file": "service.go",
            "caller_symbol": "ProcessTransaction",
            "pass_patch": (
                "package main\n\n"
                "import \"errors\"\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func Deduct(acc *Account, amount int) error {\n"
                "    if acc.Balance < amount {\n"
                "        return errors.New(\"insufficient balance\")\n"
                "    }\n"
                "    acc.Balance -= amount\n"
                "    return nil\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    // Verified transfer with explicit error checking\n"
                "    if err := Deduct(acc, amount); err != nil {\n"
                "        return err\n"
                "    }\n"
                "    return nil\n"
                "}\n"
            ),
            "go_ignored_error_patch": (
                "package main\n\n"
                "import \"errors\"\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func Deduct(acc *Account, amount int) error {\n"
                "    if acc.Balance < amount {\n"
                "        return errors.New(\"insufficient balance\")\n"
                "    }\n"
                "    acc.Balance -= amount\n"
                "    return nil\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    err := Deduct(acc, amount)\n"
                "    _ = err // Ignored / shadowed error: proceeds without nil check\n"
                "    return nil\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int, force bool) error {\n"
                "    return nil\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func ProcessTransaction(acc *Account, amount int) error {\n"
                "    return TransferFunds(acc, amount, true)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    return ProcessTransaction(acc, amount)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "import \"errors\"\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func Deduct(acc *Account, amount int) error {\n"
                "    if acc.Balance < amount {\n"
                "        return errors.New(\"insufficient balance\")\n"
                "    }\n"
                "    acc.Balance -= amount\n"
                "    return nil\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    err := Deduct(acc, amount)\n"
                "    _ = err\n"
                "    return nil\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    exec.Command(\"sh\", \"-c\", \"echo audit\").Run()\n"
                "    return nil\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var globalAccCounter int\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    go func() {\n"
                "        globalAccCounter += amount\n"
                "    }()\n"
                "    return nil\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    for i := 0; i < 300; i++ {\n"
                "        _ = i\n"
                "    }\n"
                "    return nil\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type Account struct {\n"
                "    ID      string\n"
                "    Balance int\n"
                "}\n\n"
                "func TransferFunds(acc *Account, amount int) error {\n"
                "    return nil\n"
                "}\n"
            ),
        },
        {
            "name": "go_inverted_defer_service",
            "files": {
                "client.go": (
                    "package main\n\n"
                    "import (\n"
                    "    \"errors\"\n"
                    "    \"io\"\n"
                    ")\n\n"
                    "type MockResponse struct {\n"
                    "    Body io.ReadCloser\n"
                    "}\n\n"
                    "func FetchRaw(url string) (*MockResponse, error) {\n"
                    "    if url == \"\" {\n"
                    "        return nil, errors.New(\"empty url\")\n"
                    "    }\n"
                    "    return &MockResponse{Body: nil}, nil\n"
                    "}\n\n"
                    "func FetchPayload(url string) (int, error) {\n"
                    "    resp, err := FetchRaw(url)\n"
                    "    if err != nil {\n"
                    "        return 0, err\n"
                    "    }\n"
                    "    if resp != nil && resp.Body != nil {\n"
                    "        defer resp.Body.Close()\n"
                    "    }\n"
                    "    return 200, nil\n"
                    "}\n"
                ),
                "fetcher.go": (
                    "package main\n\n"
                    "func ExecuteFetch(target string) (int, error) {\n"
                    "    return FetchPayload(target)\n"
                    "}\n"
                ),
            },
            "target_file": "client.go",
            "target_symbol": "FetchPayload",
            "caller_file": "fetcher.go",
            "caller_symbol": "ExecuteFetch",
            "pass_patch": (
                "package main\n\n"
                "import (\n"
                "    \"errors\"\n"
                "    \"io\"\n"
                ")\n\n"
                "type MockResponse struct {\n"
                "    Body io.ReadCloser\n"
                "}\n\n"
                "func FetchRaw(url string) (*MockResponse, error) {\n"
                "    if url == \"\" {\n"
                "        return nil, errors.New(\"empty url\")\n"
                "    }\n"
                "    return &MockResponse{Body: nil}, nil\n"
                "}\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    // Correct defer order: verify err before deferring Close\n"
                "    resp, err := FetchRaw(url)\n"
                "    if err != nil {\n"
                "        return 0, err\n"
                "    }\n"
                "    if resp.Body != nil {\n"
                "        defer resp.Body.Close()\n"
                "    }\n"
                "    return 200, nil\n"
                "}\n"
            ),
            "go_inverted_defer_patch": (
                "package main\n\n"
                "import (\n"
                "    \"errors\"\n"
                "    \"io\"\n"
                ")\n\n"
                "type MockResponse struct {\n"
                "    Body io.ReadCloser\n"
                "}\n\n"
                "func FetchRaw(url string) (*MockResponse, error) {\n"
                "    if url == \"\" {\n"
                "        return nil, errors.New(\"empty url\")\n"
                "    }\n"
                "    return &MockResponse{Body: nil}, nil\n"
                "}\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    resp, err := FetchRaw(url)\n"
                "    defer resp.Body.Close() // Inverted defer order placed before err != nil check\n"
                "    if err != nil {\n"
                "        return 0, err\n"
                "    }\n"
                "    return 200, nil\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "func FetchPayload(url string, timeout int) (int, error) {\n"
                "    return timeout, nil\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func ExecuteFetch(target string) (int, error) {\n"
                "    return FetchPayload(target, 50, 100)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    return ExecuteFetch(url)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "func DummyFetch() int {\n"
                "    return 0\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "import (\n"
                "    \"errors\"\n"
                "    \"io\"\n"
                ")\n\n"
                "type MockResponse struct {\n"
                "    Body io.ReadCloser\n"
                "}\n\n"
                "func FetchRaw(url string) (*MockResponse, error) {\n"
                "    if url == \"\" {\n"
                "        return nil, errors.New(\"empty url\")\n"
                "    }\n"
                "    return &MockResponse{Body: nil}, nil\n"
                "}\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    resp, err := FetchRaw(url)\n"
                "    defer resp.Body.Close()\n"
                "    if err != nil {\n"
                "        return 0, err\n"
                "    }\n"
                "    return 200, nil\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    return -1, nil\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    exec.Command(\"curl\", url).Run()\n"
                "    return 200, nil\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var fetchCounter int\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    go func() {\n"
                "        fetchCounter++\n"
                "    }()\n"
                "    return 200, nil\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "func FetchPayload(url string) (int, error) {\n"
                "    return 500, nil\n"
                "}\n"
            ),
        },
        {
            "name": "go_receiver_drift_service",
            "files": {
                "session.go": (
                    "package main\n\n"
                    "type SessionService struct {\n"
                    "    Token  string\n"
                    "    Expiry int64\n"
                    "}\n\n"
                    "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                    "    s.Token = tok\n"
                    "    s.Expiry = exp\n"
                    "    return true\n"
                    "}\n"
                ),
                "manager.go": (
                    "package main\n\n"
                    "func RenewSession(s *SessionService, tok string, exp int64) bool {\n"
                    "    return s.UpdateToken(tok, exp)\n"
                    "}\n"
                ),
            },
            "target_file": "session.go",
            "target_symbol": "UpdateToken",
            "caller_file": "manager.go",
            "caller_symbol": "RenewSession",
            "pass_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    // Correct pointer receiver mutates caller state\n"
                "    if tok == \"\" {\n"
                "        return false\n"
                "    }\n"
                "    s.Token = tok\n"
                "    s.Expiry = exp\n"
                "    return true\n"
                "}\n"
            ),
            "go_receiver_drift_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    // Value receiver drift: s is a copy, mutations silently discarded\n"
                "    s.Token = tok\n"
                "    s.Expiry = exp\n"
                "    return true\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64, force bool) bool {\n"
                "    return true\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func RenewSession(s *SessionService, tok string, exp int64) bool {\n"
                "    return s.UpdateToken(tok, exp, true, false)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    return RenewSession(s, tok, exp)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    s.Token = tok\n"
                "    s.Expiry = exp\n"
                "    return true\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    exec.Command(\"sh\", \"-c\", \"echo token\").Run()\n"
                "    return true\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "var sessionCounter int\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    go func() {\n"
                "        sessionCounter++\n"
                "    }()\n"
                "    return true\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    for i := 0; i < 300; i++ {\n"
                "        _ = i\n"
                "    }\n"
                "    return true\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type SessionService struct {\n"
                "    Token  string\n"
                "    Expiry int64\n"
                "}\n\n"
                "func (s *SessionService) UpdateToken(tok string, exp int64) bool {\n"
                "    return false\n"
                "}\n"
            ),
        },
        {
            "name": "go_channel_leak_service",
            "files": {
                "pipeline.go": (
                    "package main\n\n"
                    "type TaskPipeline struct {\n"
                    "    BufferSize int\n"
                    "}\n\n"
                    "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                    "    ch := make(chan int, 1)\n"
                    "    select {\n"
                    "    case ch <- taskID:\n"
                    "        return true\n"
                    "    default:\n"
                    "        return false\n"
                    "    }\n"
                    "}\n"
                ),
                "worker.go": (
                    "package main\n\n"
                    "func ProcessBatch(p *TaskPipeline, taskID int) bool {\n"
                    "    return p.SubmitTask(taskID)\n"
                    "}\n"
                ),
            },
            "target_file": "pipeline.go",
            "target_symbol": "SubmitTask",
            "caller_file": "worker.go",
            "caller_symbol": "ProcessBatch",
            "pass_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    // Non-blocking buffered submit with fallback\n"
                "    ch := make(chan int, 1)\n"
                "    select {\n"
                "    case ch <- taskID:\n"
                "        return true\n"
                "    default:\n"
                "        return false\n"
                "    }\n"
                "}\n"
            ),
            "go_channel_leak_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    // Goroutine and unbuffered channel leak/deadlock without receiver or select timeout\n"
                "    unbuf := make(chan int)\n"
                "    go func() {\n"
                "        unbuf <- taskID\n"
                "    }()\n"
                "    return true\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int, urgent bool) bool {\n"
                "    return true\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func ProcessBatch(p *TaskPipeline, taskID int) bool {\n"
                "    return p.SubmitTask(taskID, true, false)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    return ProcessBatch(p, taskID)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    unbuf := make(chan int)\n"
                "    go func() {\n"
                "        unbuf <- taskID\n"
                "    }()\n"
                "    return true\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    return false\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    exec.Command(\"sh\", \"-c\", \"echo submit\").Run()\n"
                "    return true\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    for i := 0; i < 300; i++ {\n"
                "        _ = i\n"
                "    }\n"
                "    return true\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type TaskPipeline struct {\n"
                "    BufferSize int\n"
                "}\n\n"
                "func (p *TaskPipeline) SubmitTask(taskID int) bool {\n"
                "    return false\n"
                "}\n"
            ),
        },
        {
            "name": "go_mutex_unlock_service",
            "files": {
                "cache.go": (
                    "package main\n\n"
                    "import \"sync\"\n\n"
                    "type SafeCache struct {\n"
                    "    mu    sync.Mutex\n"
                    "    store map[string]string\n"
                    "}\n\n"
                    "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                    "    c.mu.Lock()\n"
                    "    defer c.mu.Unlock()\n"
                    "    if key == \"\" {\n"
                    "        return false\n"
                    "    }\n"
                    "    c.store[key] = val\n"
                    "    return true\n"
                    "}\n"
                ),
                "client.go": (
                    "package main\n\n"
                    "func WriteCache(c *SafeCache, key string, val string) bool {\n"
                    "    return c.SetItem(key, val)\n"
                    "}\n"
                ),
            },
            "target_file": "cache.go",
            "target_symbol": "SetItem",
            "caller_file": "client.go",
            "caller_symbol": "WriteCache",
            "pass_patch": (
                "package main\n\n"
                "import \"sync\"\n\n"
                "type SafeCache struct {\n"
                "    mu    sync.Mutex\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    // Safe mutex locking with deferred unlock\n"
                "    c.mu.Lock()\n"
                "    defer c.mu.Unlock()\n"
                "    if key == \"\" {\n"
                "        return false\n"
                "    }\n"
                "    c.store[key] = val\n"
                "    return true\n"
                "}\n"
            ),
            "go_mutex_unlock_patch": (
                "package main\n\n"
                "import \"sync\"\n\n"
                "type SafeCache struct {\n"
                "    mu    sync.Mutex\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    c.mu.Lock()\n"
                "    if key == \"\" {\n"
                "        // Mutex unlock omission on early return: causes permanent deadlock on next call\n"
                "        return false\n"
                "    }\n"
                "    c.store[key] = val\n"
                "    c.mu.Unlock()\n"
                "    return true\n"
                "}\n"
            ),
            "arity_patch": (
                "package main\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string, overwrite bool) bool {\n"
                "    return true\n"
                "}\n"
            ),
            "keyword_patch": (
                "package main\n\n"
                "func WriteCache(c *SafeCache, key string, val string) bool {\n"
                "    return c.SetItem(key, val, true, false)\n"
                "}\n"
            ),
            "circular_patch": (
                "package main\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    return WriteCache(c, key, val)\n"
                "}\n"
            ),
            "deleted_patch": (
                "package main\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "package main\n\n"
                "import \"sync\"\n\n"
                "type SafeCache struct {\n"
                "    mu    sync.Mutex\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    c.mu.Lock()\n"
                "    if key == \"\" {\n"
                "        return false\n"
                "    }\n"
                "    c.store[key] = val\n"
                "    c.mu.Unlock()\n"
                "    return true\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "package main\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    return false\n"
                "}\n"
            ),
            "security_surface_patch": (
                "package main\n\n"
                "import \"os/exec\"\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    exec.Command(\"sh\", \"-c\", \"echo cache\").Run()\n"
                "    return true\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "package main\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    for i := 0; i < 300; i++ {\n"
                "        _ = i\n"
                "    }\n"
                "    return true\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "package main\n\n"
                "type SafeCache struct {\n"
                "    store map[string]string\n"
                "}\n\n"
                "func (c *SafeCache) SetItem(key string, val string) bool {\n"
                "    return false\n"
                "}\n"
            ),
        },
    ],
    "rust": [
        {
            "name": "math_service",
            "files": {
                "lib.rs": (
                    "pub fn compute(x: i32, y: i32) -> i32 {\n"
                    "    x + y\n"
                    "}\n\n"
                    "pub fn double(v: i32) -> i32 {\n"
                    "    v * 2\n"
                    "}\n"
                ),
                "main.rs": (
                    "mod lib;\n\n"
                    "fn run() -> i32 {\n"
                    "    lib::compute(10, 20)\n"
                    "}\n"
                ),
            },
            "target_file": "lib.rs",
            "target_symbol": "compute",
            "caller_file": "main.rs",
            "caller_symbol": "run",
            "pass_patch": (
                "pub fn compute(x: i32, y: i32) -> i32 {\n"
                "    // Optimized compute\n"
                "    x + y\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "arity_patch": (
                "pub fn compute(x: i32, y: i32, z: i32) -> i32 {\n"
                "    x + y + z\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "keyword_patch": (
                "mod lib;\n\n"
                "fn run() -> i32 {\n"
                "    lib::compute(10, 20, 30)\n"
                "}\n"
            ),
            "circular_patch": (
                "mod main;\n\n"
                "pub fn compute(x: i32, y: i32) -> i32 {\n"
                "    x + y\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "deleted_patch": (
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "pub fn compute(x: i32, y: i32) -> i32 {\n"
                "    if x < 0 { x - y } else { x + y }\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "security_surface_patch": (
                "use std::process::Command;\n\n"
                "pub fn compute(x: i32, y: i32) -> i32 {\n"
                "    let _ = Command::new(\"echo\").arg(\"compute\").output();\n"
                "    x + y\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "static mut COUNTER: i32 = 0;\n\n"
                "pub fn compute(x: i32, y: i32) -> i32 {\n"
                "    unsafe {\n"
                "        COUNTER += x + y;\n"
                "    }\n"
                "    x + y\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "pub fn compute(mut x: i32, y: i32) -> i32 {\n"
                "    for _ in 0..200 {\n"
                "        for _ in 0..10 {\n"
                "            x += 0;\n"
                "        }\n"
                "    }\n"
                "    x + y\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "pub fn compute(x: i32, y: i32) -> i32 {\n"
                "    0\n"
                "}\n\n"
                "pub fn double(v: i32) -> i32 {\n"
                "    v * 2\n"
                "}\n"
            ),
        },
        {
            "name": "config_service",
            "files": {
                "config.rs": (
                    "pub struct Config {\n"
                    "    pub port: u16,\n"
                    "}\n\n"
                    "impl Config {\n"
                    "    pub fn validate(&self, env: &str) -> bool {\n"
                    "        true\n"
                    "    }\n"
                    "}\n"
                ),
                "server.rs": (
                    "mod config;\n\n"
                    "fn init(cfg: &config::Config) -> bool {\n"
                    "    cfg.validate(\"prod\")\n"
                    "}\n"
                ),
            },
            "target_file": "config.rs",
            "target_symbol": "validate",
            "caller_file": "server.rs",
            "caller_symbol": "init",
            "pass_patch": (
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        // Check validity\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str, strict: bool) -> bool {\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "keyword_patch": (
                "mod config;\n\n"
                "fn init(cfg: &config::Config) -> bool {\n"
                "    cfg.validate(\"prod\", true, false)\n"
                "}\n"
            ),
            "circular_patch": (
                "mod server;\n\n"
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "deleted_patch": (
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        if env == \"prod\" { false } else { true }\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "use std::process::Command;\n\n"
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        let _ = Command::new(\"sh\").arg(\"-c\").arg(format!(\"echo {}\", env)).output();\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "static mut VALIDATED_COUNT: u32 = 0;\n\n"
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        unsafe {\n"
                "            VALIDATED_COUNT += 1;\n"
                "        }\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        let mut sum = 0;\n"
                "        for i in 0..200 {\n"
                "            for j in 0..10 {\n"
                "                sum += i + j;\n"
                "            }\n"
                "        }\n"
                "        let _ = sum;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "pub struct Config {\n"
                "    pub port: u16,\n"
                "}\n\n"
                "impl Config {\n"
                "    pub fn validate(&self, env: &str) -> bool {\n"
                "        false\n"
                "    }\n"
                "}\n"
            ),
        },
        {
            "name": "rust_concurrency_service",
            "files": {
                "sync_queue.rs": (
                    "pub struct TaskQueue {\n"
                    "    pub count: usize,\n"
                    "}\n\n"
                    "impl TaskQueue {\n"
                    "    pub fn push(&mut self, val: i32) -> bool {\n"
                    "        self.count += 1;\n"
                    "        let _ = val;\n"
                    "        true\n"
                    "    }\n"
                    "}\n"
                ),
                "pipeline.rs": (
                    "mod sync_queue;\n"
                    "use sync_queue::TaskQueue;\n\n"
                    "pub fn submit(q: &mut TaskQueue, task: i32) -> bool {\n"
                    "    q.push(task)\n"
                    "}\n"
                ),
            },
            "target_file": "sync_queue.rs",
            "target_symbol": "push",
            "caller_file": "pipeline.rs",
            "caller_symbol": "submit",
            "pass_patch": (
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        // Enqueue task safely\n"
                "        self.count += 1;\n"
                "        let _ = val;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32, priority: bool) -> bool {\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "keyword_patch": (
                "mod sync_queue;\n"
                "use sync_queue::TaskQueue;\n\n"
                "pub fn submit(q: &mut TaskQueue, task: i32) -> bool {\n"
                "    q.push(task, true, false)\n"
                "}\n"
            ),
            "circular_patch": (
                "mod pipeline;\n\n"
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        pipeline::submit(self, val);\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "deleted_patch": (
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        if val < 0 {\n"
                "            return false;\n"
                "        }\n"
                "        self.count += 1;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "use std::process::Command;\n\n"
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        let _ = Command::new(\"echo\").arg(\"queue\").output();\n"
                "        self.count += 1;\n"
                "        let _ = val;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "static mut GLOBAL_QUEUE_SIZE: usize = 0;\n\n"
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        unsafe {\n"
                "            GLOBAL_QUEUE_SIZE += 1;\n"
                "        }\n"
                "        self.count += 1;\n"
                "        let _ = val;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        for _ in 0..500 {\n"
                "            std::hint::black_box(1);\n"
                "        }\n"
                "        self.count += 1;\n"
                "        let _ = val;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        false\n"
                "    }\n"
                "}\n"
            ),
            "rust_concurrency_hazard_patch": (
                "static mut SHARED_LOCK_ID: u32 = 0;\n\n"
                "pub struct TaskQueue {\n"
                "    pub count: usize,\n"
                "}\n\n"
                "impl TaskQueue {\n"
                "    pub fn push(&mut self, val: i32) -> bool {\n"
                "        unsafe {\n"
                "            SHARED_LOCK_ID = SHARED_LOCK_ID.wrapping_add(1);\n"
                "        }\n"
                "        self.count += 1;\n"
                "        let _ = val;\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
        },
        {
            "name": "rust_resource_service",
            "files": {
                "buffer.rs": (
                    "pub struct DataBuffer {\n"
                    "    pub size: usize,\n"
                    "}\n\n"
                    "impl DataBuffer {\n"
                    "    pub fn allocate(size: usize) -> Self {\n"
                    "        DataBuffer { size }\n"
                    "    }\n"
                    "}\n"
                ),
                "writer.rs": (
                    "mod buffer;\n"
                    "use buffer::DataBuffer;\n\n"
                    "pub fn init_buffer(cap: usize) -> DataBuffer {\n"
                    "    DataBuffer::allocate(cap)\n"
                    "}\n"
                ),
            },
            "target_file": "buffer.rs",
            "target_symbol": "allocate",
            "caller_file": "writer.rs",
            "caller_symbol": "init_buffer",
            "pass_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        // Safe buffer constructor\n"
                "        DataBuffer { size }\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize, tag: &str) -> Self {\n"
                "        DataBuffer { size }\n"
                "    }\n"
                "}\n"
            ),
            "keyword_patch": (
                "mod buffer;\n"
                "use buffer::DataBuffer;\n\n"
                "pub fn init_buffer(cap: usize) -> DataBuffer {\n"
                "    DataBuffer::allocate(cap, \"raw\", 100)\n"
                "}\n"
            ),
            "circular_patch": (
                "mod writer;\n\n"
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        writer::init_buffer(size)\n"
                "    }\n"
                "}\n"
            ),
            "deleted_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        if size > 1000 {\n"
                "            return DataBuffer { size: 1000 };\n"
                "        }\n"
                "        DataBuffer { size }\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        let _raw = Box::into_raw(Box::new(size));\n"
                "        DataBuffer { size }\n"
                "    }\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "static mut BUF_ALLOC_COUNT: usize = 0;\n\n"
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        unsafe { BUF_ALLOC_COUNT += 1; }\n"
                "        DataBuffer { size }\n"
                "    }\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        let leaked = Box::leak(Box::new(vec![0u8; 1000]));\n"
                "        let _ = leaked;\n"
                "        DataBuffer { size }\n"
                "    }\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        DataBuffer { size: 0 }\n"
                "    }\n"
                "}\n"
            ),
            "rust_resource_leak_patch": (
                "pub struct DataBuffer {\n"
                "    pub size: usize,\n"
                "}\n\n"
                "impl DataBuffer {\n"
                "    pub fn allocate(size: usize) -> Self {\n"
                "        let leaked = Box::leak(Box::new(size));\n"
                "        DataBuffer { size: *leaked }\n"
                "    }\n"
                "}\n"
            ),
        },
        {
            "name": "rust_truthiness_service",
            "files": {
                "validator.rs": (
                    "pub struct Session {\n"
                    "    pub token: String,\n"
                    "    pub is_valid: bool,\n"
                    "}\n\n"
                    "impl Session {\n"
                    "    pub fn authenticate(&self) -> bool {\n"
                    "        if self.token.is_empty() {\n"
                    "            return false;\n"
                    "        }\n"
                    "        self.is_valid\n"
                    "    }\n"
                    "}\n"
                ),
                "checker.rs": (
                    "mod validator;\n"
                    "use validator::Session;\n\n"
                    "pub fn verify_session(s: &Session) -> bool {\n"
                    "    s.authenticate()\n"
                    "}\n"
                ),
            },
            "target_file": "validator.rs",
            "target_symbol": "authenticate",
            "caller_file": "checker.rs",
            "caller_symbol": "verify_session",
            "pass_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        // Verifies session validity\n"
                "        if self.token.is_empty() {\n"
                "            return false;\n"
                "        }\n"
                "        self.is_valid\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self, scope: &str) -> bool {\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "keyword_patch": (
                "mod validator;\n"
                "use validator::Session;\n\n"
                "pub fn verify_session(s: &Session) -> bool {\n"
                "    s.authenticate(\"admin\", \"read\")\n"
                "}\n"
            ),
            "circular_patch": (
                "mod checker;\n\n"
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        checker::verify_session(self)\n"
                "    }\n"
                "}\n"
            ),
            "deleted_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        if self.token.is_empty() {\n"
                "            return true; // inverted truthiness\n"
                "        }\n"
                "        self.is_valid\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "static mut AUTH_CHECKS: usize = 0;\n\n"
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        unsafe { AUTH_CHECKS += 1; }\n"
                "        self.is_valid\n"
                "    }\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        let mut sum = 0;\n"
                "        for i in 0..500 {\n"
                "            sum += i;\n"
                "        }\n"
                "        let _ = sum;\n"
                "        self.is_valid\n"
                "    }\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        false\n"
                "    }\n"
                "}\n"
            ),
            "rust_truthiness_drift_patch": (
                "pub struct Session {\n"
                "    pub token: String,\n"
                "    pub is_valid: bool,\n"
                "}\n\n"
                "impl Session {\n"
                "    pub fn authenticate(&self) -> bool {\n"
                "        if !self.token.is_empty() {\n"
                "            return false;\n"
                "        }\n"
                "        !self.is_valid\n"
                "    }\n"
                "}\n"
            ),
        },
        {
            "name": "rust_api_service",
            "files": {
                "api_client.rs": (
                    "pub struct Client {\n"
                    "    pub host: String,\n"
                    "}\n\n"
                    "impl Client {\n"
                    "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                    "        !endpoint.is_empty() && !self.host.is_empty()\n"
                    "    }\n"
                    "}\n"
                ),
                "app.rs": (
                    "mod api_client;\n"
                    "use api_client::Client;\n\n"
                    "pub fn execute_request(c: &Client) -> bool {\n"
                    "    c.dispatch(\"/health\")\n"
                    "}\n"
                ),
            },
            "target_file": "api_client.rs",
            "target_symbol": "dispatch",
            "caller_file": "app.rs",
            "caller_symbol": "execute_request",
            "pass_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        // Dispatches HTTP request to endpoint\n"
                "        !endpoint.is_empty() && !self.host.is_empty()\n"
                "    }\n"
                "}\n"
            ),
            "arity_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str, retries: u32) -> bool {\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "keyword_patch": (
                "mod api_client;\n"
                "use api_client::Client;\n\n"
                "pub fn execute_request(c: &Client) -> bool {\n"
                "    c.dispatch(\"/health\", 3, 5)\n"
                "}\n"
            ),
            "circular_patch": (
                "mod app;\n\n"
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        let _ = endpoint;\n"
                "        app::execute_request(self)\n"
                "    }\n"
                "}\n"
            ),
            "deleted_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n"
            ),
            "silent_logic_drift_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        if endpoint == \"/health\" {\n"
                "            return false;\n"
                "        }\n"
                "        !endpoint.is_empty() && !self.host.is_empty()\n"
                "    }\n"
                "}\n"
            ),
            "security_surface_patch": (
                "use std::process::Command;\n\n"
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        let _ = Command::new(\"curl\").arg(endpoint).output();\n"
                "        true\n"
                "    }\n"
                "}\n"
            ),
            "concurrency_hazard_patch": (
                "static mut REQ_COUNT: usize = 0;\n\n"
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        unsafe { REQ_COUNT += 1; }\n"
                "        !endpoint.is_empty()\n"
                "    }\n"
                "}\n"
            ),
            "performance_regression_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        for _ in 0..400 {\n"
                "            std::hint::black_box(endpoint);\n"
                "        }\n"
                "        !endpoint.is_empty()\n"
                "    }\n"
                "}\n"
            ),
            "breaking_public_api_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        false\n"
                "    }\n"
                "}\n"
            ),
            "rust_api_drift_patch": (
                "pub struct Client {\n"
                "    pub host: String,\n"
                "}\n\n"
                "impl Client {\n"
                "    pub fn dispatch(&self, endpoint: &str) -> bool {\n"
                "        let _ = endpoint;\n"
                "        false\n"
                "    }\n"
                "}\n"
            ),
        },
    ],
}


TARGETED_MUTATIONS_MAP: Dict[str, Tuple[str, str, str]] = {
    # TypeScript Targeted Subtle Mutations (Langkah 3)
    # a) Type widening & any escape
    "ts_type_widening_service": ("ts_type_widening_patch", "breaking_public_api", "BreakingPublicAPI"),
    # b) Unchecked optional chaining drift
    "ts_optional_chaining_service": ("ts_optional_chaining_patch", "silent_logic_drift", "SilentLogicDrift"),
    # c) Promise/async unhandled floating rejection
    "ts_floating_promise_service": ("ts_floating_promise_patch", "concurrency_hazard", "ConcurrencyHazard"),
    # d) Object property deletion / dynamic destructuring alteration
    "ts_destructuring_service": ("ts_destructuring_patch", "breaking_public_api", "BreakingPublicAPI"),
    # Python Targeted Subtle Mutations (Langkah 3)
    # a) Keyword argument drift (**kwargs and parameter renaming/omission)
    "py_kwargs_service": ("py_kwargs_drift_patch", "breaking_public_api", "BreakingPublicAPI"),
    # b) Mutable default arguments & dictionary mutation drift
    "py_mutable_default_service": ("py_mutable_default_patch", "performance_regression", "PerformanceRegression"),
    # c) Truthiness and silent logic drift (is None vs truthy check)
    "py_truthiness_service": ("py_truthiness_drift_patch", "silent_logic_drift", "SilentLogicDrift"),
    # d) Revert-mimicking subtle patches (inverts security sanitization hotfix)
    "py_revert_mimic_service": ("py_revert_mimic_patch", "real_revert", "SecuritySurface"),
    # Go Targeted Subtle Mutations (5 Idiomatic Semantic Mutation Generators)
    # a) Ignored / Shadowed Error
    "go_ignored_error_service": ("go_ignored_error_patch", "silent_logic_drift", "SilentLogicDrift"),
    # b) Inverted Defer Order
    "go_inverted_defer_service": ("go_inverted_defer_patch", "performance_regression", "PerformanceRegression"),
    # c) Pointer vs Value Receiver Drift
    "go_receiver_drift_service": ("go_receiver_drift_patch", "silent_logic_drift", "SilentLogicDrift"),
    # d) Goroutine & Unbuffered Channel Leak/Deadlock
    "go_channel_leak_service": ("go_channel_leak_patch", "concurrency_hazard", "ConcurrencyHazard"),
    # e) Mutex Unlock Omission on early branch return
    "go_mutex_unlock_service": ("go_mutex_unlock_patch", "concurrency_hazard", "ConcurrencyHazard"),
    # Rust Targeted Subtle Mutations
    # a) Mutex / lock ordering concurrency hazard
    "rust_concurrency_service": ("rust_concurrency_hazard_patch", "concurrency_hazard", "ConcurrencyHazard"),
    # b) std::mem::forget / resource leak
    "rust_resource_service": ("rust_resource_leak_patch", "performance_regression", "PerformanceRegression"),
    # c) Option / boolean truthiness drift
    "rust_truthiness_service": ("rust_truthiness_drift_patch", "silent_logic_drift", "SilentLogicDrift"),
    # d) Public trait method / visibility breaking drift
    "rust_api_service": ("rust_api_drift_patch", "breaking_public_api", "BreakingPublicAPI"),
}


class DatasetGenerator:
    """
    Synthesizes and mines balanced datasets of Micro-DSL representations
    paired with binary labels and risk scores for ModernBERT decision heads.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        seed: int = 42,
        positive_label: int = 1,
        negative_label: int = 0,
        filter_symbolic_gate: bool = False,
    ):
        self.languages = [l.lower() for l in (languages or ["python", "typescript", "go", "rust"])]
        self.seed = seed
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.filter_symbolic_gate = filter_symbolic_gate
        random.seed(seed)

    def generate_pairs_for_template(
        self,
        template: Dict[str, Any],
        language: str,
        include_subtle: bool = False,
        variation_idx: int = 0,
    ) -> List[DatasetRecord]:
        """
        Instantiate a multi-file template in an isolated temporary workspace,
        execute positive and negative mutations through TopoSliceEngine,
        and generate structured DatasetRecord objects.
        """
        records: List[DatasetRecord] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            comment_tok = "//" if language in ("typescript", "javascript", "go", "rust") else "#"
            v_comment = f"{comment_tok} variation #{variation_idx}\n" if variation_idx > 0 else ""

            for fname, fcontent in template["files"].items():
                fpath = ws_root / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                file_text = v_comment + fcontent if v_comment else fcontent
                fpath.write_text(file_text, encoding="utf-8")

            engine = TopoSliceEngine(workspace_root=ws_root)
            engine.indexer.scan_workspace()

            # 1. Positive: Clean pass (label=1, risk=0.0-0.2)
            pass_patch = v_comment + template["pass_patch"] if v_comment else template["pass_patch"]
            pass_rep = engine.verify(template["target_file"], pass_patch)
            if pass_rep.status == "APPROVED":
                records.append(
                    DatasetRecord(
                        input_dsl=pass_rep.linearized_subgraph,
                        label=self.positive_label,
                        risk_score=random.uniform(0.02, 0.15),
                        category="clean_pass",
                        language=language,
                        taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                        symbolic_gate_passed=True,
                        source_type="clean_commit",
                    )
                )

            # Structural mutations (only included if not filtering by symbolic gate)
            if not self.filter_symbolic_gate:
                # 2. Negative: Arity breaking (label=0, risk=0.8-1.0)
                arity_rep = engine.verify(template["target_file"], template["arity_patch"])
                if arity_rep.status == "REJECTED":
                    records.append(
                        DatasetRecord(
                            input_dsl=arity_rep.linearized_subgraph,
                            label=self.negative_label,
                            risk_score=random.uniform(0.85, 0.98),
                            category="arity_breaking",
                            language=language,
                            taxonomy_labels={"BreakingPublicAPI": 0.95, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                            symbolic_gate_passed=False,
                            source_type="synthetic",
                        )
                    )

                # 3. Negative: Keyword / Call argument mismatch (label=0, risk=0.8-1.0)
                caller_file = template.get("caller_file", template["target_file"])
                kw_rep = engine.verify(caller_file, template["keyword_patch"])
                if kw_rep.status == "REJECTED":
                    records.append(
                        DatasetRecord(
                            input_dsl=kw_rep.linearized_subgraph,
                            label=self.negative_label,
                            risk_score=random.uniform(0.82, 0.96),
                            category="keyword_changes",
                            language=language,
                            taxonomy_labels={"BreakingPublicAPI": 0.90, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                            symbolic_gate_passed=False,
                            source_type="synthetic",
                        )
                    )

                # 4. Negative: Circular import / call (label=0, risk=0.8-1.0)
                circ_rep = engine.verify(template["target_file"], template["circular_patch"])
                if circ_rep.status == "REJECTED":
                    records.append(
                        DatasetRecord(
                            input_dsl=circ_rep.linearized_subgraph,
                            label=self.negative_label,
                            risk_score=random.uniform(0.88, 0.99),
                            category="circular_imports",
                            language=language,
                            taxonomy_labels={"BreakingPublicAPI": 0.65, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.75},
                            symbolic_gate_passed=False,
                            source_type="synthetic",
                        )
                    )

                # 5. Negative: Deleted symbol with existing callers (label=0, risk=0.8-1.0)
                del_rep = engine.verify(template["target_file"], template["deleted_patch"])
                if del_rep.status == "REJECTED":
                    records.append(
                        DatasetRecord(
                            input_dsl=del_rep.linearized_subgraph,
                            label=self.negative_label,
                            risk_score=random.uniform(0.85, 0.98),
                            category="deleted_symbol",
                            language=language,
                            taxonomy_labels={"BreakingPublicAPI": 0.95, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                            symbolic_gate_passed=False,
                            source_type="synthetic",
                        )
                    )

            # Subtle semantic mutations mapped to ADR-0003 5-class risk taxonomy
            # (Gray-area hard negatives that pass Stage 1 & 2 symbolic gate)
            if include_subtle or self.filter_symbolic_gate:
                tmpl_name = template.get("name", "")
                if tmpl_name in TARGETED_MUTATIONS_MAP:
                    subtle_mutations = [TARGETED_MUTATIONS_MAP[tmpl_name]]
                else:
                    subtle_mutations = [
                        ("silent_logic_drift_patch", "silent_logic_drift", "SilentLogicDrift"),
                        ("security_surface_patch", "security_surface", "SecuritySurface"),
                        ("concurrency_hazard_patch", "concurrency_hazard", "ConcurrencyHazard"),
                        ("performance_regression_patch", "performance_regression", "PerformanceRegression"),
                        ("breaking_public_api_patch", "breaking_public_api", "BreakingPublicAPI"),
                    ]
                seen_patches = set()
                for patch_key, cat_name, tax_class in subtle_mutations:
                    patch_content = template.get(patch_key)
                    if not patch_content or patch_content in seen_patches:
                        continue
                    seen_patches.add(patch_content)
                    patch_to_verify = v_comment + patch_content if v_comment else patch_content
                    mut_rep = engine.verify(template["target_file"], patch_to_verify)
                    gate_passed = (mut_rep.status == "APPROVED")
                    if self.filter_symbolic_gate and not gate_passed:
                        continue
                    tax = {c: 0.0 for c in TAXONOMY_CLASSES}
                    tax[tax_class] = round(random.uniform(0.85, 0.96), 4)
                    records.append(
                        DatasetRecord(
                            input_dsl=mut_rep.linearized_subgraph,
                            label=self.negative_label,
                            risk_score=random.uniform(0.82, 0.95),
                            category=cat_name,
                            language=language,
                            taxonomy_labels=tax,
                            symbolic_gate_passed=gate_passed,
                            source_type="mutation_subtle",
                        )
                    )

        return records

    def generate_subtle_pairs_for_template(
        self,
        template: Dict[str, Any],
        language: str,
    ) -> List[DatasetRecord]:
        """Generate subtle gray-area semantic mutation records that pass symbolic gate."""
        return self.generate_pairs_for_template(template, language, include_subtle=True)

    def generate_targeted_typescript_mutations(
        self,
        count_per_type: int = 1,
    ) -> List[DatasetRecord]:
        """
        Generate targeted subtle mutations for TypeScript covering:
        a) Type widening & any escape (ts_type_widening_service)
        b) Unchecked optional chaining drift (ts_optional_chaining_service)
        c) Promise/async unhandled floating rejection (ts_floating_promise_service)
        d) Object property deletion / dynamic destructuring alteration (ts_destructuring_service)
        100% compliant with Stage 1-2 symbolic gate (symbolic_gate_passed=True).
        """
        ts_templates = [
            t for t in TEMPLATES.get("typescript", [])
            if t["name"] in TARGETED_MUTATIONS_MAP
        ]
        records: List[DatasetRecord] = []
        for i in range(count_per_type):
            for tmpl in ts_templates:
                pairs = self.generate_pairs_for_template(
                    tmpl, "typescript", include_subtle=True, variation_idx=i
                )
                for r in pairs:
                    if r.symbolic_gate_passed:
                        records.append(r)
        return records

    def generate_targeted_python_mutations(
        self,
        count_per_type: int = 1,
    ) -> List[DatasetRecord]:
        """
        Generate targeted subtle mutations for Python covering:
        a) Keyword argument & parameter renaming/omission drift (py_kwargs_service)
        b) Mutable default arguments & dictionary mutation drift (py_mutable_default_service)
        c) Truthiness and silent logic drift (py_truthiness_service)
        d) Revert-mimicking subtle patches (py_revert_mimic_service)
        100% compliant with Stage 1-2 symbolic gate (symbolic_gate_passed=True).
        """
        py_templates = [
            t for t in TEMPLATES.get("python", [])
            if t["name"] in TARGETED_MUTATIONS_MAP
        ]
        records: List[DatasetRecord] = []
        for i in range(count_per_type):
            for tmpl in py_templates:
                pairs = self.generate_pairs_for_template(
                    tmpl, "python", include_subtle=True, variation_idx=i
                )
                for r in pairs:
                    if r.symbolic_gate_passed:
                        records.append(r)
        return records

    def generate_targeted_go_mutations(
        self,
        count_per_type: int = 1,
    ) -> List[DatasetRecord]:
        """
        Generate targeted subtle mutations for Go covering:
        a) Ignored / Shadowed Error (go_ignored_error_service)
        b) Inverted Defer Order (go_inverted_defer_service)
        c) Pointer vs Value Receiver Drift (go_receiver_drift_service)
        d) Goroutine & Unbuffered Channel Leak/Deadlock (go_channel_leak_service)
        e) Mutex Unlock Omission on early branch return (go_mutex_unlock_service)
        100% compliant with Stage 1-2 symbolic gate (symbolic_gate_passed=True).
        """
        go_templates = [
            t for t in TEMPLATES.get("go", [])
            if t["name"] in TARGETED_MUTATIONS_MAP
        ]
        records: List[DatasetRecord] = []
        for i in range(count_per_type):
            for tmpl in go_templates:
                pairs = self.generate_pairs_for_template(
                    tmpl, "go", include_subtle=True, variation_idx=i
                )
                for r in pairs:
                    if r.symbolic_gate_passed:
                        records.append(r)
        return records

    def generate_targeted_rust_mutations(
        self,
        count_per_type: int = 1,
    ) -> List[DatasetRecord]:
        """
        Generate targeted subtle mutations for Rust covering:
        a) Mutex / lock ordering concurrency hazard (rust_concurrency_service)
        b) Memory / resource leak via leaked allocations (rust_resource_service)
        c) Option / boolean truthiness drift (rust_truthiness_service)
        d) Public method / API drift (rust_api_service)
        100% compliant with Stage 1-2 symbolic gate (symbolic_gate_passed=True).
        """
        rust_templates = [
            t for t in TEMPLATES.get("rust", [])
            if t["name"] in TARGETED_MUTATIONS_MAP
        ]
        records: List[DatasetRecord] = []
        for i in range(count_per_type):
            for tmpl in rust_templates:
                pairs = self.generate_pairs_for_template(
                    tmpl, "rust", include_subtle=True, variation_idx=i
                )
                for r in pairs:
                    if r.symbolic_gate_passed:
                        records.append(r)
        return records

    def generate_targeted_mutations(
        self,
        languages: Optional[List[str]] = None,
        count_per_type: int = 1,
    ) -> List[DatasetRecord]:
        """Generate targeted subtle mutations for configured or requested languages."""
        target_langs = [l.lower() for l in (languages or self.languages)]
        records: List[DatasetRecord] = []
        if "typescript" in target_langs:
            records.extend(self.generate_targeted_typescript_mutations(count_per_type=count_per_type))
        if "python" in target_langs:
            records.extend(self.generate_targeted_python_mutations(count_per_type=count_per_type))
        if "go" in target_langs:
            records.extend(self.generate_targeted_go_mutations(count_per_type=count_per_type))
        if "rust" in target_langs:
            records.extend(self.generate_targeted_rust_mutations(count_per_type=count_per_type))
        return records

    def expand_dataset(
        self,
        train_records: List[DatasetRecord],
        val_records: List[DatasetRecord],
        num_ts_samples: int = 400,
        num_py_samples: int = 400,
        num_go_samples: int = 0,
        num_rust_samples: int = 0,
        val_ratio: float = 0.2,
    ) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
        """
        Expand training and validation datasets with targeted subtle mutations across Tier 1 languages.
        Maintains exact 50% PASS / 50% REJECT class balance across both splits and per language,
        and guarantees 100% symbolic gate compliance (symbolic_gate_passed=True).
        """
        ts_templates = [t for t in TEMPLATES.get("typescript", []) if t["name"] in TARGETED_MUTATIONS_MAP]
        py_templates = [t for t in TEMPLATES.get("python", []) if t["name"] in TARGETED_MUTATIONS_MAP]
        go_templates = [t for t in TEMPLATES.get("go", []) if t["name"] in TARGETED_MUTATIONS_MAP]
        rust_templates = [t for t in TEMPLATES.get("rust", []) if t["name"] in TARGETED_MUTATIONS_MAP]

        batch_list = []
        if num_ts_samples > 0 and ts_templates:
            ts_count = max(1, (num_ts_samples // 2) // len(ts_templates))
            batch_list.append(self.generate_targeted_typescript_mutations(count_per_type=ts_count))
        if num_py_samples > 0 and py_templates:
            py_count = max(1, (num_py_samples // 2) // len(py_templates))
            batch_list.append(self.generate_targeted_python_mutations(count_per_type=py_count))
        if num_go_samples > 0 and go_templates:
            go_count = max(1, (num_go_samples // 2) // len(go_templates))
            batch_list.append(self.generate_targeted_go_mutations(count_per_type=go_count))
        if num_rust_samples > 0 and rust_templates:
            rust_count = max(1, (num_rust_samples // 2) // len(rust_templates))
            batch_list.append(self.generate_targeted_rust_mutations(count_per_type=rust_count))

        new_train: List[DatasetRecord] = []
        new_val: List[DatasetRecord] = []

        # Stratified balance per language ensures no language is crowded out
        for lang_recs in batch_list:
            valid = [r for r in lang_recs if r.symbolic_gate_passed]
            pos = [r for r in valid if r.label == self.positive_label]
            neg = [r for r in valid if r.label == self.negative_label]
            min_n = min(len(pos), len(neg))
            pos = pos[:min_n]
            neg = neg[:min_n]

            val_n = max(1, int(min_n * val_ratio))
            new_val.extend(pos[:val_n] + neg[:val_n])
            new_train.extend(pos[val_n:] + neg[val_n:])

        expanded_train = list(train_records) + new_train
        expanded_val = list(val_records) + new_val

        random.shuffle(expanded_train)
        random.shuffle(expanded_val)

        return expanded_train, expanded_val

    def generate_synthetic_dataset(
        self,
        num_samples: int = 100,
        include_subtle: bool = True,
    ) -> List[DatasetRecord]:
        """Generate balanced dataset across configured languages."""
        positives: List[DatasetRecord] = []
        negatives: List[DatasetRecord] = []
        target_pos = max(1, num_samples // 2)
        target_neg = max(1, num_samples - target_pos)

        target_pos_per_lang = max(1, (target_pos + len(self.languages) - 1) // len(self.languages))
        target_neg_per_lang = max(1, (target_neg + len(self.languages) - 1) // len(self.languages))

        for lang in self.languages:
            lang_templates = TEMPLATES.get(lang, [])
            if not lang_templates:
                continue

            lang_pos: List[DatasetRecord] = []
            lang_neg: List[DatasetRecord] = []

            tmpl_idx = 0
            while len(lang_pos) < target_pos_per_lang or len(lang_neg) < target_neg_per_lang:
                tmpl = lang_templates[tmpl_idx % len(lang_templates)]
                pairs = self.generate_pairs_for_template(tmpl, lang, include_subtle=include_subtle)
                for r in pairs:
                    if self.filter_symbolic_gate and not r.symbolic_gate_passed:
                        continue
                    if r.label == self.positive_label and len(lang_pos) < target_pos_per_lang:
                        lang_pos.append(r)
                    elif r.label == self.negative_label and len(lang_neg) < target_neg_per_lang:
                        lang_neg.append(r)
                tmpl_idx += 1
                max_iterations = max(50, (target_pos_per_lang + target_neg_per_lang) * 2)
                if tmpl_idx > max_iterations:
                    break

            positives.extend(lang_pos)
            negatives.extend(lang_neg)

        combined = positives[:target_pos] + negatives[:target_neg]
        random.shuffle(combined)
        return combined

    def mine_git_history(
        self,
        repo_path: Path,
        max_samples: int = 20,
        filter_symbolic_gate: Optional[bool] = None,
    ) -> List[DatasetRecord]:
        """
        Extract real-world commit pairs (revert commits, hotfix commits, clean merged commits)
        from git history, verifying each diff through TopoSliceEngine.
        """
        repo_path = repo_path.resolve()
        if not (repo_path / ".git").exists():
            return []

        should_filter = self.filter_symbolic_gate if filter_symbolic_gate is None else filter_symbolic_gate
        records: List[DatasetRecord] = []
        engine = TopoSliceEngine(workspace_root=repo_path)
        engine.indexer.scan_workspace()

        # 1. Mine Revert Commits (real bugs that were reverted)
        try:
            cmd = ["git", "log", "--grep=revert", "-i", "-n", "150", "--format=%H|%s"]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    if "|" not in line:
                        continue
                    rev_hash, subject = line.split("|", 1)
                    if not re.search(r"(?i)\b(revert|reverted|reverting)\b|^revert\b", subject):
                        continue

                    # Retrieve full commit message body to find referenced buggy commit hash
                    b_msg_cmd = ["git", "show", "-s", "--format=%B", rev_hash]
                    b_msg_res = subprocess.run(b_msg_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    commit_body = b_msg_res.stdout if b_msg_res.returncode == 0 else ""

                    rev_match = re.search(
                        r"(?i)(?:this reverts commit|reverts commit|reverting commit)\s+([0-9a-f]{7,40})",
                        commit_body,
                    )
                    bug_hash = rev_match.group(1) if rev_match else None

                    # If bug_hash found, enrich context with original bug commit body
                    if bug_hash:
                        b_ctx_cmd = ["git", "show", "-s", "--format=%B", bug_hash]
                        b_ctx_res = subprocess.run(b_ctx_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if b_ctx_res.returncode == 0 and b_ctx_res.stdout.strip():
                            commit_body = f"{commit_body}\n{b_ctx_res.stdout}"

                    # If bug_hash found, resolve original buggy commit diffs with base content
                    files_to_check: List[Tuple[str, str, Optional[str]]] = []  # (fname, diff_text, orig_content)
                    if bug_hash:
                        f_cmd = ["git", "show", "--name-only", "--format=", bug_hash]
                        f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if f_res.returncode == 0:
                            for fn in f_res.stdout.splitlines():
                                fn = fn.strip()
                                if fn:
                                    d_cmd = ["git", "show", "-p", bug_hash, "--", fn]
                                    d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                    if d_res.returncode == 0 and d_res.stdout.strip():
                                        o_cmd = ["git", "show", f"{bug_hash}~1:{fn}"]
                                        o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                        orig_c = o_res.stdout if o_res.returncode == 0 else None
                                        files_to_check.append((fn, d_res.stdout, orig_c))

                    # If original buggy diff was not resolvable via bug_hash, use reverse diff of revert commit
                    if not files_to_check:
                        f_cmd = ["git", "show", "--name-only", "--format=", rev_hash]
                        f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if f_res.returncode == 0:
                            for fn in f_res.stdout.splitlines():
                                fn = fn.strip()
                                if fn:
                                    d_cmd = ["git", "show", "-R", "-p", rev_hash, "--", fn]
                                    d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                    if d_res.returncode == 0 and d_res.stdout.strip():
                                        o_cmd = ["git", "show", f"{rev_hash}:{fn}"]
                                        o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                                        orig_c = o_res.stdout if o_res.returncode == 0 else None
                                        files_to_check.append((fn, d_res.stdout, orig_c))

                    for fname, diff_text, orig_content in files_to_check:
                        lang = detect_language(fname)
                        if not lang or lang not in self.languages:
                            continue
                        try:
                            rep = engine.verify(fname, diff_text, original_content=orig_content)
                            gate_passed = (rep.status == "APPROVED")
                            if should_filter and not gate_passed:
                                continue
                            tax = DatasetRecord.assign_taxonomy_labels(
                                category="real_revert",
                                label=self.negative_label,
                                risk_score=0.88,
                                context_text=f"{subject}\n{commit_body}",
                                invariant_violations=rep.invariant_violations,
                            )
                            records.append(
                                DatasetRecord(
                                    input_dsl=rep.linearized_subgraph,
                                    label=self.negative_label,
                                    risk_score=random.uniform(0.85, 0.95),
                                    category="real_revert",
                                    language=lang,
                                    taxonomy_labels=tax,
                                    symbolic_gate_passed=gate_passed,
                                    source_type="real_revert",
                                )
                            )
                            if len(records) >= max(1, max_samples // 3):
                                break
                        except Exception:
                            continue
                    if len(records) >= max(1, max_samples // 3):
                        break
        except Exception:
            pass

        # 2. Mine Hotfix Commits paired with pre-fix buggy state
        try:
            cmd = ["git", "log", "--grep=fix", "-i", "-n", "150", "--format=%H|%s"]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    if "|" not in line or "revert" in line.lower():
                        continue
                    fix_hash, subject = line.split("|", 1)
                    # Filter out false positives like 'prefix', 'fixture', etc.
                    if not re.search(r"(?i)\b(fix|hotfix|bugfix|patch)\b|^(fix|hotfix|bugfix)(\(.*\))?:", subject):
                        continue
                    if re.search(r"(?i)\b(fixture|fixtures|prefix|postfix|suffix)\b", subject) and not re.search(r"(?i)\b(bug|hotfix|fix:)\b", subject):
                        continue

                    # Retrieve full commit message body for taxonomy assignment
                    b_msg_cmd = ["git", "show", "-s", "--format=%B", fix_hash]
                    b_msg_res = subprocess.run(b_msg_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    commit_body = b_msg_res.stdout if b_msg_res.returncode == 0 else ""

                    f_cmd = ["git", "show", "--name-only", "--format=", fix_hash]
                    f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    if f_res.returncode != 0:
                        continue
                    files = [f.strip() for f in f_res.stdout.splitlines() if f.strip()]
                    for fname in files:
                        lang = detect_language(fname)
                        if not lang or lang not in self.languages:
                            continue

                        # a) Positive sample: The hotfix patch itself
                        d_cmd = ["git", "show", "-p", fix_hash, "--", fname]
                        d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if d_res.returncode == 0 and d_res.stdout.strip():
                            diff_text = d_res.stdout
                            o_cmd = ["git", "show", f"{fix_hash}~1:{fname}"]
                            o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                            orig_c = o_res.stdout if o_res.returncode == 0 else None
                            try:
                                rep_fix = engine.verify(fname, diff_text, original_content=orig_c)
                                gate_passed_fix = (rep_fix.status == "APPROVED")
                                if not should_filter or gate_passed_fix:
                                    records.append(
                                        DatasetRecord(
                                            input_dsl=rep_fix.linearized_subgraph,
                                            label=self.positive_label,
                                            risk_score=random.uniform(0.02, 0.15),
                                            category="real_hotfix",
                                            language=lang,
                                            taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                                            symbolic_gate_passed=gate_passed_fix,
                                            source_type="real_hotfix",
                                        )
                                    )
                            except Exception:
                                pass

                        # b) Paired negative sample: Pre-fix buggy state (reverse of fix patch)
                        d_bug_cmd = ["git", "show", "-R", "-p", fix_hash, "--", fname]
                        d_bug_res = subprocess.run(d_bug_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if d_bug_res.returncode == 0 and d_bug_res.stdout.strip():
                            bug_diff_text = d_bug_res.stdout
                            o_bug_cmd = ["git", "show", f"{fix_hash}:{fname}"]
                            o_bug_res = subprocess.run(o_bug_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                            orig_bug_c = o_bug_res.stdout if o_bug_res.returncode == 0 else None
                            try:
                                rep_bug = engine.verify(fname, bug_diff_text, original_content=orig_bug_c)
                                gate_passed_bug = (rep_bug.status == "APPROVED")
                                if not should_filter or gate_passed_bug:
                                    tax_bug = DatasetRecord.assign_taxonomy_labels(
                                        category="real_revert",
                                        label=self.negative_label,
                                        risk_score=0.88,
                                        context_text=f"{subject}\n{commit_body}",
                                        invariant_violations=rep_bug.invariant_violations,
                                    )
                                    records.append(
                                        DatasetRecord(
                                            input_dsl=rep_bug.linearized_subgraph,
                                            label=self.negative_label,
                                            risk_score=random.uniform(0.85, 0.95),
                                            category="real_revert",
                                            language=lang,
                                            taxonomy_labels=tax_bug,
                                            symbolic_gate_passed=gate_passed_bug,
                                            source_type="real_hotfix",
                                        )
                                    )
                            except Exception:
                                pass

                        if len(records) >= max(2, (max_samples * 2) // 3):
                            break
                    if len(records) >= max(2, (max_samples * 2) // 3):
                        break
        except Exception:
            pass

        # 3. Mine Clean Merged Commits
        try:
            cmd = ["git", "log", "--no-merges", "-n", "150", "--format=%H|%s"]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    if "|" not in line:
                        continue
                    c_hash, subject = line.split("|", 1)
                    if re.search(r"(?i)\b(fix|hotfix|bugfix|patch|revert|bug)\b|^(fix|hotfix)(\(.*\))?:", subject):
                        continue
                    f_cmd = ["git", "show", "--name-only", "--format=", c_hash]
                    f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    if f_res.returncode != 0:
                        continue
                    files = [f.strip() for f in f_res.stdout.splitlines() if f.strip()]
                    for fname in files:
                        lang = detect_language(fname)
                        if not lang or lang not in self.languages:
                            continue
                        d_cmd = ["git", "show", "-p", c_hash, "--", fname]
                        d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if d_res.returncode != 0 or not d_res.stdout.strip():
                            continue
                        diff_text = d_res.stdout
                        o_cmd = ["git", "show", f"{c_hash}~1:{fname}"]
                        o_res = subprocess.run(o_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        orig_c = o_res.stdout if o_res.returncode == 0 else None
                        try:
                            rep = engine.verify(fname, diff_text, original_content=orig_c)
                            if rep.status == "APPROVED":
                                records.append(
                                    DatasetRecord(
                                        input_dsl=rep.linearized_subgraph,
                                        label=self.positive_label,
                                        risk_score=random.uniform(0.01, 0.10),
                                        category="clean_pass",
                                        language=lang,
                                        taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                                        symbolic_gate_passed=True,
                                        source_type="clean_commit",
                                    )
                                )
                                if len(records) >= max_samples:
                                    break
                        except Exception:
                            continue
                    if len(records) >= max_samples:
                        break
        except Exception:
            pass

        return records[:max_samples]

    def mine_repository(
        self,
        repo_path: Path,
        max_samples: int = 50,
        mine_git: bool = True,
        filter_symbolic_gate: Optional[bool] = None,
    ) -> List[DatasetRecord]:
        """
        Mine an existing codebase in Python, TypeScript, Go, or Rust,
        discovering functions with callers to generate authentic mutation pairs
        and real-world git commit pairs (revert commits, hotfix commits, clean commits).
        """
        repo_path = repo_path.resolve()
        filter_gate = self.filter_symbolic_gate if filter_symbolic_gate is None else filter_symbolic_gate
        mined_records: List[DatasetRecord] = []

        # Mine git commits if available
        if mine_git and (repo_path / ".git").exists():
            git_samples = self.mine_git_history(
                repo_path,
                max_samples=max_samples // 2,
                filter_symbolic_gate=filter_gate,
            )
            mined_records.extend(git_samples)

        engine = TopoSliceEngine(workspace_root=repo_path)
        engine.indexer.scan_workspace()

        # Find symbols that have callers across workspace
        for sym_id, sym in list(engine.indexer._definitions.items()):
            if sym.kind not in ("function", "method") or not sym.file_path:
                continue

            callers = engine.indexer.get_callers(sym.qualname)
            if not callers:
                continue

            lang = detect_language(sym.file_path)
            if not lang or lang not in self.languages:
                continue

            # Read file content
            fpath = repo_path / sym.file_path
            if not fpath.exists():
                continue

            try:
                orig_content = fpath.read_text(encoding="utf-8")
            except Exception:
                continue

            lines = orig_content.splitlines(keepends=True)
            if sym.lineno > len(lines):
                continue

            # 1. Clean pass mutation: add harmless comments/docstrings inside symbol body
            target_idx = sym.lineno
            if lang == "python":
                for idx in range(sym.lineno - 1, min(len(lines), sym.lineno + 10)):
                    l_strip = lines[idx].strip()
                    if ":" in l_strip and not l_strip.startswith("#"):
                        target_idx = idx + 1
                        break
            else:
                for idx in range(sym.lineno - 1, min(len(lines), sym.lineno + 10)):
                    if "{" in lines[idx]:
                        target_idx = idx + 1
                        break

            comment_token = "#" if lang == "python" else "//"
            mutated_pass = (
                "".join(lines[:target_idx])
                + f"    {comment_token} topocache-clean-pass\n"
                + "".join(lines[target_idx:])
            )
            pass_rep = engine.verify(sym.file_path, mutated_pass)
            if pass_rep.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=pass_rep.linearized_subgraph,
                        label=self.positive_label,
                        risk_score=random.uniform(0.01, 0.15),
                        category="clean_pass",
                        language=lang,
                        taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                        symbolic_gate_passed=True,
                        source_type="clean_commit",
                    )
                )

            # Additional clean pass variant: docstring / second comment
            mutated_pass2 = (
                "".join(lines[:target_idx])
                + f"    {comment_token} verified clean invariant\n"
                + "".join(lines[target_idx:])
            )
            pass_rep2 = engine.verify(sym.file_path, mutated_pass2)
            if pass_rep2.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=pass_rep2.linearized_subgraph,
                        label=self.positive_label,
                        risk_score=random.uniform(0.01, 0.15),
                        category="clean_pass",
                        language=lang,
                        taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                        symbolic_gate_passed=True,
                        source_type="clean_commit",
                    )
                )

            # Structural mutations (only if not filter_gate)
            if not filter_gate:
                # 2. Arity breaking mutation: add a required parameter breaking existing callers
                def_line = lines[sym.lineno - 1]
                new_def_line = None
                if lang == "python" and f"def {sym.name}(" in def_line:
                    if "(self, " in def_line:
                        new_def_line = def_line.replace("(self, ", "(self, required_break_arg: int, ", 1)
                    elif "(self)" in def_line:
                        new_def_line = def_line.replace("(self)", "(self, required_break_arg: int)", 1)
                    elif "(cls, " in def_line:
                        new_def_line = def_line.replace("(cls, ", "(cls, required_break_arg: int, ", 1)
                    elif "(cls)" in def_line:
                        new_def_line = def_line.replace("(cls)", "(cls, required_break_arg: int)", 1)
                    else:
                        new_def_line = def_line.replace(f"def {sym.name}(", f"def {sym.name}(required_break_arg: int, ", 1)
                elif lang in ("typescript", "javascript") and f"{sym.name}(" in def_line:
                    new_def_line = def_line.replace(f"{sym.name}(", f"{sym.name}(requiredBreakArg: number, ", 1)
                elif lang == "go" and f"{sym.name}(" in def_line:
                    new_def_line = def_line.replace(f"{sym.name}(", f"{sym.name}(requiredBreakArg int, ", 1)
                elif lang == "rust" and f"fn {sym.name}(" in def_line:
                    if "&self," in def_line:
                        new_def_line = def_line.replace("&self,", "&self, required_break_arg: i32,", 1)
                    elif "&self)" in def_line:
                        new_def_line = def_line.replace("&self)", "&self, required_break_arg: i32)", 1)
                    elif "&mut self," in def_line:
                        new_def_line = def_line.replace("&mut self,", "&mut self, required_break_arg: i32,", 1)
                    elif "&mut self)" in def_line:
                        new_def_line = def_line.replace("&mut self)", "&mut self, required_break_arg: i32)", 1)
                    else:
                        new_def_line = def_line.replace(f"fn {sym.name}(", f"fn {sym.name}(required_break_arg: i32, ", 1)

                if new_def_line and new_def_line != def_line:
                    mutated_arity = "".join(lines[: sym.lineno - 1]) + new_def_line + "".join(lines[sym.lineno :])
                    arity_rep = engine.verify(sym.file_path, mutated_arity)
                    if arity_rep.status == "REJECTED":
                        mined_records.append(
                            DatasetRecord(
                                input_dsl=arity_rep.linearized_subgraph,
                                label=self.negative_label,
                                risk_score=random.uniform(0.85, 0.98),
                                category="arity_breaking",
                                language=lang,
                                taxonomy_labels={"BreakingPublicAPI": 0.95, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                                symbolic_gate_passed=False,
                                source_type="synthetic",
                            )
                        )

                # 3. Keyword / Call Argument Mismatch mutation
                if lang == "python" and f"def {sym.name}(" in def_line:
                    new_kw_line = def_line.replace(f"def {sym.name}(", f"def {sym.name}(*, required_kw_arg: int, ", 1)
                    mutated_kw = "".join(lines[: sym.lineno - 1]) + new_kw_line + "".join(lines[sym.lineno :])
                    kw_rep = engine.verify(sym.file_path, mutated_kw)
                    if kw_rep.status == "REJECTED":
                        mined_records.append(
                            DatasetRecord(
                                input_dsl=kw_rep.linearized_subgraph,
                                label=self.negative_label,
                                risk_score=random.uniform(0.82, 0.96),
                                category="keyword_changes",
                                language=lang,
                                taxonomy_labels={"BreakingPublicAPI": 0.90, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                                symbolic_gate_passed=False,
                                source_type="synthetic",
                            )
                        )

                for c in callers[:3]:
                    if not (c.caller and "::" in c.caller):
                        continue
                    c_file, _ = c.caller.split("::", 1)
                    c_path = repo_path / c_file
                    if not c_path.exists():
                        continue
                    try:
                        c_lines = c_path.read_text(encoding="utf-8").splitlines(keepends=True)
                    except Exception:
                        continue
                    if c.lineno <= len(c_lines):
                        c_line = c_lines[c.lineno - 1]
                        if f"{sym.name}(" in c_line:
                            if lang == "python":
                                new_c_line = c_line.replace(f"{sym.name}(", f"{sym.name}(unexpected_kw_arg=True, ", 1)
                            elif lang in ("typescript", "javascript"):
                                new_c_line = c_line.replace(f"{sym.name}(", f"{sym.name}('unexpected_arg', 99999, ", 1)
                            else:
                                new_c_line = c_line.replace(f"{sym.name}(", f"{sym.name}(99999, 88888, ", 1)
                            mut_caller = "".join(c_lines[: c.lineno - 1]) + new_c_line + "".join(c_lines[c.lineno :])
                            kw_caller_rep = engine.verify(c_file, mut_caller)
                            if kw_caller_rep.status == "REJECTED":
                                mined_records.append(
                                    DatasetRecord(
                                        input_dsl=kw_caller_rep.linearized_subgraph,
                                        label=self.negative_label,
                                        risk_score=random.uniform(0.82, 0.96),
                                        category="keyword_changes",
                                        language=lang,
                                        taxonomy_labels={"BreakingPublicAPI": 0.90, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                                        symbolic_gate_passed=False,
                                        source_type="synthetic",
                                    )
                                )
                                break

                # 4. Circular Import / Call Mutation
                for c in callers[:3]:
                    if not (c.caller and "::" in c.caller):
                        continue
                    c_file, c_sym = c.caller.split("::", 1)
                    c_name = c_sym.split(".")[-1]

                    if c_name and c_name.isidentifier() and c_name != sym.name:
                        call_stmt = f"    {c_name}()\n" if lang not in ("rust", "typescript", "javascript") else f"    {c_name}();\n"
                        mut_call_circ = "".join(lines[:target_idx]) + call_stmt + "".join(lines[target_idx:])
                        circ_rep = engine.verify(sym.file_path, mut_call_circ)
                        if circ_rep.status == "REJECTED" and circ_rep.cycles_detected:
                            mined_records.append(
                                DatasetRecord(
                                    input_dsl=circ_rep.linearized_subgraph,
                                    label=self.negative_label,
                                    risk_score=random.uniform(0.88, 0.99),
                                    category="circular_imports",
                                    language=lang,
                                    taxonomy_labels={"BreakingPublicAPI": 0.65, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.75},
                                    symbolic_gate_passed=False,
                                    source_type="synthetic",
                                )
                            )
                            break

                    if c_file != sym.file_path:
                        mut_imp_circ = None
                        if lang == "python" and c_file.endswith(".py"):
                            mod_name = c_file.replace("/", ".").removesuffix(".py").removeprefix("src.")
                            mut_imp_circ = f"import {mod_name}\n" + orig_content
                        elif lang in ("typescript", "javascript") and c_file.endswith((".ts", ".tsx", ".js")):
                            try:
                                rel_imp = os.path.relpath(c_file, Path(sym.file_path).parent).replace("\\", "/")
                                rel_imp = rel_imp.removesuffix(".ts").removesuffix(".tsx").removesuffix(".js")
                                if not rel_imp.startswith("."):
                                    rel_imp = "./" + rel_imp
                                mut_imp_circ = f"import * as _cycle_import from '{rel_imp}';\n" + orig_content
                            except Exception:
                                pass

                        if mut_imp_circ:
                            circ_rep = engine.verify(sym.file_path, mut_imp_circ)
                            if circ_rep.status == "REJECTED" and circ_rep.cycles_detected:
                                mined_records.append(
                                    DatasetRecord(
                                        input_dsl=circ_rep.linearized_subgraph,
                                        label=self.negative_label,
                                        risk_score=random.uniform(0.88, 0.99),
                                        category="circular_imports",
                                        language=lang,
                                        taxonomy_labels={"BreakingPublicAPI": 0.65, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.75},
                                        symbolic_gate_passed=False,
                                        source_type="synthetic",
                                    )
                                )
                                break

                # 5. Deleted symbol mutation: remove the symbol definition
                end_line = min(sym.end_lineno, len(lines))
                mutated_del = "".join(lines[: sym.lineno - 1]) + "".join(lines[end_line:])
                del_rep = engine.verify(sym.file_path, mutated_del)
                if del_rep.status == "REJECTED":
                    mined_records.append(
                        DatasetRecord(
                            input_dsl=del_rep.linearized_subgraph,
                            label=self.negative_label,
                            risk_score=random.uniform(0.85, 0.98),
                            category="deleted_symbol",
                            language=lang,
                            taxonomy_labels={"BreakingPublicAPI": 0.95, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                            symbolic_gate_passed=False,
                            source_type="synthetic",
                        )
                    )

            # Subtle semantic mutations mapped to ADR-0003 risk taxonomy
            # (Gray-area hard negatives that pass Stage 1 & 2 symbolic gate)
            # 6. Silent Logic Drift
            if lang == "python":
                mut_logic_stmt = "    if False: pass\n    _logic_flag = True\n"
            elif lang in ("typescript", "javascript"):
                mut_logic_stmt = "    if (false) {} (globalThis as any)._drift = true;\n"
            elif lang == "go":
                mut_logic_stmt = "    _ = 1 + 0\n"
            else:
                mut_logic_stmt = "    let _drift_flag = true;\n"
            mut_logic = "".join(lines[:target_idx]) + mut_logic_stmt + "".join(lines[target_idx:])
            logic_rep = engine.verify(sym.file_path, mut_logic)
            if logic_rep.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=logic_rep.linearized_subgraph,
                        label=self.negative_label,
                        risk_score=random.uniform(0.82, 0.95),
                        category="silent_logic_drift",
                        language=lang,
                        taxonomy_labels={"BreakingPublicAPI": 0.0, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.90},
                        symbolic_gate_passed=True,
                        source_type="mutation_subtle",
                    )
                )

            # 7. Security Surface
            if lang == "python":
                mut_sec_stmt = f"    import os\n    os.system('echo check_{sym.name} > /dev/null')\n"
            elif lang in ("typescript", "javascript"):
                mut_sec_stmt = f"    eval('var _surface = \\'{sym.name}\\';');\n"
            elif lang == "go":
                mut_sec_stmt = f"    _ = \"SELECT * FROM users WHERE id = '{sym.name}'\"\n"
            else:
                mut_sec_stmt = f"    let _sec_query = format!(\"SELECT * FROM users WHERE id = '{{}}'\", \"{sym.name}\");\n"
            mut_sec = "".join(lines[:target_idx]) + mut_sec_stmt + "".join(lines[target_idx:])
            sec_rep = engine.verify(sym.file_path, mut_sec)
            if sec_rep.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=sec_rep.linearized_subgraph,
                        label=self.negative_label,
                        risk_score=random.uniform(0.85, 0.96),
                        category="security_surface",
                        language=lang,
                        taxonomy_labels={"BreakingPublicAPI": 0.0, "SecuritySurface": 0.92, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                        symbolic_gate_passed=True,
                        source_type="mutation_subtle",
                    )
                )

            # 8. Concurrency Hazard
            if lang == "python":
                mut_conc_stmt = "    global _SHARED_MUTEX_STATE\n    _SHARED_MUTEX_STATE = getattr(__builtins__, '_leak', 0) + 1\n"
            elif lang in ("typescript", "javascript"):
                mut_conc_stmt = "    (globalThis as any)._sharedState = ((globalThis as any)._sharedState || 0) + 1;\n"
            elif lang == "go":
                mut_conc_stmt = "    go func() { _ = 1 }()\n"
            else:
                mut_conc_stmt = "    static mut HAZARD_CNT: i32 = 0; unsafe { HAZARD_CNT += 1; }\n"
            mut_conc = "".join(lines[:target_idx]) + mut_conc_stmt + "".join(lines[target_idx:])
            conc_rep = engine.verify(sym.file_path, mut_conc)
            if conc_rep.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=conc_rep.linearized_subgraph,
                        label=self.negative_label,
                        risk_score=random.uniform(0.82, 0.95),
                        category="concurrency_hazard",
                        language=lang,
                        taxonomy_labels={"BreakingPublicAPI": 0.0, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.88, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                        symbolic_gate_passed=True,
                        source_type="mutation_subtle",
                    )
                )

            # 9. Performance Regression
            if lang == "python":
                mut_perf_stmt = "    for _p1 in range(200):\n        for _p2 in range(10): pass\n"
            elif lang in ("typescript", "javascript"):
                mut_perf_stmt = "    for (let _p1 = 0; _p1 < 200; _p1++) { for (let _p2 = 0; _p2 < 10; _p2++) {} }\n"
            elif lang == "go":
                mut_perf_stmt = "    for _p1 := 0; _p1 < 200; _p1++ { for _p2 := 0; _p2 < 10; _p2++ {} }\n"
            else:
                mut_perf_stmt = "    for _p1 in 0..200 { for _p2 in 0..10 {} }\n"
            mut_perf = "".join(lines[:target_idx]) + mut_perf_stmt + "".join(lines[target_idx:])
            perf_rep = engine.verify(sym.file_path, mut_perf)
            if perf_rep.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=perf_rep.linearized_subgraph,
                        label=self.negative_label,
                        risk_score=random.uniform(0.80, 0.94),
                        category="performance_regression",
                        language=lang,
                        taxonomy_labels={"BreakingPublicAPI": 0.0, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.86, "SilentLogicDrift": 0.0},
                        symbolic_gate_passed=True,
                        source_type="mutation_subtle",
                    )
                )

            # 10. Breaking Public API
            if lang == "python":
                mut_api_stmt = "    _api_contract_drift = None\n"
            elif lang in ("typescript", "javascript"):
                mut_api_stmt = "    const _api_contract_drift: any = undefined;\n"
            elif lang == "go":
                mut_api_stmt = "    var _api_contract_drift *int = nil\n    _ = _api_contract_drift\n"
            else:
                mut_api_stmt = "    let _api_contract_drift: Option<i32> = None;\n"
            mut_api = "".join(lines[:target_idx]) + mut_api_stmt + "".join(lines[target_idx:])
            api_rep = engine.verify(sym.file_path, mut_api)
            if api_rep.status == "APPROVED":
                mined_records.append(
                    DatasetRecord(
                        input_dsl=api_rep.linearized_subgraph,
                        label=self.negative_label,
                        risk_score=random.uniform(0.82, 0.95),
                        category="breaking_public_api",
                        language=lang,
                        taxonomy_labels={"BreakingPublicAPI": 0.89, "SecuritySurface": 0.0, "ConcurrencyHazard": 0.0, "PerformanceRegression": 0.0, "SilentLogicDrift": 0.0},
                        symbolic_gate_passed=True,
                        source_type="mutation_subtle",
                    )
                )

            if len(mined_records) >= max_samples:
                break

        if filter_gate:
            mined_records = [r for r in mined_records if r.symbolic_gate_passed]

        return mined_records

    def generate_and_export(
        self,
        output_dir: Path,
        num_samples: int = 100,
        val_ratio: float = 0.2,
        repo_path: Optional[Path] = None,
        repo_paths: Optional[List[Path]] = None,
        include_subtle: bool = True,
    ) -> Tuple[int, int]:
        """
        Generate complete balanced dataset and write to dataset_train.jsonl and dataset_val.jsonl.
        Returns: (train_count, val_count)
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        all_records: List[DatasetRecord] = []

        all_repo_paths = list(repo_paths or [])
        if repo_path and repo_path not in all_repo_paths:
            all_repo_paths.append(repo_path)

        # Mine repositories if provided
        for rp in all_repo_paths:
            if rp.exists():
                per_repo_samples = max(20, num_samples // max(1, len(all_repo_paths)))
                mined = self.mine_repository(
                    rp,
                    max_samples=per_repo_samples,
                    filter_symbolic_gate=self.filter_symbolic_gate,
                )
                all_records.extend(mined)

        # Supplement with synthetic mutations across all languages to hit target count and balance
        remaining = max(0, num_samples - len(all_records))
        if remaining > 0 or not all_records:
            synthetic = self.generate_synthetic_dataset(
                num_samples=max(num_samples, remaining),
                include_subtle=include_subtle,
            )
            all_records.extend(synthetic)

        if self.filter_symbolic_gate:
            all_records = [r for r in all_records if r.symbolic_gate_passed]

        # Balance positive and negative samples
        positives = [r for r in all_records if r.label == self.positive_label]
        negatives = [r for r in all_records if r.label == self.negative_label]

        min_len = min(len(positives), len(negatives))
        if min_len > 0:
            chosen_pos = positives[:min_len]
            chosen_neg = negatives[:min_len]
        else:
            chosen_pos = positives
            chosen_neg = negatives

        # Stratified train / validation split to maintain exact class balance in both splits
        val_pos = max(1, int(len(chosen_pos) * val_ratio)) if chosen_pos else 0
        val_neg = max(1, int(len(chosen_neg) * val_ratio)) if chosen_neg else 0

        val_records = chosen_pos[:val_pos] + chosen_neg[:val_neg]
        train_records = chosen_pos[val_pos:] + chosen_neg[val_neg:]

        random.shuffle(train_records)
        random.shuffle(val_records)

        # Export to JSONL
        train_file = output_dir / "dataset_train.jsonl"
        val_file = output_dir / "dataset_val.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for rec in train_records:
                f.write(json.dumps(rec.to_dict()) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for rec in val_records:
                f.write(json.dumps(rec.to_dict()) + "\n")

        return len(train_records), len(val_records)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for dataset generation and targeted mutations."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Mine and generate balanced multi-language training datasets for Code Oracle / Laya ModernBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=Path, default=None, help="Path to an existing code repository to mine.")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./dataset_output"), help="Directory to emit dataset files.")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Target total number of balanced samples.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction allocated to validation set.")
    parser.add_argument("--languages", "-l", type=str, default="python,typescript,go,rust", help="Comma-separated languages.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--positive-label", type=int, default=1, help="Positive label.")
    parser.add_argument("--negative-label", type=int, default=0, help="Negative label.")
    parser.add_argument("--filter-symbolic-gate", action="store_true", default=False, help="Filter mutations that fail symbolic gate.")
    parser.add_argument("--include-subtle", action="store_true", default=True, help="Include subtle mutations.")
    parser.add_argument("--no-subtle", dest="include_subtle", action="store_false", help="Disable subtle mutations.")
    parser.add_argument("--targeted", action="store_true", default=False, help="Generate targeted subtle mutations for TypeScript and Python.")

    args = parser.parse_args(argv)
    lang_list = [l.strip().lower() for l in args.languages.split(",") if l.strip()]

    generator = DatasetGenerator(
        languages=lang_list,
        seed=args.seed,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        filter_symbolic_gate=args.filter_symbolic_gate,
    )

    if args.targeted:
        print(f"[*] Generating targeted mutations for languages: {', '.join(lang_list)}...")
        records = generator.generate_targeted_mutations(
            languages=lang_list,
            count_per_type=max(1, (args.num_samples // 2) // (4 * max(1, len(lang_list)))),
        )
        print(f"[+] Successfully generated {len(records)} targeted mutation records.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train_file = args.output_dir / "dataset_train.jsonl"
        val_file = args.output_dir / "dataset_val.jsonl"
        val_count = max(1, int(len(records) * args.val_ratio))
        val_recs = records[:val_count]
        train_recs = records[val_count:]
        with open(train_file, "w", encoding="utf-8") as f:
            for r in train_recs:
                f.write(json.dumps(r.to_dict()) + "\n")
        with open(val_file, "w", encoding="utf-8") as f:
            for r in val_recs:
                f.write(json.dumps(r.to_dict()) + "\n")
        return 0

    train_count, val_count = generator.generate_and_export(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        val_ratio=args.val_ratio,
        repo_path=args.repo,
        include_subtle=args.include_subtle,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
