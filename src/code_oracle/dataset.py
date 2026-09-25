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

    @staticmethod
    def _default_taxonomy_for_category(category: str, label: int, risk_score: float) -> Dict[str, float]:
        base = {c: 0.0 for c in TAXONOMY_CLASSES}
        if label == 1:
            return base

        score = max(0.6, min(1.0, risk_score))
        cat_lower = category.lower()
        if "concurrency_leak" in cat_lower or "goroutine" in cat_lower or "thread" in cat_lower or "deadlock" in cat_lower or "race" in cat_lower or "mutex" in cat_lower or "concurr" in cat_lower:
            base["ConcurrencyHazard"] = score
        elif "resource_leak" in cat_lower or "memory_leak" in cat_lower or "resource" in cat_lower or "memory" in cat_lower or "perf" in cat_lower or "quadratic" in cat_lower or "loop" in cat_lower or "timeout" in cat_lower or "slow" in cat_lower or "leak" in cat_lower:
            base["PerformanceRegression"] = score
        elif "api" in cat_lower or "arity" in cat_lower or "keyword" in cat_lower or "type_drift" in cat_lower or "deleted" in cat_lower or "param" in cat_lower or "signature" in cat_lower:
            base["BreakingPublicAPI"] = score
        elif "security" in cat_lower or "surface" in cat_lower or "taint" in cat_lower or "side_effect" in cat_lower or "cve" in cat_lower or "vuln" in cat_lower or "auth" in cat_lower or "sanitize" in cat_lower:
            base["SecuritySurface"] = score
        elif "logic" in cat_lower or "drift" in cat_lower:
            base["SilentLogicDrift"] = score
        elif "circular" in cat_lower:
            base["BreakingPublicAPI"] = round(score * 0.7, 4)
            base["SilentLogicDrift"] = round(score * 0.8, 4)
        else:
            base["SilentLogicDrift"] = score
        return base

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
    ],
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
    ) -> List[DatasetRecord]:
        """
        Instantiate a multi-file template in an isolated temporary workspace,
        execute positive and negative mutations through TopoSliceEngine,
        and generate structured DatasetRecord objects.
        """
        records: List[DatasetRecord] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            for fname, fcontent in template["files"].items():
                fpath = ws_root / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(fcontent, encoding="utf-8")

            engine = TopoSliceEngine(workspace_root=ws_root)
            engine.indexer.scan_workspace()

            # 1. Positive: Clean pass (label=1, risk=0.0-0.2)
            pass_rep = engine.verify(template["target_file"], template["pass_patch"])
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
                subtle_mutations = [
                    ("silent_logic_drift_patch", "silent_logic_drift", "SilentLogicDrift"),
                    ("security_surface_patch", "security_surface", "SecuritySurface"),
                    ("concurrency_hazard_patch", "concurrency_hazard", "ConcurrencyHazard"),
                    ("performance_regression_patch", "performance_regression", "PerformanceRegression"),
                    ("breaking_public_api_patch", "breaking_public_api", "BreakingPublicAPI"),
                ]
                for patch_key, cat_name, tax_class in subtle_mutations:
                    patch_content = template.get(patch_key)
                    if not patch_content:
                        continue
                    mut_rep = engine.verify(template["target_file"], patch_content)
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
            cmd = ["git", "log", "--grep=revert", "-i", "-n", "30", "--format=%H|%s"]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    if "|" not in line:
                        continue
                    rev_hash, subject = line.split("|", 1)
                    if not re.search(r"(?i)\b(revert|reverted|reverting)\b|^revert\b", subject):
                        continue
                    f_cmd = ["git", "show", "--name-only", "--format=", rev_hash]
                    f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    if f_res.returncode != 0:
                        continue
                    files = [f.strip() for f in f_res.stdout.splitlines() if f.strip()]
                    for fname in files:
                        lang = detect_language(fname)
                        if not lang or lang not in self.languages:
                            continue
                        d_cmd = ["git", "show", "-p", rev_hash, "--", fname]
                        d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if d_res.returncode != 0 or not d_res.stdout.strip():
                            continue
                        diff_text = d_res.stdout

                        # Check if this revert references the original buggy commit
                        rev_match = re.search(r"This reverts commit ([0-9a-f]{7,40})", diff_text)
                        if rev_match:
                            bug_hash = rev_match.group(1)
                            b_cmd = ["git", "show", "-p", bug_hash, "--", fname]
                            b_res = subprocess.run(b_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                            if b_res.returncode == 0 and b_res.stdout.strip():
                                diff_text = b_res.stdout

                        try:
                            rep = engine.verify(fname, diff_text)
                            gate_passed = (rep.status == "APPROVED")
                            if should_filter and not gate_passed:
                                continue
                            tax = DatasetRecord._default_taxonomy_for_category(subject, 0, 0.88)
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

        # 2. Mine Hotfix Commits
        try:
            cmd = ["git", "log", "--grep=fix", "-i", "-n", "30", "--format=%H|%s"]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    if "|" not in line or "revert" in line.lower():
                        continue
                    fix_hash, subject = line.split("|", 1)
                    # Filter out false positives like 'prefix', 'fixture', etc.
                    if not re.search(r"(?i)\b(fix|hotfix|bugfix|patch)\b|^(fix|hotfix)(\(.*\))?:", subject):
                        continue
                    if re.search(r"(?i)\b(fixture|fixtures|prefix|postfix|suffix)\b", subject) and not re.search(r"(?i)\b(bug|hotfix|fix:)\b", subject):
                        continue
                    f_cmd = ["git", "show", "--name-only", "--format=", fix_hash]
                    f_res = subprocess.run(f_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                    if f_res.returncode != 0:
                        continue
                    files = [f.strip() for f in f_res.stdout.splitlines() if f.strip()]
                    for fname in files:
                        lang = detect_language(fname)
                        if not lang or lang not in self.languages:
                            continue
                        d_cmd = ["git", "show", "-p", fix_hash, "--", fname]
                        d_res = subprocess.run(d_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5)
                        if d_res.returncode != 0 or not d_res.stdout.strip():
                            continue
                        diff_text = d_res.stdout
                        try:
                            rep = engine.verify(fname, diff_text)
                            gate_passed = (rep.status == "APPROVED")
                            if should_filter and not gate_passed:
                                continue
                            records.append(
                                DatasetRecord(
                                    input_dsl=rep.linearized_subgraph,
                                    label=self.positive_label,
                                    risk_score=random.uniform(0.02, 0.15),
                                    category="real_hotfix",
                                    language=lang,
                                    taxonomy_labels={c: 0.0 for c in TAXONOMY_CLASSES},
                                    symbolic_gate_passed=gate_passed,
                                    source_type="real_hotfix",
                                )
                            )
                            if len(records) >= max(2, (max_samples * 2) // 3):
                                break
                        except Exception:
                            continue
                    if len(records) >= max(2, (max_samples * 2) // 3):
                        break
        except Exception:
            pass

        # 3. Mine Clean Merged Commits
        try:
            cmd = ["git", "log", "--no-merges", "-n", "30", "--format=%H|%s"]
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
                        try:
                            rep = engine.verify(fname, diff_text)
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
