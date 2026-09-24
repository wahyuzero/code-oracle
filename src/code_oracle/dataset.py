"""
Multi-Language Dataset Mining & Synthetic Mutation Engine.
Generates balanced positive (PASS) and negative (REJECT) pairs across Tier 1 languages
(Python, TypeScript, Go, Rust) for Laya ModernBERT fine-tuning.
"""

import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from code_oracle.engine import TopoSliceEngine
from code_oracle.languages import SUPPORTED_EXTENSIONS, detect_language


@dataclass
class DatasetRecord:
    """Represents a single training or validation instance for Laya ModernBERT."""
    input_dsl: str
    label: int  # 1 for PASS (positive), 0 for REJECT (negative)
    risk_score: float  # 0.0 - 0.2 for PASS, 0.8 - 1.0 for REJECT
    category: str  # "clean_pass", "arity_breaking", "keyword_changes", "circular_imports", "deleted_symbol"
    language: str  # "python", "typescript", "go", "rust"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dsl": self.input_dsl,
            "label": self.label,
            "risk_score": round(self.risk_score, 4),
            "category": self.category,
            "language": self.language,
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
    ):
        self.languages = [l.lower() for l in (languages or ["python", "typescript", "go", "rust"])]
        self.seed = seed
        self.positive_label = positive_label
        self.negative_label = negative_label
        random.seed(seed)

    def generate_pairs_for_template(
        self,
        template: Dict[str, Any],
        language: str,
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
                    )
                )

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
                    )
                )

        return records

    def generate_synthetic_dataset(self, num_samples: int = 100) -> List[DatasetRecord]:
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
                pairs = self.generate_pairs_for_template(tmpl, lang)
                for r in pairs:
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

    def mine_repository(
        self,
        repo_path: Path,
        max_samples: int = 50,
    ) -> List[DatasetRecord]:
        """
        Mine an existing codebase in Python, TypeScript, Go, or Rust,
        discovering functions with callers to generate authentic mutation pairs
        across 5 distinct categories (clean_pass, arity_breaking, keyword_changes,
        circular_imports, deleted_symbol).
        """
        repo_path = repo_path.resolve()
        engine = TopoSliceEngine(workspace_root=repo_path)
        engine.indexer.scan_workspace()

        mined_records: List[DatasetRecord] = []

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
                    )
                )

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
                        )
                    )

            # 3. Keyword / Call Argument Mismatch mutation
            # 3A: Callee keyword-only required argument (Python)
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
                        )
                    )
            # 3B: Caller callsite argument mismatch (across all languages)
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
                                )
                            )
                            break

            # 4. Circular Import / Call Mutation
            for c in callers[:3]:
                if not (c.caller and "::" in c.caller):
                    continue
                c_file, c_sym = c.caller.split("::", 1)
                c_name = c_sym.split(".")[-1]

                # 4A: Call cycle: call the caller inside the callee
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
                            )
                        )
                        break

                # 4B: Import cycle: import caller module from callee file
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
                    )
                )

            if len(mined_records) >= max_samples:
                break

        return mined_records

    def generate_and_export(
        self,
        output_dir: Path,
        num_samples: int = 100,
        val_ratio: float = 0.2,
        repo_path: Optional[Path] = None,
        repo_paths: Optional[List[Path]] = None,
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
                mined = self.mine_repository(rp, max_samples=per_repo_samples)
                all_records.extend(mined)

        # Supplement with synthetic mutations across all languages to hit target count and balance
        remaining = max(0, num_samples - len(all_records))
        if remaining > 0 or not all_records:
            synthetic = self.generate_synthetic_dataset(num_samples=max(num_samples, remaining))
            all_records.extend(synthetic)

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
