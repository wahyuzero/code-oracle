"""
Multi-language root and entrypoint heuristic detector.
Identifies execution roots across Python, TypeScript/JavaScript, Go, and Rust:
- Test files and test functions
- CLI commands and framework decorators (@click, @app.command)
- HTTP routes (@app.get, @app.post, @router.get, @bp.route)
- Public root exports (__init__.py, index.ts, mod.rs, lib.rs, main.go)
- Public uppercase Go symbols in root package
- Dunder/magic methods and language constructors
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from code_oracle.models import Symbol

# Regex patterns for route and framework decorators
ROUTE_DECORATOR_PATTERN = re.compile(
    r"@\s*(?:(?:[\w\.]+\.)?(?:app|router|bp|api|server|web)\s*\.\s*(?:get|post|put|delete|patch|options|head|route|websocket)|(?:Get|Post|Put|Delete|Patch|Controller|Route)\b)",
    re.IGNORECASE,
)
CLI_DECORATOR_PATTERN = re.compile(
    r"@\s*(?:[\w\.]+\.)?(?:click|typer|app|cli|main)\s*\.\s*(?:command|group|option|argument)\b",
    re.IGNORECASE,
)
TEST_DECORATOR_PATTERN = re.compile(
    r"@\s*(?:pytest\s*\.\s*fixture|fixture)\b|#\[(?:tokio::test|test|actix_web::|rocket::)",
    re.IGNORECASE,
)


class EntrypointDetector:
    """Detects framework entrypoints, test suites, and root exports."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self._source_cache: Dict[str, List[str]] = {}

    def get_source_lines(self, rel_path: str) -> List[str]:
        """Fetch and cache lines of a source file."""
        norm_path = rel_path.replace("\\", "/")
        if norm_path in self._source_cache:
            return self._source_cache[norm_path]

        full_p = self.workspace_root / norm_path
        if full_p.is_file():
            try:
                lines = full_p.read_text(encoding="utf-8", errors="replace").splitlines()
                self._source_cache[norm_path] = lines
                return lines
            except Exception:
                pass
        self._source_cache[norm_path] = []
        return []

    def is_test_file(self, file_path: str) -> bool:
        """Check if file is located in a test directory or matches test naming."""
        clean = file_path.replace("\\", "/").lower()
        parts = clean.split("/")

        # Directory checks
        if any(p in ("tests", "test", "__tests__", "spec") for p in parts):
            return True

        # Extension/filename checks
        file_name = parts[-1]
        if file_name.startswith("test_") or file_name.endswith("_test.py"):
            return True
        if file_name.endswith("_test.go"):
            return True
        if (
            file_name.endswith((".spec.ts", ".test.ts", ".spec.tsx", ".test.tsx"))
            or file_name.endswith((".spec.js", ".test.js", ".spec.jsx", ".test.jsx"))
        ):
            return True
        if file_name.endswith("_test.rs"):
            return True

        return False

    def get_symbol_decorators(self, symbol: Symbol) -> List[str]:
        """
        Scan lines preceding the symbol's lineno for decorators.
        Robustly handles multi-line decorator blocks and comments.
        """
        lines = self.get_source_lines(symbol.file_path)
        if not lines or symbol.lineno < 1 or symbol.lineno > len(lines):
            return []

        decorators: List[str] = []
        idx = symbol.lineno - 2  # 0-indexed line immediately above symbol
        accum: List[str] = []
        paren_depth = 0
        bracket_depth = 0

        while idx >= 0:
            line = lines[idx].strip()
            if not line:
                if not accum:
                    idx -= 1
                    continue
                else:
                    accum.append(line)
                    idx -= 1
                    continue

            # Standalone comments between decorators
            if not accum and (line.startswith("#") or line.startswith("//")):
                idx -= 1
                continue

            open_parens = line.count("(")
            close_parens = line.count(")")
            open_brackets = line.count("[")
            close_brackets = line.count("]")

            paren_depth += (close_parens - open_parens)
            bracket_depth += (close_brackets - open_brackets)

            accum.append(line)

            # Check if this line is the beginning of a decorator (@ or #[)
            if line.startswith("@") or line.startswith("#["):
                if paren_depth <= 0 and bracket_depth <= 0:
                    dec_text = "\n".join(reversed(accum)).strip()
                    decorators.append(dec_text)
                    accum = []
                    paren_depth = 0
                    bracket_depth = 0
                idx -= 1
                continue

            # If not inside a decorator call and line is not a decorator header, decorator block ended
            if paren_depth <= 0 and bracket_depth <= 0:
                break

            idx -= 1

        return decorators

    def is_test_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol represents a test function, benchmark, or test suite."""
        if self.is_test_file(symbol.file_path):
            return True

        name = symbol.name
        # Python / JS / Rust test naming
        if name.startswith("test_") or (name.startswith("Test") and symbol.kind in ("class", "function")):
            return True

        # Go test naming
        if symbol.file_path.endswith(".go"):
            if name.startswith(("Test", "Benchmark", "Fuzz", "Example")):
                return True

        # Pytest fixture or Rust #[test] check in decorator/calls
        if any(c.callee.endswith(("fixture", "pytest.fixture")) for c in symbol.calls):
            return True

        # Check source lines for test decorators
        decorators = self.get_symbol_decorators(symbol)
        if any(TEST_DECORATOR_PATTERN.search(dec) for dec in decorators):
            return True

        return False

    def is_property_method(self, symbol: Symbol) -> bool:
        """Check if symbol is decorated with @property, @cached_property, @setter, etc."""
        if not symbol.is_method:
            return False
        decorators = self.get_symbol_decorators(symbol)
        for dec in decorators:
            clean = dec.lower()
            if any(p in clean for p in ("@property", "@cached_property", ".setter", ".deleter", "@abstractmethod", "@overload")):
                return True
        return False

    def is_cli_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol is a CLI command or program entrypoint."""
        name = symbol.name
        clean_file = symbol.file_path.replace("\\", "/").lower()
        parts = clean_file.split("/")
        filename = parts[-1]

        # Standalone main/cli functions
        if name in ("main", "cli", "run_cli", "app"):
            return True

        # Go / Rust runtime entrypoints
        if name == "main" and symbol.kind == "function":
            return True
        if symbol.file_path.endswith(".go") and name == "init":
            return True

        # Known CLI files or CLI command naming conventions
        if filename in ("cli.py", "main.py", "__main__.py", "main.go", "main.rs", "commands.py", "cmd.py"):
            if (
                name in ("main", "cli", "run", "execute", "start")
                or name.startswith(("cmd_", "do_"))
                or name.endswith(("_command", "_cmd"))
            ):
                return True

        # General CLI command naming convention across any file
        if name.startswith("cmd_") and symbol.kind in ("function", "async_function"):
            return True

        # Click / Typer / CLI decorators in calls
        for c in symbol.calls:
            if any(kw in c.callee for kw in ("click.command", "click.group", "app.command", "cli.command", "typer.command")):
                return True

        # Inspect source lines for @click or @app.command
        decorators = self.get_symbol_decorators(symbol)
        if any(CLI_DECORATOR_PATTERN.search(dec) for dec in decorators):
            return True

        return False

    def is_route_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol is a web/HTTP route handler (@app.get, @router.post, etc.)."""
        # Check call references
        for c in symbol.calls:
            callee_lower = c.callee.lower()
            if any(
                route_kw in callee_lower
                for route_kw in (
                    "app.get", "app.post", "app.put", "app.delete", "app.patch", "app.route",
                    "router.get", "router.post", "router.put", "router.delete", "router.patch",
                    "bp.route", "bp.get", "bp.post",
                )
            ):
                return True

        # Inspect source lines for route decorators
        decorators = self.get_symbol_decorators(symbol)
        if any(ROUTE_DECORATOR_PATTERN.search(dec) for dec in decorators):
            return True

        return False

    def is_root_export(self, symbol: Symbol) -> bool:
        """Check if symbol is defined in a public root export file."""
        clean = symbol.file_path.replace("\\", "/")
        parts = clean.split("/")
        filename = parts[-1]

        # Python root exports
        if filename == "__init__.py":
            return True

        # TypeScript / JavaScript root exports
        if filename in ("index.ts", "index.tsx", "index.js", "index.jsx", "main.ts", "main.tsx"):
            return True

        # Rust root exports
        if filename in ("mod.rs", "lib.rs", "main.rs"):
            return True

        # Go main file
        if filename == "main.go":
            return True

        return False

    def is_public_go_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol is an exported Go symbol in the root package."""
        if not symbol.file_path.endswith(".go"):
            return False

        clean = symbol.file_path.replace("\\", "/")
        # Root package check (in root directory or top-level file)
        is_root_dir = "/" not in clean or clean.startswith("./") and clean.count("/") == 1
        if is_root_dir or clean.startswith("main."):
            if symbol.name and symbol.name[0].isupper():
                return True

        return False

    def is_magic_method(self, symbol: Symbol) -> bool:
        """Check if symbol is a language dunder/magic method or constructor."""
        name = symbol.name
        # Python dunder methods
        if name.startswith("__") and name.endswith("__"):
            return True

        # TypeScript / JavaScript constructor
        if name == "constructor":
            return True

        return False

    def is_entrypoint(self, symbol: Symbol) -> bool:
        """
        Evaluate if a symbol qualifies as an execution or API root.
        Roots are the starting seeds for graph reachability analysis.
        """
        # Module-level blocks are always execution entrypoints
        if symbol.kind == "module" or symbol.name == "<module>":
            return True

        # Test suites and functions
        if self.is_test_symbol(symbol):
            return True

        # CLI entrypoints
        if self.is_cli_symbol(symbol):
            return True

        # HTTP route handlers
        if self.is_route_symbol(symbol):
            return True

        # Public root exports
        if self.is_root_export(symbol):
            return True

        # Public Go symbols in root package
        if self.is_public_go_symbol(symbol):
            return True

        # Dunder and magic methods
        if self.is_magic_method(symbol):
            return True

        return False


def is_entrypoint(symbol: Symbol, detector: Optional[EntrypointDetector] = None) -> bool:
    """Convenience helper to check if a symbol is an entrypoint root."""
    det = detector or EntrypointDetector()
    return det.is_entrypoint(symbol)
