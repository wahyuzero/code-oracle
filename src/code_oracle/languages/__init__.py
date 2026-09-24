"""
Modular Multi-Language AST Extraction Engine for Code Oracle.
Provides unified AST parsing across Tier 1 languages: Python, TypeScript/JavaScript, Go, and Rust.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

from code_oracle.languages.go import extract_go_imports, extract_go_symbols, validate_go_syntax
from code_oracle.languages.python import (
    extract_python_imports,
    extract_python_symbols,
    validate_python_syntax,
)
from code_oracle.languages.rust import extract_rust_imports, extract_rust_symbols, validate_rust_syntax
from code_oracle.languages.typescript import (
    extract_typescript_imports,
    extract_typescript_symbols,
    validate_typescript_syntax,
)
from code_oracle.models import ImportReference, Symbol

EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}

SUPPORTED_LANGUAGES: Set[str] = {"python", "typescript", "javascript", "go", "rust"}
SUPPORTED_EXTENSIONS: Set[str] = set(EXTENSION_TO_LANGUAGE.keys())


def detect_language(file_path: str) -> Optional[str]:
    """Detect the language of a file from its extension or path."""
    if not file_path:
        return "python"
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def extract_symbols(
    source: str,
    file_path: str = "",
    language: Optional[str] = None,
) -> List[Symbol]:
    """
    Extract symbol entities from source code across any supported Tier 1 language.
    Dispatches to language-specific Tree-sitter or AST extractor based on file extension.
    """
    lang = language or detect_language(file_path) or "python"

    if lang == "python":
        return extract_python_symbols(source, file_path=file_path)
    elif lang in ("typescript", "javascript"):
        return extract_typescript_symbols(source, file_path=file_path)
    elif lang == "go":
        return extract_go_symbols(source, file_path=file_path)
    elif lang == "rust":
        return extract_rust_symbols(source, file_path=file_path)
    else:
        # Fallback to Python AST
        return extract_python_symbols(source, file_path=file_path)


def extract_imports(
    source: str,
    file_path: str = "",
    language: Optional[str] = None,
) -> List[ImportReference]:
    """
    Extract import statements from source code across any supported Tier 1 language.
    Dispatches to language-specific Tree-sitter or AST extractor based on file extension.
    """
    lang = language or detect_language(file_path) or "python"

    if lang == "python":
        return extract_python_imports(source, file_path=file_path)
    elif lang in ("typescript", "javascript"):
        return extract_typescript_imports(source, file_path=file_path)
    elif lang == "go":
        return extract_go_imports(source, file_path=file_path)
    elif lang == "rust":
        return extract_rust_imports(source, file_path=file_path)
    else:
        return extract_python_imports(source, file_path=file_path)


def validate_syntax(
    source: str,
    file_path: str = "",
    language: Optional[str] = None,
) -> Optional[str]:
    """
    Validate code syntax for any supported Tier 1 language.
    Returns None if source is syntactically valid, or a descriptive error message if invalid.
    """
    lang = language or detect_language(file_path) or "python"

    if lang == "python":
        return validate_python_syntax(source, file_path=file_path)
    elif lang in ("typescript", "javascript"):
        return validate_typescript_syntax(source, file_path=file_path)
    elif lang == "go":
        return validate_go_syntax(source, file_path=file_path)
    elif lang == "rust":
        return validate_rust_syntax(source, file_path=file_path)
    else:
        return validate_python_syntax(source, file_path=file_path)
