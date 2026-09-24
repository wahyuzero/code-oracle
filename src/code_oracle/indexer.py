"""
Stage 2: Workspace Indexer / Symbol Cache.
Maintains an incremental, mtime-hashed inverted symbol index in .code_oracle/index.json.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from code_oracle.languages import SUPPORTED_EXTENSIONS
from code_oracle.locator import extract_imports_from_ast, extract_symbols_from_ast
from code_oracle.models import CallReference, ImportReference, Parameter, Symbol

IGNORE_DIRS = {
    ".git",
    ".code_oracle",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "node_modules",
}


def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class WorkspaceIndexer:
    """
    Incremental, mtime and content-hash cached symbol indexer.
    Keeps an inverted index of symbol definitions, callers, importers, and inheritance.
    """

    def __init__(self, workspace_root: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.cache_dir = Path(cache_dir or (self.workspace_root / ".code_oracle")).resolve()
        self.index_file = self.cache_dir / "index.json"

        # Internal state
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._file_symbols: Dict[str, List[Symbol]] = {}
        self._file_imports: Dict[str, List[ImportReference]] = {}
        self._import_graph: Dict[str, List[str]] = {}
        self._definitions: Dict[str, Symbol] = {}
        self._name_to_symbols: Dict[str, List[Symbol]] = {}
        self._callers: Dict[str, List[CallReference]] = {}
        self._importers: Dict[str, List[ImportReference]] = {}
        self._subclasses: Dict[str, List[Symbol]] = {}

        self.load_cache()

    def load_cache(self) -> bool:
        """Load index from .code_oracle/index.json if present."""
        if not self.index_file.exists():
            return False

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == 1 and isinstance(data.get("files"), dict):
                self._file_cache = data["files"]
                self._rebuild_indices()
                return True
        except Exception:
            # Corrupted index, start fresh
            self._file_cache = {}
        return False

    def save_cache(self) -> None:
        """Atomically persist index cache to .code_oracle/index.json."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temp_file = self.cache_dir / "index.json.tmp"
            data = {
                "version": 1,
                "timestamp": time.time(),
                "workspace": str(self.workspace_root),
                "files": self._file_cache,
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.index_file)
        except Exception:
            # Non-fatal if cache write fails (e.g. read-only fs)
            pass

    def clean(self) -> bool:
        """
        Safely deletes .code_oracle/ cache directory (Rollback Resilience).
        Restores workspace to default state with zero destructive impact.
        """
        self._file_cache.clear()
        self._file_symbols.clear()
        self._file_imports.clear()
        self._import_graph.clear()
        self._definitions.clear()
        self._name_to_symbols.clear()
        self._callers.clear()
        self._importers.clear()
        self._subclasses.clear()

        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                return True
            except Exception:
                return False
        return True

    def scan_workspace(self, force: bool = False) -> Dict[str, Any]:
        """
        Incrementally scan all python files in workspace.
        Re-indexes only files with changed mtime or content hash.
        """
        start_time = time.perf_counter()
        scanned_count = 0
        reindexed_count = 0

        current_rel_files: Set[str] = set()

        for root, dirs, files in os.walk(self.workspace_root):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                full_path = Path(root) / file
                rel_path = str(full_path.relative_to(self.workspace_root)).replace("\\", "/")
                current_rel_files.add(rel_path)
                scanned_count += 1

                try:
                    mtime = full_path.stat().st_mtime
                except OSError:
                    continue

                cached_entry = self._file_cache.get(rel_path)

                if (
                    not force
                    and cached_entry
                    and cached_entry.get("mtime") == mtime
                ):
                    # Cache hit by mtime
                    continue

                # Read and check content hash
                try:
                    content = full_path.read_text(encoding="utf-8")
                except Exception:
                    continue

                content_hash = compute_file_hash(content)
                if (
                    not force
                    and cached_entry
                    and cached_entry.get("hash") == content_hash
                ):
                    # Hash matches, update mtime only
                    cached_entry["mtime"] = mtime
                    continue

                # Need re-index
                reindexed_count += 1
                symbols = extract_symbols_from_ast(content, file_path=rel_path)
                imports = extract_imports_from_ast(content, file_path=rel_path)
                self._file_symbols[rel_path] = symbols
                self._file_imports[rel_path] = imports
                self._file_cache[rel_path] = {
                    "mtime": mtime,
                    "hash": content_hash,
                    "symbols": [self._serialize_symbol(s) for s in symbols],
                    "imports": [self._serialize_import(imp) for imp in imports],
                }

        # Remove deleted files from cache
        stale_files = set(self._file_cache.keys()) - current_rel_files
        for sf in stale_files:
            del self._file_cache[sf]
            self._file_symbols.pop(sf, None)
            self._file_imports.pop(sf, None)
            self._import_graph.pop(sf, None)

        if reindexed_count > 0 or stale_files or force:
            self._rebuild_indices()
            self.save_cache()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "scanned": scanned_count,
            "reindexed": reindexed_count,
            "symbols_indexed": len(self._definitions),
            "latency_ms": elapsed_ms,
        }

    def _serialize_import(self, imp: ImportReference) -> Dict[str, Any]:
        """Serialize an ImportReference object."""
        return {
            "module": imp.module,
            "name": imp.name,
            "asname": imp.asname,
            "lineno": imp.lineno,
            "file_path": imp.file_path,
            "level": imp.level,
        }

    def _deserialize_import(self, d: Dict[str, Any]) -> ImportReference:
        """Deserialize a dict into an ImportReference object."""
        return ImportReference(
            module=d.get("module"),
            name=d["name"],
            asname=d.get("asname"),
            lineno=d.get("lineno", 0),
            file_path=d.get("file_path", ""),
            level=d.get("level", 0),
        )

    def _serialize_symbol(self, s: Symbol) -> Dict[str, Any]:
        """Serialize a Symbol object to dict for cache storage."""
        return {
            "name": s.name,
            "qualname": s.qualname,
            "file_path": s.file_path,
            "kind": s.kind,
            "lineno": s.lineno,
            "end_lineno": s.end_lineno,
            "signature": s.signature,
            "min_args": s.min_args,
            "max_args": s.max_args,
            "accepted_kwargs": list(s.accepted_kwargs) if s.accepted_kwargs is not None else None,
            "required_kwargs": list(s.required_kwargs),
            "return_type": s.return_type,
            "is_method": s.is_method,
            "is_static": s.is_static,
            "bases": s.bases,
            "params": [
                {
                    "name": p.name,
                    "annotation": p.annotation,
                    "default": p.default,
                    "has_default": p.has_default,
                    "is_vararg": p.is_vararg,
                    "is_kwarg": p.is_kwarg,
                    "is_kwonly": p.is_kwonly,
                    "is_posonly": p.is_posonly,
                }
                for p in s.params
            ],
            "calls": [
                {
                    "callee": c.callee,
                    "args_count": c.args_count,
                    "kwargs": c.kwargs,
                    "lineno": c.lineno,
                    "caller": c.caller,
                    "has_vararg": c.has_vararg,
                    "has_kwarg": c.has_kwarg,
                }
                for c in s.calls
            ],
        }

    def _deserialize_symbol(self, d: Dict[str, Any]) -> Symbol:
        """Deserialize a dict into a Symbol object."""
        params = [
            Parameter(
                name=p["name"],
                annotation=p.get("annotation"),
                default=p.get("default"),
                has_default=p.get("has_default", False),
                is_vararg=p.get("is_vararg", False),
                is_kwarg=p.get("is_kwarg", False),
                is_kwonly=p.get("is_kwonly", False),
                is_posonly=p.get("is_posonly", False),
            )
            for p in d.get("params", [])
        ]
        calls = [
            CallReference(
                callee=c["callee"],
                args_count=c["args_count"],
                kwargs=c.get("kwargs", []),
                lineno=c.get("lineno", 0),
                caller=c.get("caller"),
                has_vararg=c.get("has_vararg", False),
                has_kwarg=c.get("has_kwarg", False),
            )
            for c in d.get("calls", [])
        ]
        accepted_kwargs = (
            set(d["accepted_kwargs"]) if d.get("accepted_kwargs") is not None else None
        )
        required_kwargs = set(d.get("required_kwargs", []))

        return Symbol(
            name=d["name"],
            qualname=d["qualname"],
            file_path=d["file_path"],
            kind=d["kind"],
            lineno=d["lineno"],
            end_lineno=d["end_lineno"],
            signature=d.get("signature", ""),
            params=params,
            min_args=d.get("min_args", 0),
            max_args=d.get("max_args"),
            accepted_kwargs=accepted_kwargs,
            required_kwargs=required_kwargs,
            return_type=d.get("return_type"),
            calls=calls,
            is_method=d.get("is_method", False),
            is_static=d.get("is_static", False),
            bases=d.get("bases", []),
        )

    def _remove_file_from_indices(self, file_path: str) -> None:
        """Incrementally prune symbols, callers, importers, and inheritance for a single file."""
        old_syms = self._file_symbols.get(file_path, [])
        for sym in old_syms:
            self._definitions.pop(sym.id, None)
            if sym.name in self._name_to_symbols:
                self._name_to_symbols[sym.name] = [s for s in self._name_to_symbols[sym.name] if s.file_path != file_path]
                if not self._name_to_symbols[sym.name]:
                    del self._name_to_symbols[sym.name]
            if sym.qualname != sym.name and sym.qualname in self._name_to_symbols:
                self._name_to_symbols[sym.qualname] = [s for s in self._name_to_symbols[sym.qualname] if s.file_path != file_path]
                if not self._name_to_symbols[sym.qualname]:
                    del self._name_to_symbols[sym.qualname]
            for base_name in sym.bases:
                if base_name in self._subclasses:
                    self._subclasses[base_name] = [s for s in self._subclasses[base_name] if s.file_path != file_path]
                    if not self._subclasses[base_name]:
                        del self._subclasses[base_name]
            for call in sym.calls:
                prefix = f"{file_path}::"
                callee = call.callee
                if callee in self._callers:
                    self._callers[callee] = [c for c in self._callers[callee] if not (c.caller and c.caller.startswith(prefix))]
                    if not self._callers[callee]:
                        del self._callers[callee]
                simple = callee.split(".")[-1].split("::")[-1]
                if simple in self._callers:
                    self._callers[simple] = [c for c in self._callers[simple] if not (c.caller and c.caller.startswith(prefix))]
                    if not self._callers[simple]:
                        del self._callers[simple]

        old_imps = self._file_imports.get(file_path, [])
        for imp in old_imps:
            if imp.name in self._importers:
                self._importers[imp.name] = [i for i in self._importers[imp.name] if i.file_path != file_path]
                if not self._importers[imp.name]:
                    del self._importers[imp.name]
            if imp.module:
                key = f"{imp.module}.{imp.name}"
                if key in self._importers:
                    self._importers[key] = [i for i in self._importers[key] if i.file_path != file_path]
                    if not self._importers[key]:
                        del self._importers[key]

        self._import_graph.pop(file_path, None)
        self._file_symbols.pop(file_path, None)
        self._file_imports.pop(file_path, None)

    def _add_file_to_indices(
        self,
        file_path: str,
        symbols: List[Symbol],
        imports: List[ImportReference],
    ) -> None:
        """Incrementally index symbols, callers, importers, and inheritance for a single file."""
        self._file_symbols[file_path] = symbols
        self._file_imports[file_path] = imports

        for sym in symbols:
            self._definitions[sym.id] = sym
            self._name_to_symbols.setdefault(sym.name, []).append(sym)
            if sym.qualname != sym.name:
                self._name_to_symbols.setdefault(sym.qualname, []).append(sym)

            for base_name in sym.bases:
                self._subclasses.setdefault(base_name, []).append(sym)

            for call in sym.calls:
                callee = call.callee
                self._callers.setdefault(callee, []).append(call)
                simple = callee.split(".")[-1].split("::")[-1]
                if simple != callee:
                    self._callers.setdefault(simple, []).append(call)

                if sym.qualname and "." in sym.qualname:
                    class_qualname = sym.qualname.rsplit(".", 1)[0]
                    if callee.startswith(("self.", "cls.")):
                        self._callers.setdefault(f"{class_qualname}.{simple}", []).append(call)
                        self._callers.setdefault(
                            f"{sym.file_path}::{class_qualname}.{simple}", []
                        ).append(call)

        for imp in imports:
            self._importers.setdefault(imp.name, []).append(imp)
            if imp.module:
                self._importers.setdefault(f"{imp.module}.{imp.name}", []).append(imp)

        # Update import graph edges for this file
        targets: List[str] = []
        for imp in imports:
            target_f = self.resolve_import_to_file(imp, file_path)
            if target_f and target_f != file_path and target_f not in targets:
                targets.append(target_f)
        self._import_graph[file_path] = targets

    def _rebuild_indices(self) -> None:
        """Rebuild definitions, name lookup, callers, importers, and inheritance indices."""
        self._definitions.clear()
        self._name_to_symbols.clear()
        self._callers.clear()
        self._importers.clear()
        self._subclasses.clear()
        self._import_graph.clear()

        for rel_path, file_data in self._file_cache.items():
            if rel_path not in self._file_symbols:
                self._file_symbols[rel_path] = [self._deserialize_symbol(s) for s in file_data.get("symbols", [])]
            if rel_path not in self._file_imports:
                self._file_imports[rel_path] = [self._deserialize_import(i) for i in file_data.get("imports", [])]

            self._add_file_to_indices(rel_path, self._file_symbols[rel_path], self._file_imports[rel_path])

    def resolve_import_to_file(
        self, imp: ImportReference, current_file: str
    ) -> Optional[str]:
        """
        Resolve an ImportReference from current_file to a concrete file path in the workspace.
        Handles relative imports (level > 0) and absolute workspace imports (level == 0).
        """
        available_files = set(self._file_cache.keys())
        cur_p = Path(current_file)
        cur_dir = cur_p.parent

        cur_ext = cur_p.suffix.lower()

        # TypeScript / JavaScript import resolution
        if cur_ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
            mod_str = imp.module or imp.name
            if mod_str.startswith("node:"):
                return None
            first_pkg = mod_str.split("/")[0]
            if first_pkg in {
                "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
                "constants", "crypto", "dgram", "diagnostics_channel", "dns", "domain",
                "events", "fs", "http", "http2", "https", "inspector", "module", "net",
                "os", "path", "perf_hooks", "process", "punycode", "querystring",
                "readline", "repl", "stream", "string_decoder", "timers", "tls",
                "trace_events", "tty", "url", "util", "v8", "vm", "wasi", "worker_threads",
                "zlib",
            }:
                return None

            rel_mod = mod_str.lstrip("./") if mod_str.startswith("./") else mod_str
            base = (cur_dir / rel_mod).as_posix()
            ts_cands = [
                base,
                f"{base}.ts",
                f"{base}.tsx",
                f"{base}.js",
                f"{base}.jsx",
                f"{base}/index.ts",
                f"{base}/index.tsx",
                f"{base}/index.js",
                rel_mod,
                f"{rel_mod}.ts",
                f"{rel_mod}.tsx",
                f"{rel_mod}.js",
                f"src/{rel_mod}.ts",
                f"src/{rel_mod}.tsx",
                f"src/{rel_mod}.js",
            ]
            for cand in ts_cands:
                norm = Path(cand).as_posix()
                if norm.startswith("./"):
                    norm = norm[2:]
                if norm in available_files:
                    return norm
            return None

        # Go import resolution
        if cur_ext == ".go":
            mod_path = imp.module or imp.name
            first_segment = mod_path.split("/")[0]
            # Standard library packages in Go should not resolve to workspace files
            go_stdlib = {
                "archive", "tar", "zip", "bufio", "builtin", "bytes", "compress",
                "bzip2", "flate", "gzip", "lzw", "zlib", "container", "heap", "list",
                "ring", "context", "crypto", "aes", "cipher", "des", "dsa", "ecdsa",
                "ed25519", "elliptic", "hmac", "md5", "rand", "rc4", "rsa", "sha1",
                "sha256", "sha512", "subtle", "tls", "x509", "database", "sql",
                "debug", "dwarf", "elf", "gosym", "macho", "pe", "plan9obj", "embed",
                "encoding", "ascii85", "asn1", "base32", "base64", "binary", "csv",
                "gob", "hex", "json", "pem", "xml", "errors", "expvar", "flag",
                "fmt", "go", "ast", "build", "constant", "doc", "format", "parser",
                "printer", "scanner", "token", "types", "hash", "adler32", "crc32",
                "crc64", "fnv", "maphash", "html", "template", "image", "color",
                "draw", "gif", "jpeg", "png", "index", "suffixarray", "io", "fs",
                "ioutil", "log", "slog", "syslog", "math", "big", "bits", "cmplx",
                "mime", "multipart", "quotedprintable", "net", "http", "cgi",
                "cookiejar", "fcgi", "httptest", "httptrace", "httputil", "pprof",
                "mail", "rpc", "jsonrpc", "smtp", "textproto", "url", "os", "exec",
                "signal", "user", "path", "filepath", "plugin", "reflect", "regexp",
                "syntax", "runtime", "cgo", "coverage", "metrics", "msan",
                "race", "trace", "sort", "strconv", "strings", "sync", "atomic",
                "syscall", "testing", "fstest", "iotest", "quick", "text", "tabwriter",
                "time", "tzdata", "unicode", "utf16", "utf8", "unsafe",
            }
            if first_segment in go_stdlib:
                return None

            # Relative Go imports
            if mod_path.startswith("./") or mod_path.startswith("../"):
                target_dir = (cur_dir / mod_path).resolve()
                for af in available_files:
                    if af.endswith(".go") and (self.workspace_root / af).parent.resolve() == target_dir:
                        return af
                return None

            # Package path matching
            pkg_target = mod_path.split("/")[-1]
            for af in available_files:
                if not af.endswith(".go"):
                    continue
                af_dir = Path(af).parent
                if af_dir == cur_dir:
                    continue
                if af_dir.name == pkg_target or af_dir.as_posix().endswith(mod_path):
                    return af
            return None

        # Rust import resolution
        if cur_ext == ".rs":
            mod_target = (imp.module or imp.name)
            first_crate = mod_target.split("::")[0]
            if first_crate in {"std", "core", "alloc", "proc_macro", "test"}:
                return None

            cleaned_mod = mod_target.replace("crate::", "").replace("super::", "../").replace("self::", "").replace("::", "/")
            rs_cands = [
                f"{cur_dir / cleaned_mod}.rs",
                f"{cur_dir / cleaned_mod}/mod.rs",
                f"{cur_dir / imp.name}.rs",
                f"{cur_dir / imp.name}/mod.rs",
                f"src/{cleaned_mod}.rs",
                f"src/{cleaned_mod}/mod.rs",
                f"src/{imp.name}.rs",
                f"src/{imp.name}/mod.rs",
                f"{cleaned_mod}.rs",
                f"{imp.name}.rs",
            ]
            for cand in rs_cands:
                norm = Path(cand).as_posix()
                if norm.startswith("./"):
                    norm = norm[2:]
                if norm in available_files:
                    return norm
            return None

        # Python import resolution (default)
        candidates = []

        if imp.level > 0:
            # Relative import
            target_dir = cur_dir
            for _ in range(imp.level - 1):
                target_dir = target_dir.parent

            if imp.module:
                rel_mod = imp.module.replace(".", "/")
                base = target_dir / rel_mod
                candidates.extend([
                    f"{base}.py",
                    f"{base}/__init__.py",
                    f"{base / imp.name}.py",
                    f"{base / imp.name}/__init__.py",
                ])
            else:
                candidates.extend([
                    f"{target_dir / imp.name}.py",
                    f"{target_dir / imp.name}/__init__.py",
                    f"{target_dir}/__init__.py",
                ])
        else:
            # Absolute import
            mods = []
            if imp.module:
                mods.append(imp.module)
                mods.append(f"{imp.module}.{imp.name}")
            else:
                mods.append(imp.name)

            for m in mods:
                rel_m = m.replace(".", "/")
                candidates.extend([
                    f"{rel_m}.py",
                    f"{rel_m}/__init__.py",
                    f"src/{rel_m}.py",
                    f"src/{rel_m}/__init__.py",
                ])

        for cand in candidates:
            norm = Path(cand).as_posix()
            if norm.startswith("./"):
                norm = norm[2:]
            if norm in available_files:
                return norm
        return None

    def resolve_callee(
        self, call: CallReference, caller_sym: Optional[Symbol] = None
    ) -> Optional[Symbol]:
        """
        High-accuracy resolution of a call site to its Symbol definition.
        Resolves direct calls, method self/cls invocations, and qualified accesses.
        """
        callee = call.callee

        # 1. Exact ID or qualname match
        if callee in self._definitions:
            return self._definitions[callee]

        # 2. Self or cls call inside a class
        if callee.startswith(("self.", "cls.")) and caller_sym:
            attr = callee.split(".", 1)[1]
            if caller_sym.qualname and "." in caller_sym.qualname:
                cls_qualname = caller_sym.qualname.rsplit(".", 1)[0]
                full_id = f"{caller_sym.file_path}::{cls_qualname}.{attr}"
                if full_id in self._definitions:
                    return self._definitions[full_id]
                matches = self._name_to_symbols.get(f"{cls_qualname}.{attr}", [])
                if matches:
                    return matches[0]

            same_file = f"{caller_sym.file_path}::{attr}"
            if same_file in self._definitions:
                return self._definitions[same_file]
            return self.get_definition(attr)

        # 3. Dotted or scoped callee: ClassName.method, mod.func, or mod::func
        if "." in callee or "::" in callee:
            sep = "::" if "::" in callee else "."
            if caller_sym:
                prefix, attr = callee.split(sep, 1)
                caller_file = caller_sym.file_path
                f_data = self._file_cache.get(caller_file, {})
                for imp_data in f_data.get("imports", []):
                    imp = self._deserialize_import(imp_data)
                    match_alias = imp.asname and imp.asname == prefix
                    match_name = not imp.asname and (imp.name == prefix or (imp.module and imp.name == prefix))
                    if match_alias or match_name:
                        target_f = self.resolve_import_to_file(imp, caller_file)
                        if target_f:
                            target_id = f"{target_f}::{attr}"
                            if target_id in self._definitions:
                                return self._definitions[target_id]

            match = self.get_definition(callee)
            if match:
                return match
            attr = callee.split(sep)[-1]
            if caller_sym:
                same_file = f"{caller_sym.file_path}::{attr}"
                if same_file in self._definitions:
                    return self._definitions[same_file]
            return self.get_definition(attr)

        # 4. Plain name call
        if caller_sym:
            same_file = f"{caller_sym.file_path}::{callee}"
            if same_file in self._definitions:
                return self._definitions[same_file]

            # Check file imports
            caller_file = caller_sym.file_path
            f_data = self._file_cache.get(caller_file, {})
            for imp_data in f_data.get("imports", []):
                imp = self._deserialize_import(imp_data)
                if (imp.asname and imp.asname == callee) or (not imp.asname and imp.name == callee):
                    target_f = self.resolve_import_to_file(imp, caller_file)
                    if target_f:
                        target_id = f"{target_f}::{imp.name}"
                        if target_id in self._definitions:
                            return self._definitions[target_id]

        return self.get_definition(callee)

    def get_definition(self, symbol_id_or_name: str) -> Optional[Symbol]:
        """Look up symbol by full ID or name."""
        if symbol_id_or_name in self._definitions:
            return self._definitions[symbol_id_or_name]
        matches = self._name_to_symbols.get(symbol_id_or_name, [])
        return matches[0] if matches else None

    def get_symbols_by_name(self, name: str) -> List[Symbol]:
        """Find all symbols matching a name across the workspace."""
        return self._name_to_symbols.get(name, [])

    def get_callers(self, symbol_name_or_qualname: str) -> List[CallReference]:
        """Find all call references targeting this symbol name or qualname."""
        results: List[CallReference] = []
        seen = set()

        def add_call(c: CallReference):
            k = (c.caller, c.lineno, c.callee, c.args_count, tuple(c.kwargs))
            if k not in seen:
                seen.add(k)
                results.append(c)

        if symbol_name_or_qualname in self._callers:
            for c in self._callers[symbol_name_or_qualname]:
                add_call(c)

        simple_name = symbol_name_or_qualname.split(".")[-1].split("::")[-1]
        if simple_name != symbol_name_or_qualname and simple_name in self._callers:
            for c in self._callers[simple_name]:
                add_call(c)

        return results

    def get_importers(self, symbol_name_or_qualname: str) -> List[ImportReference]:
        """Find all import references targeting this symbol name or qualname."""
        results: List[ImportReference] = []
        seen = set()

        def add_imp(imp: ImportReference):
            k = (imp.file_path, imp.lineno, imp.name, imp.module)
            if k not in seen:
                seen.add(k)
                results.append(imp)

        if symbol_name_or_qualname in self._importers:
            for imp in self._importers[symbol_name_or_qualname]:
                add_imp(imp)

        simple_name = symbol_name_or_qualname.split(".")[-1]
        if simple_name != symbol_name_or_qualname and simple_name in self._importers:
            for imp in self._importers[simple_name]:
                add_imp(imp)

        return results

    def get_subclasses(self, class_name: str) -> List[Symbol]:
        """Find all known subclasses inheriting from class_name."""
        return self._subclasses.get(class_name, [])

    def resolve_class_init(self, class_sym: Symbol) -> Optional[Symbol]:
        """Find constructor definition (__init__ for Python, constructor for TS/JS), checking inheritance."""
        # 1. Direct constructor in class
        init_names = ["__init__", "constructor"]
        for init_name in init_names:
            init_id = f"{class_sym.file_path}::{class_sym.qualname}.{init_name}"
            if init_id in self._definitions:
                return self._definitions[init_id]
            direct_match = self.get_definition(f"{class_sym.qualname}.{init_name}")
            if direct_match:
                return direct_match

        # 2. Check base classes
        for base_name in getattr(class_sym, "bases", []):
            base_sym = self.get_definition(base_name)
            if base_sym and base_sym.kind == "class":
                base_init = self.resolve_class_init(base_sym)
                if base_init:
                    return base_init
        return None

    def get_file_symbols(self, file_path: str) -> List[Symbol]:
        """Return all symbols in a given file."""
        p = Path(file_path)
        if p.is_absolute():
            try:
                clean_path = str(p.resolve().relative_to(self.workspace_root.resolve())).replace("\\", "/")
            except ValueError:
                clean_path = str(file_path).replace("\\", "/")
        else:
            full_p = (self.workspace_root / file_path).resolve()
            try:
                clean_path = str(full_p.relative_to(self.workspace_root.resolve())).replace("\\", "/")
            except ValueError:
                clean_path = str(file_path).replace("\\", "/")

        cached = self._file_cache.get(clean_path)
        if not cached:
            return []
        return [self._deserialize_symbol(s) for s in cached.get("symbols", [])]

    def overlay_transient_symbols(
        self,
        file_path: str,
        symbols: List[Symbol],
        imports: Optional[List[ImportReference]] = None,
    ) -> None:
        """
        In-memory overlay of modified symbols for a file without modifying disk.
        Allows zero-side-effect evaluation of unverified patches.
        """
        try:
            p = Path(file_path)
            clean_path = (
                str(p.resolve().relative_to(self.workspace_root.resolve())).replace("\\", "/")
                if p.is_absolute()
                else str(file_path).replace("\\", "/")
            )
        except ValueError:
            clean_path = str(file_path).replace("\\", "/")

        self._remove_file_from_indices(clean_path)
        self._add_file_to_indices(clean_path, symbols, imports or [])
        self._file_cache[clean_path] = {
            "mtime": 0.0,
            "hash": "transient",
            "symbols": [self._serialize_symbol(s) for s in symbols],
            "imports": [self._serialize_import(imp) for imp in (imports or [])],
        }

    def restore_transient_symbols(
        self,
        file_path: str,
        backup_symbols: List[Symbol],
        backup_imports: List[ImportReference],
        cached_backup: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fast incremental rollback of transient symbols and imports to restore default disk state.
        """
        try:
            p = Path(file_path)
            clean_path = (
                str(p.resolve().relative_to(self.workspace_root.resolve())).replace("\\", "/")
                if p.is_absolute()
                else str(file_path).replace("\\", "/")
            )
        except ValueError:
            clean_path = str(file_path).replace("\\", "/")

        self._remove_file_from_indices(clean_path)
        if backup_symbols or backup_imports:
            self._add_file_to_indices(clean_path, backup_symbols, backup_imports)

        if cached_backup is not None:
            self._file_cache[clean_path] = cached_backup
        else:
            self._file_cache.pop(clean_path, None)
