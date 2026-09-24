"""
Comprehensive unit and integration tests for multi-language AST extraction
across Python, TypeScript/JavaScript, Go, and Rust.
"""

import tempfile
from pathlib import Path

from code_oracle.engine import TopoSliceEngine
from code_oracle.indexer import WorkspaceIndexer
from code_oracle.languages import (
    SUPPORTED_EXTENSIONS,
    detect_language,
    extract_imports,
    extract_symbols,
    validate_syntax,
)
from code_oracle.locator import locate_affected_symbols
from code_oracle.slicer import slice_neighborhood
from code_oracle.linearizer import linearize_subgraph


# ============================================================================
# 1. Language Detection & Syntax Validation Tests
# ============================================================================

def test_detect_language():
    assert detect_language("foo.py") == "python"
    assert detect_language("service.ts") == "typescript"
    assert detect_language("component.tsx") == "typescript"
    assert detect_language("index.js") == "javascript"
    assert detect_language("view.jsx") == "javascript"
    assert detect_language("server.go") == "go"
    assert detect_language("main.rs") == "rust"
    assert detect_language("readme.md") is None
    assert detect_language("") == "python"


def test_syntax_validation():
    # Valid syntax
    assert validate_syntax("def f(): pass\n", "foo.py") is None
    assert validate_syntax("function f() {}\n", "foo.ts") is None
    assert validate_syntax("package main\nfunc main() {}\n", "foo.go") is None
    assert validate_syntax("fn main() {}\n", "foo.rs") is None

    # Invalid syntax
    err_py = validate_syntax("def f(;\n", "foo.py")
    assert err_py is not None and "SyntaxError" in err_py

    err_ts = validate_syntax("function f( { ::: \n", "foo.ts")
    assert err_ts is not None and "SyntaxError" in err_ts

    err_go = validate_syntax("package main\nfunc ( { :::\n", "foo.go")
    assert err_go is not None and "SyntaxError" in err_go

    err_rs = validate_syntax("fn main( { :::\n", "foo.rs")
    assert err_rs is not None and "SyntaxError" in err_rs


# ============================================================================
# 2. TypeScript / JavaScript AST Extraction Tests
# ============================================================================

def test_typescript_functions_and_classes():
    ts_code = """
import { helper, Client as BaseClient } from './utils';
import defItem from './defaultMod';
const reqMod = require('path');

export interface User {
    id: string;
}

export class UserService extends BaseClient implements IUser {
    private db: any;

    constructor(db: any, options: object = {}) {
        super(db);
        this.db = db;
    }

    static getInstance(): UserService {
        return new UserService(null);
    }

    async getUser(id: string, detail: boolean = false): Promise<User> {
        helper(id);
        this.save({ key: 'val' });
        return null;
    }

    private save(data: any): void {}
}

export function calculateTax(amount: number, rate: number = 0.1): number {
    return amount * rate;
}

export const arrowMultiplier = (x: number, y: number = 2) => {
    return x * y;
};
"""
    symbols = extract_symbols(ts_code, "service.ts")
    sym_map = {s.name: s for s in symbols}

    # Verify class
    assert "UserService" in sym_map
    user_svc = sym_map["UserService"]
    assert user_svc.kind == "class"
    assert "BaseClient" in user_svc.bases
    assert "IUser" in user_svc.bases

    # Verify methods
    methods = [s for s in symbols if s.is_method]
    method_names = {m.name for m in methods}
    assert "constructor" in method_names
    assert "getUser" in method_names
    assert "save" in method_names

    # Check getUser method attributes
    get_user = next(s for s in symbols if s.name == "getUser")
    assert get_user.qualname == "UserService.getUser"
    assert get_user.min_args == 1  # 'detail' has default
    assert get_user.max_args == 2
    assert any(c.callee == "helper" for c in get_user.calls)

    # Check static method
    get_instance = next(s for s in symbols if s.name == "getInstance")
    assert get_instance.is_static is True
    assert get_instance.is_method is False

    # Check standalone and arrow functions
    assert "calculateTax" in sym_map
    tax_fn = sym_map["calculateTax"]
    assert tax_fn.min_args == 1
    assert tax_fn.max_args == 2

    assert "arrowMultiplier" in sym_map
    arrow_fn = sym_map["arrowMultiplier"]
    assert arrow_fn.min_args == 1
    assert arrow_fn.max_args == 2

    # Verify imports
    imports = extract_imports(ts_code, "service.ts")
    imp_names = {i.name for i in imports}
    assert "helper" in imp_names
    assert "Client" in imp_names
    assert "defItem" in imp_names
    assert any(i.asname == "BaseClient" for i in imports)


def test_typescript_variadic_and_optional():
    ts_code = """
function variadicLog(prefix: string, ...messages: string[]): void {
    console.log(prefix, ...messages);
}

function optionalConfig(name: string, timeout?: number): void {
    variadicLog(name);
}
"""
    symbols = extract_symbols(ts_code, "logger.ts")
    sym_map = {s.name: s for s in symbols}

    var_log = sym_map["variadicLog"]
    assert var_log.min_args == 1
    assert var_log.max_args is None
    assert any(p.is_vararg for p in var_log.params)

    opt_cfg = sym_map["optionalConfig"]
    assert opt_cfg.min_args == 1
    assert opt_cfg.max_args == 2
    assert opt_cfg.params[1].has_default is True


# ============================================================================
# 3. Go AST Extraction Tests
# ============================================================================

def test_go_ast_extraction():
    go_code = """
package main

import (
    "fmt"
    calc "math/rand"
)

type Server struct {
    port int
}

type Runner interface {
    Run() error
}

func (s *Server) Start(host string, timeout int) error {
    fmt.Println(host)
    calc.Intn(timeout)
    return nil
}

func Add(a, b int, rest ...int) (int, error) {
    return a + b, nil
}
"""
    symbols = extract_symbols(go_code, "server.go")
    sym_map = {s.name: s for s in symbols}

    # Verify structs and interfaces
    assert "Server" in sym_map
    assert sym_map["Server"].kind == "struct"
    assert "Runner" in sym_map
    assert sym_map["Runner"].kind == "interface"

    # Verify method
    assert "Start" in sym_map
    start_method = sym_map["Start"]
    assert start_method.is_method is True
    assert start_method.qualname == "Server.Start"
    assert start_method.min_args == 2
    assert start_method.max_args == 2
    assert any(c.callee == "fmt.Println" for c in start_method.calls)
    assert any(c.callee == "calc.Intn" for c in start_method.calls)

    # Verify function with multi-parameter declaration and variadic
    assert "Add" in sym_map
    add_fn = sym_map["Add"]
    assert add_fn.is_method is False
    assert add_fn.min_args == 2
    assert add_fn.max_args is None
    assert any(p.is_vararg for p in add_fn.params)

    # Verify imports
    imports = extract_imports(go_code, "server.go")
    imp_map = {i.name: i for i in imports}
    assert "fmt" in imp_map
    assert "rand" in imp_map or "calc" in imp_map
    if "calc" in imp_map:
        assert imp_map["calc"].asname == "calc"


# ============================================================================
# 4. Rust AST Extraction Tests
# ============================================================================

def test_rust_ast_extraction():
    rust_code = """
use std::collections::HashMap;
use crate::models::{User, Account as Acc};
mod helper;

pub struct Config {
    pub port: u16,
}

pub trait Service {
    fn execute(&self) -> bool;
}

impl Config {
    pub fn new(port: u16) -> Self {
        Config { port }
    }

    pub fn start(&self, host: &str) -> bool {
        log(host);
        self.listen();
        true
    }

    fn listen(&self) {}
}

pub fn log(msg: &str) {
    println!("{}", msg);
}
"""
    symbols = extract_symbols(rust_code, "service.rs")
    sym_map = {s.name: s for s in symbols}

    # Verify struct and trait
    assert "Config" in sym_map
    assert sym_map["Config"].kind == "struct"
    assert "Service" in sym_map
    assert sym_map["Service"].kind == "interface"

    # Verify impl methods
    assert "new" in sym_map
    new_fn = sym_map["new"]
    assert new_fn.qualname == "Config.new"
    assert new_fn.is_static is True
    assert new_fn.is_method is False
    assert new_fn.min_args == 1

    assert "start" in sym_map
    start_method = sym_map["start"]
    assert start_method.qualname == "Config.start"
    assert start_method.is_method is True
    assert start_method.min_args == 1  # 1 formal arg excluding self
    assert any(c.callee == "log" for c in start_method.calls)
    assert any(c.callee == "self.listen" for c in start_method.calls)

    # Verify standalone function
    assert "log" in sym_map
    log_fn = sym_map["log"]
    assert log_fn.min_args == 1
    assert log_fn.max_args == 1

    # Verify use and mod imports
    imports = extract_imports(rust_code, "service.rs")
    imp_names = {i.name for i in imports}
    assert "HashMap" in imp_names
    assert "User" in imp_names
    assert "Account" in imp_names or "Acc" in imp_names
    assert "helper" in imp_names


# ============================================================================
# 5. Boundary Locator Across Languages
# ============================================================================

def test_locator_multilang():
    # TypeScript locator
    ts_orig = "export function run(x: number): number {\n    return x + 1;\n}\n"
    ts_patch = "export function run(x: number): number {\n    // modified\n    return x + 2;\n}\n"
    res_ts = locate_affected_symbols("main.ts", ts_patch, original_content=ts_orig)
    assert res_ts.syntax_error is None
    assert len(res_ts.affected_symbols) == 1
    assert res_ts.affected_symbols[0].name == "run"

    # Go locator
    go_orig = "package main\n\nfunc Run() int {\n    return 1\n}\n"
    go_patch = "package main\n\nfunc Run() int {\n    return 2\n}\n"
    res_go = locate_affected_symbols("main.go", go_patch, original_content=go_orig)
    assert res_go.syntax_error is None
    assert len(res_go.affected_symbols) == 1
    assert res_go.affected_symbols[0].name == "Run"

    # Rust locator
    rs_orig = "pub fn run() -> i32 {\n    1\n}\n"
    rs_patch = "pub fn run() -> i32 {\n    2\n}\n"
    res_rs = locate_affected_symbols("main.rs", rs_patch, original_content=rs_orig)
    assert res_rs.syntax_error is None
    assert len(res_rs.affected_symbols) == 1
    assert res_rs.affected_symbols[0].name == "run"


# ============================================================================
# 6. TopoSlice Verification Engine on Multi-Language Projects
# ============================================================================

def test_engine_typescript_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        engine = TopoSliceEngine(workspace_root=tmp)

        (tmp / "utils.ts").write_text(
            "export function formatName(name: string): string {\n"
            "    return name.trim();\n"
            "}\n"
        )
        (tmp / "main.ts").write_text(
            "import { formatName } from './utils';\n\n"
            "export function execute() {\n"
            "    return formatName('Alice');\n"
            "}\n"
        )
        engine.indexer.scan_workspace()

        # 1. Clean patch -> APPROVED
        pass_patch = (
            "export function formatName(name: string): string {\n"
            "    // Safe log\n"
            "    return name.trim();\n"
            "}\n"
        )
        rep_pass = engine.verify("utils.ts", pass_patch)
        assert rep_pass.status == "APPROVED"
        assert len(rep_pass.invariant_violations) == 0
        assert "[DIFF_TARGET] utils.ts::formatName (MODIFIED)" in rep_pass.linearized_subgraph

        # 2. Arity breaking patch -> REJECTED
        arity_patch = (
            "export function formatName(name: string, prefix: string): string {\n"
            "    return prefix + name.trim();\n"
            "}\n"
        )
        rep_arity = engine.verify("utils.ts", arity_patch)
        assert rep_arity.status == "REJECTED"
        assert any("ARITY_MISMATCH" in v for v in rep_arity.invariant_violations)

        # 3. Circular dependency patch -> REJECTED
        circ_patch = (
            "import { execute } from './main';\n\n"
            "export function formatName(name: string): string {\n"
            "    return name.trim();\n"
            "}\n"
        )
        rep_circ = engine.verify("utils.ts", circ_patch)
        assert rep_circ.status == "REJECTED"
        assert any("CIRCULAR_DEPENDENCY" in v for v in rep_circ.invariant_violations)


def test_engine_go_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        engine = TopoSliceEngine(workspace_root=tmp)

        (tmp / "calc.go").write_text(
            "package main\n\n"
            "func Add(a, b int) int {\n"
            "    return a + b\n"
            "}\n"
        )
        (tmp / "main.go").write_text(
            "package main\n\n"
            "func Run() int {\n"
            "    return Add(10, 20)\n"
            "}\n"
        )
        engine.indexer.scan_workspace()

        # Clean patch
        pass_patch = (
            "package main\n\n"
            "func Add(a, b int) int {\n"
            "    // comment\n"
            "    return a + b\n"
            "}\n"
        )
        rep_pass = engine.verify("calc.go", pass_patch)
        assert rep_pass.status == "APPROVED"

        # Arity breaking
        arity_patch = (
            "package main\n\n"
            "func Add(a, b, c int) int {\n"
            "    return a + b + c\n"
            "}\n"
        )
        rep_arity = engine.verify("calc.go", arity_patch)
        assert rep_arity.status == "REJECTED"
        assert any("ARITY_MISMATCH" in v for v in rep_arity.invariant_violations)


def test_engine_rust_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        engine = TopoSliceEngine(workspace_root=tmp)

        (tmp / "lib.rs").write_text(
            "pub fn compute(x: i32) -> i32 {\n"
            "    x * 2\n"
            "}\n"
        )
        (tmp / "main.rs").write_text(
            "mod lib;\n\n"
            "fn run() -> i32 {\n"
            "    lib::compute(10)\n"
            "}\n"
        )
        engine.indexer.scan_workspace()

        # Clean pass
        pass_patch = (
            "pub fn compute(x: i32) -> i32 {\n"
            "    // comment\n"
            "    x * 2\n"
            "}\n"
        )
        rep_pass = engine.verify("lib.rs", pass_patch)
        assert rep_pass.status == "APPROVED"

        # Deleted symbol referenced by caller
        del_patch = (
            "// compute function deleted\n"
            "pub fn other() {}\n"
        )
        rep_del = engine.verify("lib.rs", del_patch)
        assert rep_del.status == "REJECTED"
        assert any("BROKEN_REFERENCE" in v for v in rep_del.invariant_violations)


def test_typescript_constructor_instantiation_and_contract_break():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        engine = TopoSliceEngine(workspace_root=tmp)

        (tmp / "user.ts").write_text(
            "export class User {\n"
            "    constructor(public id: string, public name: string) {}\n"
            "}\n"
        )
        (tmp / "main.ts").write_text(
            "import { User } from './user';\n\n"
            "export function makeUser() {\n"
            "    return new User('1', 'Alice');\n"
            "}\n"
        )
        engine.indexer.scan_workspace()

        # 1. Clean patch must be APPROVED (constructor accepts 2 args)
        clean_patch = (
            "export class User {\n"
            "    // Safe comment\n"
            "    constructor(public id: string, public name: string) {}\n"
            "}\n"
        )
        rep_clean = engine.verify("user.ts", clean_patch)
        assert rep_clean.status == "APPROVED"
        assert len(rep_clean.invariant_violations) == 0

        # 2. Breaking constructor signature change must be REJECTED (requires token)
        break_patch = (
            "export class User {\n"
            "    constructor(public id: string, public name: string, public token: string) {}\n"
            "}\n"
        )
        rep_break = engine.verify("user.ts", break_patch)
        assert rep_break.status == "REJECTED"
        assert any("ARITY_MISMATCH" in v for v in rep_break.invariant_violations)


def test_typescript_enums_constants_and_dangling_comment_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        engine = TopoSliceEngine(workspace_root=tmp)

        (tmp / "constants.ts").write_text(
            "// Note: nonexistentSymbol might be added in the future\n"
            "export enum Status { ACTIVE, INACTIVE }\n"
            "export const MAX_RETRIES = 5;\n"
            "export function getStatus() { return Status.ACTIVE; }\n"
        )
        (tmp / "app.ts").write_text(
            "import { Status, MAX_RETRIES } from './constants';\n\n"
            "export function isMax(c: number) { return c >= MAX_RETRIES; }\n"
        )
        engine.indexer.scan_workspace()

        # Check extracted symbols in index
        syms = engine.indexer.get_file_symbols("constants.ts")
        sym_names = {s.name for s in syms}
        assert "Status" in sym_names
        assert "MAX_RETRIES" in sym_names
        assert any(s.kind == "enum" for s in syms)
        assert any(s.kind == "constant" for s in syms)

        # Patch importing nonexistent symbol must be REJECTED even if present in comments
        bad_patch = (
            "import { Status, nonexistentSymbol } from './constants';\n"
            "export function test() { return Status.ACTIVE; }\n"
        )
        rep = engine.verify("app.ts", bad_patch)
        assert rep.status == "REJECTED"
        assert any("BROKEN_REFERENCE" in v and "nonexistentSymbol" in v for v in rep.invariant_violations)


def test_go_stdlib_imports_no_false_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "calc.go").write_text(
            "package main\n\n"
            "import (\n"
            "    \"strings\"\n"
            ")\n\n"
            "func Add(a, b string) string {\n"
            "    return strings.TrimSpace(a) + strings.TrimSpace(b)\n"
            "}\n"
        )
        (tmp / "main.go").write_text(
            "package main\n\n"
            "import (\n"
            "    \"fmt\"\n"
            ")\n\n"
            "func Run() {\n"
            "    fmt.Println(Add(\"hello \", \"world\"))\n"
            "}\n"
        )
        engine = TopoSliceEngine(workspace_root=tmp)
        engine.indexer.scan_workspace()

        # Verify a safe patch to calc.go does not trigger false circular dependency
        safe_patch = (
            "package main\n\n"
            "import (\n"
            "    \"strings\"\n"
            ")\n\n"
            "func Add(a, b string) string {\n"
            "    // Safe log comment\n"
            "    return strings.TrimSpace(a) + strings.TrimSpace(b)\n"
            "}\n"
        )
        rep = engine.verify("calc.go", safe_patch)
        assert rep.status == "APPROVED"
        assert len(rep.invariant_violations) == 0


def test_go_const_var_and_type_aliases():
    go_code = """
package main

const Port = 8080
const (
    A = 1
    B = 2
)
var Config = "test"
type UserID string
type Runner interface { Run() }
type Server struct { port int }
"""
    symbols = extract_symbols(go_code, "config.go")
    sym_map = {s.name: s for s in symbols}

    assert "Port" in sym_map and sym_map["Port"].kind == "constant"
    assert "A" in sym_map and sym_map["A"].kind == "constant"
    assert "B" in sym_map and sym_map["B"].kind == "constant"
    assert "Config" in sym_map and sym_map["Config"].kind == "variable"
    assert "UserID" in sym_map and sym_map["UserID"].kind == "type_alias"
    assert "Runner" in sym_map and sym_map["Runner"].kind == "interface"
    assert "Server" in sym_map and sym_map["Server"].kind == "struct"


def test_rust_generic_impl_and_constants():
    rust_code = """
pub struct Wrapper<T> {
    val: T,
}

pub trait Process<T> {
    fn process(&self, item: T) -> bool;
}

impl<T: std::fmt::Display> Process<T> for Wrapper<T> {
    fn process(&self, item: T) -> bool {
        true
    }
}

pub const MAX_RETRIES: u32 = 3;
pub static INSTANCE: i32 = 1;
pub type UserResult<T> = Result<T, String>;
"""
    symbols = extract_symbols(rust_code, "lib.rs")
    sym_map = {s.name: s for s in symbols}

    assert "Wrapper" in sym_map and sym_map["Wrapper"].kind == "struct"
    assert "Process" in sym_map and sym_map["Process"].kind == "interface"
    assert "process" in sym_map
    proc_method = sym_map["process"]
    assert proc_method.qualname == "Wrapper.process"
    assert "Process" in proc_method.bases
    assert "MAX_RETRIES" in sym_map and sym_map["MAX_RETRIES"].kind == "constant"
    assert "INSTANCE" in sym_map and sym_map["INSTANCE"].kind == "variable"
    assert "UserResult" in sym_map and sym_map["UserResult"].kind == "type_alias"

