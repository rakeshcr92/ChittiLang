#!/usr/bin/env python3
"""
TeluguLang v0.3
================

New in v0.3 (this release):
---------------------------
✅ **Enhanced `cheppu`** – comma‑separated multiple print args.
✅ **Functions** – `seva name(params) { ... }` to define; `tirugu expr` to return.
✅ **Function Calls** – `name(arg1, arg2)` in expressions.
✅ **Interactive REPL** – run with no file to enter REPL (`.exit` to quit).

Existing Core:
--------------
* Variable assignment:     `veru x = 10`
* Print:                   `cheppu x`  OR  `cheppu "X=", x`
* While loop:              `podugu (x > 0) { ... }`
* If/Else:                 `ayithe (...) { ... } lekapothe { ... }`
* Numbers, strings, booleans (`nijam`, `abaddam`)
* Arithmetic + comparisons
* Line comments beginning `#`
* Bilingual error messages (English + తెలుగు translit)

Usage:
------
    python telugulang.py program.tlg
    python telugulang.py            # REPL

Example (functions + if/else):
--------------------------------
veru n = 5

seva fact(x) {
    ayithe (x <= 1) {
        tirugu 1
    }
    tirugu x * fact(x - 1)
}

cheppu "fact(", n, ") =", fact(n)
"""

import sys
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Callable

########################################
# Keywords & Tokens
########################################

KEYWORDS = {
    "cheppu": "PRINT",      # print
    "veru": "ASSIGN",        # var assign
    "podugu": "WHILE",       # while
    "ayithe": "IF",          # if
    "lekapothe": "ELSE",     # else
    "nijam": "TRUE",         # true
    "abaddam": "FALSE",      # false
    "seva": "FUNCTION",      # function def
    "tirugu": "RETURN",      # return
}

TOKEN_SPEC = [
    ("NUMBER",      r"\d+(?:\.\d+)?"),
    ("STRING",      r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""),
    ("ID",          r"[A-Za-z_][A-Za-z0-9_]*"),
    ("EQ",          r"=="),
    ("NE",          r"!="),
    ("GE",          r">="),
    ("LE",          r"<="),
    ("GT",          r">"),
    ("LT",          r"<"),
    ("ASSIGN_OP",   r"="),
    ("PLUS",        r"\+"),
    ("MINUS",       r"-"),
    ("STAR",        r"\*"),
    ("SLASH",       r"/"),
    ("PERCENT",     r"%"),
    ("COMMA",       r","),
    ("L_PAREN",     r"\("),
    ("R_PAREN",     r"\)"),
    ("L_BRACE",     r"\{"),
    ("R_BRACE",     r"\}"),
    ("SEMICOLON",   r";"),
    ("NEWLINE",     r"\n"),
    ("SKIP",        r"[ \t\r]+"),
    ("MISMATCH",    r"."),  # must be last
]

TOKEN_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC),
    re.UNICODE,
)

########################################
# Token class & Lexer
########################################

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int
    def __repr__(self):  # pragma: no cover
        return f"Token({self.type}, {self.value!r}, ln={self.line}, col={self.column})"


def lex(source: str) -> List[Token]:
    """Convert source to tokens. Strips `#` comments to end of line."""
    tokens: List[Token] = []
    lines = source.splitlines(keepends=True)
    cleaned = []
    for line in lines:
        idx = line.find('#')
        if idx != -1:
            # preserve newline if present
            newline = "\n" if line.endswith("\n") else ""
            line = line[:idx] + newline
        cleaned.append(line)
    cleaned_source = "".join(cleaned)

    line_num = 1
    line_start = 0
    for mo in TOKEN_REGEX.finditer(cleaned_source):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start + 1

        if kind == "NEWLINE":
            tokens.append(Token("NEWLINE", value, line_num, column))
            line_num += 1
            line_start = mo.end()
            continue
        elif kind == "SKIP":
            continue
        elif kind == "ID":
            lower = value.lower()
            if lower in KEYWORDS:
                tokens.append(Token(KEYWORDS[lower], value, line_num, column))
            else:
                tokens.append(Token("IDENT", value, line_num, column))
            continue
        elif kind == "NUMBER":
            tokens.append(Token("NUMBER", value, line_num, column))
            continue
        elif kind == "STRING":
            tokens.append(Token("STRING", value, line_num, column))
            continue
        elif kind == "MISMATCH":
            raise TLScanError(line_num, column, value)
        else:
            tokens.append(Token(kind, value, line_num, column))

    tokens.append(Token("EOF", "", line_num, 1))
    return tokens

########################################
# Error Classes (bilingual)
########################################

class TeluguLangError(Exception):
    pass

class TLScanError(TeluguLangError):
    def __init__(self, line: int, col: int, bad: str):
        self.line = line
        self.col = col
        self.bad = bad
    def __str__(self):
        return (
            f"Lexical error at line {self.line}, column {self.col}: unexpected character {self.bad!r}.\n"
            f"తెలుగు: వరుస {self.line}, కాలమ్ {self.col} వద్ద గుర్తు తెలియని అక్షరం {self.bad!r}."
        )

class TLParseError(TeluguLangError):
    def __init__(self, token: Token, message_en: str, message_te: str):
        self.token = token
        self.message_en = message_en
        self.message_te = message_te
    def __str__(self):  # pragma: no cover
        return (
            f"Syntax error near {self.token.type}({self.token.value!r}) at line {self.token.line}, col {self.token.column}: {self.message_en}\n"
            f"తెలుగు: లైన్ {self.token.line}, కాలమ్ {self.token.column}: {self.message_te}"
        )

class TLRuntimeError(TeluguLangError):
    def __init__(self, message_en: str, message_te: str):
        self.message_en = message_en
        self.message_te = message_te
    def __str__(self):  # pragma: no cover
        return f"Runtime error: {self.message_en}\nతెలుగు: {self.message_te}"

# Internal signal for returning from a function
class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

########################################
# AST Nodes
########################################

@dataclass
class Node: ...

@dataclass
class Program(Node):
    statements: List[Node]

@dataclass
class PrintStmt(Node):
    expressions: List[Node]  # multi‑arg print

@dataclass
class AssignStmt(Node):
    name: str
    expr: Node

@dataclass
class WhileStmt(Node):
    condition: Node
    body: List[Node]

@dataclass
class IfStmt(Node):
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]]

@dataclass
class FuncDecl(Node):
    name: str
    params: List[str]
    body: List[Node]

@dataclass
class ReturnStmt(Node):
    expr: Optional[Node]  # None ⇒ return None

@dataclass
class CallExpr(Node):
    callee: str
    args: List[Node]

@dataclass
class BinOp(Node):
    left: Node
    op: str
    right: Node

@dataclass
class UnaryOp(Node):
    op: str
    operand: Node

@dataclass
class Literal(Node):
    value: Any

@dataclass
class Var(Node):
    name: str

########################################
# Parser
########################################

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # token helpers --------------------------------------------------
    def current(self) -> Token:
        return self.tokens[self.pos]
    def advance(self) -> Token:
        tok = self.current()
        if tok.type != "EOF":
            self.pos += 1
        return tok
    def match(self, *types: str) -> bool:
        if self.current().type in types:
            self.advance(); return True
        return False
    def expect(self, ttype: str, en: str, te: str) -> Token:
        tok = self.current()
        if tok.type != ttype:
            raise TLParseError(tok, en, te)
        self.advance(); return tok
    def consume_optional_newlines(self):
        while self.match("NEWLINE"): pass

    # top ------------------------------------------------------------
    def parse(self) -> Program:
        stmts: List[Node] = []
        self.consume_optional_newlines()
        while self.current().type != "EOF":
            stmts.append(self.statement())
            if self.match("SEMICOLON"): pass
            self.consume_optional_newlines()
        return Program(stmts)

    # statements -----------------------------------------------------
    def statement(self) -> Node:
        tok = self.current()
        t = tok.type
        if t == "PRINT":
            return self.print_stmt()
        elif t == "ASSIGN":
            return self.assign_stmt()
        elif t == "WHILE":
            return self.while_stmt()
        elif t == "IF":
            return self.if_stmt()
        elif t == "FUNCTION":
            return self.func_decl()
        elif t == "RETURN":
            return self.return_stmt()
        else:
            raise TLParseError(tok,
                "Expected a statement (cheppu, veru, podugu, ayithe, seva)...",
                "స్టేట్మెంట్ కావాలి (cheppu, veru, podugu, ayithe, seva)...")

    def print_stmt(self) -> PrintStmt:
        self.expect("PRINT", "Missing 'cheppu' keyword.", "'cheppu' కీవర్డ్ కావాలి.")
        exprs = [self.expression()]
        while self.match("COMMA"):
            exprs.append(self.expression())
        return PrintStmt(exprs)

    def assign_stmt(self) -> AssignStmt:
        self.expect("ASSIGN", "Missing 'veru' keyword.", "'veru' కీవర్డ్ కావాలి.")
        name_tok = self.expect("IDENT", "Expected variable name after 'veru'.", "'veru' తరువాత పేరు ఇవ్వాలి.")
        self.expect("ASSIGN_OP", "Expected '=' in assignment.", "అసైన్ చేయడానికి '=' అవసరం.")
        expr = self.expression()
        return AssignStmt(name_tok.value, expr)

    def while_stmt(self) -> WhileStmt:
        self.expect("WHILE", "Missing 'podugu' keyword.", "'podugu' కీవర్డ్ కావాలి.")
        self.expect("L_PAREN", "Expected '(' after 'podugu'.", "'podugu' తరువాత '(' ఇవ్వాలి.")
        cond = self.expression()
        self.expect("R_PAREN", "Expected ')' after condition.", "కండిషన్ తర్వాత ')' కావాలి.")
        body = self.block()
        return WhileStmt(cond, body)

    def if_stmt(self) -> IfStmt:
        self.expect("IF", "Missing 'ayithe' keyword.", "'ayithe' కీవర్డ్ కావాలి.")
        self.expect("L_PAREN", "Expected '(' after 'ayithe'.", "'ayithe' తరువాత '(' ఇవ్వాలి.")
        cond = self.expression()
        self.expect("R_PAREN", "Expected ')' after condition.", "కండిషన్ తర్వాత ')' కావాలి.")
        then_body = self.block()
        else_body = None
        if self.match("ELSE"):
            else_body = self.block()
        return IfStmt(cond, then_body, else_body)

    def func_decl(self) -> FuncDecl:
        self.expect("FUNCTION", "Missing 'seva' keyword.", "'seva' కీవర్డ్ కావాలి.")
        name_tok = self.expect("IDENT", "Function name expected.", "ఫంక్షన్ పేరు కావాలి.")
        self.expect("L_PAREN", "Expected '(' after function name.", "ఫంక్షన్ పేరుకు తర్వాత '(' ఇవ్వాలి.")
        params: List[str] = []
        if self.current().type != "R_PAREN":
            p = self.expect("IDENT", "Parameter name expected.", "పరామీటర్ పేరు కావాలి.")
            params.append(p.value)
            while self.match("COMMA"):
                p = self.expect("IDENT", "Parameter name expected.", "పరామీటర్ పేరు కావాలి.")
                params.append(p.value)
        self.expect("R_PAREN", "Expected ')' after params.", ") తర్వాత కావాలి.")
        body = self.block()
        return FuncDecl(name_tok.value, params, body)

    def return_stmt(self) -> ReturnStmt:
        self.expect("RETURN", "Missing 'tirugu' keyword.", "'tirugu' కీవర్డ్ కావాలి.")
        # return expr optional (blank line / newline / ; / })
        if self.current().type in ("NEWLINE", "SEMICOLON", "R_BRACE"):
            return ReturnStmt(None)
        expr = self.expression()
        return ReturnStmt(expr)

    def block(self) -> List[Node]:
        self.expect("L_BRACE", "Expected '{' to start block.", "బ్లాక్ మొదలుపెట్టడానికి '{' ఇవ్వాలి.")
        self.consume_optional_newlines()
        stmts: List[Node] = []
        while self.current().type not in ("R_BRACE", "EOF"):
            stmts.append(self.statement())
            if self.match("SEMICOLON"): pass
            self.consume_optional_newlines()
        self.expect("R_BRACE", "Expected '}' to end block.", "బ్లాక్ ముగించడానికి '}' అవసరం.")
        return stmts

    # expressions ----------------------------------------------------
    def expression(self) -> Node:
        return self.parse_comparison()
    def parse_comparison(self) -> Node:
        node = self.parse_term_addsub()
        while self.current().type in ("EQ", "NE", "GT", "GE", "LT", "LE"):
            op_tok = self.advance()
            right = self.parse_term_addsub()
            node = BinOp(node, op_tok.type, right)
        return node
    def parse_term_addsub(self) -> Node:
        node = self.parse_factor_muldiv()
        while self.current().type in ("PLUS", "MINUS"):
            op_tok = self.advance(); right = self.parse_factor_muldiv(); node = BinOp(node, op_tok.type, right)
        return node
    def parse_factor_muldiv(self) -> Node:
        node = self.parse_unary()
        while self.current().type in ("STAR", "SLASH", "PERCENT"):
            op_tok = self.advance(); right = self.parse_unary(); node = BinOp(node, op_tok.type, right)
        return node
    def parse_unary(self) -> Node:
        if self.current().type in ("PLUS", "MINUS"):
            op_tok = self.advance(); operand = self.parse_unary(); return UnaryOp(op_tok.type, operand)
        return self.parse_primary()
    def parse_primary(self) -> Node:
        tok = self.current()
        if tok.type == "NUMBER":
            self.advance(); return Literal(float(tok.value)) if '.' in tok.value else Literal(int(tok.value))
        elif tok.type == "STRING":
            self.advance(); raw = tok.value; body = raw[1:-1] if raw and raw[0] == raw[-1] and raw[0] in ('"', "'") else raw
            body = body.replace('\\n','\n').replace('\\t','\t').replace('\\"','"').replace("\\'", "'")
            return Literal(body)
        elif tok.type == "TRUE":
            self.advance(); return Literal(True)
        elif tok.type == "FALSE":
            self.advance(); return Literal(False)
        elif tok.type == "IDENT":
            # could be var OR call
            name = tok.value
            self.advance()
            if self.match("L_PAREN"):
                args: List[Node] = []
                if self.current().type != "R_PAREN":
                    args.append(self.expression())
                    while self.match("COMMA"):
                        args.append(self.expression())
                self.expect("R_PAREN", "Expected ')' after arguments.", "ఆర్గుమెంట్స్ తర్వాత ')' కావాలి.")
                return CallExpr(name, args)
            return Var(name)
        elif tok.type == "L_PAREN":
            self.advance(); expr = self.expression(); self.expect("R_PAREN", "Expected ')' after expression.", "ఎక్స్‌ప్రెషన్ తర్వాత ')' కావాలి."); return expr
        else:
            raise TLParseError(tok, "Unexpected token in expression.", "ఎక్స్‌ప్రెషన్‌లో అనుకోని టోకెన్.")

########################################
# Environment (scoped)
########################################

class Environment:
    def __init__(self, parent: Optional['Environment']=None):
        self.parent = parent
        self.vars: Dict[str, Any] = {}
    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise TLRuntimeError(f"Variable '{name}' is not defined.", f"'{name}' నిర్వచించలేదు.")
    def set(self, name: str, value: Any):
        self.vars[name] = value
    def assign_existing(self, name: str, value: Any) -> bool:
        if name in self.vars:
            self.vars[name] = value; return True
        if self.parent is not None:
            return self.parent.assign_existing(name, value)
        return False

########################################
# Callable object to hold user functions
########################################

@dataclass
class UserFunction:
    params: List[str]
    body: List[Node]
    closure: Environment  # environment at def time (for globals)

    def call(self, interp: 'Interpreter', args: List[Any]):
        if len(args) != len(self.params):
            raise TLRuntimeError(
                f"Expected {len(self.params)} args, got {len(args)}.",
                f"{len(self.params)} ఆర్గుమెంట్లు కావాలి, {len(args)} వచ్చాయి."
            )
        local_env = Environment(self.closure)
        for name, val in zip(self.params, args):
            local_env.set(name, val)
        # Execute body in local env
        try:
            interp._exec_block(self.body, local_env)
        except _ReturnSignal as rs:
            return rs.value
        return None  # implicit None

########################################
# Interpreter
########################################

class Interpreter:
    def __init__(self):
        self.env = Environment()   # global env
        self.functions: Dict[str, UserFunction] = {}

    def run(self, program: Program):
        self._exec_block(program.statements, self.env)

    def _exec_block(self, stmts: List[Node], env: Environment):
        prev_env = self.env
        self.env = env
        try:
            for stmt in stmts:
                self.exec_stmt(stmt)
        finally:
            self.env = prev_env

    def exec_stmt(self, stmt: Node):
        if isinstance(stmt, PrintStmt):
            vals = [self.eval_expr(e) for e in stmt.expressions]
            print(*vals)
        elif isinstance(stmt, AssignStmt):
            val = self.eval_expr(stmt.expr)
            # Always assign in *current* env; does not walk parent (like 'var')
            self.env.set(stmt.name, val)
        elif isinstance(stmt, WhileStmt):
            while self.truthy(self.eval_expr(stmt.condition)):
                self._exec_block(stmt.body, Environment(self.env))
        elif isinstance(stmt, IfStmt):
            if self.truthy(self.eval_expr(stmt.condition)):
                self._exec_block(stmt.then_body, Environment(self.env))
            elif stmt.else_body is not None:
                self._exec_block(stmt.else_body, Environment(self.env))
        elif isinstance(stmt, FuncDecl):
            self.functions[stmt.name] = UserFunction(stmt.params, stmt.body, self.env)
        elif isinstance(stmt, ReturnStmt):
            val = None if stmt.expr is None else self.eval_expr(stmt.expr)
            raise _ReturnSignal(val)
        else:
            raise TLRuntimeError(
                f"Unknown statement type: {type(stmt).__name__}.",
                f"తెలియని స్టేట్మెంట్ రకం: {type(stmt).__name__}."
            )

    def eval_expr(self, node: Node) -> Any:
        if isinstance(node, Literal):
            return node.value
        elif isinstance(node, Var):
            return self.env.get(node.name)
        elif isinstance(node, UnaryOp):
            v = self.eval_expr(node.operand)
            if node.op == "PLUS": return +v
            if node.op == "MINUS": return -v
            raise TLRuntimeError(f"Unsupported unary {node.op}.", f"సపోర్ట్ చేయని unary {node.op}.")
        elif isinstance(node, BinOp):
            l = self.eval_expr(node.left); r = self.eval_expr(node.right); op = node.op
            try:
                if op == "PLUS": return l + r
                if op == "MINUS": return l - r
                if op == "STAR": return l * r
                if op == "SLASH": return l / r
                if op == "PERCENT": return l % r
                if op == "EQ": return l == r
                if op == "NE": return l != r
                if op == "GT": return l > r
                if op == "GE": return l >= r
                if op == "LT": return l < r
                if op == "LE": return l <= r
            except Exception as exc:  # pragma: no cover
                raise TLRuntimeError(
                    f"Error evaluating op {op}: {exc}",
                    f"ఆపరేటర్ {op} లెక్కలోపం: {exc}"
                )
            raise TLRuntimeError(f"Unsupported operator {op}.", f"సపోర్ట్ చేయని ఆపరేటర్ {op}.")
        elif isinstance(node, CallExpr):
            if node.callee not in self.functions:
                raise TLRuntimeError(
                    f"Function '{node.callee}' not defined.",
                    f"ఫంక్షన్ '{node.callee}' నిర్వచించలేదు."
                )
            fn = self.functions[node.callee]
            args = [self.eval_expr(a) for a in node.args]
            return fn.call(self, args)
        else:
            raise TLRuntimeError(
                f"Unknown expression node {type(node).__name__}.",
                f"తెలియని ఎక్స్‌ప్రెషన్ {type(node).__name__}."
            )

    @staticmethod
    def truthy(val: Any) -> bool:
        return bool(val)

########################################
# Execution Helpers
########################################

def run_source(src: str):
    tokens = lex(src)
    parser = Parser(tokens)
    program = parser.parse()
    interp = Interpreter()
    interp.run(program)

########################################
# REPL
########################################

def repl():
    print("TeluguLang REPL (v0.3). Type .exit to quit.")
    buffer_lines: List[str] = []
    open_braces = 0
    interp = Interpreter()  # persistent state across inputs
    while True:
        prompt = ">> " if open_braces == 0 else ".. "
        try:
            line = input(prompt)
        except EOFError:
            print(); break
        if line.strip() == ".exit":
            break
        buffer_lines.append(line + "\n")
        open_braces += line.count('{') - line.count('}')
        if open_braces > 0:
            continue  # keep collecting until braces close
        # try execute
        src = "".join(buffer_lines)
        buffer_lines.clear()
        try:
            tokens = lex(src)
            parser = Parser(tokens)
            program = parser.parse()
            # run in existing interpreter env (so state persists)
            interp.run(program)
        except TeluguLangError as e:
            print(e)
        except Exception as e:
            print("Internal error:", e)

########################################
# Main
########################################

def main(argv: List[str]):
    if len(argv) == 1:  # REPL
        repl(); return 0
    if len(argv) != 2:
        print("Usage: python telugulang.py <source.tlg>")
        print("తెలుగు: python telugulang.py <source.tlg> అని నడపండి")
        return 1
    path = argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        print(f"Could not read file: {e}")
        print(f"తెలుగు: ఫైల్ చదవలేకపోయాను: {e}")
        return 1
    try:
        run_source(src)
    except TeluguLangError as e:
        print(e, file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
