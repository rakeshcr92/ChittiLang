#!/usr/bin/env python3
"""
TeluguLang v0.1 (ASCII transliteration)
======================================

A tiny, educational, Telugu-flavored toy programming language inspired by BhaiLang.
This initial version supports:

* Variable assignment:     veru x = 10
* Print:                   cheppu x
* While loops:             podugu (x > 0) { ... }
* Integer + float literals
* String literals in double or single quotes
* Arithmetic:              + - * / %
* Comparisons:             == != > >= < <=
* Boolean literals:        nijam (true), abaddam (false)
* Line comments starting with #
* Bilingual (English + Telugu) error messages

Planned for later versions:
---------------------------
* If / Else (ayithe / lekapothe)
* Functions (seva)
* Return (tirigi)
* REPL mode
* Standard library helpers

Usage:
------
    python telugulang.py path/to/program.tlg

Example program (save as example.tlg):
--------------------------------------
veru x = 5
cheppu "Countdown from"  # prints a string literal
cheppu x

podugu (x > 0) {
    cheppu x
    veru x = x - 1
}
cheppu "Done!"

Expected output:
----------------
Countdown from
5
5
4
3
2
1
Done!

NOTE: This interpreter is intentionally small and hackable—perfect for learning.
"""

import sys
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Dict, Callable

########################################
# Token Definitions
########################################

# Reserved keywords (all lowercase ASCII)
KEYWORDS = {
    "cheppu": "PRINT",      # print
    "veru": "ASSIGN",        # variable decl/assign
    "podugu": "WHILE",       # while loop
    "ayithe": "IF",          # (planned)
    "lekapothe": "ELSE",     # (planned)
    "nijam": "TRUE",         # boolean true
    "abaddam": "FALSE",      # boolean false
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
    ("L_PAREN",     r"\("),
    ("R_PAREN",     r"\)"),
    ("L_BRACE",     r"\{"),
    ("R_BRACE",     r"\}"),
    ("SEMICOLON",   r";"),
    ("NEWLINE",     r"\n"),
    ("SKIP",        r"[ \t\r]+"),
    ("MISMATCH",    r"."),  # MUST be last
]

TOKEN_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC),
    re.UNICODE,
)

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Token({self.type}, {self.value!r}, ln={self.line}, col={self.column})"


def lex(source: str) -> List[Token]:
    """Convert source code to a list of tokens.

    Handles # comments (to end of line) prior to token regex.
    """
    tokens: List[Token] = []
    # Remove comments (replace with nothing until newline)
    # We'll do this line-by-line to preserve newline structure & columns for error reporting.
    lines = source.splitlines(keepends=True)
    cleaned = []
    for ln, line in enumerate(lines, start=1):
        # find first unescaped #
        # simpler: treat any # as comment start (not within strings) — good enough for v0.1
        idx = line.find('#')
        if idx != -1:
            line = line[:idx] + "\n" if line.endswith("\n") else line[:idx]
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
# Errors (bilingual)
########################################

class TeluguLangError(Exception):
    """Base class for all TeluguLang errors."""
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
    expressions: List[Node]  # allow multiple args -> print each separated by space

@dataclass
class AssignStmt(Node):
    name: str
    expr: Node

@dataclass
class WhileStmt(Node):
    condition: Node
    body: List[Node]

# Reserved for later
@dataclass
class IfStmt(Node):
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]]

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
# Parser (Recursive Descent w/ Pratt-style expr parsing)
########################################

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # --- token helpers --------------------------------------------------
    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.current()
        if tok.type != "EOF":
            self.pos += 1
        return tok

    def match(self, *types: str) -> bool:
        if self.current().type in types:
            self.advance()
            return True
        return False

    def expect(self, ttype: str, en: str, te: str) -> Token:
        tok = self.current()
        if tok.type != ttype:
            raise TLParseError(tok, en, te)
        self.advance()
        return tok

    def consume_optional_newlines(self):
        while self.match("NEWLINE"):
            pass

    # --- top-level ------------------------------------------------------
    def parse(self) -> Program:
        stmts: List[Node] = []
        self.consume_optional_newlines()
        while self.current().type != "EOF":
            stmt = self.statement()
            stmts.append(stmt)
            # optional semicolon or newline terminators
            if self.match("SEMICOLON"):
                pass
            self.consume_optional_newlines()
        return Program(stmts)

    # --- statements -----------------------------------------------------
    def statement(self) -> Node:
        tok = self.current()
        ttype = tok.type
        if ttype == "PRINT":
            return self.print_stmt()
        elif ttype == "ASSIGN":
            return self.assign_stmt()
        elif ttype == "WHILE":
            return self.while_stmt()
        elif ttype == "IF":  # not yet wired in runtime, but we can parse for forward compat
            return self.if_stmt()
        else:
            # Attempt expression statement? For now, error.
            raise TLParseError(
                tok,
                "Expected a statement (cheppu, veru, podugu, ayithe)...",
                "స్టేట్మెంట్ కావాలి (cheppu, veru, podugu, ayithe)...",
            )

    def print_stmt(self) -> PrintStmt:
        self.expect("PRINT", "Missing 'cheppu' keyword.", "'cheppu' కీవర్డ్ కావాలి.")
        exprs = [self.expression()]
        # allow multiple expressions separated by commas (optional feature)
        while self.match("COMMA"):  # COMMA not defined yet; leaving hook
            exprs.append(self.expression())
        return PrintStmt(exprs)

    def assign_stmt(self) -> AssignStmt:
        self.expect("ASSIGN", "Missing 'veru' keyword.", "'veru' కీవర్డ్ కావాలి.")
        name_tok = self.expect(
            "IDENT", "Expected variable name after 'veru'.", "'veru' తరువాత వేరియబుల్ పేరు ఇవ్వాలి."
        )
        self.expect(
            "ASSIGN_OP", "Expected '=' in assignment.", "అసైన్ చేయడానికి '=' అవసరం."
        )
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

    def block(self) -> List[Node]:
        self.expect("L_BRACE", "Expected '{' to start block.", "బ్లాక్ మొదలుపెట్టడానికి '{' ఇవ్వాలి.")
        self.consume_optional_newlines()
        stmts: List[Node] = []
        while self.current().type not in ("R_BRACE", "EOF"):
            stmts.append(self.statement())
            if self.match("SEMICOLON"):
                pass
            self.consume_optional_newlines()
        self.expect("R_BRACE", "Expected '}' to end block.", "బ్లాక్ ముగించడానికి '}' అవసరం.")
        return stmts

    # --- expressions ----------------------------------------------------
    # Pratt parser style: precedence climbing
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
            op_tok = self.advance()
            right = self.parse_factor_muldiv()
            node = BinOp(node, op_tok.type, right)
        return node

    def parse_factor_muldiv(self) -> Node:
        node = self.parse_unary()
        while self.current().type in ("STAR", "SLASH", "PERCENT"):
            op_tok = self.advance()
            right = self.parse_unary()
            node = BinOp(node, op_tok.type, right)
        return node

    def parse_unary(self) -> Node:
        if self.current().type in ("PLUS", "MINUS"):
            op_tok = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op_tok.type, operand)
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.current()
        if tok.type == "NUMBER":
            self.advance()
            if '.' in tok.value:
                return Literal(float(tok.value))
            return Literal(int(tok.value))
        elif tok.type == "STRING":
            self.advance()
            # Strip quotes & unescape simple escapes
            raw = tok.value
            if raw[0] == raw[-1] and raw[0] in ('"', "'"):
                body = raw[1:-1]
            else:
                body = raw
            body = body.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
            return Literal(body)
        elif tok.type == "TRUE":
            self.advance()
            return Literal(True)
        elif tok.type == "FALSE":
            self.advance()
            return Literal(False)
        elif tok.type == "IDENT":
            self.advance()
            return Var(tok.value)
        elif tok.type == "L_PAREN":
            self.advance()
            expr = self.expression()
            self.expect("R_PAREN", "Expected ')' after expression.", "ఎక్స్‌ప్రెషన్ తర్వాత ')' కావాలి.")
            return expr
        else:
            raise TLParseError(tok, "Unexpected token in expression.", "ఎక్స్‌ప్రెషన్‌లో అనుకోని టోకెన్.")

########################################
# Interpreter
########################################

class Environment:
    def __init__(self):
        self.vars: Dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name not in self.vars:
            raise TLRuntimeError(
                f"Variable '{name}' is not defined.",
                f"'{name}' వేరియబుల్ నిర్వచించలేదు."
            )
        return self.vars[name]

    def set(self, name: str, value: Any):
        self.vars[name] = value


class Interpreter:
    def __init__(self):
        self.env = Environment()

    def run(self, program: Program):
        for stmt in program.statements:
            self.exec_stmt(stmt)

    def exec_stmt(self, stmt: Node):
        if isinstance(stmt, PrintStmt):
            vals = [self.eval_expr(e) for e in stmt.expressions]
            print(*vals)
        elif isinstance(stmt, AssignStmt):
            val = self.eval_expr(stmt.expr)
            self.env.set(stmt.name, val)
        elif isinstance(stmt, WhileStmt):
            while self.truthy(self.eval_expr(stmt.condition)):
                for inner in stmt.body:
                    self.exec_stmt(inner)
        elif isinstance(stmt, IfStmt):  # future feature; safe no-op if not used
            if self.truthy(self.eval_expr(stmt.condition)):
                for inner in stmt.then_body:
                    self.exec_stmt(inner)
            elif stmt.else_body is not None:
                for inner in stmt.else_body:
                    self.exec_stmt(inner)
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
            val = self.eval_expr(node.operand)
            if node.op == "PLUS":
                return +val
            elif node.op == "MINUS":
                return -val
            else:
                raise TLRuntimeError(
                    f"Unsupported unary operator {node.op}.",
                    f"ఉపయోగించలేనిది unary ఆపరేటర్ {node.op}."
                )
        elif isinstance(node, BinOp):
            left = self.eval_expr(node.left)
            right = self.eval_expr(node.right)
            op = node.op
            try:
                if op == "PLUS":
                    return left + right
                elif op == "MINUS":
                    return left - right
                elif op == "STAR":
                    return left * right
                elif op == "SLASH":
                    return left / right
                elif op == "PERCENT":
                    return left % right
                elif op == "EQ":
                    return left == right
                elif op == "NE":
                    return left != right
                elif op == "GT":
                    return left > right
                elif op == "GE":
                    return left >= right
                elif op == "LT":
                    return left < right
                elif op == "LE":
                    return left <= right
                else:
                    raise TLRuntimeError(
                        f"Unsupported operator {op}.",
                        f"సపోర్ట్ చేయని ఆపరేటర్ {op}."
                    )
            except Exception as exc:  # catch Python TypeErrors etc.
                raise TLRuntimeError(
                    f"Error evaluating binary op {op}: {exc}",
                    f"బైనరీ ఆపరేటర్ {op} విలయలోపం: {exc}"
                ) from exc
        else:
            raise TLRuntimeError(
                f"Unknown expression node {type(node).__name__}.",
                f"తెలియని ఎక్స్‌ప్రెషన్ {type(node).__name__}."
            )

    @staticmethod
    def truthy(val: Any) -> bool:
        return bool(val)

########################################
# Driver
########################################

def run_source(src: str):
    tokens = lex(src)
    parser = Parser(tokens)
    program = parser.parse()
    interp = Interpreter()
    interp.run(program)


def main(argv: List[str]):
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
