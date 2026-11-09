"""Minimal C++/CUDA lexer: drops comments and preprocessor lines, keeps positions."""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Tok:
    kind: str   # id | num | op | str
    text: str
    line: int
    col: int


_RX = re.compile(r"""
  (?P<pp>^[ \t]*\#(?:\\\n|[^\n])*)
 |(?P<ws>\n|[ \t\r]+)
 |(?P<lc>//[^\n]*)
 |(?P<bc>/\*.*?\*/)
 |(?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
 |(?P<num>0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*)
 |(?P<id>[A-Za-z_]\w*)
 |(?P<op><<<|>>>|<<=|>>=|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|::|.)
""", re.X | re.M | re.S)

_IGN = re.compile(r"gpusec-ignore(?:[:\s]+([A-Za-z0-9,\- ]*))?")


def tokenize(src: str):
    """Return (tokens, ignores). ignores maps line -> set of rule ids ('*' = all)."""
    toks, ignores = [], {}
    line, ls = 1, 0
    for m in _RX.finditer(src):
        kind, text = m.lastgroup, m.group()
        if kind in ("lc", "bc"):
            im = _IGN.search(text)
            if im:
                rules = {r for r in re.split(r"[,\s]+", (im.group(1) or "").upper()) if r.startswith("GS-")} or {"*"}
                for ln in (line, line + text.count("\n") + 1):
                    ignores.setdefault(ln, set()).update(rules)
        elif kind in ("id", "num", "op", "str"):
            toks.append(Tok(kind, text, line, m.start() - ls + 1))
        n = text.count("\n")
        if n:
            line += n
            ls = m.start() + text.rindex("\n") + 1
    return toks, ignores
