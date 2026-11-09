"""Lightweight CUDA parser: finds __global__ kernels and builds a statement tree."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, List, Optional
from .lexer import Tok

OPEN = {"(", "[", "{"}
CLOSE = {")", "]", "}"}


@dataclass
class Node:
    kind: str                      # block|if|for|while|do|switch|simple|return|break|continue
    line: int = 0
    toks: List[Tok] = field(default_factory=list)   # simple statement / for-init
    cond: List[Tok] = field(default_factory=list)
    incr: List[Tok] = field(default_factory=list)
    body: List["Node"] = field(default_factory=list)
    els: Optional["Node"] = None
    parts: int = 0                 # for-loops: number of ';'-separated header parts


@dataclass
class Param:
    name: str
    is_ptr: bool


@dataclass
class Kernel:
    name: str
    params: List[Param]
    body: Node
    line: int


def close_idx(t, i):
    """Index of the bracket closing t[i]."""
    depth = 0
    for j in range(i, len(t)):
        if t[j].kind == "str":
            continue
        x = t[j].text
        if x in OPEN:
            depth += 1
        elif x in CLOSE:
            depth -= 1
            if depth == 0:
                return j
    return len(t) - 1


def split_top(toks, sep):
    out, d = [[]], 0
    for t in toks:
        if t.kind != "str":
            if t.text in OPEN:
                d += 1
            elif t.text in CLOSE:
                d -= 1
            elif t.text == sep and d == 0:
                out.append([])
                continue
        out[-1].append(t)
    return out


def walk(n: Node) -> Iterator[Node]:
    yield n
    for c in n.body:
        yield from walk(c)
    if n.els:
        yield from walk(n.els)


def _stmts(t):
    out, i = [], 0
    while i < len(t):
        n, j = _stmt(t, i)
        if n is not None:
            out.append(n)
        i = j if j > i else i + 1
    return out


def _stmt(t, i):
    tok = t[i]
    x = tok.text
    if x == "{":
        j = close_idx(t, i)
        return Node("block", tok.line, body=_stmts(t[i + 1:j])), j + 1
    if x == "if":
        i += 1
        if i < len(t) and t[i].text == "constexpr":
            i += 1
        j = close_idx(t, i)
        then, k = _stmt(t, j + 1) if j + 1 < len(t) else (None, j + 1)
        els = None
        if k < len(t) and t[k].text == "else":
            els, k = _stmt(t, k + 1) if k + 1 < len(t) else (None, k + 1)
        return Node("if", tok.line, cond=t[i + 1:j], body=[then] if then else [], els=els), k
    if x in ("for", "while", "switch"):
        j = close_idx(t, i + 1)
        head = t[i + 2:j]
        body, k = _stmt(t, j + 1) if j + 1 < len(t) else (None, j + 1)
        n = Node(x, tok.line, body=[body] if body else [])
        if x == "for":
            parts = split_top(head, ";")
            n.parts = len(parts)
            if len(parts) == 3:
                n.toks, n.cond, n.incr = parts
        else:
            n.cond = head
        return n, k
    if x == "do":
        body, k = _stmt(t, i + 1) if i + 1 < len(t) else (None, i + 1)
        cond = []
        if k < len(t) and t[k].text == "while":
            j = close_idx(t, k + 1)
            cond, k = t[k + 2:j], j + 1
            if k < len(t) and t[k].text == ";":
                k += 1
        return Node("do", tok.line, cond=cond, body=[body] if body else []), k
    if x in ("case", "default"):
        j = i
        while j < len(t) and t[j].text != ":":
            j += 1
        return Node("simple", tok.line), j + 1
    if x == ";":
        return None, i + 1
    d, j = 0, i
    while j < len(t):
        y = t[j]
        if y.kind != "str":
            if y.text in OPEN:
                d += 1
            elif y.text in CLOSE:
                if d == 0:
                    break
                d -= 1
            elif y.text == ";" and d == 0:
                break
        j += 1
    toks = t[i:j]
    k = j + 1 if j < len(t) and t[j].text == ";" else max(j, i + 1)
    kind = x if x in ("return", "break", "continue") else "simple"
    return Node(kind, tok.line, toks=toks), k


def _params(toks) -> List[Param]:
    out = []
    for seg in split_top(toks, ","):
        for k, t in enumerate(seg):          # drop default values
            if t.text == "=":
                seg = seg[:k]
                break
        ids = [t for t in seg if t.kind == "id"]
        if not ids or ids[-1].text == "void":
            continue
        out.append(Param(ids[-1].text, any(t.text in ("*", "[") for t in seg)))
    return out


def _kernel(t, i):
    j = i + 1
    while j < len(t):
        if t[j].text == "(":
            if t[j - 1].text in ("__launch_bounds__", "__maxnreg__", "__cluster_dims__"):
                j = close_idx(t, j) + 1
                continue
            break
        if t[j].text in (";", "{"):
            return None
        j += 1
    else:
        return None
    pc = close_idx(t, j)
    k = pc + 1
    while k < len(t) and t[k].text not in ("{", ";"):
        k += 1
    if k >= len(t) or t[k].text == ";":
        return None                          # prototype only
    e = close_idx(t, k)
    body = Node("block", t[k].line, body=_stmts(t[k + 1:e]))
    return Kernel(t[j - 1].text, _params(t[j + 1:pc]), body, t[i].line), e + 1


def parse_kernels(t) -> List[Kernel]:
    out, i = [], 0
    while i < len(t):
        if t[i].kind == "id" and t[i].text == "__global__":
            r = _kernel(t, i)
            if r:
                out.append(r[0])
                i = r[1]
                continue
        i += 1
    return out
