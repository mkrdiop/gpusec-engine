"""Detection rules. One KernelAnalyzer per __global__ kernel.

Core idea: a light taint analysis computes, for every variable, whether its value is
  0 = uniform across the block, 1 = data-dependent (loaded from memory), 2 = thread-variant (threadIdx).
Rules then use that level to separate real hazards from harmless conditionals
(e.g. a barrier under `if (n > 0)` is fine, under `if (threadIdx.x < 32)` it is not).
"""
from __future__ import annotations
import re
from .lexer import Tok
from .models import Finding, Severity
from .parser import Kernel, Node, close_idx, split_top

SYNC = {"__syncthreads", "__syncthreads_count", "__syncthreads_and", "__syncthreads_or"}
ASSIGN = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
OPEN = {"(", "[", "{"}
CLOSE = {")", "]", "}"}
REL = {"<", "<=", ">", ">="}
BINOPS = {"<", "<=", ">", ">=", "==", "!=", "&&", "||", "=", "+=", "-=", "?", ":"}
API_RE = re.compile(r"^cuda[A-Z]\w*$")
API_IGNORE = {"cudaGetLastError", "cudaPeekAtLastError", "cudaGetErrorString", "cudaGetErrorName"}
API_LOW = {"cudaFree", "cudaFreeHost", "cudaFreeAsync", "cudaStreamDestroy", "cudaEventDestroy", "cudaDeviceReset"}


def render(toks):
    out, prev = [], None
    for t in toks:
        if prev is not None and ((prev.kind in ("id", "num") and t.kind in ("id", "num"))
                                 or t.text in BINOPS or prev.text in BINOPS):
            out.append(" ")
        out.append(t.text)
        prev = t
    return "".join(out)


def const_int(toks, consts):
    """Evaluate a tiny integer expression made of literals, known constants and + - * / % << >>."""
    if not toks:
        return None
    s = ""
    for t in toks:
        if t.kind == "num":
            s += re.sub(r"[uUlL]+$", "", t.text)
        elif t.kind == "id" and t.text in consts:
            s += str(consts[t.text])
        elif t.text in ("+", "-", "*", "%", "(", ")", "<<", ">>"):
            s += t.text
        elif t.text == "/":
            s += "//"
        else:
            return None
    try:
        v = eval(s, {"__builtins__": {}}, {})
        return v if isinstance(v, int) else None
    except Exception:
        return None


def assignments(toks):
    """Yield (name, op, rhs_tokens, lhs_tokens) for each declarator/assignment in a statement."""
    out = []
    for seg in split_top(toks, ","):
        d = 0
        for k, t in enumerate(seg):
            if t.kind == "str":
                continue
            if t.text in OPEN:
                d += 1
            elif t.text in CLOSE:
                d -= 1
            elif d == 0 and t.text in ASSIGN and k > 0:
                lhs = seg[:k]
                if lhs[-1].kind == "id" and (len(lhs) < 2 or lhs[-2].text not in (".", "->")):
                    out.append((lhs[-1].text, t.text, seg[k + 1:], lhs))
                break
    return out


def _exits(n: Node) -> bool:
    if n.kind in ("return", "break", "continue"):
        return True
    return n.kind == "block" and bool(n.body) and _exits(n.body[-1])


def _has_exit(n: Node, nested=False) -> bool:
    if n.kind == "return":
        return True
    if n.kind == "break" and not nested:
        return True
    inner = nested or n.kind in ("for", "while", "do", "switch")
    return any(_has_exit(c, inner) for c in n.body) or (n.els is not None and _has_exit(n.els, inner))


class KernelAnalyzer:
    def __init__(self, kernel: Kernel, path: str, consts: dict, lines: list):
        self.k, self.path, self.consts, self.lines = kernel, path, consts, lines
        self.findings, self._seen = [], set()
        self.ptrs = {p.name for p in kernel.params if p.is_ptr}
        self.level, self.deps, self.count, self.alias, self.shared = {}, {}, {}, {}, {}
        self._collect()

    # ------------------------------------------------------------------ analysis setup
    def _collect(self):
        self.assigns = []

        def note(toks, conds):
            for name, op, rhs, lhs in assignments(toks):
                self.assigns.append((name, op, rhs, lhs, conds))
                self.count[name] = self.count.get(name, 0) + 1

        def rec(n, conds):
            k = n.kind
            if k == "block":
                for c in n.body:
                    rec(c, conds)
            elif k == "if":
                c2 = conds + [n.cond]
                for c in n.body:
                    rec(c, c2)
                if n.els:
                    rec(n.els, c2)
            elif k in ("for", "while", "do", "switch"):
                if k == "for":
                    note(n.toks, conds)
                    note(n.incr, conds)
                c2 = conds + [n.cond] if n.cond else conds
                for c in n.body:
                    rec(c, c2)
            else:
                note(n.toks, conds)
                if any(t.text == "__shared__" for t in n.toks):
                    self._decl_shared(n.toks)

        rec(self.k.body, [])
        for name, *_ in self.assigns:
            self.level.setdefault(name, 0)
        for _ in range(6):
            changed = False
            for name, op, rhs, lhs, conds in self.assigns:
                lv = self.level_of(rhs)
                for c in conds:
                    lv = max(lv, self.level_of(c))
                if lv > self.level[name]:
                    self.level[name], changed = lv, True
                if name not in self.ptrs and self._is_ptr_decl(lhs) and any(
                        t.kind == "id" and t.text in self.ptrs for t in rhs):
                    self.ptrs.add(name)
                    changed = True
                self.deps.setdefault(name, set()).update(t.text for t in rhs if t.kind == "id")
            if not changed:
                break
        for name, op, rhs, lhs, conds in self.assigns:
            if self.count[name] == 1 and op == "=" and rhs and not any(t.text == "[" for t in rhs):
                self.alias[name] = "".join(t.text for t in rhs)

    @staticmethod
    def _is_ptr_decl(lhs):
        if len(lhs) < 3 or lhs[0].kind != "id":
            return False
        return lhs[-2].text == "*" or (lhs[-2].text in ("__restrict__", "const")
                                       and any(t.text == "*" for t in lhs[:-2]))

    def _decl_shared(self, toks):
        for k, t in enumerate(toks):
            if t.text == "[" and k > 0 and toks[k - 1].kind == "id":
                j = close_idx(toks, k)
                self.shared[toks[k - 1].text] = const_int(toks[k + 1:j], self.consts)
                return
        ids = [t for t in toks if t.kind == "id" and t.text not in ("__shared__", "extern", "volatile")]
        if len(ids) >= 2:
            self.shared[ids[-1].text] = 1

    def level_of(self, toks):
        lv, i = 0, 0
        while i < len(toks):
            t = toks[i]
            if t.kind == "id":
                if t.text == "threadIdx":
                    lv = max(lv, 2)
                elif (t.text in self.ptrs or t.text in self.shared) and i + 1 < len(toks) and toks[i + 1].text == "[":
                    j = close_idx(toks, i + 1)
                    if self.level_of(toks[i + 2:j]) >= 1:
                        lv = max(lv, 1)
                    i = j + 1
                    continue
                elif t.text in self.level:
                    lv = max(lv, self.level[t.text])
            i += 1
        return lv

    def closure(self, names):
        seen, stack = set(), [n for n in names if n in self.level]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(d for d in self.deps.get(n, ()) if d in self.level and d not in seen)
        return seen

    def canon(self, toks):
        s = "".join(t.text for t in toks)
        for _ in range(2):
            s = re.sub(r"(?<![.\w])[A-Za-z_]\w*", lambda m: self.alias.get(m.group(), str(self.consts.get(m.group(), m.group()))), s)
        return s

    # ------------------------------------------------------------------ reporting
    def _report(self, rule, sev, conf, tok, msg, fix, cwe):
        key = (rule, tok.line, tok.col)
        if key in self._seen:
            return
        self._seen.add(key)
        ev = self.lines[tok.line - 1] if 0 < tok.line <= len(self.lines) else ""
        self.findings.append(Finding(rule, sev, conf, self.path, tok.line, tok.col, self.k.name, msg, fix, cwe, ev.rstrip()))

    # ------------------------------------------------------------------ traversal
    def run(self):
        self._visit(self.k.body, [], [])
        ev = []
        self._events(self.k.body, ev)
        self._race(ev)
        return self.findings

    def _visit(self, n, guards, scopes):
        k = n.kind
        if k == "block":
            local, lscopes = [], []
            for c in n.body:
                self._visit(c, guards + local, scopes + lscopes)
                if c.kind == "if" and not c.els and c.body and _exits(c.body[0]):
                    local.append(c.cond)
                    lscopes.append(("exit", c.cond, c.line))
        elif k == "if":
            self._scan(n.cond, guards)
            g, s = guards + [n.cond], scopes + [("cond", n.cond, n.line)]
            for c in n.body:
                self._visit(c, g, s)
            if n.els:
                self._visit(n.els, g, s)
        elif k in ("for", "while", "do", "switch"):
            if k == "for":
                self._scan(n.toks, guards)
                self._scan(n.incr, guards)
            self._scan(n.cond, guards)
            self._check_loop(n)
            g = guards + ([n.cond] if n.cond else [])
            s = scopes + ([("cond", n.cond, n.line)] if n.cond else [])
            for c in n.body:
                self._visit(c, g, s)
        else:
            if not any(t.text == "__shared__" for t in n.toks):
                self._scan(n.toks, guards)
            self._check_sync(n.toks, scopes)

    # ------------------------------------------------------------------ GS-MEM-001 / GS-MEM-002
    def _scan(self, toks, guards):
        for i, t in enumerate(toks):
            if (t.kind == "id" and i + 1 < len(toks) and toks[i + 1].text == "["
                    and (t.text in self.ptrs or t.text in self.shared)
                    and not (i and toks[i - 1].text in (".", "->"))):
                self._check_subscript(toks, i, guards)

    def _is_guard(self, cond, names, text):
        if not any(t.text in REL for t in cond):
            return False
        if names & {t.text for t in cond if t.kind == "id"}:
            return True
        return bool(text) and text in self.canon(cond)

    def _check_subscript(self, toks, i, guards):
        base = toks[i]
        j = close_idx(toks, i + 1)
        inner = toks[i + 2:j]
        if not inner:
            return
        k = j + 1
        while k < len(toks) and toks[k].text == "[":
            k = close_idx(toks, k) + 1
        nxt = toks[k].text if k < len(toks) else ""
        addr_of = i >= 1 and toks[i - 1].text == "&" and (
            i < 2 or (toks[i - 2].kind != "id" and toks[i - 2].text not in (")", "]")))
        is_write = nxt in ASSIGN or nxt in ("++", "--") or addr_of

        if base.text in self.shared:
            size, val = self.shared[base.text], const_int(inner, self.consts)
            if size and val is not None and not 0 <= val < size:
                self._report("GS-MEM-002", Severity.HIGH, "high", base,
                             f"`{base.text}[{render(inner)}]` indexes element {val} of a __shared__ array declared with {size} elements.",
                             f"Keep the index below {size}, or enlarge `{base.text}`.", "CWE-119")
            return

        if self.level_of(inner) == 0:
            return
        if {t.text for t in inner} & {"%", "min", "clamp"}:
            return
        names = self.closure({t.text for t in inner if t.kind == "id"})
        text = self.canon(inner)
        prefix = toks[:i]
        extra = [prefix] if any(t.text in ("&&", "?") for t in prefix) else []
        if any(self._is_guard(g, names, text) for g in list(guards) + extra):
            return
        var = next((t.text for t in inner if t.kind == "id" and t.text in self.level), "idx")
        what = "write to" if is_write else "read from"
        self._report("GS-MEM-001", Severity.HIGH if is_write else Severity.MEDIUM, "medium", base,
                     f"Thread-derived index `{render(inner)}` is used for a {what} `{base.text}` with no bounds check against its size. "
                     "If the launch has more threads than elements, the extra threads access memory outside the allocation.",
                     f"Guard the access (`if ({var} >= n) return;` or `if ({var} < n) {{ ... }}`) and pass the element count to the kernel.",
                     "CWE-787" if is_write else "CWE-125")

    # ------------------------------------------------------------------ GS-SYNC-001
    def _check_sync(self, toks, scopes):
        for i, t in enumerate(toks):
            if not (t.kind == "id" and t.text in SYNC and i + 1 < len(toks) and toks[i + 1].text == "("):
                continue
            best = None
            for kind, cond, line in scopes:
                lv = self.level_of(cond)
                if lv == 0:
                    continue
                if kind == "cond":
                    sev = Severity.HIGH if lv == 2 else Severity.MEDIUM
                else:
                    sev = Severity.MEDIUM if lv == 2 else Severity.LOW
                if best is None or sev > best[0]:
                    best = (sev, kind, cond, line, lv)
            if not best:
                continue
            sev, kind, cond, line, lv = best
            c = render(cond)
            if kind == "exit":
                msg = (f"Threads satisfying `{c}` (line {line}) can leave the kernel before this barrier. CUDA does not guarantee "
                       "defined behaviour unless every thread of the block reaches `__syncthreads()`.")
            elif lv == 2:
                msg = (f"`{t.text}()` is only reached under `{c}` (line {line}), which depends on the thread index. "
                       "If some threads skip the barrier the block can hang or behave undefined.")
            else:
                msg = (f"`{t.text}()` is under `{c}` (line {line}), which depends on loaded data and may differ between threads. "
                       "Verify that all threads of a block take the same path.")
            self._report("GS-SYNC-001", sev, "medium" if lv == 2 else "low", t, msg,
                         "Make all threads reach the barrier: keep `__syncthreads()` outside the conditional and guard only the memory accesses.",
                         "CWE-833")

    # ------------------------------------------------------------------ GS-LOOP-001
    def _check_loop(self, n):
        if n.kind not in ("for", "while", "do"):
            return
        c = [t.text for t in n.cond]
        infinite = (c in (["true"], ["1"])) if n.kind != "for" else (n.parts == 3 and not n.cond)
        if infinite and not any(_has_exit(b) for b in n.body):
            tok = Tok("id", n.kind, n.line, 1)
            self._report("GS-LOOP-001", Severity.HIGH, "medium", tok,
                         "Loop has no exit condition, `break` or `return`. A kernel stuck here keeps the GPU busy until a watchdog or reset.",
                         "Add a bounded exit condition or an iteration cap. If this is an intentional persistent kernel, add `// gpusec-ignore GS-LOOP-001`.",
                         "CWE-835")

    # ------------------------------------------------------------------ GS-RACE-001
    def _events(self, n, ev):
        k = n.kind
        if k == "block":
            for c in n.body:
                self._events(c, ev)
        elif k == "if":
            self._acc(n.cond, ev)
            for c in n.body:
                self._events(c, ev)
            if n.els:
                self._events(n.els, ev)
        elif k in ("for", "while", "do", "switch"):
            if k == "for":
                self._acc(n.toks, ev)
            for _ in range(1 if k == "switch" else 2):   # 2 passes expose cross-iteration hazards
                self._acc(n.cond, ev)
                for c in n.body:
                    self._events(c, ev)
                if k == "for":
                    self._acc(n.incr, ev)
        else:
            self._acc(n.toks, ev)

    def _acc(self, toks, ev):
        if any(t.text == "__shared__" for t in toks):
            return
        reads, writes = [], []
        for i, t in enumerate(toks):
            if not (t.kind == "id" and t.text in self.shared and i + 1 < len(toks) and toks[i + 1].text == "["):
                continue
            if i and toks[i - 1].text in (".", "->", "&"):
                continue
            idx, k = "", i + 1
            while k < len(toks) and toks[k].text == "[":
                j = close_idx(toks, k)
                idx += "[" + self.canon(toks[k + 1:j]) + "]"
                k = j + 1
            nxt = toks[k].text if k < len(toks) else ""
            item = (t.text, idx, t)
            if nxt == "=":
                writes.append(item)
            elif nxt in ASSIGN or nxt in ("++", "--"):
                reads.append(item)
                writes.append(item)
            else:
                reads.append(item)
        ev.extend(("r",) + r for r in reads)
        ev.extend(("w",) + w for w in writes)
        if any(t.kind == "id" and t.text in SYNC for t in toks):
            ev.append(("bar",))

    def _race(self, ev):
        pending = {}
        for e in ev:
            if e[0] == "bar":
                pending.clear()
            elif e[0] == "w":
                pending.setdefault(e[1], []).append((e[2], e[3].line))
            else:
                _, arr, idx, tok = e
                const = re.compile(r"^(\[\d+\])+$")
                other = next(((i, l) for i, l in pending.get(arr, [])
                              if i != idx and not (const.match(i) and const.match(idx))), None)
                if other and idx not in {i for i, _ in pending[arr]}:
                    self._report("GS-RACE-001", Severity.MEDIUM, "medium", tok,
                                 f"`{arr}{idx}` is read after `{arr}{other[0]}` was written (line {other[1]}) with no `__syncthreads()` in between. "
                                 "If the element was written by another thread, the read can see stale data (data race).",
                                 "Insert `__syncthreads()` between the write phase and the read phase, and again before the buffer is overwritten.",
                                 "CWE-362")


# ---------------------------------------------------------------------- GS-API-001 (host code)
def check_api(tokens, path, lines):
    out = []
    for i, t in enumerate(tokens):
        if t.kind != "id" or not API_RE.match(t.text) or t.text in API_IGNORE:
            continue
        if i + 1 >= len(tokens) or tokens[i + 1].text != "(":
            continue
        if i and tokens[i - 1].text not in (";", "{", "}"):
            continue
        sev = Severity.LOW if t.text in API_LOW else Severity.MEDIUM
        ev = lines[t.line - 1].rstrip() if 0 < t.line <= len(lines) else ""
        out.append(Finding("GS-API-001", sev, "high", path, t.line, t.col, "", 
                           f"The status returned by `{t.text}` is ignored; a failed allocation, copy or launch goes unnoticed and later kernels run on invalid memory.",
                           f"Check the result, e.g. `CUDA_CHECK({t.text}(...));`, and call `cudaGetLastError()` after kernel launches.",
                           "CWE-252", ev))
    return out
