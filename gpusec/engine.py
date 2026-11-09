from __future__ import annotations
import os
import re
from .lexer import tokenize
from .models import Finding
from .parser import parse_kernels
from .rules import KernelAnalyzer, check_api, const_int

_DEF = re.compile(r"^[ \t]*#\s*define\s+([A-Za-z_]\w*)[ \t]+([^\n]+?)\s*$", re.M)
_CONST = re.compile(r"\b(?:constexpr|const)\s+(?:unsigned\s+|signed\s+)?(?:int|long|size_t|unsigned)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
SKIP_DIRS = {".git", "node_modules", "build", "dist", "__pycache__", ".venv"}


def _constants(src):
    raw = [(m.group(1), m.group(2)) for m in _DEF.finditer(src)] + [(m.group(1), m.group(2)) for m in _CONST.finditer(src)]
    consts = {}
    for _ in range(2):
        for name, expr in raw:
            toks, _ign = tokenize(expr)
            v = const_int(toks, consts)
            if v is not None:
                consts[name] = v
    return consts


def analyze_source(src: str, path: str = "<memory>"):
    toks, ignores = tokenize(src)
    lines = src.splitlines()
    consts = _constants(src)
    findings = check_api(toks, path, lines)
    for k in parse_kernels(toks):
        findings += KernelAnalyzer(k, path, consts, lines).run()
    findings = [f for f in findings
                if not (ignores.get(f.line) and ("*" in ignores[f.line] or f.rule_id in ignores[f.line]))]
    findings.sort(key=lambda f: (f.file, f.line, f.col, f.rule_id))
    return findings


def iter_files(paths, exts):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in exts:
                    yield os.path.join(root, fn)


def analyze_paths(paths, exts=(".cu", ".cuh")):
    out = []
    for f in iter_files(paths, set(exts)):
        with open(f, encoding="utf-8", errors="replace") as fh:
            out += analyze_source(fh.read(), os.path.relpath(f).replace(os.sep, "/"))
    return out
