from __future__ import annotations
import argparse
import sys
from .dynamic import TOOLS, correlate, parse_output, run_sanitizer
from .engine import analyze_paths
from .models import RULES, Severity
from .report import to_json, to_sarif, to_text

_CONF = {"low": 0, "medium": 1, "high": 2}


def _emit(text, out):
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpusec", description="Static security analysis for CUDA kernels.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="analyze .cu/.cuh files")
    s.add_argument("paths", nargs="+")
    s.add_argument("--format", choices=["text", "json", "sarif"], default="text")
    s.add_argument("--output")
    s.add_argument("--fail-on", choices=["low", "medium", "high", "never"], default="high")
    s.add_argument("--min-confidence", choices=list(_CONF), default="low")
    s.add_argument("--sanitizer-log", help="Compute Sanitizer output used to confirm findings")
    s.add_argument("--ext", action="append", help="extra file extension, e.g. .cpp")
    sub.add_parser("rules", help="list rules")
    z = sub.add_parser("sanitize", help="run Compute Sanitizer on a built executable")
    z.add_argument("--tools", default="memcheck,racecheck,synccheck")
    z.add_argument("--log", default="sanitizer.log")
    z.add_argument("target", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)

    if a.command == "rules":
        for r in RULES.values():
            print(f"{r.id:12} {r.cwe:8} {r.name}\n             {r.description}")
        return 0

    if a.command == "sanitize":
        cmd = [x for x in a.target if x != "--"]
        if not cmd:
            p.error("give the executable after --, e.g. gpusec sanitize -- ./app")
        chunks = []
        for tool in a.tools.split(","):
            if tool not in TOOLS:
                p.error(f"unknown tool {tool}")
            chunks.append(f"######## {tool}\n" + run_sanitizer(cmd, tool))
        with open(a.log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(chunks))
        print(f"sanitizer log written to {a.log}")
        return 0

    findings = analyze_paths(a.paths, tuple([".cu", ".cuh"] + (a.ext or [])))
    if a.sanitizer_log:
        with open(a.sanitizer_log, encoding="utf-8", errors="replace") as fh:
            correlate(findings, parse_output(fh.read()))
    findings = [f for f in findings if _CONF[f.confidence] >= _CONF[a.min_confidence]]
    _emit({"text": to_text, "json": to_json, "sarif": to_sarif}[a.format](findings), a.output)
    if a.fail_on == "never":
        return 0
    return 1 if any(f.severity >= Severity[a.fail_on.upper()] for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
