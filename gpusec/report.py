from __future__ import annotations
import json
from collections import Counter
from .models import RULES, Severity
from . import __version__

_LEVEL = {Severity.HIGH: "error", Severity.MEDIUM: "warning", Severity.LOW: "note"}
_SCORE = {Severity.HIGH: "8.0", Severity.MEDIUM: "5.0", Severity.LOW: "3.0"}


def to_text(findings):
    if not findings:
        return "GPUsec: no findings.\n"
    out = []
    for f in findings:
        tag = "  [confirmed by Compute Sanitizer]" if f.confirmed else ""
        out.append(f"{f.file}:{f.line}:{f.col}: {f.severity.name} [{f.rule_id}] {RULES[f.rule_id].name}{tag}")
        out.append(f"    kernel: {f.kernel or '(host)'} | confidence: {f.confidence} | {f.cwe}")
        if f.evidence:
            out.append(f"    {f.line:>5} | {f.evidence.strip()}")
        out.append(f"    {f.message}")
        out.append(f"    fix: {f.fix}\n")
    c = Counter(f.severity.name for f in findings)
    out.append(f"GPUsec: {len(findings)} finding(s): " + ", ".join(f"{c[s]} {s}" for s in ("HIGH", "MEDIUM", "LOW") if c[s]))
    return "\n".join(out) + "\n"


def to_json(findings):
    return json.dumps({"tool": "gpusec", "version": __version__, "findings": [f.to_dict() for f in findings]}, indent=2)


def to_sarif(findings):
    rules = [{"id": r.id, "name": r.name, "shortDescription": {"text": r.name},
              "fullDescription": {"text": r.description}, "help": {"text": r.description},
              "properties": {"tags": ["security", "cuda", r.cwe]}} for r in RULES.values()]
    results = []
    for f in findings:
        results.append({
            "ruleId": f.rule_id, "level": _LEVEL[f.severity],
            "message": {"text": f"{f.message} Fix: {f.fix}"},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": f.file},
                                                "region": {"startLine": f.line, "startColumn": f.col}}}],
            "properties": {"security-severity": _SCORE[f.severity], "confidence": f.confidence,
                           "kernel": f.kernel, "cwe": f.cwe, "confirmedDynamically": f.confirmed}})
    return json.dumps({"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
                       "runs": [{"tool": {"driver": {"name": "GPUsec", "version": __version__,
                                                     "informationUri": "https://www.gpusec.com", "rules": rules}},
                                 "results": results}]}, indent=2)
