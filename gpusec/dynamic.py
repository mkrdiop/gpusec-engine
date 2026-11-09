"""Dynamic side: run NVIDIA Compute Sanitizer and correlate its errors with static findings.

NOTE: the text parser is best-effort and based on the usual Compute Sanitizer output layout
("========= Invalid __global__ write of size 4 bytes", "... by thread (x,y,z) in block ...").
Validate it against your CUDA version on real hardware before relying on it.
"""
from __future__ import annotations
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
_CATEGORY = [(re.compile(r"Invalid __\w+__ (?:read|write)|out.of.bounds", re.I), "oob"),
             (re.compile(r"Uninitialized", re.I), "uninit"),
             (re.compile(r"Race reported|hazard", re.I), "race"),
             (re.compile(r"Barrier error|Divergent thread", re.I), "sync")]
_KERNEL = re.compile(r"\bin (?:(?P<file>[^\s:]+):(?P<line>\d+):)?(?P<kernel>[A-Za-z_]\w*)\s*\(")
RULE_CATEGORY = {"GS-MEM-001": "oob", "GS-MEM-002": "oob", "GS-RACE-001": "race", "GS-SYNC-001": "sync"}


@dataclass
class DynamicError:
    category: str
    kernel: Optional[str]
    file: Optional[str]
    line: Optional[int]
    text: str


def run_sanitizer(cmd: List[str], tool: str = "memcheck", timeout: int = 900) -> str:
    p = subprocess.run(["compute-sanitizer", "--tool", tool, "--print-limit", "100", *cmd],
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def parse_output(text: str) -> List[DynamicError]:
    """Split the log into blocks separated by bare '=========' lines; one error per block."""
    blocks, cur = [], []
    for raw in text.splitlines():
        if not raw.startswith("========="):
            continue
        body = raw[9:].strip()
        if body:
            cur.append(body)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    errors = []
    for blk in blocks:
        for k, line in enumerate(blk):
            cat = next((c for rx, c in _CATEGORY if rx.search(line)), None)
            if cat:
                m = next((_KERNEL.search(l) for l in blk[k:] if _KERNEL.search(l)), None)
                errors.append(DynamicError(cat, m.group("kernel") if m else None, m.group("file") if m else None,
                                           int(m.group("line")) if m and m.group("line") else None, line))
                break
    return errors


def correlate(findings, errors: List[DynamicError]):
    """Mark static findings as confirmed when a dynamic error of the same kind hit the same kernel."""
    for f in findings:
        cat = RULE_CATEGORY.get(f.rule_id)
        if cat and any(e.category == cat and e.kernel == f.kernel for e in errors):
            f.confirmed, f.confidence = True, "high"
    return findings
