from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import IntEnum


class Severity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    cwe: str
    description: str


RULES = {r.id: r for r in [
    Rule("GS-MEM-001", "Unbounded global memory access", "CWE-787",
         "A global-memory pointer is indexed with a thread-derived index and no bounds guard is visible. "
         "Writes map to CWE-787, reads to CWE-125."),
    Rule("GS-MEM-002", "Constant index outside a __shared__ array", "CWE-119",
         "A __shared__ array is accessed with a constant index outside its declared size."),
    Rule("GS-SYNC-001", "Barrier in divergent control flow", "CWE-833",
         "__syncthreads() is reachable only by some threads of a block (thread-dependent branch, loop "
         "trip count or early exit). The kernel can hang or behave undefined."),
    Rule("GS-RACE-001", "Shared-memory hazard without barrier", "CWE-362",
         "A __shared__ element is read after a write to a different index with no __syncthreads() between."),
    Rule("GS-LOOP-001", "Kernel loop without exit", "CWE-835",
         "An unconditional loop with no break or return can hold the GPU indefinitely (denial of service)."),
    Rule("GS-API-001", "Unchecked CUDA runtime call", "CWE-252",
         "The error code returned by a CUDA runtime call is ignored, so failures go unnoticed."),
]}


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    confidence: str          # low | medium | high
    file: str
    line: int
    col: int
    kernel: str
    message: str
    fix: str
    cwe: str = ""
    evidence: str = ""
    confirmed: bool = False  # set when Compute Sanitizer reproduced it

    def to_dict(self):
        d = asdict(self)
        d["severity"] = self.severity.name
        return d
