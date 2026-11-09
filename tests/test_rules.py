import os
import pytest
from gpusec.engine import analyze_source
from gpusec.dynamic import parse_output, correlate

HERE = os.path.dirname(__file__)


def load(name):
    with open(os.path.join(HERE, "samples", name)) as f:
        return analyze_source(f.read(), name)


def by_kernel(findings):
    d = {}
    for f in findings:
        d.setdefault(f.kernel, set()).add(f.rule_id)
    return d


def test_safe_code_has_no_findings():
    assert load("safe.cu") == []


def test_vulnerable_kernels():
    d = by_kernel(load("vulnerable.cu"))
    assert d["scale"] == {"GS-MEM-001"}
    assert "GS-SYNC-001" in d["divergent_sync"]
    assert d["early_exit"] == {"GS-SYNC-001"}
    assert d["race"] == {"GS-RACE-001"}
    assert d["shared_oob"] == {"GS-MEM-002"}
    assert d["spin"] == {"GS-LOOP-001"}
    assert d["tail_loop"] == {"GS-SYNC-001"}
    assert d[""] == {"GS-API-001"}


def test_write_is_high_read_is_medium():
    f = [x for x in load("vulnerable.cu") if x.kernel == "scale"]
    assert {x.severity.name for x in f} == {"HIGH", "MEDIUM"}
    assert {x.cwe for x in f} == {"CWE-787", "CWE-125"}


def test_api_severity():
    api = [x for x in load("vulnerable.cu") if x.rule_id == "GS-API-001"]
    assert sorted(x.severity.name for x in api) == ["LOW", "MEDIUM"]


def test_dynamic_confirmation():
    log = """========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x70 in scale.cu:6:scale(float *, const float *, int)
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f is out of bounds
=========
========= ERROR SUMMARY: 24 errors
"""
    errs = parse_output(log)
    assert errs and errs[0].category == "oob" and errs[0].kernel == "scale"
    fs = correlate(load("vulnerable.cu"), errs)
    assert all(f.confirmed for f in fs if f.kernel == "scale")
    assert not any(f.confirmed for f in fs if f.kernel == "race")
