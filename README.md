# GPUsec static engine (MVP)

Pure-Python static analyzer for CUDA (`.cu`, `.cuh`). No LLVM needed: a lexer, a lightweight
statement parser and a taint analysis that labels each variable *uniform* (0), *data-dependent* (1)
or *thread-variant* (2, derived from `threadIdx`). Rules use that label, which removes most false
positives of plain regex checks (a barrier under `if (n > 0)` is fine; under `if (tid < 32)` it is not).

## Use
    pip install .
    gpusec scan src/ --format text            # or json / sarif
    gpusec scan . --format sarif --output gpusec.sarif --fail-on high
    gpusec rules
    gpusec sanitize -- ./app                  # runs Compute Sanitizer (GPU + CUDA toolkit required)
    gpusec scan src --sanitizer-log sanitizer.log   # marks findings "confirmed"

Exit code 1 when a finding reaches `--fail-on` (default `high`). Suppress with
`// gpusec-ignore GS-MEM-001 reason` on the line or the line above.

## Rules
| Id | Detects | CWE |
|---|---|---|
| GS-MEM-001 | thread-indexed global access with no visible bounds guard (write = HIGH, read = MEDIUM) | 787 / 125 |
| GS-MEM-002 | constant index outside a `__shared__` array | 119 |
| GS-SYNC-001 | `__syncthreads()` under thread-dependent branch, loop or early exit | 833 |
| GS-RACE-001 | `__shared__` read of another index after a write, no barrier between | 362 |
| GS-LOOP-001 | `while(true)` / `for(;;)` with no exit in a kernel | 835 |
| GS-API-001 | ignored return of a `cuda*` runtime call (host code) | 252 |

Recognized guards: `if (i >= n) return;`, `if (i < n) {...}`, loop conditions, `i < n && a[i]`,
`i < n ? a[i] : x`, `%`, `min`, `clamp`.

## Known limits (be honest with users)
* Heuristic, not a proof: a guard is recognized when a relational condition mentions the index variable
  (it does not verify that the bound equals the allocation size). Static OOB findings are therefore
  `medium` confidence until Compute Sanitizer reproduces them.
* Analyzes `__global__` kernels only; `__device__` helpers, macros that hide kernel loops
  (`CUDA_KERNEL_LOOP`), pointer arithmetic and aliasing across function calls are not followed.
* Write-after-read hazards, warp-level primitives and cooperative groups are not modeled.
* The Compute Sanitizer log parser is based on the usual text layout; validate it on your CUDA version.

## Next steps
Clang AST/LLVM-IR backend behind the same rule interface for inter-procedural analysis, `__device__`
functions, a labeled corpus to measure precision/recall, and a SARIF baseline mode for incremental adoption.
