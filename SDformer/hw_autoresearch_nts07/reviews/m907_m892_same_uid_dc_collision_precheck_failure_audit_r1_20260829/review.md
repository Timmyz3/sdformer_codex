# M907 — M892 same-UID DC collision precheck failure audit

Verdict: **PASS failure audit (100/100)**.

The M892 pre-attempt gate pipes `ps ... -o args=` into an `rg -q` expression
whose text contains `dc_shell`.  The `ps` snapshot can see that same `rg`
process, so the expression matches its own argv.  No external DC process is
needed.  On a clean host the old expression returned 0 (collision), while an
exact `/proc/<pid>/exe`/`comm` identity scan returned 1 (clean).

The failure occurred before the launch lock, license query, attempt token, and
DC invocation.  Therefore the original M892 one-shot authority is unconsumed.
The only admitted repair is additive: preserve every sealed M892 file and place
a pinned compatibility shim in front of the one defective `rg` invocation.
All other `rg` calls delegate to the pinned `/usr/bin/rg`.  Collision decisions
must use executable/comm identity, never argv text.

This audit establishes no DC, PPA, speedup, energy, or paper claim.
