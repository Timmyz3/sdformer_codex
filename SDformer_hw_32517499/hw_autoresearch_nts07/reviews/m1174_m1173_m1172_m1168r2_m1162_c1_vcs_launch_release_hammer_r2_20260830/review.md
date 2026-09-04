# M1174 fresh release hammer

Verdict: **GO (100/100, P0=0, P1=0)**.

The exact M1173 release (`31302e76…`) and its recursively sealed author receipt bind the M1172 source hammer (`review d82bf311…`, outer `1b8ef5ac…`), the compile-repaired R2 sources, and runner `4a661d50…`. The old consumed R1 attempt is explicitly non-reusable. The R2 attempt, result, work, and quarantine namespaces were fresh.

The runner contains exactly one VCS compile and one timed simv invocation, uses the foundry `UNIT_DELAY` model, consumes the attempt only after byte/seal, same-UID, and 64-GiB gates, and recursively seals both failure quarantine and canonical success. Static checking passed 1,104 checks and rejected 43 controlled mutations. No runner, VCS, simv, EDA tool, or license query was executed by this hammer.

This hammer authorizes only one future functional VCS compile and one simv run after live revalidation. It does not verify function, timing, cycles, PPA, power, energy, speedup, or any paper claim.
