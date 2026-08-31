# M1242 — independent hammer of M1239 R11 checker hardening

Verdict: **NO-GO for release authoring.** The unchanged R11 candidate remains
structurally admissible, and M1239 rejects the three exact destructive
mutations requested by M1235. However, the checker still accepts four nearby
destructive variants that preserve its lexical tokens while disabling the
behavior. Score: **88/100**; P0=0, P1=1, P2=0.

## What passed

The candidate TB remains exactly
`850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776`.
M528, M935, M1162, the R3 SVA, R10 TB, and `docs/359` retain their frozen
hashes. The M1239 author package and source contract verify through both seal
layers. The canonical checker passes. All 18 declared unittests pass, and the
original set of 15 M1232 test names is still present.

The independent campaign confirms that all three named M1235 holes are now
closed: disabling `random_request_window_active`, deleting the post-retirement
response-count oracle, and replacing `repeat (hold_cycles)` with `repeat (0)`
are each rejected with a non-empty checker error list.

## P1 — lexical tokens can mask disabled behavior

Four independent in-memory mutations were accepted with an empty error list:

1. enable the request window and immediately assign it back to zero;
2. compute a positive `hold_cycles` and immediately overwrite it with zero;
3. leave the enable statement only in a comment and execute a zero assignment;
4. leave the positive repeat statement only in a comment and execute
   `repeat (0)`.

The first two show that the checker does not prove unique executable
assignment or execution order. The last two show that comments can satisfy
the current substring predicates. These are not formatting-only variants:
each disables a claimed exercised oracle while retaining the strings the
checker searches for. Therefore the checker is still overfit to tokens and
cannot authorize release publication.

The narrow repair remains checker/tests-only. Keep the exact candidate TB SHA
frozen; strip comments before executable-anchor checks (or parse the task),
require unique ordered assignments, reject a later zero overwrite before the
window/hold use, add negative tests for all four variants, reseal, and request
a new independent hammer.

## Boundary

No candidate source, DUT RTL, M935, M1162, SVA, workload, or frozen document
was changed. No VCS, simv, EDA, GPU, or remote command was invoked. Functional
VCS, timing, cycles, speedup, PPA, energy, system speedup, headline, and paper
admission remain false.
