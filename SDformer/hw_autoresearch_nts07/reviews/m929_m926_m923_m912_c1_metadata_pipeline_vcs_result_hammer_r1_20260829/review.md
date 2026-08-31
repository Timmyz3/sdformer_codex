# M929 — M926/M923/M912 C1 metadata-pipeline raw VCS result hammer

## Verdict

`PASS_M929_M926_M923_M912_FUNCTIONAL_VCS_RESULT_HAMMER`, **100/100,
P0=0, P1=0, P2=0**.  The canonical M926 result is admitted only as a
foundry-`UNIT_DELAY` functional VCS result.  M929 did not run the runner, VCS,
`simv`, another EDA tool, or a license query, and did not modify the result,
RTL, SVA, TB, contract, release, frozen predecessor, or `docs/359`.

## Result integrity and fresh identity

The canonical result has 112 regular-file entries in its recursive manifest
and exactly 112 actual non-symlink regular files when the two seal files are
excluded.  Every entry verifies, the outer manifest seal verifies, and there
is no missing or extra regular file.  The two VCS-created symbolic links are
intentionally not followed by the `find -P -type f` contract:

- `csrc/_883037_archive_1.so`
- `simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml`

The unique M926 attempt marker exists and pins runner SHA
`08c71339...685`.  It was created at `2026-08-29T04:16:23Z`, before the
canonical result.  There is no M926 failure quarantine.  The frozen M923
one-shot marker and M923 canonical result remain absent; therefore M923 itself
was not consumed.  M926 is the fresh additive identity that consumed exactly
one attempt.

## Compile, simulation, attacks, and coverage

The fail-closed runner accepts both elements of the compile and simulation
pipelines only when they are zero; canonical promotion occurs only after these
checks and all token gates.  Thus the admitted compile and simulation pipeline
return codes are both zero.  The compile log shows VCS
`V-2023.12-SP1_Full64`, parses the pinned foundry 128x128 1RW model and all
pinned RTL/SVA/TB identities, and has no warning, error, or fatal marker.

The exact runner compiles with `+define+UNIT_DELAY`; neither
`+notimingcheck` nor `+no_notifier` appears in the invocation or result logs.
This is explicitly functional macro-model evidence, not timing evidence.

The simulation has exactly one PASS, coverage, P2-strength, held-final, and
M923 phase-correction token.  It reports no assertion failure, error, or fatal.
All 14 required normal cover fields are nonzero.  P2 reports 20 consecutive
distinct reads and 189 response-identity checks.  All six attack classes occur
exactly once: dirty-reserved, stale-epoch, overflow, wrong-parent,
read-before-write, and parent-only-nonzero.  The repaired wrong-parent phase
token proves row 1 captured parent 63 with `relation_ok=0` within three
watchdog iterations.

All six reported SVA cover properties match at least once: task done 7,
two-cycle initial fill 9, promote without bubble 260, same-edge PF replacement
196, dead-then-live 3, and deadline-then-write 7.  The terminal PASS reports
271 commits, 7 completed tasks, 8 two-cycle fills, and all six attacks.

## Exact source and release chain

The receipt's runner, contract, RTL, SVA, and TB hashes match the live pinned
files.  The M926 contract's inner and outer seals verify.  The independent
M927 source hammer's recursive manifest and outer seal verify, and its exact
review/manifest/outer trio is `5a788613...c6ee`, `32996610...1c010`, and
`ef0c4dec...3225`.  The separately authored M928 release is double-sealed and
binds that exact trio together with the M926 runner/contract and M912/M923
source identities.

M919 remains in its sealed failure quarantine with exit code 22 and
`functional_vcs_verified=false`.  The independent M922 forensic designation
`FAILED_OR_INCOMPLETE_DO_NOT_CITE` remains authoritative and unchanged.

## Claim boundary

M929 admits only: **the M912 metadata-pipelined 1RW product-capture island
passes this directed/random/adversarial foundry-`UNIT_DELAY` functional VCS
workload under the pinned source identity**.

It admits no timing, cycle count, speedup, PPA, energy, full-system result,
system speedup, headline, or paper-citable performance claim.  In particular,
the CPU-ledger 1.74x candidate is not upgraded by this result.
