# M939 — M925/M896 decoder GTLS first-row result hammer

## Verdict

**Result integrity PASS 100/100, P0/P1/P2 = 0/0/0; production and paper
admission FAIL by frozen scope.**  M925 completed exactly the one authorized
`M854_FIRST_D0_A1_T0` full-row diagnostic.  It did not run a decoder-complete
or full-population workload and cannot supply a production Table-A row,
hardware speedup, system speedup, energy, or paper-ready PPA result.

This review was read-only.  It did not start the simulator, EDA, GPU, remote
work, or a network operation, and it did not modify the result, attempt,
source, production tables, or `docs/359`.

## Identity, seals and process drain

The canonical result contains the five exact payloads named by its manifest;
the manifest and outer seal verify with no missing, extra, or symlinked
payload.  The consumed one-shot attempt also verifies.  Its M925 source
contract, M927 release, M930 fresh-source hammer and M928 final-launch hammer
all bind through their exact identities and recursive seals.

The worker was the actual Python 3.10 process in a private process group and
session (`PID=PGID=SID=1544683`).  It returned zero, was reaped, and the group
was proven empty before publication.  That PID is absent at review time.  The
result namespace contains only the canonical result and consumed attempt; no
orphan stage, failure, or log namespace remains.

## Exact one-row diagnostic

For the single D0, A1-OSG, timestep-0 row, independent arithmetic reproduces:

- expanded requests: **38,672,612**
- compressed transactions: **9,582,057**
- total diagnostic cycles: **20,548,766**
- address transaction SHA-256:
  `78b90d378956948fc3eab3d7a1bd6f88c8bcf4d32871e971641c9b1a62dfaa6e`
- commit sequence SHA-256:
  `aa69b355efd62b428e2909ee4c1dbecdf34ec3e1e8681b0c78ace19a444ff861`

The cycle classes sum exactly to the reported total: 18,502,452 active-service,
2,046,313 dependency-completion, one compute, and zero memory, psum-bank and
weight-bank cycles.  The 4.0359405 expanded/compressed ratio is a representation
diagnostic, not a speedup.

## Runtime and safety

The measured elapsed time is 937.461076 s with 10,716,244 KiB process peak RSS
and 532,147,294 B counted live scheduler state.  Relative to the 932.078357 s
M883 host anchor, the diagnostic host ratio is **0.994258×**, so it does not
show a host-runtime gain.  The historical 9.320784 s 100× hypothesis had
already failed and was not retried here.

All 13 resource snapshots retain a heartbeat, one group member, and zero
timeout/resource flags.  The lowest observed available memory and commit
headroom are 408,452,916 KiB and 113,303,024 KiB; peak sampled group RSS is
9,205,024 KiB.  Completion at 937.46 s is below the 2,715 s operational safety
timeout.

## Claim boundary and minimum extension

M925 proves that the frozen M896 scheduler can complete and exactly seal one
D0 row under bounded process control.  It does **not** prove coverage of D1,
D2, D3, another scene/sample/timestep, the full decoder population, or a
same-resource system aggregation.  Neither its diagnostic cycles nor its host
runtime may be inserted into production Table-A.

The smallest informative next step is a fresh identity—not a retry or reuse of
M925—with one exact row from each of D0, D1, D2 and D3 at the same sample,
A1-OSG configuration and timestep.  D1 exactness is a prerequisite.  Each row
and the aggregate should carry transaction/commit hashes and cycle classes.
That four-layer slice would close layer coverage only; it would still remain a
diagnostic.  A sealed multi-sequence stratified slice and then a
decoder-complete same-resource aggregation are required before Table-A
admission can be considered.
