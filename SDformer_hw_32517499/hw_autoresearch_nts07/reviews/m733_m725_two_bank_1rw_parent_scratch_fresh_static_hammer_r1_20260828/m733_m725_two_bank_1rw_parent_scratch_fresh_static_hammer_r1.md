# M733｜M725 two-bank 1RW parent scratch fresh static hammer r1

Date: 2026-08-28

## Verdict

**PASS_STATIC_ONLY, 95/100.** P0/P1/P2 = **0/3/3**.

This review authorizes exactly one stratified CPU fast-kill with at most three
workers, using the exact analyzer and contract identities listed below.  It
does not authorize RTL, VCS, GPU, DC, Formality, PT, PTPX, a full-population
replay, or any paper performance claim.

The proposed mechanism is physically different from the rejected pseudo-dual
port interpretation of M528: each selected bank contains all nine 128-bit
slices of one 1152-bit parent row.  A read and a write overlap only when their
row hashes choose different complete banks.  No read/write overlap is granted
inside one 1RW bank.

## Frozen review identity

- Analyzer SHA256: `92c3dbeb7c595d2a41a32d493ad0a43cf1caa7512179f36568681358cb698c29`
- Contract SHA256: `6f48bda6541042e613cd9045e61ede2a92116795c3feee38c1c184af91503e34`
- M504 analyzer SHA256: `9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e`
- M505 analyzer SHA256: `9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced`
- Frozen row ledger SHA256: `6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334`
- M505 result SHA256: `b8a29f2fafc0e7d051d66ed206cd5c25efb866d4a1ab02082aa71bad4b14eb61`
- M528 same-ledger result SHA256: `778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1`
- docs/359 SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
- Required interpreter path: `/opt/anaconda3/bin/python` (observed Python 3.12.7)
- Observed interpreter target SHA256: `873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161`
- Observed NumPy version: 1.26.4

The result directory was absent at review time.  Static compilation under the
required interpreter succeeded.  The author analyzer was not imported or
executed, and no CPU/GPU/EDA workload was launched by this review.

## Audit findings

### 1. Bank organization and port legality — pass

`bank(row) = parity(row & xor_mask)` selects one of two complete row banks.
The scheduler permits `read && write` only when the read parent and write row
select different banks.  Same-bank overlap is blocked and asserted.  Forwarding
does not consume a macro port.  Thus there is no hidden 2RW/1R1W macro and no
reuse of the nine bit slices as independent row banks.

The accounting is arithmetically consistent with the frozen M528 capacity:
`213376 + 18432 = 231808 B`, leaving `13952 B` under the `245760 B` budget.
This is a capacity result only; the second bank's physical area, delay, muxing,
and dynamic energy remain unmeasured.

### 2. M505 schedule and baseline — pass for a fast-kill

The candidate reuses M504's clean-room parent map and popcount order, and
recomputes the M505 dead-write-only baseline on the same 64-row tile.  The
candidate conserves arithmetic issues, parent edges, live writes, synchronous
one-cycle reads, the two-entry ordered response queue, and producer forwarding.
Each candidate is required not to regress the recomputed single-bank cycle
count.  No ideal fused M473 ceiling is used as the positive denominator.

The predictive same-bank expression does not explicitly include the queue
capacity predicate.  Ordered one-parent-per-consumer service makes this
conservative for the immediate-next deadline case; if the implicit invariant
ever fails, it can only add a hold or trip the non-regression assertion, not
create a free speedup.  This should nevertheless be made explicit before any
full replay or RTL translation.

### 3. Population and DSE — pass with a strict scope limit

The planned population is 11 deterministic partitions for each of all ten
samples and all four bottleneck Conv operators: 440 of 17,280 phases.  Every
selected phase processes all 3,000 rows in the same 64-row task grain.  Spawn
workers reopen the frozen ledger with `os.pread`; no large ledger is pickled
into workers.  The top-level `__main__` guard and module-level worker functions
are compatible with `ProcessPoolExecutor(..., mp_context=spawn)`.

Twelve compile-time XOR masks are evaluated, then the lowest-cycle mask is
selected on this same profiling population.  This is legal profile-guided DSE,
not a runtime oracle.  It is also not held-out evidence and is not exhaustive
over all 63 nonzero masks.  Therefore a positive point may only nominate one
mask for the already-contracted full-population same-ledger replay; it cannot
support cross-sequence generalization or a paper number.

### 4. Gates and claim boundary — pass

GO requires all three conditions: macro-rounded capacity within 240 KiB,
local issue-window speedup at least 1.05x, and stall reduction at least 30%.
The result explicitly labels full population, full-pipeline cycle,
same-ledger speedup, RTL/VCS/Synopsys PPA, energy, system speedup, and paper
headline as false.  A failed performance gate kills the idea before RTL.

## Nonblocking findings

- **P1-1:** the contract names the Python path but does not SHA-pin the
  interpreter/NumPy or assert `sys.executable`; this authorization therefore
  applies only to the exact command below on the reviewed host.
- **P1-2:** the selected XOR mask is tuned and scored on the same 440-phase
  profile.  A full-population replay and later cross-sequence check are needed
  before any generality claim.
- **P1-3:** 231,808 B proves capacity only.  Matched macro area, mux/control
  area, frequency, and access energy are mandatory if the full replay passes.
- **P2-1:** M505 and M528 result files are SHA-frozen but not semantically
  parsed by M725.  The local baseline is safely recomputed from frozen source,
  but downstream admission must reconcile its totals with the sealed results.
- **P2-2:** the analyzer has overwrite protection but no mechanically consumed
  one-shot launch token.  The exact output path and contract SHA must be
  checked by the result hammer.
- **P2-3:** `PASS_STRATIFIED_CPU_FASTKILL` denotes successful execution, not a
  positive gate decision.  Downstream readers must use
  `decision.go_full_same_ledger_replay`.

## Sole authorized command

From `/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07`, and only
when VCS, Design Compiler, and PrimeTime PX are not active:

```bash
/opt/anaconda3/bin/python \
  system_simulator/scripts/analyze_m725_h67_two_bank_1rw_parent_scratch_fastkill.py \
  --contract contracts/m725_h67_two_bank_1rw_parent_scratch_fastkill_contract_r1_20260828.json \
  --out results/m725_h67_two_bank_1rw_parent_scratch_fastkill_r1_20260828 \
  --workers 3
```

Before launch, recheck the analyzer SHA and contract SHA against this review.
After launch, a fresh independent result hammer must verify the output seals,
all conservation equalities, the 440-phase population, per-mask rows, gate
logic, and claim boundary before any full-population replay is authorized.
