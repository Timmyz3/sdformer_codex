# M2069 M2063 init0 mapped-energy failure hammer

## Verdict

**PASS, 97/100, P0/P1/P2 = 0/1/2.** M2063 is a conclusively
consumed failure. Its exact identity is permanently no-retry and it is not
citable as mapped functionality, SAIF, PTPX, power or energy evidence. This
review launches no VCS, EDA, license query or GPU task and does not modify any
M2063 source/result artifact or `docs/359`.

## Evidence integrity

Both namespaces are healthy and exhaustive:

- The attempt token has one sealed member; its inner manifest and outer seal
  verify, and it says `M2063_ATTEMPT_CONSUMED_NO_RETRY`.
- The failure quarantine has five sealed evidence members; its inner manifest
  and outer seal verify.
- The sealed fingerprint covers 112 raw-work members: 110 regular files,
  2 symlinks and 237,392,430 regular-file bytes. Recomputing the entire raw
  tree produces an exact ordered match.
- The quarantined compile, runtime and license logs are byte-identical to the
  retained raw-work copies.

The attempt used one license preflight, one ordinary compile and one ordinary
simulation. It produced zero SAIF and zero PTPX runs; the TSBG axis was never
attempted.

## This is not another X failure

M2058 failed at the execute boundary on a coalesced control X/Z check. M2061
localized the first failure to `ordinary.cycle_count` at the first settled
execute negedge. M2063 is materially different:

1. It completes all 192 descriptor loads at cycle 383.
2. It enters the ordinary SAIF window.
3. It runs to 20,679 SVA attempts and reaches `base_done_cycle`.
4. It then fails at 62,037,010 ps in line 275 with
   `M2063 ordinary mapped completion ledger drift`.

The runtime log contains no `mapped X/Z` fatal. Therefore the quarantine's
generic `failure.json` text, `runtime fatal/XZ ordinary_lru4`, is inaccurate
for this attempt and must not be propagated into reports.

Deterministic zero initialization removed the earlier four-state observability
barrier well enough to run the workload. It did **not** repair or prove mapped
dynamic equivalence.

## What drifted

The ordinary completion task checks one ten-way disjunction:

1. measured execute cycles = 20,292;
2. row accesses = 149;
3. issues = 1,278;
4. products = 29,472;
5. cache misses = 149;
6. cache hits = 0;
7. cache evictions = 145;
8. weight-bundle beats = 1,788;
9. scalar-bank requests = 14,304;
10. scalar-bank responses = 14,304.

At least one comparison rejected. The source TB's M2057 reference completes at
cycle 20,676 and reports 20,292 base cycles for the same slot42 workload.
However, M2063 prints no actual completion values, no member name and no first
divergent cycle; neither sealed quarantine nor its exactly fingerprinted raw
tree contains a waveform. The failing member and whether multiple members
drifted are therefore **not uniquely recoverable**. Inferring that the cycle
term alone failed from elapsed wall-clock simulation time would be unsound.

No arithmetic-mismatch, duplicate-commit, protocol, stale or overflow fatal
appears before line 275. That narrows the observed failure to completion/counter
identity, but it does not prove arithmetic correctness because the wrapper
preempts the source TB's complete final ledger and PASS sequence.

## Findings

- **P1:** `failure.json` misclassifies the visible completion-ledger fatal as
  runtime X/Z. The log is authoritative for failure class.
- **P2:** the ten-way completion check has no per-member actual/expected dump.
- **P2:** no waveform or differential trace can locate the first RTL/mapped
  divergence.

## ISCAS and next-step decision

M2063 itself is permanently no-retry. For the ISCAS window I also recommend
closing this mapped-power branch: three progressively improved no-retry
attempts have yielded zero SAIF/PTPX results, and a fourth power runner would
repeat acquisition before locating the divergence.

This failure does **not** invalidate the M2057 RTL-cycle result or independently
admitted logic-only C2 DC area/throughput evidence. The paper should stay at
RTL-cycle plus pre-macro logic-area scope, explicitly leave measured mapped
power unavailable, and prioritize FC2 continuation plus manuscript closure.

If power work is resumed after the deadline, begin in a new diagnostic-only
namespace. Print all ten actual/expected ledgers and capture the first
cycle-by-cycle RTL/mapped divergence before proposing another SAIF/PTPX
attempt. This review authorizes neither that successor nor any new tool run.

`docs/359` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
