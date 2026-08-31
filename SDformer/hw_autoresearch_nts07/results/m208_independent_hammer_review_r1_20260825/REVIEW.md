# M208 independent hammer review

## Verdict

**58/100 — FAIL_P0_RTL_DEADLOCK; revoke M207/M208/M209 admission.**

The original M208 evidence is internally consistent for its stated 256 cases:
all `4 geometries × 4 modes × 16 seeds` are present exactly once, the Synopsys
VCS log has the same identity domain, and all 256 reported control-cycle values
match the production recurrence.  Its negative claim fields are also intact.

That calibration is not the legal RTL input domain.  Every generated bitmap is
very sparse (at most three events per beat), all tokens are isolated, and both
downstream ready signals are permanently asserted.  The maximum observed M202
queue occupancy is only two.

## Independent evidence

The independent Synopsys VCS suite generated 152 new structured/random
payloads without importing or executing the production M208 Python model.  It
also checked exact event conservation, no duplicate source, output-block
packet stability, tag/count identity, and a same-cycle terminal header chain.
An independently written transaction-level oracle compared cycle, group,
packet, maximum-queue, descriptor-hold, two-buffer-full, and terminal-collapse
results.

- 152/152 cases completed; 0 oracle mismatches below the overflow boundary.
- 38 cases per legal output-block geometry.
- 140 two-buffer-full, 120 partial-tail, 4 odd-full/trailing-zero, 4 zero,
  35 stage-0 multi-window, 148 terminal-collapse cases.
- Six cases reached queue occupancy at least six (original M208 maximum: two).
- The held legal next header was accepted on the same edge as terminal done;
  both chained tokens completed with no protocol fault.

## P0: legal bank-dense payload deadlocks M207

A legal descriptor can contain twelve events in one physical bank.  Therefore
a legal four-descriptor packet can add `4 × 12 = 48` events to that bank.  The
reviewed M207 RTL declares `descriptor_bank_sum[bank]` as five bits, truncating
48 to 16.

The dedicated VCS reproduction sends two full four-descriptor windows at the
legal 48-event packet bound.  It accepts and replays 192 groups, but never
retires the token.  At the terminal deadlock:

- `upstream_done_seen = 1` and both windows remain closed;
- both bitmaps have been emptied, so `candidate_count = 0`;
- the truncated bank counts underflow to 224 in each buffer;
- `candidate_total = 448`, so no new group can load and token done is false.

This is a legal-payload functional failure, not a performance-model delta.
Consequently M208 is admissible only as a historical statement about its exact
256 sparse vectors.  It cannot admit the M207 legal domain, frozen H67 replay,
M209 numbers, complete FC2, PPA/energy, system speedup, or a headline claim.

## Read-only M210 repair audit

The M210 derivative functionally addresses this P0: the sum is six bits, a
96-event/window capacity guard is present, SVA bounds the packet sum at 48, and
the dedicated bank48 VCS run records two accepted 48-event packets, 192 groups,
and one correct completion.  Its inherited 256-case sweep is also 0-mismatch.

The reviewed r2 evidence is not yet exact-input sealed.  The result directory's
own `SHA256SUMS` is clean, but `input_sha256.txt` expects
`tb_m210_fc2_bank48_adversarial.sv = 9e57ea93...`; the current file is
`ff7e8284...`.  The current TB adds a latency field absent from the sealed log.

M210 admission therefore requires:

1. rerun bank48 VCS from the current exact TB and reseal inputs/logs;
2. run this independent 152-vector payload/oracle suite on M210;
3. rerun frozen H67 replay with M210 model/RTL hashes;
4. keep complete-FC2, physical, energy, system, and headline fields false until
   their own evidence exists.

## Score and findings

- Interface/control concept: 16/20
- Verification identity and breadth: 14/20
- Legal-domain functional correctness: 4/25
- Reproducibility: 10/15
- Claim discipline: 10/10
- Complete-FC2/physical evidence: 4/10

P0: legal bank48 traffic deadlocks M207; revoke M208/M209 admission.

P1: “256-case calibration” is a structured sparse sample, not exhaustive state
or legal-payload coverage.

P2: control-cycle calibration does not establish complete FC2, PPA, energy,
system speedup, or a paper headline.
