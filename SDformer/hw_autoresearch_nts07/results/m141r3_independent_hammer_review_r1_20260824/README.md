# M141r3 independent hammer review

Status: `78/100`, `P0=0`, `P1=2`, `P2=5`.

The production analyzer reproduced byte-for-byte. An independent implementation then rebuilt the 20 heldout records and all 69,120 descriptor recurrences without importing M141's schedule class. It exactly reproduced M132 serial service, B2/B3/B4 cycles, weight-beat arithmetic, PWP/correction work, bank occupancy, barriers, and the three quoted ratios.

The arithmetic result is real under the frozen aggregate recurrence. The B3 RTL choice is not closed because the model starts PWP before descriptor fill finishes on 49,521 descriptors and does not define token arrival, forwarding, bank ports, or buffer representation. It also inherits M140's unimplemented already-sparse-mask producer. A conservative full-materialization sensitivity reduces B3 from 2.6362x/1.8412x to 2.4292x/1.6966x and changes the B3/B4 trade; those sensitivity numbers are not proposed as replacement headlines.

Independent positives:

- `weight_beats_per_key=3` closes exactly: 8,271,296 groups produce 24,813,888 weight tokens;
- folded/startup work is 99,916,708 and total correction work is 124,730,596;
- peak live bank ownership is exactly 2/3/4, with no same-bank early reuse;
- all 300 zero-correction descriptors also have zero PWP work, so the empty branch does not create a heldout collision;
- all 160 window barriers drain PWP/correction; producer lookahead is bounded to B descriptors; and
- the quoted 2.636228947x, 1.841237749x, and 8.361900143x ratios are arithmetic-exact but remain module/service-island cycle-model metrics.

Files:

- `m141r3_independent_hammer_review_r1.json`: score, findings, and claim boundary;
- `independent_recompute_and_attack.json`: exact independent counters and attacks;
- `audit_m141r3_independent.py`: reproducible independent audit;
- `production_exact_rerun_receipt_r1.json`: byte-exact production rerun receipt; and
- `immutable_manifest.sha256`: hashes for this evidence package.

No production analyzer, contract, result, RTL, or docs/359 file was changed.
