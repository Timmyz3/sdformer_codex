# M1056 exact packed-psum 1RW arbitration source receipt

## Verdict

`PASS_M1056_SOURCE_ONLY__M1057_REQUIRED_NO_FULL_REPLAY`

The additive M1056 source closes the modeling hole identified by M1051 at the
small-oracle level. It does not run or admit the full 51.84M-row result.

Each M1016 psum event now carries a packed macro group, logical bank, physical
row address, program order, base-ready cycle, and dependency set. Four packed
groups each expose exactly one 1RW grant per cycle. Requests enter a fixed FIFO
by arrival cycle and then program order; caller list order has no authority.
Different addresses still serialize on the shared port, while a same-address
read waits at least one cycle after its predecessor write.

## Backpressure closure

A delayed read delays its dependent write. The last actual write grant updates
task completion; task completion updates the next task's work start; that shift
updates all later event cycles and the final sample commit. The model never
computes a corrected result as `raw_cycles + 403922`.

The directed cascade makes the distinction concrete:

| Item | Value |
|---|---:|
| M1016 nominal cycles | 20 |
| Arbitrated cycles | 22 |
| Nominal excess accesses | 16 |
| Delayed accesses | 24 |
| Task starts | 0, 11 |
| Nominal task ends | 8, 19 |
| Effective task ends | 9, 20 |

The cycle penalty is 2, not 16. Arbitration can occupy slack or create
cascaded stalls, so an excess-access count is not a latency increment.

## Fairness and gates

Candidate, strongest-zero, and same-coordinate-bit must use identical common
service counts/digests and the same four-group 1RW arbiter configuration.
Design-specific work can produce different completion cycles only after this
same-resource arbitration.

`capacity_bytes_pass` and `port_feasibility_pass` are separate outputs. A
214,912-byte sum is below 240 KiB, but that fact alone never admits a port
calendar, matched cycles, speedup, or PPA.

Thirteen directed tests cover no-conflict identity, multiplicity 2 and 3,
reversed cross-task input, different-address collision, same-address RAW,
dependency deadlock, delayed completion cascade, three-design asymmetry, and
capacity/port gate separation.

## Next gate

An independently authored M1057 hammer must attack all directed anchors and
the frozen identities. Only after M1057 may a separate release authorize one
CPU full replay. This receipt provides no launch authority.

No full replay, EDA, GPU, or remote job ran. M1016, M1040/M1051 evidence, and
`docs/359` were not modified.
