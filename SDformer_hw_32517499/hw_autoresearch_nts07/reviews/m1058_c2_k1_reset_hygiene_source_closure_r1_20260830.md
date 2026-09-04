# M1058 C2 K1 reset-hygiene source closure

Status: **PASS source RTL/VCS; mapped-gate claim remains closed pending independent M1059.**

## First-principles localization

The M1046 K1 gate watchdog is reproducible with UCLI and SAIF disabled. At 25.5 ns all three fault D inputs are 0. At 26.0 ns the K1 service fault D net `n95278` becomes X while the memory-adapter and core-adapter fault D nets remain 0. The service fault state itself is still 0 at that observation edge. Tracing `n95278` through `U101902/U114245/U102574/U95924/U114244` confines the first unknown to the service illegal-request/fault cone, before any accepted memory request.

Source audit found that the service resets FIFO pointers, count, and validity authority but does not reset the four FIFO payload arrays. Those invalid payloads are semantically masked in RTL, but remain X in four-state gate simulation and can reconverge through decoded array-select logic. The M1058 change explicitly resets only those four arrays. This is a source reset-hygiene hypothesis until fresh mapped replay passes; it is not yet a mapped-fix claim.

## Additive change

Only `fifo_tag_q`, `fifo_block_q`, `fifo_bank_id_q`, and `fifo_channel_q` entries are added to the synchronous reset branch. After module-name normalization and removal of that reset block:

- the service is byte-identical to M519;
- the standalone wrapper is byte-identical to M519;
- the K1 top is byte-identical to M519;
- the matched shell is byte-identical to M803.

ARCH_MODE 1 and 2 continue to instantiate the frozen K8 and K1x8 modules.

## VCS evidence

Both old K1 and M1058 K1 passed the same five reference workloads:

| case | events | old cycles | M1058 cycles |
|---:|---:|---:|---:|
| 0 | 20 | 259 | 259 |
| 1 | 41 | 737 | 737 |
| 2 | 90 | 3153 | 3153 |
| 3 | 110 | 7569 | 7569 |
| 4 | 0 | 14 | 14 |

All numeric, tuple, weight, accepted-unknown, and protocol mismatch totals are zero.

A separate validation-only VCS build used random register initialization and five reset length/phase/seed combinations. All five passed the same exact anchors. This use of `+vcs+initreg+random` is an attack oracle only; it is prohibited from production mapped replay and is not the repair.

## Admission boundary

M1058 does not launch DC, mapped VCS, SAIF, or PTPX. M1059 must independently check the exact source/release identities before authorizing a fresh K1 synthesis and mapped five-case replay without any initreg option. If that mapped replay still produces X, the FIFO-reset hypothesis is rejected and this line stops rather than adding broader resets blindly.

