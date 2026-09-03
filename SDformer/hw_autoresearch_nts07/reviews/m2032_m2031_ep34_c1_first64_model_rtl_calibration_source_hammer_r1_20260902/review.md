# M2032 independent M2031 C1 calibration-source review

## Verdict

**PASS, 95/100; P0/P1/P2 = 0/1/1.** No blocking source defect was
found. The 64-row fixture is byte-exact to the first 64 rows of the
double-sealed M1590 support16 ledger. I independently implemented both the
maximum-popcount/earliest-row parent matcher and the one-read/write liveness
schedule without importing the candidate audit or frozen M505 model. The
result exactly reproduces the candidate constants:

| Counter | Independent value |
|---|---:|
| Input / residual nonzeros | 565 / 192 |
| Exact-parent rows / parent edges | 4 / 58 |
| Issue accepts / stalls | 196 / 14 |
| Dead writes / macro reads / macro writes | 31 / 54 / 33 |
| Forwards / deadline holds | 4 / 6 |
| Liveness cycles | 210 |
| Psum commits / row completions | 64 / 64 |

The independent hammer passes under both platform Python 3.6 and Anaconda
Python 3.12. The supplied source audit itself passes under the pinned Anaconda
Python 3.12 interpreter.

## Static SV audit

The wildcard DUT connection is complete for the pinned current DUT ports, and
no `force` or `release` seam is present. Data-valid follows request-valid;
source data is reconstructed from the selected residual source while parent
psum supplies the common subset. The checker validates all 96 signed lanes on
every accepted psum commit, demands atomic row completion, requires the ep34
task-done token, then waits for a negative edge before sampling all terminal
counters. Per-execution and global watchdogs are present.

The scratch wrapper instantiates nine copies of the pinned 128x128 1RW foundry
model. That model exposes the expected module and its `UNIT_DELAY` branch with
`SRAM_DELAY=0.0100`. This is a source review only: no VCS compile, simulation,
license query, or GPU job was launched here.

## Nonblocking findings

P1: the candidate audit starts with `/usr/bin/env python3`, but this host's
default `python3` is 3.6.8 and rejects `from __future__ import annotations`.
The exact audit passes with `/opt/anaconda3/bin/python3.12`; a future runner
must pin that interpreter rather than trust the shebang.

P2: the testbench fixture and foundry macro are absolute-path dependencies.
The next runner must fail closed on existence and exact SHA before consuming a
license. This affects portability, not the reviewed logic.

## Authorization and paper boundary

With P0=0, authoring one exact-SHA, fail-closed VCS runner is authorized.
Executing EDA is not authorized by this review.

Even if that future run passes, it establishes only event-count and signed
numeric calibration on one real ep34 64-row tile for the frozen M528 r2
reference island. It does **not** promote the M1590 `1.694510x` CPU cycle-model
ratio to an RTL speedup, does not validate a latest mapped successor, and does
not establish full-network/system performance, timing, energy, or a headline
claim. docs/359 remains unchanged.
