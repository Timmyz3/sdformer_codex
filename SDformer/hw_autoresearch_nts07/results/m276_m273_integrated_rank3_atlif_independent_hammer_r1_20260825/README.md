# M276 independent hammer review of M273 integrated rank3 ATLIF

Verdict: **the integrated standalone module is strong directed functional RTL,
but it is not yet speedup/PPA/accuracy evidence and its legal-traffic fault
output is not glitch-clean.** Evidence quality is `92/100`; hardware admission
is `58/100`; P0/P1/P2 are `0/5/3`.

## Independent Synopsys replay

The frozen producer launcher was copied to a fresh relocated tree and rerun with
Synopsys VCS V-2023.12-SP1. The exact producer test, all SVA covers and its
sealed manifest passed. A wrong expected RTL SHA exited `10` before compile.

An independently written TB then exercised the frozen RTL with eight
configuration attacks and five raw framing/tag attacks. It also used two
independent numeric profiles instead of the producer's reference functions.
It checked:

- 9,600 stage1 accumulator states and 6,400 stage2 event decisions;
- 113 even and 149 odd RNE tie cases;
- 715 Q8 and 278 Q24 saturation cases;
- raw-bank and intermediate-bank dual-ready oldest-first arbitration;
- same-cycle one-entry product replacement;
- registered FIFO empty behavior and full simultaneous pop/push;
- complete context drain before release.

There were zero sampled-edge numeric, tag, beat, valid-mask, ordering,
conservation or lifecycle mismatches.

## Exact cycle boundary

| Context | Cycles |
|---|---:|
| Zero tiles | 7 |
| One tile, continuous/ready-high | 24 |
| Four tiles, continuous/ready-high | 39 |
| One tile plus two config bubbles | 26 |
| Forty tiles, result ready one cycle in eight | 1618 |

Thus `5*N+19` is reproduced for nonempty, gap-free, ready-high contexts. It is
not a universal context formula: the RTL accepts a zero-tile release in seven
cycles, while the formula gives 19 and producer SVA requires at least 24. The
1618-cycle pressure point belongs only to its frozen backpressure schedule; it
is not an arbitrary-backpressure maximum.

## Protocol finding

Legal sustained-valid traffic produced 47 half-cycle `protocol_error` pulses in
the 40-tile pressure test. After an accepted beat, the RTL advances its beat
state while the source legally retains the old `last/data` until later in the
cycle. The combinational `fault_event` then sees a false framing mismatch and
directly qualifies `protocol_error`, `result_valid` and the issue paths.

The pulse clears before the next active edge, so `protocol_error_q` is not set
and the producer's posedge-only SVA does not observe it. This is a P1 integration
issue: register fault reporting or validate captured beat/next-state data, then
add an intra-cycle stability check.

## Claim boundary

Paper-safe wording is limited to an executable standalone integrated module,
zero directed sampled-edge mismatches, and the explicitly conditioned cycle
measurements above. There is no area-matched Fixed RTL, checkpoint-derived
rank3 configuration, trained accuracy admission, full trace replay, DC/STA, or
macro-aware power/energy. No speedup, PPA, energy, system, accuracy or headline
claim is admitted.

The final reproducible flow is `run_m276_m273_fresh_relocation_vcs.sh`; the
sealed VCS evidence is under `vcs_run_r7_final`. Producer files and
`docs/359_DATE终局冻结_20260813.md` were not modified. No open-source RTL tool was
invoked.
