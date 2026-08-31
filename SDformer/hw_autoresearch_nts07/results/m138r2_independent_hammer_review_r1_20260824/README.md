# M138r2 independent hammer review r1

Outcome: **87/100, P0=0, P1=1, P2=5**. GO for exact-SHA fixed-latency functionality and the flattened 3 ns standard-cell logic-only result. NO-GO for reset-safe macro integration, macro-inclusive PPA, throughput/physical/system speedup or a DATE headline.

## What reproduced

- A fresh VCS compile reproduces 96 vectors/outputs, 212 accepted beats, 208 macro requests, 8,832 signed lane checks, four escapes, 63 exact 2/2/2/3 vector-start interval checks, 13 stall cycles and all six nonzero SVA covers.
- Independent attacks test the exact last-data/first-padding boundary at 8, 9, 10 and 11 bits. All four legal boundary bits assemble correctly; all four first-padding bits quarantine immediately, become sticky and launch no following read.
- The r2 DC manifest verifies. Across its log and reports, TIM-209/OPT-150/ELAB-312/Error counts are all zero; the superseded r1 has one TIM-209 and six OPT-150 occurrences.
- Independent DC reduction is 15,295.644056 um2, 19,042 cells, 15,644 combinational cells, 3,398 sequential cells, zero macros, setup +0.4516 ns and hold +0.0000 ns at the frozen ideal-clock ZeroWireload 3 ns cut.
- Versus the separately synthesized M133+M137 sums, integration reduces area by 287.279992 um2 (1.843556%), cells by 365 (1.880765%), sequential cells by 12 (0.351906%) and combinational cells by 353 (2.206664%). This is only an integration diagnostic.

## Active P1: reset epoch still aliases

M138r2 keeps the unchanged M137 token generator. An old token-0 response delayed across reset aliases the first new token-0 request. The independent attack completes a new width-8 vector with the new tag but stale pre-reset data, with `protocol_error=0`.

Repair before macro integration: add macro flush request/ack and block requests until stale returns are invalidated, or guarantee and assert an atomic macro+frontend reset; also add a non-aliasing epoch/generation identity.

## Ready/valid and padding result

The r1 self-loop was `assembler_protocol_error -> quarantine -> assembler input valid/ready -> assembler_protocol_error`. R2 cuts that path with registered `downstream_fault_q`. The remaining response-ready to request-ready path is feed-forward: response validity comes from registered pending/skid state and the macro response, not current request readiness. DC's loop check is clean, and 128 valid-low payload perturbations keep the idle ready/macro boundary stable.

Metadata/state faults are genuinely pre-SRAM zero-read. Final padding is different: it can only be checked after the final 512-bit return, so the final read is already spent. The independent test shows no *following* read escapes, but the design must not claim all faults are rejected before SRAM.

## Claim boundary

Citable now: the exact commercial-VCS counters and 2/2/2/3 schedule under a fixed one-cycle aggregate behavioral macro; the four-width data/padding boundary result; and the exact flattened TSMC28 logic-only area/cell/slack numbers with zero macros, ideal clock and ZeroWireload.

Not citable: failed r1 DC numbers; reset safety; foundry macro behavior; per-bank skew; macro-inclusive PPA; routed Fmax; power/energy; or any 2x throughput/module/accelerator/system speedup. M137's two-to-one number is isolated first-delivery latency while steady-state II remains one.

Full findings are in `m138r2_independent_hammer_review_r1.json`; exact reconciliation is in `m138r2_independent_evidence_audit_r1.json`.
