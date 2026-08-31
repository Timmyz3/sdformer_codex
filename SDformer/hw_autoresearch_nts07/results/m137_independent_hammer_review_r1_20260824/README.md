# M137 independent hammer review r1

Outcome: **89/100, P0=0, P1=1, P2=5**. GO for exact-SHA fixed-latency functionality and the matched standard-cell logic-only DC A/B. NO-GO for reset-safe macro integration, macro-inclusive PPA, throughput or system speedup.

## Reproduced results

- Production VCS independently recompiles and reproduces 128 requests/outputs, 2,048 word checks, 120 II1 checks, all directed attacks and seven nonzero covers.
- An independent 65,538-request run naturally crosses token `ffff -> 0000`: 65,537 consecutive II1 intervals and every one-cycle response pass.
- Full stalled skid payload and all metadata remain stable. A skid pop can accept a new request on the same edge and immediately recover to fallthrough II1.
- Matched M136/M137 VCS measures delivery `2 -> 1` cycles while both remain II1. This is a 50% bridge-latency reduction, not a throughput gain.
- Both 20-entry DC evidence manifests verify. Independent reductions are area `22.4492748926%`, cells `10.0613824393%`, and sequential cells `46.9471947195%`.
- Read-only candidate DDC paths meet the current 3 ns port cut: data fallthrough +1.9747 ns, response-valid +1.9074 ns, ready feedback +2.1465 ns and token-to-ready +1.7924 ns.

## P1: reset epoch alias

M137 resets `next_token_q` to zero but has no macro flush/ack or epoch. A pre-reset token-0 response delayed across reset can match the first post-reset token-0 request. The independent attack observes `protocol_error=0` and accepts stale payload with new metadata.

Repair this before integration: flush and acknowledge the external macro response pipeline before re-enabling requests, or guarantee and assert an atomic macro+bridge reset; add a generation identity that cannot alias across reset.

## Performance interpretation

The standard-cell result is strong and fair: same 1,303 ports, same RTL cut, same TSMC28 libraries, SDC, DC Tcl, 3 ns clock, ideal clock, ZeroWireload and zero macros. The 569 fewer sequential cells exactly reconcile to replacing the two-entry FIFO with one skid.

The isolated bridge latency ratio is mathematically 2.0x, but throughput is unchanged at II1. Foundry macros, token wrapper, per-bank valid/skew, routing, propagated clock and power are absent, so none of the percentages is a full accelerator or system speedup.

Detailed findings are in `m137_independent_hammer_review_r1.json`; exact evidence reconciliation is in `m137_independent_evidence_audit_r1.json`.
