# M139 independent hammer review r1

Outcome: **90/100, P0=0, P1=1, P2=5**. GO for the narrowly scoped M139 RTL/VCS claim and matched 3 ns logic-only DC comparison. Conditional GO for calling the bridge "epoch-safe": that wording is valid only when the external 16-bank wrapper's flush acknowledgement truthfully means all pre-reset responses are irreversibly drained. NO-GO for an unconditional epoch-safety, macro-complete PPA, energy, Fmax, kernel speedup or system-speedup claim.

## Independently reproduced

- Exact frozen M136/M137/M139 RTL, M139 SVA/TB/contracts, sealed completion markers and `docs/359` identities passed preflight. All sealed VCS/DC evidence manifests verified; `docs/359` remained at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- Synopsys VCS V-2023.12-SP1 reproduced the production PASS exactly: 65,667 requests/outputs, 1,050,672 words, 65,538 wrap-campaign requests, one natural 16-bit wrap, 65,657 II1 checks, seven flushes, 16 stall cycles and 15 skid cycles. All nine required covers were nonzero and there was no assertion failure.
- A separately written VCS hammer drove prolonged initial-high acknowledgement, eight stale-drain cycles, completion collision, postcompletion response, unsolicited RUN response, RUN-time acknowledgement, reset with an outstanding response, reset with a populated skid, false completion replay and an independent 65,538-request wrap stream. It passed all expected safety checks and observed the admitted false-completion limitation once.
- M136, M137 and M139 were rerun through the identical M139 flattened DC Tcl/SDC flow. The independently extracted areas, cell counts and setup/hold slacks exactly match the sealed receipts.

## P1: acknowledgement truth is still a system assumption

The low/high/low handshake proves that an acknowledgement transition occurred; it cannot prove that the absent macro wrapper actually drained every bank. The independent attack accepted an old token-zero request, reset, supplied a formally clean but false low/high/low completion, accepted a new token-zero request, and then replayed the old token-zero data. M139 returned the **new tag with old data and `protocol_error=0`**.

This is not a hidden contract violation: the frozen VCS contract explicitly sets `false_wrapper_completion_protected=false`, `actual_16_bank_flush_wrapper=false`, and prohibits claiming protection from false completion. It is nevertheless the principal integration risk. Before an unconditional epoch-safe claim, implement the actual 16-bank flush aggregator and prove that acknowledgement cannot rise until every bank/return queue is empty and cannot precede a later pre-reset response.

## What is genuinely closed

- An acknowledgement that is already high after reset is not accepted; recovery waits for a fresh sampled low.
- Responses during precompletion WAIT_LOW/WAIT_HIGH are quarantined and drained. A response colliding with the high completion sample, a response before acknowledgement drops, acknowledgement reassertion in RUN, and an unsolicited response in RUN all fail closed.
- The unchanged M137 data path is held in reset outside RUN, so request/response service and stale internal skid metadata cannot cross the recovery boundary.
- Under the behavioral one-cycle responder, normal operation remains one accepted request per cycle and one-cycle delivery across a natural `ffff -> 0000` token wrap.

These are directed commercial-simulation results, not an exhaustive proof of all temporal interleavings.

## Matched logic-only DC

| Design | Cell area (um2) | Cells | Comb | Seq | Setup slack (ns) | Hold slack (ns) |
|---|---:|---:|---:|---:|---:|---:|
| M136 | 6,098.148005 | 7,494 | 6,282 | 1,212 | +1.3130 | +0.0004 |
| M137 | 4,729.157996 | 6,740 | 6,097 | 643 | +1.3691 | +0.0009 |
| M139 | 4,466.070017 | 6,893 | 6,247 | 646 | +1.3331 | +0.0001 |

Against M137, M139 is 263.087979 um2 smaller (**-5.563104%**) despite 153 more total cells (**+2.270030%**) and three more sequential cells (**+0.466563%**). Against M136 it uses **26.763502% less cell area**, 8.019749% fewer cells and 46.699670% fewer sequential cells. All three meet the 3 ns constraint in the matched rerun.

The M139-vs-M137 area reduction is an exact synthesis observation, not evidence that safety has negative physical or energy cost: cell sizing/mapping changed, cell count increased, and this flow has ideal unpropagated clock, ZeroWireload, zero macros and no routing or power model.

## Performance boundary

M139 adds no normal pipeline stage and preserves M137's behavioral II1/one-cycle path. It does **not** accelerate it. Recovery needs at least three sampled transition edges and has no maximum latency or timeout if acknowledgement is missing. No workload-composed recovery overhead was measured.

Therefore the citable performance statement is only: "the reset repair preserves the previously admitted normal behavioral II1 and one-cycle response under the same one-cycle macro responder." Physical frequency, macro latency, energy and all kernel/system acceleration ratios remain false.

## Findings

- **P0 (0):** none found in the admitted, truthful-ack scope.
- **P1 (1):** false macro completion followed by same-token stale replay silently aliases; the actual 16-bank flush wrapper/proof is absent.
- **P2 (5):** no bounded recovery liveness/timeout; no exhaustive formal interleaving proof; no RTL-to-netlist Formality seal; DC is ideal-clock/ZeroWireload/zero-macro and has no PT/PTPX/route evidence; unchanged II/latency and uncomposed recovery cost do not support a speedup claim.

## Required next actions

1. Implement a per-bank flush/empty aggregator with a monotonic completion contract, then rerun the false-completion replay and delayed-response attacks against that wrapper.
2. Add formal properties tying `macro_flush_ack` to all-bank/return-queue emptiness and proving that no pre-reset response can occur after the completion edge.
3. Decide and verify the missing/stuck acknowledgement policy: deliberate indefinite quarantine or a bounded, externally observable timeout/escalation.
4. Seal M139 RTL-to-mapped-netlist equivalence, then add real SRAM timing/power and matched PT/PTPX evidence if any physical claim is needed.
5. Keep `physical_speedup=false`, `system_speedup=false` and `headline=false`; M139 is a correctness repair, not the acceleration headline.

The machine-readable record is `m139_independent_hammer_review_r1.json`; all files are pinned by `manifest.sha256`.
