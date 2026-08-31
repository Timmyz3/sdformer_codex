# M186 independent hammer review

Verdict: **CONDITIONAL_FLAT_EXECUTABLE_REFERENCE_ONLY**, score **80/100**.
M186 is executable and numerically correct inside one uninterrupted reset
epoch, but it is **not yet an unconditional fail-closed reference** because an
old response can alias a new request across reset.  It remains ineligible for
complete-FC2, physical-speedup, system-speedup, paper-PPA or headline use.

## Evidence that passed

- Both sealed receipts and their manifests/input lists verify.  The VCS
  `RUN_COMPLETE.txt` SHA-256 is `4ff995f84ef3ce9a49770bfa7e975e73f25e0fc1677038b1991c34785ddc3f8c`;
  the DC receipt SHA-256 is
  `403879ef252a83baa305f563739dd67ebd02363e14ca7a0339b7ddc85cce15c9`.
- A fresh VCS build using the exact full-coverage command reproduces the
  sealed run exactly: 190 requests, 190 responses, 190 results, 997 replayed
  source terms, 147 same-cycle response/request replacements, 22 non-prefix
  requests, both fault attacks and all ten nonzero coverpoints.  There are no
  assertion failures.
- A separately written VCS test exhausts all 255 nonempty 8-bank masks and
  checks 24,480 lane results against an independent integer calculation.  It
  observes 255/255/255 request/response/result conservation, 255 completions,
  all 64 masks with bit7 set and bit6 clear, request/result stalls, reset flush
  with no new pending request, and sticky rejection of an unsolicited stale
  response.  There are no assertion failures.
- The flat DC receipt is internally consistent: 37,144.673821 um2, 42,428
  cells, 4,298 sequential cells, 143 logic levels, 2.52 ns critical path,
  +0.0002 ns setup slack and +0.0000 ns hold slack at 3.000 ns.  Reports show
  zero macros, zero mapped multipliers, zero max/min/cap/transition/fanout
  violations, an ideal clock and `ZeroWireload`.

## P0: reset response aliases a new request

The response interface has neither request tag nor epoch.  The RTL only tests
`weight_response_valid && !pending_valid_q`.  A response that arrives after
reset while no new request is pending is therefore rejected, which is the
case covered by the primary test.  That is not the hard case.

The independent negative probe performs this legal temporal sequence:

1. request A is accepted and remains outstanding;
2. local reset clears A's pending slot;
3. request B is accepted after reset;
4. A's delayed response arrives while B is pending.

VCS proves that the old payload is accepted, labeled with B's tag/mask,
produces 96 corrupted lanes, and leaves `protocol_error=0`.  Existing SVA also
passes because `ap_response_requires_pending` sees B pending; the interface
contains no identity with which to distinguish A from B.

M186 can therefore be cited only under an explicit external contract that the
weight service is synchronously reset and guarantees that no pre-reset
response can appear after post-reset admission.  The robust fix is an echoed
request identity/epoch plus comparison, or a flush request/ack that keeps new
admission closed until the service has discarded all old responses.  A fixed
quarantine without a proved response-latency bound is insufficient.  Add the
cross-reset A/reset/B/stale-A sequence to admission VCS/SVA.

## Performance audit

The arithmetic is independently reproduced:

- M184 + M185 standalone sum: 37,156.643801 um2.
- Flat M186 saves only 11.969980 um2, or 0.032214912%, versus that sum.  Flat
  boundary optimization therefore supplies essentially no material K8-area
  amortization.
- M180 + M169 K4 standalone sum: 32,940.809935 um2.  M186 is 1.127618717x,
  or 12.761871655% larger.
- The bounded analytic schedule ratio is 127,581,198 / 97,607,807 =
  1.307079853x.  Combining that numerator with the unmatched standalone K4
  area yields **1.159150548x conditional throughput/logic-area**.  The
  independently recomputed break-even K8 area is 43,056.268999 um2.

The 1.159150548x number is not yet a fair measured density result.  The K4
denominator is a sum of separately synthesized blocks rather than a matched
flat K4 issue island.  More importantly, the schedule assumes one group
result per cycle, while M186 has one outstanding request and no real SRAM.  It
can sustain one request per cycle only when the external service supplies an
accepted response every cycle after the initial latency.  Any response gap or
latency greater than one cycle is exposed because the single slot cannot hide
it.  A latency-aware K4/K8 comparison with the same response service is P0 for
performance admission.

The apparent response slot stores only tag/block/mask metadata.  The 6,144-bit
K8 weight response and the 2,304-bit Acc24 context are broad combinational
inputs; neither payload storage nor SRAM/context access is included in area or
energy.  Weight SRAM, descriptor producer, context store, BN2, residual,
PAFT-specific `sn2` threshold identity and valid825 remain outside M186.

## Remaining P1/P2 gates

- **P1:** build a matched flat K4 issue island under the same constraints and
  drive both K4/K8 with identical, explicit response-latency/backpressure
  traces; report service cycles as well as frontend schedule cycles.
- **P1:** replace the 6,144+2,304-bit ideal external payloads with a concrete
  banked SRAM/context interface and account for macro area, access energy and
  routing.  The current setup margin is only 0.0002 ns under ideal-clock,
  zero-wireload DC.
- **P1:** run PAFT `sn2`-specific threshold census and valid825.  A non-unit
  threshold invalidates the current multiplier-free numeric identity unless
  it is exactly folded into weights.
- **P1:** close Formality and macro-aware PT/SAIF/PTPX before any physical or
  energy claim.  DC reports 4,298-load clock and 4,305-load internal high-
  fanout nets whose delay calculation is capped at 1,000 loads.
- **P2:** resolve the signed/unsigned conversion warnings in M184/M185, and
  upstream the exhaustive-mask and cross-reset tests.  Random stall counters
  are reproducible only with the exact coverage instrumentation; admission
  should prioritize invariant conservation/coverage conditions over incidental
  random counts when instrumentation changes.

`docs/359_DATE终局冻结_20260813.md` remains unchanged at SHA-256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

