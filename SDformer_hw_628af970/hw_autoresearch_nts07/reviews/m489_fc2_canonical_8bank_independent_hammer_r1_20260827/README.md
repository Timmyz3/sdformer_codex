# M489 canonical eight-bank FC2 independent hammer (r1)

## Verdict

**92/100 — GO to matched Synopsys DC/STA/SAIF/PTPX; NO-GO for a cycle-speedup, complete-FC2, system, or headline claim.**

Receipt-blind source review and an independent Synopsys VCS V-2023.12-SP1 rebuild/rerun reproduce the five directed rows exactly.  The candidate and baseline both terminate in eight instances of the same 128-bit, fixed-L4, eight-slot scalar-bank model.  They see the same globally phased request permission, response visibility, result backpressure, and done backpressure.  No compile warning, combinational-loop diagnostic, assertion failure, numerical mismatch, expanded transaction-multiset mismatch, weight mismatch, live-slot reuse, or conservation failure was observed.

The result is deliberately negative in cycles: the shared-state K8 candidate takes 59/143/505/1246 cycles, versus 51/131/486/1231 cycles for K1x8 at B=1/2/4/8.  Therefore the measured ratios 0.864407/0.916084/0.962376/0.987961 are **K8 throughput relative to K1x8**, not positive speedups.  Equivalently, K8 throughput is 13.6%/8.4%/3.8%/1.2% lower; if expressed as latency overhead, K8 is 15.7%/9.2%/3.9%/1.2% higher.  The production contract's phrase “1.2 to 13.6 percent slower” must be replaced by one of those unambiguous formulations.

## What was independently checked

- Current source identities were hashed independently; frozen `docs/359_DATE终局冻结_20260813.md` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- The production sealed directory passes both its inner manifest and outer seal checks.  The independent run does not invoke the production runner or consume its receipt.
- Both sides use eight `m349_fc2_scalar_bank_memory_model` instances with identical `BANK_ID`, width, L4 latency, newest-first policy, global request-allow phase, and response-visibility phase.  Runs start after reset with the phase counter reset, so candidate/baseline are cycle-phase matched.
- The inclusive cycle definition is one clocked monitor from accepted header through accepted token-done.  Inputs are driven at negedge and sampled at posedge, avoiding the earlier active-region endpoint race.
- Expanded request/response multisets, expected signed weights, Acc24 results, request/response conservation, all eight memory pending counts, and live-slot reuse are checked.  Candidate and baseline deliberately use different tags; equality is checked over the actual semantic tuple `(block,slice,channel)` and numerical result.
- M218, all eight M219 services, M342/M349 tops, M216 frontends, and M488 have live SVA cover evidence.  The independent seed was changed to 489927, but the testbench is directed, so this is a rebuild/replay check rather than randomized stimulus diversity.

## Reproduced observations

| B | Events | K8 cycles | K1x8 cycles | K8/K1x8 throughput |
|---:|---:|---:|---:|---:|
| 1 | 20 | 59 | 51 | 0.864407 |
| 2 | 41 | 143 | 131 | 0.916084 |
| 4 | 90 | 505 | 486 | 0.962376 |
| 8 | 110 | 1246 | 1231 | 0.987961 |
| 1 | 0 | 14 | 14 | 1.000000 |

The four-nonzero-row geometric-mean throughput ratio is 0.931504.  The cycle-summed ratio is 1899/1953 = 0.972350; this aggregate is tied only to these four directed rows and is not a frozen-trace or system-weighted number.

Observed stress includes 705 memory-request stalls, 45 result stalls, 1162 raw stalls, 882 candidate full-eight-bank bundles, 885 baseline eight-bank same-cycle issues, 994 candidate and 7024 baseline younger-before-older retirements, and four protocol attacks per the final PASS line.  M488-specific cover records 211 cycles of a fully stalled pending request, 883 all-eight response cycles, 373 decreasing-slot response pairs, 355 retire-then-reuse events, and four protocol attacks.

## Limits and risks

1. The integrated M489 test has zero hits for M488 `cp_partial_request_distribution` and `cp_core_response_stall`.  It therefore does not independently exercise unequal per-bank acceptance within one bundle or a stalled assembled core response.  Those behaviors exist in the separate M488 directed milestone, but a final composite regression should add both before paper freeze.
2. “Same memory” means identical bank topology, latency model, and exogenous visibility schedules, not identical request trajectories.  The candidate coalesces one multi-bank transaction through O8/FIFO4; the baseline has eight independent O8/FIFO4 services (aggregate O64/FIFO32).  This is a legitimate architecture trade-off and a strong performance baseline, but it is not an iso-internal-buffer comparison.
3. Only five directed workload pairs are present.  There is no frozen 120-record FC2 replay, multi-sequence workload, complete FC2/FFN path, SRAM macro, physical timing, or energy evidence.
4. Both sides perform the same expected number of scalar bank reads.  M489 cannot claim weight-SRAM traffic or read-energy reduction from this test; any benefit must come from shared controller/context/Acc24 logic and must include the adapter overhead.
5. VCS compile emitted no loop diagnostic, and the reviewed ready/valid path breaks the former response-dependent request combinational cycle.  This is not a formal combinational-loop proof; DC timing/constraint checks and, ideally, Formality remain required.

## Physical admission gate

Use identical 3.0 ns TSMC28 libraries, clock/input/output constraints, eight scalar-bank pins, loads, and memory-macro assumptions for M489 and M349.

- For geometric-mean throughput/logic-area efficiency to exceed K1x8, require `A_K8/A_K1x8 < 0.931504`.  For the four-row cycle-summed efficiency, require `< 0.972350`; to win every nonzero row, require `< 0.864407`.
- Report both logic-only and macro-inclusive area.  The same eight SRAM macros cancel in traffic but may dilute a logic-area advantage at chip level.
- SAIF must replay identical semantic events and external bank-read counts.  Report energy per completed token/output work, not power alone.  Per row, K8 energy wins only if `P_K8/P_K1x8 < cycles_K1x8/cycles_K8`; equal read count forbids crediting memory traffic savings.
- Require clean setup/hold and all five timing constraints for both.  If the K8 physical result misses the selected throughput/area and energy gates, retain M489 only as a fairness closeout/negative DSE, not a paper contribution.

## Scoring

| Dimension | Score | Reason |
|---|---:|---|
| Reproducibility and identity | 10/10 | Independent VCS rebuild/replay and production double-seal check pass. |
| External-memory fairness | 18/20 | Same eight-bank model and exogenous schedules; internal O8 vs O64 capacity is intentionally different. |
| Numerical/transaction correctness | 24/25 | Strong semantic and conservation checks; external scalar tuple checking is partly inferred through adapter identity and result checks. |
| Protocol/SVA stress | 17/20 | Broad active SVA and OOO/backpressure attacks; two integrated adapter covers remain zero. |
| Cycle measurement and wording | 13/15 | Race-free inclusive definition; only directed rows and production “slower” wording needs correction. |
| Claim discipline | 10/10 | Contract correctly denies positive cycle, complete-FC2, physical, system, and headline claims. |
| **Total** | **92/100** | **Scoped GO to physical efficiency gate only.** |

