# M497 canonical K1 versus K1x8: independent hammer r1

Audit date: 2026-08-27 (Asia/Shanghai)

Score: **93/100**

Verdict: **CONDITIONAL GO** for M496 matched DC.  M497 itself passes its
directed functional and identity gate.  The M496 executable gate remains closed
until its K1 elaboration is changed from M494/M490 to the validated M499
no-reuse wrapper and all affected identities are locked again.

## 1. Receipt-blind recomputation

The four nonzero rows in `sim.log` are:

| Output blocks | Canonical K1 | Replicated K1x8 | K1x8/K1 |
|---:|---:|---:|---:|
| 1 | 253 | 51 | 4.960784313725490x |
| 2 | 773 | 131 | 5.900763358778626x |
| 4 | 3154 | 486 | 6.489711934156379x |
| 8 | 7659 | 1231 | 6.221770917952884x |

The independently recomputed geometric mean is
`(253/51 * 773/131 * 3154/486 * 7659/1231)^(1/4)` =
**5.863399625158185x**.  The aggregate ratio is
`(253+773+3154+7659)/(51+131+486+1231)` =
**6.234333859926277x**.  These are different, both correct statistics.  Use
5.8634x when the text says geometric mean and 6.2343x only when it explicitly
says aggregate cycle ratio.  The zero-event endpoint is 14/14 cycles.

The cycle monitor uses one positive-edge owner, starts at accepted header, ends
at accepted `token_done`, and adds one to make both endpoints inclusive.  The
global visibility gates reset to the same phase for each architecture.  The
candidate and baseline run sequentially, but receive the same deterministic
request, response, result, and done visibility schedules relative to reset.

## 2. Functional and transaction evidence

The exact run passes 10 clean architecture-cases, two midflight POR cases and
four protocol attacks.  It records zero numeric mismatches, zero transaction
multiset mismatches and zero weight mismatches.  Conservation checks close the
expected read, request, response, result and pending counts.  Both endpoints
exercise out-of-order response retirement; the candidate covers 8,052
single-source requests and the replicated endpoint covers 885 full eight-bank
issue cycles.

The testbench computes signed reference accumulators independently from a
deterministic weight function.  It checks each accepted response payload,
per-(block,slice,channel) request/response multiplicity, final Acc24 outputs and
completion counts.  This is materially stronger than checking final output
alone.

## 3. M499 loop repair

Relative to M490, M499 changes the request slot gate from

`!slot_valid_q[slot] || (core_rsp_accept && complete_slot == slot)`

to

`!slot_valid_q[slot]`.

That removes the direct response-accept-to-request-ready bypass in the outer
adapter.  M219 retains its internal accepted-response free-slot bypass and the
scalar SRAM model retains response/request replacement.  Waiting one registered
edge at the outer boundary is therefore a conservative, architecture-preserving
way to break the K1 integration feedback cone.  The observed evidence supports
the functional statement: the old M494/M490 development image did not complete
the clean K1 case within the attempted run, whereas exact M499 completes every
case and all scoreboards/SVAs pass.

Claim limit: there is no independent formal combinational-loop report or
Formality result in M497.  Write “removes the outer same-edge reuse dependency
and restores exact VCS progress,” not “formally proves the only possible
three-layer loop.”

M490 and M491 hashes remain the same as those pinned by M492, so this K1-only
repair does not mutate the admitted equal-bandwidth K8 evidence.

## 4. SVA, warnings and coverage audit

The runner rejects compile errors, requires exactly one bracketed warning, and
allowlists both the exact BTNL class and the exact absent M218 bind target.  The
observed compile log contains only that warning.  It is benign for this
elaboration because M497 instantiates M219 services but no M218 service; the
M218 assertion source is inherited through the shared baseline assertion file.

All runner-required candidate and baseline SVA covers have nonzero matches.
Important candidate counts include 4,906 internal M219 same-cycle replacements,
2,660 outer retire-then-next-cycle slot reuses, 7,744 cut-through responses,
2,751 out-of-order bundle-response patterns, 1,123 pending-request stalls and
four protocol-attack observations.  There are no reported assertion failures.

Known coverage holes are bounded:

- M499 full-eight-bank request and eight-response-same-cycle covers are zero by
  construction for a single-source K1 endpoint.
- M499 same-cycle outer slot reuse is zero by design; the next-cycle reuse cover
  is nonzero.
- The adapter-level core-response-stall cover is zero.  M219 result stall and
  request/raw backpressure are covered, so this does not invalidate the current
  arithmetic/cycle claim, but a future randomized adapter regression should hit
  and check the held completed-bundle path before paper-ready signoff.
- One seed and directed payloads are insufficient for exhaustive correctness;
  M497 remains a directed VCS gate, not formal verification or frozen H67 replay.

## 5. Identity and seal

The sealed manifest and its outer seal both verify.  The current runner SHA
equals the runner SHA recorded by the result.  All source, assertion, testbench,
filelist, contract, and frozen-doc preflight observations equal their expected
SHA.  `docs/359_DATE终局冻结_20260813.md` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## 6. Claim boundary

Admitted:

- exact-SHA Synopsys VCS for a directed FC2 slice;
- exact arithmetic and transaction-multiset equivalence for the tested cases;
- the four inclusive directed cycle rows;
- 5.863399625x geometric K1x8/K1 cycle scaling under an 8x peak
  bank/service-bandwidth increase;
- M499 as the validated low-bandwidth endpoint for the later three-axis Pareto.

Forbidden:

- same-resource speedup;
- frozen H67 or full-network speedup;
- complete FC2 or complete FFN performance;
- physical, power, energy or paper-ready PPA;
- multiplying 5.8634x by another local ratio;
- presenting 6.2343x without labeling it aggregate weighting;
- claiming that M499 is faster than M490 (it deliberately removes a bypass).

## 7. Hard gate to M496

At the audit snapshot, M495 `ARCH_MODE=0` still instantiates
`m494_fc2_k1_cutthrough_8bank_raw4_acc24`, and the M496 exact runner still pins
M494.  Therefore:

1. **NO-GO to the current M496 runner as-is.**
2. Change only the K1 elaboration to
   `m499_fc2_k1_no_reuse_8bank_raw4_acc24` and include the M499 adapter/wrapper
   in the DC filelist.
3. Recompute and lock M495, filelist, runner and contract identities before
   execution.  The contract should explicitly name M499 as the K1 point.
4. Keep the same M495 external port shape, library, operating condition, 3 ns
   SDC, compile sequence and reporting gates for all three modes.
5. Require all three setup/hold and five constraint classes to pass.  Report
   area and sequential-cell ratios before applying the already admitted cycle
   evidence.
6. A passing M496 remains logic-only pre-macro evidence.  SAIF/PTPX, common SRAM
   macro modeling and receipt-blind hammer remain separate gates.

After items 2–4 are sealed, M497 gives a **GO** to start M496 matched DC.

## 8. Score breakdown

| Category | Score |
|---|---:|
| Identity, preflight and seal | 20/20 |
| Numeric/transaction conservation | 25/25 |
| Cycle endpoint and schedule fairness | 18/20 |
| SVA/adversarial coverage | 17/20 |
| Claim discipline | 10/10 |
| Causal proof and downstream readiness | 3/5 |
| **Total** | **93/100** |

The deductions are for directed/single-seed scope, the unhit adapter response
hold cover, the absence of a formal loop proof, and the stale M494 identity in
the downstream M496 assets at the audit snapshot.
