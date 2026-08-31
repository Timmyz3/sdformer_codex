# M189 independent hammer review

Verdict: **86/100 — PASS as an honest negative cost audit; REJECT as the K7 implementation candidate.**

M189 is functionally sound at its stated module boundary. I independently checked all 254 legal 8-bank masks with a fresh VCS testbench tied to RTL SHA `a9e419e...`: 97,920 internal stable-order compaction terms, 24,576 signed output lanes, 253 II=1 replacements, nine held-result cycles and six empty/full/overflow fail-close checks all passed. The sealed run also retains 15/15 nonzero SVA covers and zero assertion-failure signatures. All 46 sealed input/output/DC-manifest hash entries pass.

The physical result is decisively negative and is reported honestly. M189 is 36,119.789785 um2, 1.331368782x the M185 K8 accumulator. M188+M189 is 46,537.469817 um2 versus M184+M185 at 37,156.643801 um2. After the exact K7 throughput factor of 0.999112212403, conditional throughput per area is only 0.797715405x. The contracts correctly keep physical/system speedup, paper-PPA and headline claims false.

## Blocking admission gates

1. M189 must remain a negative reference; it cannot be the K7 candidate.
2. M190 must synthesize to **strictly below 26,705.976561 um2** at the matched 3 ns logic-only corner just to break even. That requires more than 9,413.813224 um2 (26.062758%) reduction from M189 and 423.839211 um2 less area than M185. A useful 1% density margin would require no more than 26,338.415605 um2.
3. Clearing the standalone threshold is only a screen. Admission still requires exact-SHA flat M188+M190 synthesis because standalone area addition can hide duplication/removal across the boundary.

## M190 concept judgment

For RTL snapshot `d607cb9f...`, lowest-hole elision is mathematically correct for all 254 legal masks: removing the lowest empty bank maps slot `s` to structural bank `s` or `s+1`, and any further holes become zero lanes. There were zero mapping or signed-sum mismatches.

The hardware description must remain precise: each lane has an adjacent-bank two-source data choice **plus invalid-to-zero gating**; a lowest-hole priority encoder and broadcast control also remain. It is therefore an attractive next screen, not a free seven-mux result. Let matched DC decide it.

## Evidence boundary

- M189 DC is ideal-clock, ZeroWireload, zero-macro and pre-layout. High-fanout timing is estimated.
- No SRAM response, complete FC2, extracted parasitics, SAIF power, or system speedup is established.
- Formality should be spent only on the implementation that survives the area gate.
- `docs/359_DATE终局冻结_20260813.md` remains SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

Machine-readable details are in `m189_independent_hammer_review.json` and `independent_recompute_result.json`. The fresh test is reproducible with `tb_m189_independent_hammer.sv`; its VCS transcript is `sim.log`.
