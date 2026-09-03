# ISCAS 2027 commit 4ea66eb2 / M2050 artifact-open independent review r4

- Date: 2026-09-03 (Asia/Shanghai)
- Reviewed commit: `4ea66eb299475130ba99acd86dd2f95a66c66870`
- Reviewer identity: `/root/iscas_recent_paper_reviewer`
- Mode: artifact-open, read-only. No EDA, GPU, license query, or network task was launched.
- Modification boundary: only this independent review artifact was created. The paper, M2050/M2030 evidence, predecessor artifacts, and `docs/359` were not modified.

## Final verdict

**Overall 4.2/5: Accept, estimated 85--90% acceptance tendency. Strong Accept probability is approximately 25--35%; the draft has not yet reached a dependable Strong Accept.**

M2050 closes the previous four-group representativeness weakness. The exact RTL distribution now covers 192 performance-independent workloads across four sealed DSEC sequences, all 12 FC1 layers, four FC2 layers supported by the same G48 frontend, and first/middle/last aligned B4 token quartets. The paper reports the weighted component result accurately and retains the negative tail, real-activity/directed-weight split, and physical boundary. It never promotes `2.5061x` to full-FC, same-area, energy, or system performance.

The result remains a component paper rather than an end-to-end accelerator result. That is acceptable for ISCAS, especially because C1, C2/TSBG, and C3 are explicitly called execution islands rather than a measured monolithic top. Strong Accept remains limited by TSBG's incomplete FC2/token/sample population and logic-only hold/power-open companion, plus C1's separated model and physical anchors.

## Reviewer scores

| Dimension | Score / 5 | Independent assessment |
|---|---:|---|
| Novelty | 3.8 | C1's finite single-1RW exact parent capture has a clear object/constraint difference. TSBG is a useful typed-signed, private-context mapping of established bundle/weight reuse, but not a new reuse paradigm. |
| Soundness | 4.6 | Arithmetic, selection, evidence classes, negative tail, and unsupported compositions are disclosed and sealed. |
| Implementation | 4.5 | M2050 supplies 192 exact VCS runs; M2030 separately supplies matched DC on the exact same M2018/M803 source identity. |
| Evaluation | 4.4 | Cross-sequence, cross-layer, and cross-token distribution is materially stronger than M2047. It still excludes eight FC2 layers, most tokens, and nine of ten captured samples per sequence. |
| Presentation | 4.1 | Four pages are visually clean and logically scoped, but dense. The artifact README is stale at the prior four-group M2047 state. |
| **Overall** | **4.2** | **Accept; not yet dependable Strong Accept.** |

## Paper/PDF/linter/seal audit

- `main.tex` and `build/main.pdf` exactly match commit `4ea66eb2`.
- The PDF is four US-letter pages, fonts are embedded, and visual inspection found no clipping or unreadable table content.
- The build log contains no overfull box, undefined-control-sequence, LaTeX error, or fatal-error finding.
- The abstract contains 234 words under the paper's tokenizer.
- The paper manifest verifies `main.tex`, `references.bib`, `check_claim_boundaries.py`, `build/main.pdf`, and `README.md`.
- The claim linter returns `PASS_ISCAS2027_CLAIM_BOUNDARY_LINTER`.
- M2050's 192 simulation logs, result manifest, and outer seal verify. The independent M2050 hammer manifest and outer seal also verify.
- M2030's matched-DC independent hammer manifest and outer seal verify.
- `docs/359` remains at SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Independent M2050 recomputation

All 192 rows are uniquely indexed by slots 0--191. Their sums reproduce the paper:

- Ordinary LRU4 cycles: `1,381,704`.
- TSBG-B4 cycles: `551,343`.
- Weighted speedup: `1,381,704 / 551,343 = 2.5060697243x`, correctly rounded to `2.5061x`.
- Time reduction: `1 - 551,343 / 1,381,704 = 60.0968804%`, correctly rounded to `60.10%`.
- Bundle requests: `121,008 -> 41,916`.
- Scalar requests: `968,064 -> 335,328`.
- Request reduction: `65.3609679%`, correctly rounded to `65.36%`.
- Workload geomean/median/max: `1.872580x / 2.212190x / 3.195063x`.
- Nineteen all-zero workloads are retained at `27 -> 27 cycles`; none is filtered from the aggregate.

The single marginal regression is also correct and disclosed:

- Slot 7, `zurich_city_09_a`, sample 0, FC1 layer 10, middle quartet at token 96000.
- Ordinary/TSBG cycles: `595 -> 596`.
- Ratio: `0.9983221477x`.
- Exactly one nonempty workload is below `1.0x`; 27 nonempty workloads equal `1.0x`, and 145 improve.

### Selection identity

The row population is exactly `4 sequences x 16 supported layers x 3 token regions = 192 workloads`:

- Samples `0/10/20/30`, the first captured sample of `zurich_city_09_a`, `interlaken_01_a`, `thun_01_b`, and `zurich_city_12_a`, respectively.
- Twelve FC1 layers: layer IDs `8,10,12,14,16,18,20,22,24,26,28,30`, with source-group counts `6,6,12,12,24,24,24,24,24,24,48,48`.
- Four G48-supported FC2 layers: layer IDs `9,11,13,15`, with `24,24,48,48` source groups.
- Eight FC2 layers above G48, IDs `17,19,21,23,25,27,29,31`, are explicitly excluded.
- Each sample/layer pair contributes exactly one first, middle, and last aligned B4 quartet.
- The selection rule is fixed without consulting performance. Smaller layers are zero-padded onto the same physical G48 geometry.

### Real-activity, signed, and weight boundary

- The fixture contains real ep34 activity/sign descriptors from the sealed capture.
- The selected natural descriptors contain `25,045` nonzero codes and **zero negative codes**; all measured nonzero values are `+1`.
- Deterministic directed INT8 weights, not captured model weights, exercise arithmetic. Scheduling and cycle count are weight-value independent.
- Signed products, reset recovery, stale/replay rejection, and the INT8 `-128` corner are exercised in a subsequent synthetic recovery phase. This proves the path but does not make the natural measured interval bipolar.
- All 192 logs contain exactly one unique M2048 workload PASS and no fatal/assertion/timeout signature under the checked failure patterns.

## Breakdown consistency

Independent weighted recomputation matches the result and paper:

| Scope | Workloads | Ordinary cycles | TSBG cycles | Weighted speedup |
|---|---:|---:|---:|---:|
| FC1 | 144 | 1,175,236 | 448,945 | 2.617773x |
| Supported FC2 | 48 | 206,468 | 102,398 | 2.016328x |
| Total | 192 | 1,381,704 | 551,343 | 2.506070x |

Sequence weighted ratios are `2.6012x`, `2.5414x`, `2.5330x`, and `2.3152x`, matching Table IV's rounded `2.601/2.541/2.533/2.315x`. Token-role weighted ratios are `2.6316x`, `2.4577x`, and `2.4122x` for first/middle/last. This supports a distribution claim, not an every-workload claim.

## M2030 same-source physical companion

M2050 and M2030 pin the same M2018 RTL SHA `96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21` and M803 SHA `cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156`.

M2030 reports a separate matched TSMC-28, 3-ns, logic-only, pre-macro DC ablation:

- Ordinary LRU4: `249,710.451846 um2`, setup WNS `+0.0264 ns`.
- TSBG-B4: `249,739.809848 um2`, setup WNS `+0.0688 ns`.
- Added logic area: `29.358002 um2`, or `0.0117568%`.
- Both diagnostic hold WNS values are `-0.0164 ns`.
- Ideal clocks, ZeroWireload, and standard-cell state arrays are used; SRAM macro cost, hold closure, power, energy, and paper-PPA readiness remain false.

The paper says `matched logic-only DC ablation adds 0.0118% area` and does not claim same area. M2050 cycles and M2030 area may be stated side by side because their RTL identities match, but their ratios must not be multiplied into system or energy efficiency.

## C1/C2/C3 and ep34 precision wording

The paper's surrounding lines remain correctly bounded:

- **C1:** `1.6945x` and `-40.99% time` remain visibly tagged as a same-ledger cycle model. The `166,514 um2`, PT setup/hold, 16,549 compare points, and bounded C1 power window are separate mapped physical anchors. The paper explicitly says the single real-mask VCS tile calibrates event counts and directed arithmetic, not the cycle ratio.
- **C2/K8:** the denominator is equal-bandwidth `K1x8`, not a single K1 lane. The modest `1.0167x` cycle result travels with `4.5507x` directed-throughput/logic-area and `77.66%` less synthesized logic, and remains logic-only with matched hold/power open.
- **C3:** it is exact fixed-T10 service coverage, not a speedup contribution. The paper correctly says its Formality comparison is between pre- and post-hold-repair mapped netlists, not RTL-to-gate.
- **ep34 precision:** the frozen software configuration has hardware quantization disabled. Only the separately evaluated deployment candidate enables Q7 scores, integer Shiftmax normalization, Q1.7 gates, and QDQ of eight named convolution/deconvolution weights. Other operators remain at checkpoint precision. The 825-frame accuracy table is explicitly a backend-mismatched compatibility gate, not causal evidence that quantization improves accuracy and not full-network INT8.
- **ATLIF identity:** the paper distinguishes 105 installed wrappers, 12 runtime-bypassed `sn2_q`, 93 invoked, and 12 invoked but graph-dead `attn_sn`, leaving 81 graph-live services. All 93 captured outputs are binary; typed signs are a downstream protocol property rather than an analog-ATLIF claim.

No M2050 number is called full-FC, full-network, system, same-area, real-weight, or energy evidence in the abstract, Table II, body, limitations, or conclusion.

## Severity findings

### P0: 0

No arithmetic, identity, seal, selection, scope, or claim-promotion failure was found.

### P1: 1

1. The sealed paper `README.md` is stale: it still describes M2047 as the latest TSBG closure and says the result remains a four-group microbenchmark. The paper and M2050 evidence now use 192 workloads. This does not invalidate the PDF, but an artifact-open reader receives contradictory guidance from a manifest-covered file.

### P2: 5

1. Exact RTL sampling uses only the first captured sample of each sequence and three B4 quartets per supported layer, not the full 40-sample/token population.
2. Eight of twelve FC2 layers exceed G48 and are excluded; therefore `all FC1 + supported FC2` must remain the scope.
3. Natural selected descriptors are unipolar and weights are directed; signed and `-128` coverage is synthetic recovery only.
4. One nonempty workload is marginally slower at `0.998322x`; every-workload speedup remains forbidden.
5. M2030 is logic-only and hold/power-open, while C1's headline ratio still uses a separate cycle model and physical anchor. These are disclosed limitations but keep physical completeness below Strong Accept.

The M2050 raw receipt retaining `RAW_PASS_PENDING_INDEPENDENT_REVIEW` and `paper_admitted=false` is not counted as a failure because the separately sealed, same-commit independent hammer explicitly admits the scoped paper wording. Updating the artifact README should link readers to that admission layer.

## Acceptance assessment

- **Accept probability:** 85--90%.
- **Strong Accept probability:** 25--35%.
- **Current class:** solid Accept, not dependable Strong Accept.

The dominant positive is no longer just a large best-case number: M2050 shows `2.5061x` weighted cycles and `65.36%` fewer requests across layer, sequence, and token-role partitions while retaining empty workloads and the negative tail. The dominant remaining weakness is that the exact distribution is still sampled and the physical companion is logic-only with no power.

## Minimum executable improvement path

1. **Immediate, no experiment:** update `paper/iscas2027/README.md` from the four-group M2047 description to the admitted M2050 192-workload scope, rerun the linter, rebuild if needed, and reseal the paper package.
2. **Highest-return no-new-RTL extension:** reuse the M2050 parametric RTL and fixed rule on the remaining sealed ep34 samples, or at minimum a predeclared multi-sample subset, preserving all-zero workloads and the `0.9983x` tail. Report percentiles and the slow-workload count. This attacks the first-sample concern without opening a new architecture.
3. **If schedule permits:** tile the eight FC2 layers above G48 through repeated G48 passes under the same public-port/cache accounting. This would allow a full FC-layer graph claim without calling it full-token or system performance.
4. **Strong-Accept physical gate:** obtain one matched, workload-driven C2/TSBG power result and close hold under an explicitly stated memory implementation. Do not multiply it by C1 or infer whole-network energy.

Steps 2--4 are improvements, not submission blockers. Do not reopen lossy pruning, another Conv matcher, or monolithic integration before the current four-page Accept draft is secured.

## Permitted concise wording

> Across 192 fixed, performance-independent ep34 component workloads spanning four DSEC sequences, all 12 FC1 layers, four G48-supported FC2 layers, and first/middle/last aligned B4 token quartets, exact post-load VCS execution on the same parametric G48/B4/LRU4 RTL reduces cycles from 1,381,704 to 551,343 (2.5061x; 60.10% less time) and scalar weight-bank requests from 968,064 to 335,328 (65.36% fewer). Nineteen all-zero workloads and the 0.9983x worst nonempty case are retained. A separate matched 3-ns TSMC-28 logic-only DC ablation on the same RTL source adds 0.0118% area and meets setup; eight FC2 layers above G48, full token/sample coverage, hold closure, macros, power, energy, and system performance remain open.

## Forbidden promotions

- Full-FC, all-FC2, full-token, full-capture, whole-network, FPS, or system speedup.
- Every-workload improvement.
- Real-weight or empirically bipolar natural ep34 cycle workload.
- Same-area, macro-inclusive, hold-closed, power, energy, or paper-ready PPA.
- Multiplication of M2050's cycle ratio by M2030 area or by any C1/C2/C3 component ratio.
- Relabeling Prosperity, Phi, FireFly-T, ELSA, SpikeX, or CICC measurements as this work's result.
