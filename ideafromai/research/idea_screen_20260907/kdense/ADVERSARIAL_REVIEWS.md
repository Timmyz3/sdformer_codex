# Adversarial reviews (K-Dense step 7)

Reviewer P03 did not originate the ideas (role assignment). Same model as P02 — this is **not** independent-lab review. Templates follow `references/idea_evaluation.md`. Review the strongest version first.

---

## I001 — Motion-XOR 32-lane triple-popcount leaf

- **Reviewer (not originator):** P03
- **Strongest version:** Keep H67’s exact Q7 formula; one combinational 32-lane leaf; `K_peer` is the previous-timestep K shadow; compare equal-resource against FireFly-T AND-PopCount; do not claim system FPS until T0.
- **Observation that would count against it:** ep34 T0 shows score-leaf share < ~2%, **and** I003’s one-block swap holds subset AEE. Then the leaf is neither a PPA story nor a necessary algorithm term.
- **Alternative explanations:** (1) FireFly-T AND-PopCount plus a cheap extra XOR outside the cited “new ALU”; (2) QKFormer linear Q-K matches AEE so the third popcount is unused capacity; (3) editors read any popcount leaf as FireFly-T regardless of the XOR term.
- **Measurement / analysis failure:** labeling ep35 census as ep34; reporting leaf Hz as network FPS; omitting Q/K projection from “attention share”.
- **Sampling / generalizability failure:** 100-sample ep35 path; S3 vs S1 occupancy differs by 15×; DSEC indoor/driving mix may not match the user’s deployment.
- **Prior evidence that challenges it:** FireFly-T AND-PopCount (arXiv:2505.12771 / IEEE TC 2026); α-XNOR co-silence (CVPR 2025); Bishop AAC (ISCA 2025); historical attention envelope ~0.59%.
- **Harm / inequity / misuse:** standard accelerator paper; unpublished traces if leaked. No clinical harm.
- **Mitigation:** T0 first; name priors in the brief; isolate leaf benches; keep traces local.
- **Residual uncertainty:** T0 unknown; whether XOR term is accuracy-necessary (I003).
- **Disposition:** retain for measurement (M0–M2). Not a title lock.
- **Response / owner / status:** Accept T0 kill-switch. Owner P01. Open until M0.

---

## I002 — Token-level skip of that leaf

- **Reviewer:** P03
- **Strongest version:** Lossless per-token enable when K is {0,0} across T or Q/K bits equal t0; same MX3P leaf; **no** 15×15 memo.
- **Observation against:** M2 shows the Shiftmax denominator grain is ~100% dirty (ep35 preview already 90.1% window-dirty, S3 0% clean). Then skip cannot be sold as a row skip; energy-gate may still be true.
- **Alternatives:** (1) energy gating without architectural claim; (2) no skip; (3) term-wise skip of overlap only (I012).
- **Measurement failure:** token skip written as Shiftmax skip; 96% single-sample skip; 56% token skip as 56% row skip.
- **Sampling failure:** ep35 ≠ ep34; S1 is almost empty and would overstate skip if averaged without per-stage tables.
- **Prior evidence:** DeltaCNN/CBinfer (pixel skip); DLSS/CICC 2026 (feature-map skip) — both wrong grain if cited as the same mechanism.
- **Harm:** silent AEE regression if skip is approximate.
- **Mitigation:** merge into I001 as enable; freeze lossless-only unless AEE gated.
- **Residual uncertainty:** true RTL row definition.
- **Disposition:** revise — merge into I001.
- **Response / owner / status:** Merged. Owner P02. Closed as a separate title; open as M1/M2.

---

## I003 — One stage-2 block → SDSA or QKFormer linear Q-K

- **Reviewer:** P03
- **Strongest version:** Frozen-weight swap of **one** deepest stage-2 block; if AEE explodes, 10-frame finetune from ep34; three-way AEE: Motion-XOR / SDSA / linear Q-K.
- **Observation against:** subset AEE moves toward public SDformerFlow 1.58 and stays there after 10 frames. Then XOR looks necessary and the “simpler hardware” story dies — which is still information.
- **Alternatives:** keep Motion-XOR; swap all 12 blocks (forbidden as first probe); SpikePool (out of scope, breaks spike Q-K).
- **Measurement failure:** 12-block swap; quoting 1.259 on a 10-frame subset; QKFormer ImageNet numbers as OF evidence.
- **Sampling failure:** one sequence is not valid825; classification checkpoints are not DSEC.
- **Prior evidence:** QKFormer NeurIPS 2024 arXiv:2403.16552; SDformerFlow arXiv:2407.15801 SDSA; Spike-driven Transformer V2.
- **Harm:** wasted A800 if done before T0; none ethical.
- **Mitigation:** one block; subset first; stop if direction is clearly up.
- **Residual uncertainty:** finetune budget to hold 1.259 unknown.
- **Disposition:** retain as cheapest algorithm-face probe.
- **Response / owner / status:** Accept. Owner P01. Open, parallel-optional with M0.

---

## I004 — Mixed T=2/T=10 FC join

- **Reviewer:** P03
- **Strongest version:** LoAS FTP is the **baseline**; candidate is mixed-horizon packing plus existing bounded temporal shared-sum; equal Y-port, equal back-pressure; no title “LoAS”.
- **Observation against:** equal-resource timeline vs direct FTP is ≤1.00× **and** a reviewer writes “LoAS + ATLIF theta” in the first paragraph.
- **Alternatives:** direct FTP; RSR++; copy LoAS uniformly at T=10 (kills the local difference).
- **Measurement failure:** add reduction (−20.8%/−42.2%) reported as cycle speedup; missing persist-Y.
- **Sampling failure:** one FC2 capture; T=2 fibers may not be the FC bottleneck.
- **Prior evidence:** LoAS MICRO 2024 arXiv:2407.14073; RSR/RSR++; Chen/Chang mux T=4/2/1 28 nm.
- **Harm:** none beyond unpublished RTL.
- **Mitigation:** pre-declare mixed-T as the only novelty claim; M4 equal-resource.
- **Residual uncertainty:** whether mixed-T is enough of a delta for TCAS-II.
- **Disposition:** retain as contingent circuit face if T0 fails I001.
- **Response / owner / status:** Accept. Owner P01. Dormant until T0.

---

## I005 — TSBG B8 broadcast

- **Reviewer:** P03
- **Strongest version:** same-IO B8 vs ordinary LRU row buffer; cycle ≥1.15× **or** ≤5% slower with ≥30% fewer weight bytes; CPU 3.89× is not the headline.
- **Observation against:** equal-resource RTL <1.15×, **or** the owner’s prior “no novelty” judgment is reused by reviewers (Gustavson/Eyeriss/TSBG).
- **Alternatives:** LRU row buffer; do nothing; train for more row reuse (changes AEE).
- **Measurement failure:** CPU serialized 3.89× as VCS.
- **Sampling failure:** frozen ep34 traces; new sparsity would change reuse.
- **Prior evidence:** Eyeriss, Gustavson, TSBG family; owner already judged unoriginal.
- **Harm:** none.
- **Mitigation:** comparator-only; M3 gate.
- **Residual uncertainty:** RTL vs CPU gap size.
- **Disposition:** retain as performance comparator, not title default.
- **Response / owner / status:** Accept. Owner P01. Open as M3 after M0 if resources exist.

---

## I007 — ATLIF in-island + late-BN θ packet

- **Reviewer:** P03
- **Strongest version:** membrane stays in the island; θ packet at BN boundary using existing A8 RTL on **real** BN/PSN intervals.
- **Observation against:** real BN intervals show recovery error that moves AEE, **or** reviewers cite Chen/Chang mixed-T 28 nm as the same island.
- **Alternatives:** freeze θ at compile; move BN earlier; no packet (always wait).
- **Measurement failure:** synthetic testbench intervals as “online BN”.
- **Sampling failure:** one BN group ≠ all stages.
- **Prior evidence:** Gist ISCA 2018; Chen/Chang Spike-IAND-Former arXiv:2503.19643.
- **Harm:** none.
- **Mitigation:** M7 real intervals; do not claim first mixed-T ASIC.
- **Residual uncertainty:** real recovery rate.
- **Disposition:** retain as fifth / contingent neuron island.
- **Response / owner / status:** Accept. Owner P01. After M0 unless BN is on the critical path.

---

## I012 — Term-wise skip of overlap / same_zero / motion_xor

- **Reviewer:** P03
- **Strongest version:** clock-gate each of the three popcounts when that term’s bit support is empty; overlap mean 0.013 on ep35 preview suggests overlap is almost always skippable; still lossless.
- **Observation against:** software Q7 already short-circuits empty terms, so RTL “skip” is zero extra information; or the three terms share a single popcount tree and cannot be gated independently.
- **Alternatives:** only K=0 skip; only overlap skip; fuse all three.
- **Measurement failure:** citing mean 0.013 as “99% skip” without a histogram of zeros.
- **Sampling failure:** ep35 preview.
- **Prior evidence:** any sparse popcount / zero-skip ALU.
- **Harm:** none.
- **Mitigation:** histogram of exact-zero terms on ep34; only then RTL enable bits.
- **Residual uncertainty:** datapath fusion in `h67_motionxor_score_q7.sv`.
- **Disposition:** retain as a measurement hanging on I001, not a separate title.
- **Response / owner / status:** Accept. Owner P02. Fold into M1.

---

I006, I008, I009–I011, I013 were not shortlisted; they receive literature checks but not a full adversarial template. Minority request: if T0 shows decoder or conv dominance, reopen I008/I006 rather than forcing I001.
