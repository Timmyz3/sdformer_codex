# TCAS-II Express Brief circuits-fit independent review r1

- Review date: 2026-09-04 (Asia/Shanghai)
- Object: `hw_autoresearch_nts07/paper/tcasii/main.tex` and its cited C1/C2/TSBG evidence
- Manuscript SHA256: `1095d47aaa7395c597a71b989adf4e7c964f9bd23738b56b74ab6d9608811a13`
- PDF SHA256: `406aba5b60b388db1e61455aec461e38e2c19b97b1e8e82171e04bac5bc753c3`
- Frozen `docs/359` SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
- Mode: independent read-only review. No manuscript/RTL/result edit, EDA/VCS/GPU run, Git mutation, or `docs/359` modification was performed. One CPU-only opportunity screen decoded the already sealed ep34 FC capture; it is explicitly non-admitted diagnostic evidence.

## Technical verdict

**C1 + C2/TSBG is a coherent TCAS-II Express Brief and does not need a third contribution. Current judgment: 3.85/5, Weak Accept / Accept-leaning, roughly 60--75% reviewer acceptance tendency.** The circuits novelty is not product sparsity or broadcast by itself. It is the exact execution of those reuse opportunities before the saved physical resource, under finite lifetime, a real 1RW contract, response identity, and private accumulator ownership.

The submission is not yet Strong Accept. The principal gap is that C2/TSBG claims to suppress SRAM requests but still lacks a matched hold-clean routed implementation and logic-plus-SRAM energy. C1 is substantially better anchored, yet its nine-SRAM mapped parent scratch is separate from the 214,912-B/105-macro complete storage ledger and its 1.6945x result remains a cycle model calibrated by one real 64-row VCS tile. These are disclosed rather than hidden, so they are limitations rather than invalid claims.

If matched C2 P&R, matched power, the C1 storage decomposition, and strict submission formatting close without weakening the present numbers, I would raise the paper to **4.2--4.35/5, credible Accept / Strong-Accept tendency, roughly 78--88%**. An FPGA prototype or tapeout is not a venue requirement. The official guide instead requires a significant circuits-and-systems advance, a nearly publication-ready first submission, and strict 4.5 content + 0.5 references; it also warns that circuits papers without a demonstrated advantage or practical impact can be returned without full review ([TCAS-II submission guide](https://ieee-cas.org/publication/TCAS-II/tcas-ii-manuscript-submission-guide), [TCAS-II scope](https://ieee-cas.org/publication/TCAS-II)). Synthesis-only and FPGA-based accelerator briefs both appear in TCAS-II: the WS-LOS SNN accelerator reports 65-nm synthesis and application energy, while the sparse-CNN scheduling brief uses a Virtex-7 implementation ([WS-LOS SNN accelerator](https://ieeexplore.ieee.org/document/10143982/), [sparse-CNN scheduling brief](https://ieeexplore.ieee.org/document/10290941/)).

## Scores

| Dimension | Current | After P0 | Reviewer reasoning |
|---|---:|---:|---|
| Novelty | 3.60 | 3.70 | Direct priors own subset-product reuse and broadcast; finite 1RW execution and typed private completion are defensible object differences |
| Circuits fit | 4.40 | 4.65 | SRAM ports, liveness, arbitration, response identity, Acc24 sharing, and pre-read suppression are strongly circuit-facing |
| Soundness | 4.60 | 4.70 | Claim classes and negative tails are unusually disciplined; extrapolated G96/G192 residual remains non-formal |
| Implementation | 3.65 | 4.40 | C1 has VCS/DC/PT/FM/PX anchors; C2 is setup-clean but hold, P&R, and power remain open |
| Evaluation | 4.05 | 4.45 | 2,880 direct workloads plus 11.16M full-token model are strong; C1 is one sequence and C2 energy is missing |
| Presentation | 3.35 | 4.25 | Narrative is coherent but defensive and underfilled; author metadata is still placeholder |
| **Overall** | **3.85** | **4.2--4.35** | **No third mechanism is needed; evidence closure changes the decision** |

## Why the two contributions are circuit-novel enough

### C1: novelty survives only as constrained single-port execution

Prosperity already owns subset/prefix parent selection, residual issue, and dependency reconstruction. C1 may not reclaim those ideas. Its valid circuit claim is the mechanism required when a parent is only temporarily resident and one 1RW macro service must arbitrate parent read, residual progress, and write/retirement:

- grant-time liveness recheck prevents a directory-time hit from becoming a stale parent;
- deadline-aware arbitration, a reserved response queue, and forwarding prevent single-port service from orphaning an accepted response;
- dead-write suppression and atomic completion preserve exact external state.

The baseline ladder supports this distinction: strongest-zero requires 648.741M modeled cycles, same-coordinate bit skipping 646.619M, and finite-1RW C1 382.849M, or 1.6945x. The 1.902x concurrent-access result is correctly presented as an unphysicalized ceiling. The remaining 12.25% is therefore a measured port/lifetime tax, not hidden ideal bandwidth.

The vulnerability is physical scope. Nine 128x128 SRAM leaves implement an 18,432-B, 96-lane parent-product scratch. The full ledger is 214,912 B: 18,432 B parent, 24,448 B metadata/reserve, 122,880 B psum, and 49,152 B weight, conservatively 105 macros and 0.988049 mm2 by the existing model. The paper discloses the separation, but Table IV should make it visible instead of forcing the reviewer to reconstruct it from limitations.

### C2/TSBG: novelty is request-before-read plus private ownership

K8 against the fair K1x8 equal-bandwidth baseline is only 1.0167x faster (1,913 versus 1,945 directed cycles). Its valid headline is 77.61% less logic and 4.541x directed throughput per logic area from sharing Acc24, endpoint, and control. Both numbers must remain together.

Broadcast, group-major scheduling, Gustavson delivery, and weight reuse are established by FireFly-T, ELSA, SpikeX, and earlier dataflows. TSBG's defensible specialization is narrower: the common row identity is resolved before `ST_FETCH_REQ`; only the delivered weight row is shared; sign, destination, tag, terminal, product, and Acc24 state remain private. Current M2018 RTL genuinely implements that boundary: a cache hit goes from `ST_FIND` to `ST_BRIDGE`, whereas only a miss enters `ST_FETCH_REQ`.

The current 2,880-workload VCS result is strong component evidence: 92.652M versus 50.505M cycles (1.8345x) and 58.13% fewer scalar bank requests with 0.0118% matched logic-area overhead. The 11.16M-quartet CPU replay removes the fixed-position concern, but 2.0874x remains a VCS-calibrated ratio of sums, not RTL, same-area, or whole-network execution. The 779,040/780,000 high-group median-residual extrapolation is properly disclosed.

## P0: required before a strong TCAS-II submission

1. **Matched ordinary/TSBG physical closure.** Use identical source, cache capacity, floorplan, clock/IO, PVT, CTS/route, and hold-repair policy. Require setup and hold WNS >= 0, clean connectivity/DRC, and post-physical equivalence for both axes. Report routed cell area and frequency; do not merge a macro-free routed island with the 288-KiB capacity model.
2. **Matched logic plus weight-store energy.** Use the same final identity and pre-register low/median/high request-density windows from different captured samples. SAIF scope must be DUT-only with no X activity. Report internal, switching, leakage, duration, and energy for each logic axis. Add the identical SRAM capacity/leakage to both axes and apply the same foundry read-energy model only to actual bank activations. One favorable slot is insufficient for a memory-saving headline.
3. **Make the C1 physical scope unmissable.** Add one compact row to the main table: `parent scratch: 18,432 B, 9 macros, integrated`; `complete ledger: 214,912 B, 105 macros, 0.988 mm2 [area model], not integrated`. The abstract may keep the nine-SRAM island result, but must never imply full-ledger PPA.
4. **Strict paper closure.** Current PDF is Letter and five pages; page-5 right is references-only, but the strict checker still fails because page-5 left content ends at 543.57 pt versus the conservative 650-pt gate. Fill the column with matched P&R/power, not provenance filler. Replace placeholder authors, e-mail, ORCID, and funding before upload.

These four items are sufficient. Full-network FPS, FPGA deployment, decoder integration, and a third sparse mechanism are not P0 for this component brief.

## P1: useful but not required to submit

1. Extend C1's same-ledger replay from `zurich_city_09_a` to the other three captured DSEC sequences. Keep the same strongest-zero/bit/C1 ladder and report ratio of sums, minimum sequence, and spread.
2. Replace the G48 `1,917+3` composite lineage with one clean compile/one simv batch if this is cheap. The existing evidence is sealed and usable, but the long lineage explanation consumes scarce brief space.
3. Give K8/K1x8 a small trace-derived service distribution or narrow `4.541x` everywhere to the five directed equal-service loads. Do not present it as a universal workload speedup.
4. Compress Related Work and the direct-prior table into one compact comparison. The paper already cites relevant TCAS-II precedents: WS-LOS reports throughput/energy for an SNN accelerator, and the delta-weight-sharing accelerator reports both memory-access reduction and energy efficiency ([WS-LOS](https://ieeexplore.ieee.org/document/10143982/), [delta-weight-sharing accelerator](https://ieeexplore.ieee.org/document/10466232/)). This reinforces that TSBG power, not another idea, is the missing metric.

## Conditional mechanism enhancement: one C2 option only

### B4-union selective bank fill -- GO for a CPU/VCS quick gate, not yet for the manuscript

This is the only new circuit change I would consider. It is an internal deepening of C2/TSBG, not a third contribution.

M2018 currently hardwires `core_req_bank_valid=8'hff` and `core_req_source_count=8` on every cache miss, although the inherited M803 adapter already accepts an arbitrary nonzero eight-bank mask and checks its popcount. All four B4 context masks are loaded before `ST_FIND`. The engine can therefore OR their active lanes for the selected source group before `ST_FETCH_REQ`, issue only the banks needed by at least one private context, and retain a per-bank valid mask with the cache row. Products and Acc24 contexts remain private, and a missing bank is fetched exactly before use.

A read-only diagnostic over the sealed 40-sample capture gives a large opportunity signal. Under the current TSBG miss schedule, full-row fill accounts for exactly 67,992,387,648 modeled scalar bank reads. Replacing each miss's 16-bank two-half fill by the B4 union popcount times six output slices gives 17,316,452,106 reads, a 74.53% reduction relative to current TSBG (FC1 70.08%, FC2 82.46%). The calculation decodes all 11.16M aligned quartets and exactly reproduces the current 67.992G full-row operand before substitution.

This is **not admitted performance evidence**. It assumes bank-granular fills under the current miss schedule and does not yet price partial-valid cache metadata, additional fills, response timing, logic power, or the fair ordinary scheduler with the same bank mask. It cannot enter the abstract or results table.

Use these fail-closed gates before any new RTL:

- build an exact ordinary-versus-TSBG CPU recurrence in which both modes receive the same union-mask capability; require at least 30% additional TSBG scalar-read reduction and no worse than 2% component-cycle regression relative to current TSBG;
- demonstrate that TSBG still gives at least 1.5x ratio-of-sums against the mask-aware ordinary baseline; otherwise the apparent gain was a weak dense-row baseline;
- if the model passes, run directed VCS with positive/negative sources, partial masks, cache hit/miss/refill, out-of-order bank responses, backpressure, and exact Acc24 comparison;
- admit to the paper only if matched routed area overhead is <=2%, setup/hold remain closed at 3 ns, and combined logic-plus-SRAM energy improves by >=15% beyond current TSBG.

If any gate fails, retain the existing TSBG unchanged. Do not rename this as a separate novelty; describe it, if admitted, as bank-selective realization of the existing pre-read admission rule.

### Other embedded candidates: NO-GO for this submission

- **Dynamic ordinary/TSBG selector:** only 3.10% of full-token quartets are marginally slower and the worst is 0.99755x. A runtime cost selector is not worth new control unless it improves aggregate modeled cycles by >=1% at <=0.5% area, which is unlikely.
- **Two-bank C1 lifetime coloring:** it could attack the 12.25% port gap, but foundry macro granularity and a new exact scheduler would invalidate the nearly closed C1 chain. Revisit only after submission, and only if an equal-bit two-bank model reduces C1 cycles by >=8% with <=10% macro-area overhead.
- **Extra hit clock gating:** cache-hit request suppression already exists. Add row-register/address/local-clock enables only if the matched PTPX breakdown shows request count falling without commensurate memory-cone energy. It is a repair, not a contribution.

## Ideas that must be prohibited

1. S2 lossy block pruning, epsilon-RQTB, attention score pruning, ATLIF rank/TDA, decoder line buffers, or a new Conv matcher as a third contribution. Each opens a new exactness/baseline/PPA debt and destroys the five-page focus.
2. Calling product sparsity, broadcast, Gustavson ordering, group-major scheduling, or weight reuse itself novel. Cite the prior and claim only the constrained circuit specialization.
3. Comparing K8 only with one K1, calling 4.541x a cycle speedup, or omitting the 1.0167x cycle result.
4. Multiplying C1, K8, TSBG, or full-token model ratios; calling any of them whole-network, FPS, silicon, or frame energy.
5. Using the 105-macro C1 model as integrated PPA, the 288-KiB C2 memory as saved capacity, or bank-request reduction as measured energy.
6. Porting to FPGA solely to satisfy an imagined TCAS-II requirement. It would introduce a new technology baseline without closing the ASIC energy question.

## Final recommendation

**Freeze the contribution count at two. Finish the already-open C2 matched P&R and power chains, make the C1 nine-macro/full-ledger split visible, and close the 4.5+0.5 page topology.** In parallel, the B4-union selective-bank idea deserves one CPU quick gate because it targets the exact resource TSBG claims to save and the existing adapter already supports masked banks. It should alter the manuscript only after a fair mask-aware ordinary baseline, VCS exactness, and routed energy pass; otherwise it stays a diagnostic and the present architecture is submitted unchanged.
