# TCAS-II full-token claim and layout audit — independent review

- Review date: 2026-09-04 (Asia/Shanghai)
- Paper source SHA256: `a0d2ef158dc8729b8343a6dbdadce8aa9222db8855e95e64c771c061b7ba2a81`
- Tracked PDF SHA256: `d319ae6861697c5f9dbb119dddb8eed77c714553a5cb61c56b0f435a235143e1`
- README SHA256: `21f39d4497bbbbc524820042eac394cb29d4df7ba58df9be1d0fb4fb4ee39cf3`
- Claim-linter SHA256: `dc8c5a26784e59474f9aff454ac1817bd12c432c1a870ebc48333ecb8a54fdcd`
- M2159 review SHA256: `8636da512a8d343552d4a0106702c6d392c4361ca525c47a9780cc33ada716b2`
- Frozen `docs/359` SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

This was a paper-only audit. I did not modify the manuscript, README, checker,
RTL, result packages, or `docs/359`; I did not run Git, EDA, VCS/simv, GPU, or
license commands. I ran the claim linter and PDF checker and inspected a local
render of all five pages. All PDF fonts are embedded.

## Verdict

**The M2159 insertion improves the paper and preserves the intended claim
boundary. Current score: 3.9/5, Weak Accept / Accept-leaning.** The new result
directly answers the largest selection concern in the 2,880-row VCS population:
the three fixed B4 locations did not manufacture the aggregate opportunity.
It does not, however, close the circuits result that TCAS-II still needs most:
matched hold-clean physical implementation and logic-plus-SRAM energy for
ordinary versus TSBG.

The two-contribution brief remains coherent. C1 is finite-lifetime, single-1RW
product capture; C2/TSBG is pre-read weight-delivery reuse with private signed
products and Acc24 contexts. No third mechanism should be added.

## Exact claim-boundary audit

### PASS: direct VCS headline remains distinct from the calibrated model

1. The abstract reports `1.8345x` only as the ratio of sums over **2,880
   same-port/cache component VCS workloads**, explicitly covering all 12 FC1
   and 12 FC2 identities. It also calls the result component-level and denies
   whole-network inference. This is the correct headline boundary.
2. `2.0874x` is absent from the abstract. It appears only in the Evaluation
   paragraph and Table III, whose caption says `[VCS-calibrated CPU model]`.
   The paragraph labels modeled cycles and ends with “not RTL or whole-network
   execution.” No C1, K8, TSBG, or full-population ratios are multiplied.
3. The model population is stated: 11.16 million aligned B4 quartets from the
   same frozen 40-sample, four-sequence capture, covering 12 FC1 and 12 FC2
   layers. The selection is declared independent of performance.
4. The distribution is not hidden: p10/p50/p90 are
   `1.0000/1.5414/2.4388x`; 3.10% are marginally slower; the worst case is
   `0.99755x`; FC1/FC2 and all four sequence ranges are reported.
5. The recurrence boundary is accurate. For G<=48, all 3,840 ordinary/TSBG VCS
   cycle fields match with zero residual. G96/G192 is only 6.981% of the full
   population and uses a median residual from 960 keyed VCS anchors. The
   min/max experiment is correctly called an **observed-envelope sensitivity,
   not a formal bound**.
6. Real ep34 activity/sign descriptors and directed timing weights are
   separated. The text does not promote the model to real-weight arithmetic,
   same-area RTL, energy, FPS, or system speedup.

The README is consistent with those boundaries: it calls `2.0874x` a
VCS-calibrated CPU model, denies RTL/same-area/full-network/energy/FPS and an
abstract headline, and keeps `1.8345x` tied to fixed-region VCS.

### Wording hardening before upload

These are not evidence failures, but they are cheap changes that prevent an
aggressive reviewer from joining adjacent claims:

1. In the full-population paragraph, change “not RTL or whole-network
   execution” to **“not RTL, same-area, or whole-network execution,”** and call
   `2.0874x` explicitly the **ratio of sums**.
2. Disclose that **779,040 of 780,000 high-group quartets** use the median
   residual (99.877% within G96/G192; 6.981% of the complete population). The
   present wording says “unseen descriptors” but leaves the exact extrapolated
   count implicit.
3. Change “G96/G192 identities execute as exact 48-source continuations” to
   “execute as **functionally exact** 48-source continuations.” This prevents
   “exact” from being read as an exact cycle-model claim.
4. In Scope and Limitations, qualify “seven slightly slower cases” as the
   **2,880-workload direct-VCS population**. The full-population model has a
   separate 3.10% slower-case rate; both are correct but currently appear
   farther apart than ideal.
5. Strengthen the linter so `2.0874` is allowed only inside Evaluation/Table
   III, requires the local strings `ratio of sums`, `same-area`, `3.10%`, and
   the high-group extrapolation count, rather than merely checking that these
   facts occur somewhere in the manuscript.

## PDF and TCAS-II format audit

The tracked PDF is Letter and exactly five pages. Page 5 right column begins
with REFERENCES, contains no body heading, and page 5 left contains no
reference entry. Thus the **reference-column topology passes**. Visual review
found no clipping, overlap, or margin escape. Figures 1 and 2 are readable at
normal zoom, although Table IV remains dense.

The strict content-fill gate still fails:

- `page5_left_content_ymax_pt = 519.660253`
- conservative required line in the checker: `650 pt`
- status: `FAIL_TCASII_SUBMISSION_PDF`

Therefore this is a five-page draft, not yet a compliant 4.5-content +
0.5-reference upload. The missing matched P&R/power material is the right way
to fill the column. Do not add defensive provenance prose merely to consume
space. Author names, affiliation details, e-mail, ORCID, and funding are still
placeholders and remain an upload blocker.

One visible presentation bug should be fixed immediately: Fig. 2 no longer
contains an evidence ladder, but its caption still says “The lower ladder
separates model, VCS, and physical evidence.” The stale sentence is plainly
false in the rendered figure.

## Does the new table improve TCAS-II fit?

**Yes, materially but modestly: approximately +0.1 overall and +0.2--0.3 in
Evaluation.** Table III turns the earlier three-position sample into a
full-aligned-B4 robustness study over the frozen capture and exposes FC1/FC2,
sequence, percentile, slower-case, and worst-case behavior. This is exactly
the kind of cycle-model-plus-RTL-anchor evaluation used by architecture-aware
circuits papers.

It is still a CPU-model table, not a physical result. Its four rows also repeat
the surrounding paragraph and contain several dash cells. Once matched P&R and
energy arrive, the strongest final layout is either (a) retain Table III as a
small robustness table and add one paired physical-energy table, or (b) fold
its aggregate/range/tail fields into the main evidence table and use the saved
space for a per-layer or per-sequence distribution plot. Do not promote
`2.0874x` into the abstract unless a new direct RTL campaign truly covers that
population.

## Prioritized remaining work

### P0 for a strong TCAS-II submission

1. **Matched ordinary/TSBG P&R:** identical source, capacity, floorplan, PVT,
   CTS/route, and hold-repair policy; setup and hold WNS >= 0; DRC/connectivity
   and equivalence closed. Report routed area overhead and frequency honestly.
2. **Matched logic plus common-SRAM energy:** use the same final identity and
   pre-registered low/median/high real-activity windows. Separate logic
   internal/switching/leakage from the identical 288-KiB SRAM leakage and
   read-dependent dynamic energy. Request reduction is not itself an energy
   result.
3. **Fill the 4.5-page body with those results:** replace open-result caveats,
   not with a third mechanism. Rebuild and run the PDF checker without
   `--draft-underfill-ok`.
4. **Finalize author metadata and cover letter.** The cover letter must not
   claim 4.5+0.5 compliance until the strict checker passes.

### P1 strengthening

1. Apply the five wording/linter hardenings above and remove the stale Fig. 2
   caption sentence.
2. If affordable, extend C1's same-ledger replay beyond one DSEC sequence. This
   remains the largest evaluation asymmetry after the full-token TSBG result.
3. Replace the 105-macro C1 storage model versus nine-macro mapped-island prose
   with one compact storage-breakdown graphic or table if space permits.
4. Deepen the K8/K1x8 service distribution only after physical and energy P0s;
   five directed loads are enough for compatibility, not broad workload
   generalization.

## Independent score

| Dimension | Current | After matched P&R/power and final format | Reason |
|---|---:|---:|---|
| Novelty | 3.6/5 | 3.7/5 | Clear object differences; full-token data does not add a new circuit |
| Circuits fit | 4.3/5 | 4.6/5 | 1RW storage, pre-read suppression, signed private contexts are well aligned |
| Soundness | 4.6/5 | 4.7/5 | Excellent boundary discipline; high-group extrapolation remains non-formal |
| Implementation | 3.6/5 | 4.4/5 | C1 is anchored; C2/TSBG still lacks hold-clean routed/power closure |
| Evaluation | 4.0/5 | 4.5/5 | Full-token model removes position bias; physical energy remains missing |
| Presentation | 3.4/5 | 4.3/5 | Readable five pages, but underfilled, placeholder authors, stale caption |
| **Overall** | **3.9/5** | **4.3/5** | **Weak Accept / Accept-leaning now; credible Accept to Strong-Accept tendency after closure** |

These probabilities are subjective reviewer tendencies, not acceptance
guarantees: approximately 60--75% in the present evidence state and 80--90%
after matched P&R/power, exact boundary hardening, and strict page closure.

## Final decision

**Keep the new full-population table and keep `1.8345x` as the abstract VCS
headline. Keep `2.0874x` exclusively as a `[VCS-calibrated CPU model]`
Evaluation result.** The paper now has enough mechanism and workload evidence;
the next score-changing work is physical and energy closure, not another idea.
