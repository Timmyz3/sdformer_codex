# TCAS-II Express Brief draft (C1 + C2/TSBG)

This directory is a **journal retarget** of the ISCAS component paper.  It does
not replace `paper/iscas2027/`.  IEEE forbids simultaneous submission: if this
manuscript is submitted to TCAS-II, the same content must **not** be submitted
to ISCAS 2027.

Official constraints (verify before upload):

- https://ieee-cas.org/publication/TCAS-II/guidelines-author
- https://ieee-cas.org/publication/TCAS-II/tcas-ii-manuscript-submission-guide
- Portal: https://ieee.atyponrex.com/journal/TCAS2
- Strict 4.5 pages content + 0.5 page references; **last column references only**
- Binary accept/reject; only minor mods on accept
- Single-blind (names, affiliation, funding, ORCID)
- Abstract 100–250 words; at least five keywords
- Not submitted elsewhere; conference expansion only after the conference, with ≥30% new material

## What changed relative to the ISCAS draft

- Class: `IEEEtran` journal, not conference; anonymous line replaced by a
  fill-in author block.
- Contribution cut: **two exact reuse objects**.  C1 captures products under a
  finite single-1RW parent lifetime; C2 combines typed K8 with context-safe
  TSBG weight delivery.  C3 is coverage, not a speedup, and is dropped.
- TSBG 1.8345× is the ratio of sums over 2,880 fixed-region component VCS
  workloads covering all 12 FC1 and 12 FC2 layer identities.  The earlier
  G48-only 1,920-workload subset is 2.4438×, and the eight-layer FC2
  continuation is 1.7657×; neither is full-token/full-FC wall time.  C2
  4.541× is directed-throughput/logic-area against equal-bandwidth K1×8.
  None of these is a full-network or silicon result.
- A separately labeled VCS-calibrated CPU model now replays every aligned B4
  quartet in the same frozen 40-sample/four-sequence capture: 11.16M quartets,
  313.604G/150.234G modeled cycles (2.0874×), and 64.68% fewer scheduled
  scalar weight reads.  G≤48 has zero fitted residual; G96/G192 uses a median
  residual with an observed min/max sensitivity of 2.0870×.  This robustness
  result is not RTL, same-area, full-network, energy, FPS, or an abstract
  headline.
- Adding the identical 288-KiB foundry-QRT capacity to both K8 and K1×8 gives
  a separately labeled logic-plus-memory area model: 39.72% less area and
  1.687× directed-throughput/area.  It is not integrated macro P&R.
- A separate directed ordinary/post-read/pre-read ablation now isolates
  suppression before the SRAM request: 2,304/2,304/576 accepted bank reads,
  with 4,608 signed products and 24 exact commits per axis. Evidence:
  `results/m2231_m2215_causal_parse_only_successor_r1_20260905`, admitted by
  `reviews/m2232_m2231_causal_parse_only_result_hammer_r1_20260905`.
  This restores postprocessing of unchanged raw VCS logs; the original
  M2215 failure remains failed. The 75% directed reduction is not a new
  population speedup, matched energy result, or abstract headline.
- Added TCAS-II / TBioCAS citations (Taylor PIM tutorial, Qiao DS-CIM, Frenkel ODIN).
- No M-numbers, no system FPS, no multiplied component ratios.

## 9.20 body vs submit

Teacher gate is **body text by 2026-09-20**, not a guaranteed submit date.
Do not upload until: author names/ORCID/funding are filled, last-column-only
refs are verified in the PDF, and the cover letter is attached.  Graphical
abstract is a submission-portal file, not a sixth manuscript page.

## Reproduce

```bash
python3.12 check_claim_boundaries.py
/tmp/tectonic-musl/tectonic main.tex --outdir build
pdfinfo build/main.pdf
python3.12 check_submission_pdf.py --draft-underfill-ok
```

Current compile (2026-09-05, musl tectonic): **5 Letter pages**, with page 5
column 2 containing references only. The strict PDF checker passes without
the draft-underfill exception; the final body reaches 703.62 pt, and page 5
has been visually inspected. This formatting result does not close open
hardware or author-metadata requirements. The FC2 continuation is complete:
the fixed-region VCS population covers all 12 FC1 and all 12 FC2 layer
identities.  Legitimate remaining content is matched TSBG logic/SRAM energy,
post-route timing/hold, and final author/thanks/funding/ORCID metadata.  No
result may be invented merely to fill the page.  Tune `\IEEEtriggeratref{N}`
only after the matched-power/P&R campaigns and independent paper review are
complete.
