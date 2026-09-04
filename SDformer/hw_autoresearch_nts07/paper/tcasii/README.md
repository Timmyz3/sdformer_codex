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
- Adding the identical 288-KiB foundry-QRT capacity to both K8 and K1×8 gives
  a separately labeled logic-plus-memory area model: 39.72% less area and
  1.687× directed-throughput/area.  It is not integrated macro P&R.
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

Current compile (2026-09-04, musl tectonic): **5 Letter pages**, with page 5
column 2 containing references only.  It remains an intermediate working
draft rather than upload-ready because page 5 column 1 is underfilled relative
to the strict 4.5-page content target.  The FC2 continuation is now complete:
the fixed-region VCS population covers all 12 FC1 and all 12 FC2 layer
identities.  Legitimate remaining content is matched TSBG logic/SRAM energy,
post-route timing/hold, and final author/thanks/funding/ORCID metadata.  No
result may be invented merely to fill the page.  Tune `\IEEEtriggeratref{N}`
only after the matched-power/P&R campaigns and independent paper review are
complete.
