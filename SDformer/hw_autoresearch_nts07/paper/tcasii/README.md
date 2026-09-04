# TCAS-II Express Brief draft (C1-only)

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
- Contribution cut: **C1 only**.  C1 is the only island with area, setup/hold,
  Formality, and a bounded energy window.  C2/TSBG hold and power remain open
  and are therefore not claimed.  C3 is coverage, not a speedup, and is dropped.
- TSBG 2.4438× / C2 4.5411× stay out of the abstract and the admitted table.
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
```

Current compile (2026-09-04, musl tectonic): **4 Letter pages**, last page is
discussion/limitations | references.  TCAS-II wants **exactly 5 pages** with
the last column references only.  Do not pad with C2/TSBG claims.  Legitimate
fill before 9.20: real author/thanks/funding/ORCID block, a slightly larger
overview figure, and one more column of 28-nm flow detail (library, corner,
Formality mapped-to-mapped).  Then tune `\IEEEtriggeratref{N}` so page 5
column 2 is references and nothing else.
