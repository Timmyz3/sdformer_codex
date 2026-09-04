# DATE 2027 claim-safe paper skeleton

This directory contains a deliberately conservative six-page writing skeleton.
It is not a submitted manuscript and does not promote component evidence into a
system claim.

## Files

- `main.tex`: six forced pages covering motivation, architecture, C1--C3,
  evaluation, comparison methodology, and closure gates.
- `tables/evidence_registry.tex`: the single claim/boundary registry used by the
  evaluation page.

## Build

From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

or, when available:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current host did not expose `pdflatex` or `latexmk` when this skeleton was
created, so no PDF is claimed.  The source intentionally uses only common LaTeX
packages (`geometry`, `booktabs`, `graphicx`, `hyperref`, and `xcolor`).

## Evidence admitted in this draft

- C1 CPU same-ledger opportunity: `1.7591725402x` `[model]`, explicitly non-RTL.
- C1 macro-aware slice: `147246.39209 um2`, setup `+0.001795 ns`; hold and
  full-storage energy remain open.
- C2 equal-bandwidth point: `1.01672765x` cycles, `4.541077998x`
  throughput/mm2, and `-77.6104%` logic area; logic-only pre-macro.
- C3 fixed-T10 DC: `62433.503388 um2`, setup `+0.0003 ns`; a later
  prelayout PT diagnostic failed the strict gate at setup `-0.001154 ns` and
  hold `-0.022628 ns`, so no timing-closed speedup is claimed.
- final checkpoint selection: ep34, AEE `1.199514`, activity `72.891G`; the
  selection is independently sealed, while E2--E8 hardware recapture/replay
  remains pending.
- Production Table A: zero admitted rows.

## Unclosed placeholders

1. Execute and independently seal the already-authorized ep34 one-shot
   hardware-trace rebind; checkpoint selection itself is closed.
2. Decoder-complete, memory-inclusive cycle/traffic row.
3. C1 full-storage and hold closure; C2/C3 hold closure.
4. Component SAIF/PTPX and memory-inclusive energy.
5. Multi-sequence, same-resource whole-network comparison.
6. Final figures, Table A, normalized comparison table, and complete bibliography.

Prosperity and Phi are related/official-artifact comparators only.  Their results
must remain attributed to the original work and cannot be described as this
project's RTL or system speedup.
