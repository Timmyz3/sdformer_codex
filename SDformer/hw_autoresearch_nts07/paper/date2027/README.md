# ISCAS 2027 claim-safe paper skeleton

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

- C1 ep34 CPU same-ledger opportunity: `1.694510x` `[model]`, explicitly
  non-RTL.  The nine-macro mapped component is `166514.312 um2`, with PT
  setup/hold `+0.027871/+0.001827 ns`, Formality `16549/16549`, and a bounded
  mapped-energy point of `29.0763 mW` / `22.0689 nJ` per 64-row window.
- C2 fresh registered-fault-matched equal-bandwidth point: `1.016728x` cycles,
  `4.550657x` throughput/mm2, and `-77.6576%` logic area; K8/K1x8 logic areas
  are `130822.775/585534.972 um2`.  Logic-only pre-macro setup is
  `+0.0018/+0.0014 ns`; diagnostic hold remains open at
  `-0.0190/-0.0177 ns`.
- C3 fixed-T10: `63756.125879 um2`; DC setup/hold
  `+0.000300/+0.034585 ns`, PT `+0.000299/+0.030474 ns`, and gate-to-gate
  Formality `11180/11180`.
- final checkpoint selection: ep34, AEE `1.199514`, activity `72.891G`; the
  selection and 40-sample activation capture are independently sealed, while
  decoder-complete address-timed replay remains pending.
- Production Table A: zero admitted rows.

## Unclosed placeholders

1. Decoder-complete, memory-inclusive cycle/traffic row.
2. Complete C2 Formality, dual-corner timing/hold repair, and mapped power after
   the registered-fault repair; close C3 mapped energy.
3. Admit TSBG only if true-protocol VCS and same-resource DC/energy pass.
4. Memory-inclusive system energy.
5. Multi-sequence, same-resource whole-network comparison.
6. Final figures, Table A, normalized comparison table, and complete bibliography.

Prosperity and Phi are related/official-artifact comparators only.  Their results
must remain attributed to the original work and cannot be described as this
project's RTL or system speedup.
