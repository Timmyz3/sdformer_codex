# ISCAS 2027 component-paper draft

This directory is intentionally separate from `paper/date2027/`.  The working
scope is a four-technical-page component paper (the last published ISCAS author
instructions are used only as a planning constraint; the ISCAS 2027 page limit
must be rechecked when its author kit is released).

Frozen paper identity:

- workload/checkpoint: Motion C12 ep34, SHA prefix `4bbaf7fc`;
- task evidence: DSEC valid825 AEE 1.199514, global firing rate 5.6709%;
- wording: binary-event optical-flow workload with a typed-signed downstream
  execution protocol;
- algorithm role: Motion C12/H60 supplies the frozen binary-event workload,
  Motion-XOR arithmetic contract and task-quality anchor; its frozen config has
  hardware quantization disabled, while the separately labelled M2045 deployment
  candidate enables the Q7/Q1.7 hardware-order Shiftmax path; it is not presented
  as a separate algorithm contribution in this hardware paper;
- ATLIF accounting: 105 installed = 12 runtime-bypassed `sn2_q` + 93 invoked;
  among the 93 invoked, a separate graph audit finds 12 `attn_sn` return values
  without consumers, leaving 81 graph-live services under the fixed normal
  inference call graph;
- contribution structure: C1; C2 with TSBG embedded; C3 as exact-service
  completeness, not an independent speedup claim.

The paper uses an ``execution islands'' formulation because C1, C2, and C3
share a typed-source contract but have not been measured as a monolithic
integrated top.  The unifying co-design link is a three-invariant mapping:
resident repeated products to C1, shared FC weight-row identity with private
signed contexts to C2+TSBG, and deterministic Fixed-T10 state order to C3.

Four-page narrative budget:

- introduction and frozen workload contract: about 0.6 page;
- C1 capacity/port-constrained product capture: about 0.8 page;
- C2 typed K8 plus TSBG weight delivery: about 0.9 page;
- C3 exact Fixed-T10 service: at most 0.2 page;
- evaluation, fair baselines and limitations: about 1.1 pages;
- related work and conclusion: about 0.4 page.

The 2026-09-03 revision was compiled with Tectonic and `IEEEtran`.
It produces four letter-size pages including eight references, with no
overfull boxes; NewTX/TeX Gyre Termes fonts are embedded.  The abstract is kept
below 250 words by the claim linter.  The deployment-accuracy row is referenced
to a historical baseline with different GPU backend flags, so it is presented
only as a compatibility gate.  A fresh artifact-open review reopened the PDF,
claim linter, paper seal, and five key evidence families and scored the paper
 4.2/5 (Accept, estimated 85--90% acceptance tendency), with P0=0.  The
earlier 3.9/5 and 4.1/5 assessments are superseded.  The review initially
identified this README's stale four-group scope as P1; this revision closes
that documentation issue.  M2050 expands the exact same-parametric-RTL TSBG
closure to 192 performance-independent ep34 workloads spanning four DSEC
sequences, all 12 FC1 layers, four G48-supported FC2 layers, and fixed
first/middle/last B4 token quartets.  Post-load VCS execute cycles fall from
1,381,704 to 551,343 (2.5061x; 60.10% less time), and scalar weight-bank
requests fall from 968,064 to 335,328 (65.36% fewer).  Nineteen empty workloads
and the 0.9983x worst nonempty case remain in the aggregate.  The M2018/M803
source SHA matches the M2030 schedule-mode DC ablation, where TSBG adds 0.0118%
logic area and both axes meet setup.  This is a sampled component distribution,
not full-FC or system speedup; eight FC2 layers above G48, real weights, hold
closure, macros, power, and energy remain open and are disclosed in the paper.
Compress prose rather than restore weak mechanisms if later edits approach the
page limit.

Reproduce the layout with any IEEE-compatible TeX installation, for example:

```bash
cd hw_autoresearch_nts07/paper/iscas2027
tectonic main.tex --outdir build
pdfinfo build/main.pdf | rg Pages
```

Do not allocate a contribution bullet or a result-table row to RQTB, lossy S2,
empty-tile skipping, decoder matchers, or rejected scheduling variants.  They
may be omitted completely; a four-page paper does not need a catalog of every
screened idea.

Do not copy the DATE draft's empty full-system table or pending claims into this
paper.  Every result in the abstract must already have a model, VCS, DC/PT, or
Formality label in the body.

Before editing headline numbers, run:

```bash
python3 hw_autoresearch_nts07/paper/iscas2027/check_claim_boundaries.py
```

The linter rejects TSBG CPU-model numbers in the abstract/main hardware table,
requires the matched logic-only/hold-open boundary for its DC support result,
requires the equal-bandwidth C2 denominator, and keeps the ep34 identity and
no-system-speedup boundary explicit.
