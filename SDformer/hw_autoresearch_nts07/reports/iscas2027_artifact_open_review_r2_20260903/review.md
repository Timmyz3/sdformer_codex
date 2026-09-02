# ISCAS 2027 artifact-open independent review r2

Date: 2026-09-03 (Asia/Shanghai)

## Verdict

The four-page submission is ready for an ISCAS component-paper submission, but
is not a Strong Accept claim.  Overall score: **3.9/5, Weak Accept**, with an
estimated **65--75% acceptance tendency**.  This independent review supersedes
the older 4.1/5 self-assessment in
`reports/iscas2027_hardware_review_r1_20260902/`.

| Dimension | Score / 5 | Finding |
|---|---:|---|
| Novelty | 3.7 | C1 has a clear finite-capacity/single-1RW object difference; C2/TSBG is a narrower typed-signed specialization of prior reuse mechanisms. |
| Soundness | 4.4 | Claim boundaries are conservative and the critical artifacts and seals re-open successfully. |
| Implementation | 4.1 | C1/C2/C3 have RTL, VCS, synthesis, STA, or Formality anchors, but there is no measured monolithic top. |
| Evaluation | 3.7 | Real traces, multiple sequences, a task-quality gate, and fair baselines are present; the headline performance remains model or directed-component evidence. |
| Presentation | 3.9 | The four-page structure is readable and internal milestone IDs are absent; the final page remains visually light. |

## Priority ledger

- **P0: 0.** The 105/93/81 split, baseline/candidate quantization identity,
  backend-mismatched accuracy boundary, C1 model-versus-physical split, C3
  mapped-to-mapped Formality scope, and TSBG model-versus-RTL split are explicit.
- **P1: 2 at review time.** No single workload closes model, RTL, and physical
  evidence on one implementation; the paper directory had not yet received a
  Git commit identity.
- **P2: 3.** The 81 count is graph-live rather than deletion-equivalent; C2/TSBG
  hold/macro/power remain open; the fourth page is underfilled.

## Re-opened evidence

The reviewer re-opened and rechecked the inner/outer seals for the ep34 capture,
C1 mapped component, C3 physical/equivalence result, matched TSBG synthesis,
and deployment-accuracy result.  The final paper claim linter and `SHA256SUMS`
also passed, and the PDF remained four US-letter pages with embedded fonts.

## Text refinements applied after review

1. `81 functionally live` was narrowed to graph-live under the fixed call graph.
2. The C2 efficiency metric now names directed-VCS throughput and DC logic-cell area.
3. The accuracy table labels its historical backend-mismatched baseline.
4. AEE, AAE, AAE-Bench., and DSEC-Fl are defined, and AEE is named as the preselected gate metric.
5. The evaluation contract states prelayout/ideal-clock/ZeroWireload scope,
   C1 macro inclusion, C2/C3 logic-only scope, and the lack of monolithic integration.

No new performance claim was introduced by these edits.
