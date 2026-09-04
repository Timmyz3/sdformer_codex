# DSEC Full-Resolution Code, Protocol, and RTL Audit

Date: 2026-07-30

## Executive conclusion

The active queue is suitable for an internal NB0 versus H67 Motion versus H66d
Local-5 comparison under the same full-resolution validation procedure. It is
not an exact reproduction of the published SDformerFlow training budget, and
the resulting local valid825 numbers are not interchangeable with the official
DSEC test-server numbers.

The queued deploy evaluations are exact only for the selected attention-core
numeric path. They do not yet establish full-network fixed-point accuracy or a
window15/T450 SystemVerilog sign-off.

## Paper-protocol compliance

| Item | Published/released protocol | Active queue | Status |
|---|---|---|---|
| Full resolution | 480x640 | 480x640 | aligned |
| Crop/fullres window | 2x9x9 -> 2x15x15 | 2x9x9 source -> 2x15x15 FT | aligned |
| Fullres fine-tune | 30 epochs | 30 epochs | aligned |
| Fullres physical batch | 1 or 2 | 2 | aligned |
| Test BN | running states disabled | `bn_policy=no_running`, 78 BN modules | aligned |
| Evaluation batch | released evaluator forces 1 | explicit `test.eval_batch_size=1` | aligned |
| Crop training budget | paper: 80 epochs | NB0/H67/H66d: 60/20/30 | not aligned |
| Optimizer | paper: AdamW, lr 1e-3, wd 1e-2, half every 10 | NB0 released-code lineage; H67/H66d custom low-LR FT | not aligned |
| Evaluation set | official DSEC test server for paper table | local validation split, 825 samples | different |

The paper text and released training YAML are themselves inconsistent: the
paper states 80 crop epochs and initial lr 1e-3, while the released YAML uses 60
epochs and lr 1e-4. The executable released configuration should be cited as the
baseline implementation, while this discrepancy remains disclosed.

## Metric contract

- `AEE` and percentage errors use the repository's historical implementation.
- `AAE` is the historical 2-D direction angle. Keep it only for comparison with
  old local runs.
- `AAE_Benchmark` implements the DSEC/Barron angle between `(u,v,1)` vectors.
  This is the local metric to compare with the definition used by the DSEC
  server, but local validation and official test numbers are still different
  datasets.
- The published SDformerFlow-v2 DSEC test result (AEE 1.602, AAE 4.871) must not
  be presented as a directly comparable baseline for local valid825.
- Existing NB0 fullres ep29 profile: AEE 1.44535, legacy AAE 6.51280,
  benchmark AAE 6.18034, 825 samples. It already ran at eval batch 1 because the
  evaluator forced that value; the new replay only adds explicit provenance.

## Checkpoint and remap audit

Strengthened audit output:
`neuron_autoresearch/experiments/dsec_fullres_paper_w15/load_chain_audit_v2.json`.

| Model | ATLIF | Shiftmax | overlay keys | missing/unexpected | remapped positional tensors | unequal |
|---|---:|---:|---:|---:|---:|---:|
| NB0 | 0 | 0 | 0 | 0/0 | 12 | 0 |
| H67 | 105 | 12 | 210 | 0/0 | 12 | 0 |
| H66d | 105 | 12 | 210 | 0/0 | 12 | 0 |

Each remapped tensor was independently interpolated from the window9 source and
compared element-wise with the loaded window15 model. The audit therefore rules
out missing overlay installation, dropped overlay weights, and an unapplied
`remap=v1` state dict for these three starts.

The generic upstream local `load_model(..., remap="v1")` branch also had an
unapplied-state-dict bug. It now calls `load_state_dict` after interpolation and
has a regression test. Current H9 training already used its own audited loader,
so this fix protects baseline and future non-H9 entrypoints rather than changing
the active H67 process.

## Queue robustness and artifact provenance

- Formal evaluation now serializes `eval_batch_size`, checkpoint-load counts,
  config SHA-256, checkpoint path/size/mtime, and the deployment scope.
- Ranking tables show legacy and benchmark AAE separately.
- A rerun can resume from the newest checkpoint that has a matching optimizer,
  scheduler, and scaler state instead of restarting from the crop source.
- Existing checkpoints and profiles are preserved. The NB0 provenance replay is
  written under `paper_valid825_b1/`, separate from historical
  `standard_valid825/`.
- The queue still serializes GPU work. H67/H66d evaluation and deploy inference
  will not overlap training.
- Standard and deploy evaluators now reject a provenance-bearing profile whose
  config hash or checkpoint identity does not match. Legacy profiles without
  identity remain readable with a warning. The fullres follower writes a stale
  replacement to an artifact-fingerprinted sibling directory instead of
  overwriting historical output.
- `threshold_freeze_after_step` in the historical H9 configs freezes only the
  separate homeostatic update; it does not freeze AdamW gradients on the
  threshold parameter. The active H67/H66d runs retain that historical
  behavior. A new opt-in `freeze_threshold_grad_after_step` branch now supports
  a true gradient freeze for future ablations without changing these runs.

## RTL-exact boundary

The hardware-order deploy path freezes:

- Q7 score with step 2^-7;
- 16-entry Q8 exp2 LUT;
- integer row sum and ceil-power-of-two normalization;
- Q1.7 gate with ties-to-even rounding;
- true invalid-candidate masking for H66d Local-5.

This establishes attention-core hardware-order numeric equivalence only. The
following remain open before using the term fullres RTL-exact:

1. Window15/T450 controller loops, destination width, address generation, SRAM
   depth, accumulator depth, and ordered-trace replay for H67.
2. Window15/T450 Local-5 line buffer, halo/boundary handling, address control,
   post-G0 term trace, and projection replay.
3. Zero-mismatch comparison between fullres Python hardware-order traces and SV
   outputs at row, term, and delivered-update boundaries.
4. Full-network quantization of convolution/projection/decoder weights,
   membrane state, thresholds, normalization, and accumulators, or an explicit
   paper boundary that the accelerator covers only the attention core.
5. An energy model that includes Shiftmax, popcount/reduction, control, SRAM,
   and data movement. Current `energy_uj` is explicitly a spike-activity proxy
   and cannot support a chip-level energy claim.

Until these close, use `attention-core hardware-order numeric` in tables and
captions, not `full-network RTL-exact`.

The concrete current scale blockers are:

- H67 top/row-engine defaults and the TTX descriptor scheduler are still fixed
  to T162; the scheduler contains a literal `row_n_tokens=162`.
- Local-5 full-line-buffer RTL/testbench use row8, destination16, and three
  synthetic windows rather than fullres T450 checkpoint traces.
- Existing fullres deploy scripts invoke the Python evaluator, not an
  Icarus/Verilator trace replay.

The hardware-side T450 sign-off checklist is maintained in
`hw_autoresearch_nts07/docs/100_DSEC全分辨率RTLExact签核阻塞与T450闭环清单_20260730.md`.

## Required next gates

1. Finish H67 FT30 and standard valid825; retain ep0, AEE-best, ep29 and ep29
   training state.
2. Finish H66d FT30 and the same evaluation.
3. Run NB0 provenance replay and H67/H66d dyadic plus hardware-order valid825.
4. Select the algorithm winner using same-split AEE first, benchmark AAE second,
   and resolution-normalized spike activity; do not rank by the incomplete
   energy proxy.
5. Promote only the winner to T450 SV trace closure and official DSEC test
   submission. Add equal-budget/seed runs only after that selection.
