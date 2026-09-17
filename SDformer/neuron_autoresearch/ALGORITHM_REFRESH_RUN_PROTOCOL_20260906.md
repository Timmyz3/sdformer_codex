# September algorithm refresh: authorized paired training

Date: 2026-09-06. User authorized introducing the candidates and trying training.
The existing ep34 remains immutable; this batch does not change hardware code.
Literature and adaptations: `literature/ALGORITHM_REFRESH_CANDIDATES_20260906.md`.

## Registered matrix

| Order | Run | Modification | Parent | Budget | Deploy shape |
|---|---|---|---|---|---|
| 1 | control | Ordinary matched fine-tuning | C12 ep34 alpha=1/8 | 5 full epochs | Unchanged |
| 2 | augment | 5% voxel-entry dropout on randomly selected 50% of training samples | Same | Same | Unchanged |
| 3 | distill | Same dropout + frozen clean-view C12 teacher | Same | Same | Unchanged |
| 4 | delay | Shared +/-1 timestep residual parameterization in every functional ATLIF temporal matrix | Same | Same | Dense materialized original matrices |

This is a finite initial screen, not proof of convergence. If continuation and
candidate both improve at the end, extend their budgets together. Do not treat
one five-epoch negative result as definitive rejection of the source paper.
R2 event-structure supervision and R3 TRE-like masking are not silently queued:
the former needs event/time and warp tests; the latter needs its random-mask
control and hardware-reuse diagnostics. They remain separate follow-ups.

## Training contract

- Full DSEC train split, 480x640, window 2x15x15, ten input temporal bins,
  physical batch 2, accumulation 1, AMP, CuPy, workers 8, prefetch 1.
- Four fresh AdamW optimizers with the same settings. This is model-weight
  fine-tuning from ep34, NOT an optimizer-state true resume.
- LR backbone/norm 1e-6, temporal-neuron 5e-7, threshold 5e-8; one decay at local
  epoch 3. These start at the prior run's post-two-decay LR scale, not its peak.
- Preserve existing binary threshold/amplitude behavior and alpha=1/8. No
  carrier, SC mixture, sparse stage-only attention replacement, or rate penalty.
- Same seed and sample-order policy. Corruption has an independent RNG; teacher
  forward runs inside an RNG-preserving context. Teacher is detached, outside
  the student optimizer/checkpoint, and reset between independent examples.
- Teacher is copied after the complete 105-ATLIF/12-attention student load and
  verified tensor-identical BEFORE BN conversion. Teacher runs no_running BN,
  batch 1, on each clean item after the trainer's existing batch preprocessing.
  This is not a claim that batch preprocessing equals the standard batch1 input
  normalization path in every detail.
- Distillation uses final same-interval flow in configured pixel units, only
  corrupted examples, valid finite train GT below 400px, teacher EPE <=1px and
  teacher error <= detached student error. Weight 0.1 ramps over 200 steps.
  The ordinary supervised loss remains active for all examples.
- Dropout occurs after existing normalization, shares the keep mask across the
  two polarities, retains shape and endpoints, and does not rescale retained
  values. It is voxel-view corruption, not raw-event thinning.
- Delay branches are initialized to zero, channel-shared and linear. They are
  applied to all 81 functionally consumed ATLIF sites, not the 24 bypassed or
  diagnostic-only sites. All105 installed modules remain. This is an explicit
  module parameterization experiment, not a claim to reproduce full MD-Mixer or
  add inference expressivity beyond the existing unrestricted temporal matrix.
- Delay export materializes the effective matrix before the SAME dense addmm.
  Only dense original keys are written to deployment checkpoints. The final
  optimizer-side artifact additionally stores the training parameterization;
  ordinary legacy --resume is intentionally rejected by the new entrypoint.

## Validation and evidence

Internal training epochs 0..4 export as global epochs 35..39. Save/evaluate
epochs37 and39 with the existing standard valid825 entrypoint, no_running BN,
batch1, unchanged metrics. The small running-BN trainer validation is disabled
only in this dedicated process; it is not used for checkpoint selection.
Official DSEC hidden test remains excluded. MVSEC is not re-tuned in this batch.

Every formal result must pass 105 installed ATLIF, 12 Shiftmax attention,
210 overlay keys, missing0/unexpected0, 825 samples. Report all four arms with
AEE, AAE-2D, AE-3D, Fl and spikes. Compare distill to augment, and each candidate
to matched control, not merely to the frozen parent's 1.199514 AEE.
No new weight artifact inherits the old ep34 hardware capture or RTL evidence.

## Files and operations

All paths below are relative to `neuron_experiments/H9_bipolar_self_attention/`:

- `overlay/models/STSwinNet_SNN/refresh_training.py`: train-only helpers.
- `entrypoints/train_algorithm_refresh.py`: new entrypoint; extends the existing
  source-overlay pattern in memory without editing legacy train/baseline files.
- `entrypoints/run_algorithm_refresh_20260906.py`: identity-bound finite queue.
- `entrypoints/verify_algorithm_refresh_smoke.py`: export/standard-load admission.
- `tests/test_algorithm_refresh.py`: CPU tests for identity, gradient, export,
  corruption, RNG isolation, units, mask behavior and entrypoint compilation.
- `configs/generated/algorithm_refresh_20260906/`: four formal and four smoke
  YAMLs plus SHA manifest. Existing configs are not overwritten.
- `results/algorithm_refresh_20260906/status.log`: queue status.
- `results/algorithm_refresh_20260906/current.json`: active phase/PID or failure.
- `results/algorithm_refresh_20260906/<mode>/train.log`: individual train log.
- `results/algorithm_refresh_20260906/<mode>/standard_valid825/`: standard metrics.
- `results/algorithm_refresh_20260906/summary.json`: results added after each arm.
- `results/algorithm_refresh_20260906/report.md`: automatically refreshed Markdown
  table; pending arms remain explicit and smoke metrics are never included.

Environment: `/opt/conda/envs/sdformerflow/bin/python`; MLflow disabled; checkpoints
and logs local. Never invoke preparation again over an existing manifest.
Queue uses a process lock, waits for an idle GPU before each new job, checks
source/config/parent SHA, and leaves an exclusive attempt receipt. Failure stops
the queue; no automatic retry, deletion, or restoration of old controllers.
It refuses new jobs below 8GiB free. Hardware watchers are left untouched.

Commands from repository root (do not duplicate an already running queue):

```bash
/opt/conda/envs/sdformerflow/bin/python -m unittest discover -s neuron_experiments/H9_bipolar_self_attention/tests -p test_algorithm_refresh.py -v
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H9_bipolar_self_attention/entrypoints/run_algorithm_refresh_20260906.py --run
```

The formal queue requires `smoke_acceptance.json`, generated only after real
8-step training on distill/delay, nonzero learning signal, unchanged dense key
schema, and standard 4-sample inference. Smoke metrics are not paper accuracy.

## Initial checks

- Nine CPU tests passed.
- Distill real-data smoke: eight steps completed, teacher tensor identity PASS,
  78 BN modules configured no_running, peak allocated GPU memory40.720GiB.
  First four steps had nonzero confidence-eligible KD pixels and finite KD loss.
- Delay real-data smoke: all81 eligible sites installed with exactly unchanged
  initial weights; coefficients updated (absmax1.9883e-6 by logged step4).
  Eight steps completed, peak allocated GPU memory40.465GiB.
- Both dense exports passed four samples through the standard evaluator:
  105 ATLIF, 12 attention, 210 overlay keys, missing0/unexpected0, finite AEE.
  The first verifier pass encountered the evaluator's string-valued metrics;
  parsing was corrected, the already completed inference was SHA-checked and
  reused, and no training/inference job was blindly retried.
- `smoke_acceptance.json`: accepted=true, manifest SHA
  `2e93113dac7b1c0326873c61f7de65bee1fe9fa4b1dee2d3e1970396df58a060`.
- Detached finite supervisor launched, PID4045485. Its launcher is
  `entrypoints/launch_algorithm_refresh_20260906.py`; source SHA/PID are recorded
  in `supervisor.json`. Queue order control -> augment -> distill -> delay.
  This process maintains report.md and terminates after the queue exits; it does
  not require an active conversation. Consult current.json for the current child.
