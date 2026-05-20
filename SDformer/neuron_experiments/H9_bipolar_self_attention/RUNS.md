# H9 Run Log

## H13 Review-Driven Queue Additions

- Source docs:
  `neuron_autoresearch/H13_SERIES_REVIEW.md` and
  `neuron_autoresearch/experiments/h13_signed_consensus_attention/H13_DEEP_ANALYSIS.md`.
- Decision:
  the docs are actionable. They identify three immediate H13 risks worth
  testing before broader attention redesign: AAE drift, Shiftmax hardware cost,
  and negative ternary event preservation.
- Controller:
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/autorun_h13n_h14.py`
  now waits for the active H13n full run, profiles H13n checkpoints including
  epoch7, then screens paper-faithful H14 strict-BSA guards before H13
  normalization/negative-feedback ablations.

### New H13 Review Experiments

| Run | Purpose | Guard config | Full config |
|---|---|---|---|
| H13r | H13n + angular loss `lambda_ang=0.2` to protect AAE | `configs/h13r_ang02_h13n_guard120.yml` | `configs/h13r_ang02_h13n_full.yml` |
| H13s | H13n + signed-consensus ShiftNorm, power-of-two denominator | `configs/h13s_shiftnorm_h13n_guard120.yml` | `configs/h13s_shiftnorm_h13n_full.yml` |
| H13t | H13n + signed-consensus popcount L1 normalization, no Shiftmax | `configs/h13t_popcount_l1_h13n_guard120.yml` | `configs/h13t_popcount_l1_h13n_full.yml` |
| H13u | H13n + independent negative firing target feedback | `configs/h13u_negtarget_h13n_guard120.yml` | `configs/h13u_negtarget_h13n_full.yml` |

Note: H13r angular-loss protection is generated but no longer prioritized in
the automatic queue. Prior H9a/i14 evidence suggests angular loss can conflict
with compat QK gating, so it should only be revisited after a stable
paper-derived attention mechanism is selected.

### H13t Code Change

- File:
  `overlay/models/STSwinNet_SNN/bsa_attention.py`
- Added mode:
  `signed_consensus_popcount_l1`
- Forward logic:
  ternary signed-consensus scores are shifted nonnegative and normalized by
  exact row-wise L1 sum. This is the no-Shiftmax ablation for testing whether
  the useful part is signed ternary consensus rather than exponent-like
  normalization.
- Unit test:
  `/opt/conda/envs/sdformerflow/bin/python -m unittest neuron_experiments/H9_bipolar_self_attention/tests/test_bsa_attention.py`
  passed.

## H9a Legacy Backup

- Backup folder:
  `neuron_experiments/H9A_legacy_best_backup/`
- Purpose:
  freeze the best historical H9a setup outside the active H9/H10 experiment
  folder so later Shiftmax edits do not obscure the reproducible reference.
- H9a behavior:
  `bsa_attention.mode: compat_qk_product` / default. This keeps baseline
  `sn2_q(sum(q))` token spike gating and applies the historical Shiftmax
  compatibility gate.
- Reference valid40:
  AEE `1.5043755`, AAE `7.6364652`, SOPs `3.0847G`, firing `0.0723596`.

## H10b QKFormer Spike-Shift H9a-Core

- Controlled full config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h10b_qkformer_spike_shift_h9a_core_full.yml`
- Guard config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h10b_qkformer_spike_shift_h9a_core_guard120.yml`
- Difference from H9a:
  same Q/K ternary ATLIF, same stage0 FFN binary, same stage3 block0 FFN
  binary, same stage0/stage2 downsample binary. Only attention mode changes
  from H9a legacy product gate to `qkformer_spike_shift`.
- Guard result on 2026-05-15:
  120 train steps passed, validation loss `1.188450`, Shiftmax installed on
  12 attention blocks, gate mean `0.632895`, max GPU memory about `41.825 GiB`,
  speed about `8.384 samples/s`.
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h10b_qkformer_spike_shift_h9a_core_full_bs8_20260515_010701_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h10b_qkformer_spike_shift_h9a_core_full_bs8_20260515_010701.log`
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h10b_qkformer_spike_shift_h9a_core_full_bs8_20260515_010701.log`

## H10c QK-BSA H9a-Core

- Purpose:
  make the Shiftmax object a ternary Q/K matrix score instead of a QKFormer
  token gate.
- Code:
  `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`
- Mode:
  `bsa_attention.mode: qk_bsa`
- Formula:
  `scores = Q_ternary @ K_ternary^T`, `weights = Shiftmax(scores)`,
  `attn = weights @ K_ternary`. This uses K as the value carrier because
  SDFormerFlow's `Spiking_QK_WindowAttention3D` has no V projection.
- Configs:
  `neuron_experiments/H9_bipolar_self_attention/configs/h10c_qk_bsa_h9a_core_guard120.yml`
  and
  `neuron_experiments/H9_bipolar_self_attention/configs/h10c_qk_bsa_h9a_core_full.yml`
- Guard result on 2026-05-15:
  120 train steps passed with batch size 8. Validation loss `1.133972`,
  Shiftmax installed on 12 attention blocks, row-sum mean `0.631393`,
  gate mean `0.003897`, max GPU memory about `58.093 GiB`, speed about
  `6.373 samples/s`.
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h10c_qk_bsa_h9a_core_full_bs8_20260515_022226_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h10c_qk_bsa_h9a_core_full_bs8_20260515_022226.log`
- After training:
  the run wrapper profiles `checkpoint_epoch29.pth` on valid40 with AEE, AAE,
  firing rate, and SOPs in
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h10c_qk_bsa_h9a_core_epoch29_valid40_20260515_022226/`.
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h10c_qk_bsa_h9a_core_full_bs8_20260515_022226.log`

## H9e H9a Compat Half-Blocks Even No-Downsample

- Purpose:
  keep the best H9a attention mechanism and test replacing half of the FFN
  blocks in every Swin stage.
- Attention:
  `bsa_attention.mode: compat_qk_product`, all 12 QK attention blocks.
- Neurons:
  all Q/K use PSN+ATLIF+ternary. FFN `mlp.sn1/sn2` use binary ATLIF for
  `layers.0.swin_blocks.0`, `layers.1.swin_blocks.0`,
  `layers.2.swin_blocks.0/2/4`, and `layers.3.swin_blocks.0`.
- Downsample:
  untouched, so this is a clean block-only replacement test.
- Configs:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9e_h9a_compat_halfblocks_even_no_down_guard120.yml`
  and
  `neuron_experiments/H9_bipolar_self_attention/configs/h9e_h9a_compat_halfblocks_even_no_down_full.yml`
- Guard result on 2026-05-15:
  120 train steps passed with batch size 8. Validation loss `1.137287`,
  Shiftmax installed on 12 attention blocks, ATLIF modules `36`, row-sum mean
  `0.629624`, max GPU memory about `42.891 GiB`, speed about
  `7.591 samples/s`.
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9e_h9a_compat_halfblocks_even_no_down_full_bs8_20260515_124359_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h9e_h9a_compat_halfblocks_even_no_down_full_bs8_20260515_124359.log`
- After training:
  the run wrapper profiles `checkpoint_epoch29.pth` on valid40 with AEE, AAE,
  firing rate, and SOPs in
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9e_halfblocks_even_epoch29_valid40_20260515_124359/`.
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h9e_h9a_compat_halfblocks_even_no_down_full_bs8_20260515_124359.log`

## H9f H9e Half-Blocks Even All-Downsample

- Purpose:
  keep H9e's half-even FFN block replacement, then add binary ATLIF to every
  existing Swin downsample node to test whether H9a's stronger SOP reduction
  mainly came from downsample sparsification.
- Attention:
  `bsa_attention.mode: compat_qk_product`, all 12 QK attention blocks.
- Neurons:
  all Q/K use PSN+ATLIF+ternary. FFN `mlp.sn1/sn2` use binary ATLIF for
  `layers.0.swin_blocks.0`, `layers.1.swin_blocks.0`,
  `layers.2.swin_blocks.0/2/4`, and `layers.3.swin_blocks.0`.
- Downsample:
  binary ATLIF on all existing downsample nodes:
  `layers.0.downsample.sn`, `layers.1.downsample.sn`, and
  `layers.2.downsample.sn`. `layers.3` has no downsample.
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9f_h9e_halfblocks_even_all_down_full.yml`
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9f_h9e_halfblocks_even_all_down_full_bs8_20260515_220431_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h9f_h9e_halfblocks_even_all_down_full_bs8_20260515_220431.log`
- Startup check:
  run reached epoch0 and logged ATLIF module count `39`, matching H9e's 36
  modules plus 3 downsample modules.
- After training:
  the run wrapper profiles `checkpoint_epoch29.pth` on valid40 with AEE, AAE,
  firing rate, and SOPs in
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9f_halfblocks_even_all_down_epoch29_valid40_20260515_220431/`.
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h9f_h9e_halfblocks_even_all_down_full_bs8_20260515_220431.log`

## H9g All-Blocks All-Downsample

- Purpose:
  stress-test the maximum H9-style sparse replacement scope: every Swin block
  FFN neuron plus every existing downsample neuron.
- Attention:
  `bsa_attention.mode: compat_qk_product`, all 12 QK attention blocks.
- Neurons:
  all Q/K use PSN+ATLIF+ternary. All 12 Swin blocks' FFN `mlp.sn1/sn2`
  use binary ATLIF. All existing downsample nodes use binary ATLIF.
- Downsample:
  binary ATLIF on `layers.0.downsample.sn`, `layers.1.downsample.sn`, and
  `layers.2.downsample.sn`. `layers.3` has no downsample.
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9g_allblocks_all_down_full.yml`
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9g_allblocks_all_down_full_bs8_20260516_124118_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h9g_allblocks_all_down_full_bs8_20260516_124118.log`
- Startup check:
  run reached epoch0 and logged ATLIF module count `51`, matching 24 Q/K
  modules plus 24 FFN modules plus 3 downsample modules.
- After training:
  the run wrapper profiles `checkpoint_epoch29.pth` on valid40 with AEE, AAE,
  firing rate, and SOPs in
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9g_allblocks_all_down_epoch29_valid40_20260516_124118/`.
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h9g_allblocks_all_down_full_bs8_20260516_124118.log`

## H9a Shiftmax Compatibility Full

- Status: running
- Run directory:
  `neuron_experiments/H9_bipolar_self_attention/results/h9a_shiftmax_compat_h8m_full_bs8_20260512_200523_setsid`
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9a_shiftmax_compat_h8m_full.yml`
- Entrypoint:
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py`
- Baseline checkpoint:
  `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`
- Save path:
  `neuron_experiments/H9_bipolar_self_attention/results/h9a_shiftmax_compat_h8m_full_bs8_20260512_200523_setsid/checkpoint_epoch{}.pth`
- Main changes:
  H8m PSN+ATLIF neuron stack, all Q/K attention modules ternary, stage0 FFN
  binary, stage3 block0 FFN binary, stage0/stage2 downsample binary, plus H9
  Shiftmax compatibility gate on all 12 `Spiking_QK_WindowAttention3D` blocks.
- Runtime choice:
  `batch_size=8`, `n_workers=8`, AMP on, `pin_memory=false`.

### Preflight

- Unit tests:
  `/opt/conda/envs/sdformerflow/bin/python -m unittest neuron_experiments.H9_bipolar_self_attention.tests.test_bsa_attention neuron_experiments.H9_bipolar_self_attention.tests.test_atlif_ternary_psn`
  passed.
- `batch_size=16` smoke OOMed.
- `batch_size=8` smoke passed, max GPU memory about 42.9 GiB, training speed
  about 4.60 samples/s for the 8-step probe.
- `batch_size=12` probe passed but was slower, max GPU memory about 64.2 GiB,
  training speed about 3.25 samples/s for the 4-step probe.

### Monitor

```bash
tail -f neuron_experiments/H9_bipolar_self_attention/results/h9a_shiftmax_compat_h8m_full_bs8_20260512_200523_setsid/train.log
```

```bash
nvidia-smi
```

## Autopilot Queue

- Status: running, waiting for H9a full to finish.
- PID:
  `1188026`
- Launcher log:
  `neuron_experiments/H9_bipolar_self_attention/results/autopilot_launcher_20260513_015737.log`
- Main log:
  `neuron_experiments/H9_bipolar_self_attention/results/autopilot_20260513_015737.log`

### Queue Order

1. Wait for H9a full run to finish.
2. Profile the latest H9a checkpoint on valid40 with AEE, AAE, firing rate,
   and SOPs.
3. Generate H9b stage/block Shiftmax subset configs.
4. Run H9b short 120-step searches:
   `stage0`, `stage1`, `stage2`, `stage3`, `stage23`, and `h8_goodmix`.
5. Profile every H9b short checkpoint on valid10.
6. Promote the best H9b run to full only if it satisfies:
   `AEE <= 1.15`, `AAE <= 7.0`, and `SOPs <= 3.75G`.
7. If promoted full finishes, profile it on valid40.

### Monitor Autopilot

```bash
tail -f /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/autopilot_20260513_015737.log
```

## H9b Stage1 Continuation

- Status: running from `checkpoint_epoch10.pth`.
- Previous partial full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9b_attn_stage1_120_full_20260513_123803_setsid/`
- Active continuation run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9b_attn_stage1_continue_fullmodel_ep10_to29_20260513_160117_setsid/`
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/generated_h9b_20260513_123803/h9b_attn_stage1_continue_ep10_to29.yml`
- Resume detail:
  `runtime.load_full_model: true` loads the full H9 module checkpoint after
  registering the Shiftmax pickle hook. Startup confirmed ATLIF threshold mean
  `0.113498`, so this preserves epoch10 neuron state instead of resetting to
  `0.1`.
- Current replacement set:
  all Q/K neurons use PSN+ATLIF+ternary; Shiftmax is enabled only on attention
  blocks `1:0` and `1:1`; H8m FFN/downsample sparse groups are still present.

### Monitor H9b Continuation

```bash
tail -f neuron_experiments/H9_bipolar_self_attention/results/h9b_attn_stage1_continue_fullmodel_ep10_to29_20260513_160117_setsid/train.log
```

## H9c Stage3 All6 No-Downsample Full

- Date: 2026-05-14 manual profile after full training.
- Run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9c_layers2_all6_ffn_no_down_full_20260513_172341_setsid/`
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9c_layers2_all6_ffn_no_down_full.yml`
- Checkpoint:
  `checkpoint_epoch29.pth`
- Training endpoint:
  train loss `1.285306`, validation loss `0.936984`.
- Profile:
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9c_all6_epoch29_valid40_20260514_manual/sops_summary.json`
- valid40 metrics:
  AEE `1.424656`, AAE `31.166865`, SOPs `3.0823G`, firing rate `0.072303`.
- Interpretation:
  SOPs are lower than baseline, and AEE improves versus baseline valid40, but
  AAE is badly degraded. This variant should not be treated as a good final
  candidate until the angular-error issue is fixed.

## H9c Stage3 Odd135 No-Downsample Full

- Date: 2026-05-14 manual profile after full training.
- Run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9c_layers2_odd135_ffn_no_down_full_20260514_010949_setsid/`
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9c_layers2_odd135_ffn_no_down_full.yml`
- Checkpoint:
  `checkpoint_epoch29.pth`
- Replacement:
  global Q/K uses PSN+ATLIF+ternary with Shiftmax; `layers.2` block `1/3/5`
  FFN `mlp.sn1/sn2` uses binary ATLIF; downsample is untouched.
- Training endpoint:
  train loss `1.226282`, validation loss `0.997214`.
- Profile:
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9c_odd135_epoch29_valid40_20260514_manual/sops_summary.json`
- valid40 metrics:
  AEE `1.474479`, AAE `33.050563`, SOPs `3.4013G`, firing rate `0.079786`.
- Interpretation:
  Interleaving stage3 FFN replacement does not fix the angular-error failure.
  Compared with H9a, odd135 has higher SOPs and much worse AAE, so the issue is
  likely tied to replacing `layers.2` FFN blocks under the current ternary/Shiftmax
  attention setting rather than simply replacing too many adjacent blocks.

## H10 QKFormer-Compatible Shiftmax

- Date: 2026-05-14.
- Root-cause check:
  SDFormerFlow uses `Spiking_QK_WindowAttention3D`, whose attention path is
  QKFormer-like: `att_token = sum(q)` followed by `attn = k * att_token`.
  It is not standard `QK^T V`.
- Fix:
  `bsa_attention.mode: qkformer_token` now applies Shiftmax to the native signed
  Q token score and gates the ternary K carrier. The old H9 product gate remains
  available as `compat_qk_product` only for historical comparison.
- Code:
  `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`
- Smoke:
  `h10_qkformer_shiftmax_layers01_allblocks_ffn_no_down_smoke.yml` passed 4 train
  steps plus validation.
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h10_qkformer_shiftmax_layers01_allblocks_ffn_no_down_full_20260514_113055_setsid/`
- Active config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h10_qkformer_shiftmax_layers01_allblocks_ffn_no_down_full.yml`
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h10_qkformer_shiftmax_layers01_allblocks_ffn_no_down_full_20260514_113055.log`

### H10 Result

- Completed checkpoint:
  `checkpoint_epoch29.pth`
- Training endpoint:
  train loss `4.765464`, validation loss `2.789779`.
- Profile:
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h10_qkformer_shiftmax_layers01_epoch29_valid40_20260514_manual/sops_summary.json`
- valid40 metrics:
  AEE `3.873903`, AAE `71.638042`, SOPs `3.4228G`, firing rate `0.080291`.
- Interpretation:
  This corrected token-Shiftmax variant is not viable as implemented. Although it
  is closer to the QKFormer carrier than the old H9 product gate, replacing
  `sn2_q(sum(q))` with a positive Shiftmax gate appears to destroy the signed
  token-gating behavior needed by the baseline.

## H9h Stage0+Stage2 All-Blocks and Downsample02 Full

- Date: 2026-05-16.
- Purpose:
  test the user's requested targeted sparse replacement: keep all Q/K attention
  replacements, replace every FFN neuron in stage0 and stage2 blocks, and replace
  downsample only in stage0 and stage2.
- Attention:
  `bsa_attention.mode: compat_qk_product`, all 12 QK attention blocks.
- Neurons:
  all Q/K use PSN+ATLIF+ternary. FFN `mlp.sn1/sn2` use binary ATLIF for
  `layers.0.swin_blocks.0/1` and `layers.2.swin_blocks.0/1/2/3/4/5`.
- Downsample:
  binary ATLIF on `layers.0.downsample.sn` and `layers.2.downsample.sn`.
  `layers.1.downsample.sn` is intentionally untouched.
- Expected module count:
  42 ATLIF modules = 24 Q/K + 16 FFN + 2 downsample.
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9h_stage0_stage2_allblocks_down02_full.yml`
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9h_stage0_stage2_allblocks_down02_full_bs8_20260516_214053_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h9h_stage0_stage2_allblocks_down02_full_bs8_20260516_214053.log`
- After training:
  the run wrapper profiles `checkpoint_epoch29.pth` on valid40 with AEE, AAE,
  firing rate, and SOPs in
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9h_stage0_stage2_allblocks_down02_epoch29_valid40_20260516_214053/`.
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h9h_stage0_stage2_allblocks_down02_full_bs8_20260516_214053.log`

## H9h-Ang Stage0+Stage2 All-Blocks and Downsample02 Full

- Date: 2026-05-16.
- Reason:
  the first H9h run was stopped early because `loss.lambda_ang` was still `0`,
  and baseline SDFormerFlow's supervised loss keeps its angular term commented
  out. H9h-Ang adds an experiment-local loss wrapper under the H9 overlay so the
  baseline folder remains untouched.
- Loss:
  `lambda_mod: 1`, `lambda_ang: 1.0`, `use_angular_loss: true`.
  The angular term is in radians and is added as
  `lambda_mod * magnitude_loss + lambda_ang * angular_loss`.
- Replacement:
  same as H9h: all Q/K, all stage0+stage2 block FFN nodes, and downsample nodes
  in stage0/stage2.
- Code:
  `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/h9_losses.py`
- Config:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9h_stage0_stage2_allblocks_down02_ang_full.yml`
- Smoke:
  `h9h_stage0_stage2_allblocks_down02_ang_smoke_20260516_2146.log` confirmed
  angular loss enabled, 42 ATLIF modules, 12 Shiftmax modules, and threshold
  mean increasing from `0.100000` to about `0.1008` during short training.
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h9h_stage0_stage2_allblocks_down02_ang_full_bs8_20260516_215048_setsid/`
- Active log:
  `neuron_experiments/H9_bipolar_self_attention/results/h9h_stage0_stage2_allblocks_down02_ang_full_bs8_20260516_215048.log`
- After training:
  the run wrapper profiles `checkpoint_epoch29.pth` on valid40 with AEE, AAE,
  firing rate, and SOPs in
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h9h_stage0_stage2_allblocks_down02_ang_epoch29_valid40_20260516_215048/`.
- Result:
  train exit code `0`, profile exit code `0`. Epoch29 valid40 metrics:
  AEE `1.537797`, AAE `7.973107`, SOPs `3.4759G`, firing rate `0.081536`.
  Compared with baseline E0, AEE improves by `0.046979`, AAE worsens by
  `0.471903`, and SOPs/firing only drop by about `4.03%`. This is less
  attractive than H9a/H9e because the angular loss helped keep AEE reasonable
  but did not deliver enough sparsity.
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h9h_stage0_stage2_allblocks_down02_ang_full_bs8_20260516_215048.log`

## H9i Stage0+Stage2 Sparse-Cap Variant

- Date: 2026-05-17.
- Reason:
  H9h-Ang thresholds were mostly capped: Q/K max `0.13`, FFN/downsample max
  `0.105`. Epoch29 checkpoint inspection showed Q/K mean `0.127821`, FFN mean
  `0.105000`, and downsample mean `0.105000`, so the small threshold increase
  was a configuration ceiling rather than failed updating.
- Configs:
  `neuron_experiments/H9_bipolar_self_attention/configs/h9i_stage0_stage2_allblocks_down02_ang_sparse_full.yml`
  and
  `neuron_experiments/H9_bipolar_self_attention/configs/h9i_stage0_stage2_allblocks_down02_ang_sparse_guard120.yml`.
- Change from H9h-Ang:
  Q/K `max_threshold: 0.18`, `activity_eta: 3.0`;
  stage0 FFN `max_threshold: 0.13`, `activity_eta: 0.06`;
  stage2 FFN `max_threshold: 0.13`, `activity_eta: 0.02`;
  downsample stage0/stage2 `max_threshold: 0.13`, `activity_eta: 0.06`.
- Guard120 result:
  exit code `0`, angular loss enabled, 42 ATLIF modules. Threshold mean rose
  from `0.100000` to `0.100915` after 120 train steps, max threshold reached
  `0.104099`, train loss `20.912154`, validation loss `10.587359`.
- Interpretation:
  the stronger sparse-cap setup is runnable and begins moving thresholds upward,
  but 120 steps is too early to confirm final sparsity. It should be promoted
  only as a guarded full run, then profiled against H9h-Ang/H9a/H9e.

## H13 Bias-Centered Bipolar Follow-Ups

- Date: 2026-05-19.
- Reason:
  zero-centered ternary Q/K treated copied PSN bias as a negative event source.
  H13 uses bias-centered ternary Q/K so the original positive spike boundary is
  preserved while negative spikes represent real signed deviation.
- Common attention:
  `bsa_attention.mode: signed_consensus_shiftmax` unless noted. The Shiftmax
  score is computed from `sign(Q) * sign(K)` consensus, while the value carrier
  remains threshold-scaled `K`.
- Common training:
  all parameters are fine-tuned from
  `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`.
- Guard results:
  H13i ShiftNorm guard gave AEE `1.553865`, AAE `7.516223`, SOPs `3.7852G`,
  firing `0.088791`; bipolar firing was fixed but ShiftNorm was weaker.
  H13n partial high-SOP FFN/downsample guard gave AEE `1.500320`,
  AAE `7.365211`, SOPs `3.6512G`, firing `0.085649`; this is the current
  full-training candidate.
  H13p target-rate `0.02` guard gave AEE `1.540671`, AAE `7.796620`,
  SOPs `3.5899G`, firing `0.084210`; too much sparsity pressure hurt accuracy.
  H13q target-rate `0.035` guard gave AEE `1.578515`, AAE `7.524042`,
  SOPs `3.6493G`, firing `0.085604`; it did not improve over H13n.
- Active full run:
  `neuron_experiments/H9_bipolar_self_attention/results/h13n_biascenter_shiftmax_target05_halfffn_down02_full_bs8_20260519_142730_setsid/`
- Monitor:
  `tail -f neuron_experiments/H9_bipolar_self_attention/results/h13n_biascenter_shiftmax_target05_halfffn_down02_full_bs8_20260519_142730_setsid/train.log`
- Full result:
  finished epoch29 on 2026-05-19. The controller profiled epochs
  `0/4/7/9/14/19/24/29` on valid40. Best checkpoint by validation profile was
  early epoch7, not epoch29:
  AEE `1.582509`, AAE `7.403115`, SOPs `3.7777G`, firing `0.088617`.
  Epoch29 overfit/degraded badly: AEE `2.575229`, AAE `13.577255`,
  SOPs `3.7052G`, firing `0.086916`. Compared with H9a epoch29 reference
  (AEE `1.504376`, AAE `7.636465`, SOPs `3.0847G`, firing `0.072360`),
  H13n does not improve the sparse story: AAE is competitive only at epoch7,
  but SOPs/firing are worse than both H9a and the E0 baseline.
- H13n profile dirs:
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h13n_full_epoch7_valid40_20260519_220148/`
  and
  `neuron_experiments/H9_bipolar_self_attention/results/profile_h13n_full_epoch29_valid40_20260519_220257/`.
- H14 follow-up interruption:
  H14a/H14b/H14c guard profiles completed, but the controller promoted H14c to
  a full run before we stopped the queue. User requested stopping on
  2026-05-19. The active full process was terminated with SIGTERM at
  `neuron_experiments/H9_bipolar_self_attention/results/h14c_strict_bsa_thetav_mild_full_bs8_20260519_222027_setsid/`
  after checkpoint epoch0 had been saved. Exit code is `-15`; do not treat this
  partial full run as a final result.

## Fast Screening Protocol

- Date: 2026-05-19.
- Reason:
  full training repeatedly wasted time because train loss did not predict
  AEE/AAE/SOPs, and several runs were best at early checkpoints rather than
  epoch29.
- New entrypoint:
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py`.
- Protocol:
  `neuron_autoresearch/FAST_SCREENING_PROTOCOL.md`.
- Usage:
  run short configs with `runtime.max_train_steps` set to small values such as
  `40/80/120`, immediately profile valid10, and only promote promising
  candidates to valid40/full training.

## H18/H21/H22 直接注意力筛选

- 日期：2026-05-19 至 2026-05-20。
- 目的：
  避免把某一次坏超参误判为“论文机制不可用”。这一批实验把“注意力模块设计”和
  “稀疏/阈值/score 超参”拆开筛选。
- 代码：
  `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`
  和
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py`。
- 配置生成脚本：
  `entrypoints/make_h18_configs.py`、
  `entrypoints/make_h21_configs.py`、
  `entrypoints/make_h22_configs.py`。
- H18 筛选结果目录：
  `neuron_experiments/H9_bipolar_self_attention/results/rapid_screen_h18_direct_h13fix_20260519_232711/`。
  关键现象是步数敏感：40-step 的 direct alpha-XNOR/A2OS2A 看起来会崩，
  AAE 约 `90+`，但 120-step 后 AEE/AAE 可以恢复。因此后续 direct attention
  不再用 40-step 单独否定。H18e 120-step valid10 精度最好
  （AEE/AAE `1.0467/6.2924`），但 SOPs 高达 `4.3253G`；H18c 120-step
  valid10 是更好的稀疏折中（AEE/AAE `1.0876/6.7315`，SOPs `3.8138G`）。
- H13 修复信号：
  H13v 低学习率 120-step valid10 达到 AEE/AAE `0.9609/5.9033`，SOPs
  `3.8293G`；补跑 valid40 后达到 AEE `1.4864`、AAE `7.2360`、SOPs
  `3.6648G`、firing `0.08597`。H13w 强稀疏反馈 120-step valid40 达到
  AEE `1.5350`、AAE `7.5568`、SOPs `3.5815G`、firing `0.08401`。这说明
  H13n 全量失败至少有训练超参和稀疏反馈因素，不能只归因于模块本身。
- H21 筛选结果：
  SpikeVideoFormer 风格 Hamming attention 的 ternary-active 版本 valid10 较好，
  但 valid40 为 AEE `1.6768`、AAE `8.4236`、SOPs `3.5858G`，精度不如
  H13v/H13w，因此暂时降级为备选，不作为当前主线。
- 当前队列：
  正在跑 H22。H22 固定 H18c direct alpha-XNOR + Shiftmax 模块，只扫超参：
  target firing rate、target-rate feedback、activity penalty、score scale、
  alpha-XNOR 静默奖励/反极性惩罚、active normalization、sign-value mode、
  学习率。H22 后自动接 H23。
- 下一步队列：
  H23 会测试“低学习率 + 强稀疏反馈”的组合，目标是保住 H13v/H18c 的精度，
  同时把 SOPs 从 `3.6G-3.8G` 往 `3G` 附近压。

## H23/H24/H25 主线短测队列

- 日期：2026-05-20。
- 触发原因：
  H22 仍停在 `3.50G-3.65G` SOPs，没有追上 H9a legacy 的 `3.0847G`。
  用户明确要求不要只换一个方案跑一下，而是每个方案内部系统遍历超参；
  同时 Q/K 必须保持三值，其他位置再做二值/三值和替换范围组合。
- 当前队列日志：
  `neuron_experiments/H9_bipolar_self_attention/results/h23_h24_h25_main_queue_20260520_005352.stdout`
- H23：
  继续跑低学习率 + 强稀疏反馈组合，目标是确认 H13v/H18c 的精度红利能不能
  和更强稀疏约束合起来。
- H24：
  回到 H9a 的低 SOPs 替换范围（Q/K 三值、stage0 FFN、stage3 block0 FFN、
  stage0/stage2 downsample），但把注意力从 H9a compat gate 换成
  `alpha_xnor_matrix_shiftmax`。这一组扫学习率、角度 loss、flow regularization、
  ATLIF target rate/activity。
- H25：
  Q/K 仍硬性三值 ATLIF + alpha-XNOR Shiftmax；细分 FFN `sn1` 升维、
  `sn2` 降维、二值/三值、downsample 是否替换。目标是判断 FFN 中到底哪个位置
  对 SOPs/AAE 最敏感。
- 三值发放检查：
  新增 `entrypoints/summarize_neuron_balance.py`，会从 train log 抽取
  ternary activity、pos、neg、binary activity、threshold。H22 已检查，pos/neg
  基本平衡，没有出现负脉冲塌缩。

## H26 降级注意力回收与自动全量

- 日期：2026-05-20。
- 用户反馈：
  之前降级的注意力不应直接放弃，稀疏可能可以通过 target rate、ATLIF
  activity penalty、正则项和三值位置组合补回来。
- 新配置生成脚本：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h26_attention_revisit_configs.py`。
- 新自动全量脚本：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py`。
- 队列日志：
  `neuron_experiments/H9_bipolar_self_attention/results/h26_attention_revisit_then_promote_20260520_010221.stdout`。
- 当前策略：
  H26 会在 H23/H24/H25 队列结束后自动开始；H26 完成后，promotion 脚本读取
  H23/H24/H25/H26 的最新 valid40 `summary.csv`，按综合分选一个候选进入
  30 epoch 全量训练。即使没有候选完全达到 H9a 的 `3.0847G` SOPs，也会选
  当前综合最优方案跑全量，避免 GPU 空转。
- H26 候选：

| 实验 | 注意力模式 | 三值/稀疏变化 | 用途 |
|---|---|---|---|
| H26a | `alpha_xnor_matrix_l1` | H9a scope + target040 | 回收 H18d |
| H26b | `a2os2a_direct` | H9a scope + target040 | 回收 H18e |
| H26c | `hamming_ternary_active_direct` | H9a scope + target040 | 回收 H21b |
| H26d | `hamming_binary_direct` | H9a scope + target040 | 硬件友好 Hamming 对照 |
| H26e | `alpha_xnor_matrix_shiftmax` | value=`sign` | 减少阈值实数乘法影响 |
| H26f | `alpha_xnor_matrix_l1` | FFN 也改三值 | 测试 FFN 三值稀疏潜力 |
| H26g | `a2os2a_direct` | FFN sn1 三值、sn2 二值 | 细分 FFN 升维/降维 |
| H26h | `hamming_ternary_active_direct` | target035 + 更强 activity | 更强稀疏 Hamming |
| H26i | `alpha_xnor_matrix_l1` | flow_regul_weight=`0.0003` | 检查 AAE/正则耦合 |

## H27 标准 BSA 范式复测

- 日期：2026-05-20。
- 背景：
  H14 已经做过 `strict_bsa_shiftmax`，即 `sign(Q) @ sign(K)^T -> Shiftmax -> @V`，
  但当时跟 H13n 替换范围绑定，没有纳入 H23-H26 的自动全量候选池。
- 旧 H14 valid40：

| 实验 | AEE | AAE | SOPs(G) | firing | 判断 |
|---|---:|---:|---:|---:|---|
| H14a threshold-V sqrt | 1.6340 | 7.9874 | 3.5330 | 0.08288 | 稀疏尚可，AEE 偏高 |
| H14b sign-V sqrt | 1.5507 | 7.7514 | 3.5440 | 0.08313 | 更硬件友好，精度可接受 |
| H14c threshold-V head/mild | 1.5213 | 7.7909 | 3.6935 | 0.08664 | 精度较好，SOPs 偏高 |

- 新 H27：
  `entrypoints/make_h27_strict_bsa_configs.py` 生成 H9a 低 SOPs 替换范围下的
  strict BSA 复测配置，扫描 `value_mode=sign/threshold`、`norm=sqrt_head_dim/head_dim/active`、
  `target_rate=0.040/0.035`。
- 队列：
  `entrypoints/run_h27_after_current_full.sh` 已排队，等待当前
  H24/H25/H26/auto-full 队列结束后执行。H27 完成后会再次运行
  `promote_best_rapid_screen.py`，把 H27 也纳入全量候选。
