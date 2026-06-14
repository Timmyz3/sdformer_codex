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

## NTS-11 两神经元部署线

- 日期：2026-06-11 至 2026-06-12。
- 目的：
  在 NTS-10d/09e（h60 + S2+S3 Shiftmax + freeze1224）基础上，把推理时的神经元类型
  收敛为**严格两神经元**：Q/K 三值 ATLIF-PSN + 其余全部二值官方 ATLIF-PSN，**不留 vanilla PSN**。
- 基线 checkpoint：
  `experiments/baseline_stride_upstream/checkpoint_epoch59.pth`（NB0 valid825 参考
  AEE `1.4872`、AAE `9.93°`）。
- 配置生成：
  `entrypoints/make_nts11_two_neuron_only_configs.py`（11a–11g）、
  `entrypoints/make_nts11_phase2_configs.py`（11h–11m）、
  `entrypoints/make_nts11_phase3_configs.py`（11n–11p，**已取消，不跑**）。
- 链式验证：
  `entrypoints/verify_nts11_chain.py` PASS（93 ATLIF 模块安装，NB0 加载缺 overlay key
  符合预期，save/reload OK）。

### 部署策略：两神经元 vs 三神经元

全模型共 105 个 `Spiking_neuron` 位点：

| 区域 | 数量 | 严格两神经元（11a–11i, 11k–11m） | 三神经元消融（11j, 11n–11p） |
|---|---:|---|---|
| Q/K attention | 24 | 三值 ATLIF-PSN | 三值 ATLIF-PSN |
| encoder FFN / downsample 等 | 69 | 二值官方 ATLIF（`path_selection: all_non_qk`） | 二值官方 ATLIF |
| decoder / pred / resblocks | 12 | 二值官方 ATLIF（同上） | **vanilla PSN 保留** |
| 推理时神经元类型数 | — | **2**（三值 + 二值） | **3**（三值 + 二值 + vanilla） |
| 安装 ATLIF 模块数 | — | 93 | 81 |

**主线决策**：只推进严格两神经元线；11j 短测精度最好但推理为三类型，不作为部署候选。

### 共享训练设定（11 系列默认）

- 注意力：`bsa_attention.mode: h60`，`bipolar_mu` 0→0.05 经 720 step 升温（继承 10d）。
- Q/K 三值：`threshold_eta=6.5e-4`，`threshold_lr_scale=50000`，`threshold_freeze_after_step=1224`（除非变体覆盖）。
- 二值组：`all_non_qk` → `official_atlif`，`threshold_eta=0`（阈值不更新）。
- 优化器：`param_groups` 默认 `neuron_lr=3e-5`、`backbone_lr=1e-6`；warmup 200 step、`start_factor=0.1`。
- 短测：`max_train_steps=1224`，valid10 profile，gate `AEE>1.75`。

### 变体一览（差异仅列相对 11b 的增量）

**11b 为 S23 主线基线**：两神经元 + Shiftmax S2+S3（8 blocks）+ freeze1224 + 默认 LR。

| ID | 严格两神经元 | Shiftmax 范围 | 主要差异 |
|---|---|---|---|
| **11a** | ✓ | S2 only（6 blocks） | 覆盖范围小于 11b，测 S3 是否必要 |
| **11b** | ✓ | S2+S3（8） | 主线基线 |
| **11c** | ✓ | S2+S3 | `neuron_lr=5e-5`，`backbone_lr=2e-6`（fast LR） |
| **11d** | ✓ | S2+S3 | `neuron_lr=2e-5`，`backbone_lr=5e-7`（slow LR） |
| **11e** | ✓ | S2+S3 | Q/K `threshold_lr_scale=25000`（减半） |
| **11f** | ✓ | S2+S3 | `threshold_freeze_after_step=816`（提前冻结 Q/K 阈值） |
| **11g** | ✓ | S2+S3 | Q/K `threshold_eta=3.25e-4`（减半） |
| **11h** | ✓ | S2+S3 | 分 stage Q/K scale：`{0:25k, 1:35k, 2:50k, 3:50k}` |
| **11i** | ✓ | S2+S3 | 恢复 10d 式 `s0_ffn` 路径组（scale 8k）+ `all_non_qk` |
| **11k** | ✓ | S2+S3 | decoder 显式路径组（scale 3k）+ `all_non_qk`；短测≈11b |
| **11l** | ✓ | S2+S3 | 11c fast LR + freeze816 组合 |
| **11m** | ✓ | S2+S3 | 11h stage scale + freeze816 组合 |

**非两神经元（仅记录，不推进）**：

| ID | 推理类型数 | 主要差异 |
|---|---:|---|
| **11j** | 3 | encoder 二值 ATLIF，decoder/pred/resblocks 保留 vanilla PSN（12 模块） |
| **11n** | 3 | 11j + fast LR |
| **11o** | 3 | 11n + warmup 720/0.05 |
| **11p** | 3 | 11o + freeze816 |

Phase-3（11n/o/p）于 2026-06-12 被用户叫停；`results/nts11_phase3_20260612_015249/` 无完整 summary。

### 短测结果 Batch-1（严格两神经元，2026-06-11）

- 目录：`results/nts11_two_neuron_20260611_203636/`
- 条件：1224 steps，valid10，自 NB0 ep59 微调。

| rank | 配置 | AEE | AAE | SOPs(G) | firing |
|---:|---|---:|---:|---:|---:|
| 1 | **11c** fastlr | 3.338 | 62.16 | 1.586 | 0.0375 |
| 2 | 11b baseline | 3.663 | 68.67 | 1.546 | 0.0366 |
| 3 | 11e qkscale25k | 3.705 | 68.72 | 1.553 | 0.0367 |
| 4 | 11g eta0325 | 3.705 | 68.72 | 1.553 | 0.0367 |
| 5 | 11a S2 only | 3.726 | 69.84 | 1.546 | 0.0366 |
| 6 | 11f freeze816 | 3.794 | 70.76 | 1.542 | 0.0365 |
| 7 | 11d slowlr | 3.955 | 74.02 | 1.526 | 0.0361 |

解读：fast LR（11c）在短测中明显领先；S2-only（11a）未优于 S23；slow LR（11d）最差。
11e 与 11g 数值相同（阈值更新幅度等价）。

### 短测结果 Batch-2（phase-2，2026-06-11）

- 目录：`results/nts11_phase2_20260611_230130/`

| rank | 配置 | 两神经元 | AEE | AAE | SOPs(G) | firing | 备注 |
|---:|---|:---:|---:|---:|---:|---:|---|
| 1 | 11j vanilla_decoder | ✗ | **2.065** | **26.84** | 2.082 | 0.0493 | 三类型，**不纳入部署线** |
| 2 | **11l** fastlr+freeze816 | ✓ | 3.300 | 62.08 | 1.592 | 0.0377 | 严格两神经元 batch-2 最优 |
| 3 | 11h stage_qkscale | ✓ | 3.630 | 68.03 | 1.552 | 0.0367 | |
| 4 | 11k decoder_soft | ✓ | 3.663 | 68.67 | 1.546 | 0.0366 | ≈11b |
| 5 | 11m stage+freeze816 | ✓ | 3.715 | 69.41 | 1.547 | 0.0366 | |
| 6 | 11i layered_s0ffn | ✓ | 3.791 | 70.50 | 1.538 | 0.0364 | |

### 跨 batch 综合（仅严格两神经元）

| 候选 | 来源 | AEE | AAE | 判断 |
|---|---|---:|---:|---|
| **11l** | batch-2 | 3.300 | 62.08 | 当前综合最优（fast LR + 提前 freeze） |
| **11c** | batch-1 | 3.338 | 62.16 | 与 11l 接近，仅缺 freeze816 |
| 11b | batch-1 | 3.663 | 68.67 | 基线参照 |

短测距 NB0（AEE 1.49）仍有较大 gap，需更长训练或 warm720 等再验证；下一批短测应沿
**11c/11l 方向**（full `all_non_qk` + fast LR ± freeze816 ± warm720），不再跑 11j/phase-3 三类型线。

### Phase-4：二值/三值范围排列组合（2026-06-12 启动）

- 配置生成：`entrypoints/make_nts11_phase4_scope_configs.py`
- 自动流水线：`entrypoints/run_nts11_scope_autopilot.py`
  （短测 valid10 → 选最优 → 全量 30 epoch → valid825 标准推理）
- 统一训练旋钮：11l 方向（`neuron_lr=5e-5`、`backbone_lr=2e-6`、`freeze816`）

**范围分区**（全模型 105 个 `Spiking_neuron`）：

| 分区 | 数量 | 说明 |
|---|---:|---|
| Q/K core（sn_q, sn_k） | 24 | 默认三值 |
| sn2_q | 12 | 11q 前为 vanilla 漏洞；11r+ 显式覆盖 |
| attn_aux（attn_sn, proj_sn） | 24 | 注意力辅助脉冲 |
| FFN mlp | 24 | s0:4 / s1:4 / s2:12 / s3:4 |
| downsample | 3 | |
| patch_embed | 6 | |
| decoder head | 12 | decoders+preds+unet resblocks |

**Phase-4 变体**（均严格两神经元，除 11q 参照组保留 sn2_q vanilla）：

| ID | 三值范围 | 二值范围 |
|---|---|---|
| **11q** | Q/K only | all_non_qk（sn2_q 仍 vanilla，对照） |
| **11r** | Q/K | sn2_q 显式二值 + all_non_qk |
| **11s** | Q/K + sn2_q | all_non_qk |
| **11t** | Q/K + attn_aux | sn2_q + 其余 |
| **11u** | Q/K + 全部 FFN | sn2_q + 其余 |
| **11v** | Q/K + s0 FFN | sn2_q + 其余 |
| **11w** | Q/K + s2 FFN | sn2_q + 其余 |
| **11x** | Q/K + decoder head | sn2_q + 其余 |
| **11y** | Q/K + 全 encoder body | sn2_q + decoder head |
| **11z** | Q/K + FFN sn1 升维 | sn2_q + 其余 |
| **11aa** | Q/K + downsample | sn2_q + 其余 |
| **11ab** | Q/K + patch_embed | sn2_q + 其余 |

短测完成后，自动从 valid10 结果中选综合分最优者做全量标准化训练（valid825）。
监控：`tail -f neuron_experiments/H9_bipolar_self_attention/results/nts11_scope_autopilot_*/status.log`

### 副服务器推荐：NTS-11ac（与 Phase-4 不重复）

- **为何选它**：Phase-4 只扫二值/三值**范围**，统一默认 warmup 200/0.1，不碰 warm720。
  11ac 在短测最优方向 11l（fastlr+freeze816）上，叠加 **warm720/0.05**（与 sc_mu 升温对齐），
  并用 **11r 严格两神经元**（105 模块，sn2_q 不再留 vanilla）。
- **与主服务器分工**：主服跑 12 组 scope 短测 → 自动升全量；副服直接跑 11ac 全量 30 epoch + valid825。
- **配置**：`configs/generated/nts11ac_hw_h60_s23_sn2qbin_fastlr_freeze816_warm720_full30.yml`
- **一键脚本**：`entrypoints/run_nts11ac_secondary_server.sh`

## NTS-11 Phase-4 二值/三值范围短测 → 全量（自动追加）

- 时间：`2026-06-12T19:18:31`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts11_scope_short_20260612_020958`
- 选中短测：`nts11aa_hw_h60_s23_scope_downsample_ternary_s1224_steps1224`（valid10 AEE `2.4954`、AAE `46.5989`）
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_20260612_065413.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_bs8_20260612_065413_setsid`
- 方法：两神经元线 scope sweep（11q–11ab），统一 fastlr+freeze816。
- 标准推理：`eval_DSEC_flow_SNN.py` full valid825。

### 短测排名（valid10）

见短测目录 `summary.md`。

### 全量 valid825

| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes(G) | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.6385 | 10.7551 | 0.5486 | 0.2176 | 0.1050 | 28.3197 | 6.1130% | 22106.56 |
| 14 | 1.6018 | 10.3221 | 0.5435 | 0.2145 | 0.1030 | 28.5316 | 6.1587% | 22431.03 |
| 19 | 1.5426 | 10.0158 | 0.5274 | 0.2037 | 0.0963 | 28.8305 | 6.2232% | 22893.16 |
| 24 | 1.5625 | 10.2521 | 0.5309 | 0.2067 | 0.0982 | 29.3558 | 6.3366% | 23430.42 |
| 28 | 1.5881 | 10.4435 | 0.5333 | 0.2062 | 0.0979 | 31.6085 | 6.8229% | 25127.22 |
| 29 | 1.5609 | 10.0094 | 0.5270 | 0.2077 | 0.0999 | 29.3708 | 6.3398% | 23367.36 |

当前全量最佳：epoch19，AEE `1.5426`、AAE `10.0158`。

### 与 baseline / 成熟 H9 线对比（valid825 同协议）

| 线 | AEE | AAE | total_spikes | firing | sparsity | effective_FLOPs | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NB0 ep59** | 1.4872 | 9.93° | 44.05G | 9.50% | 90.50% | 116.79G | 37638 |
| **10d ep29** | **1.4781** | **9.69°** | 39.30G | 8.48% | 91.52% | 104.26G | 32301 |
| 07b ep29 | 1.4855 | 9.74° | 36.80G | 7.94% | 92.06% | 97.63G | 31581 |
| **11aa ep19** | 1.5426 | 10.02° | **28.83G** | **6.22%** | **93.78%** | **76.50G** | **22893** |

相对 NB0，11aa ep19：AEE `+0.055`、AAE `+0.09°`，但 spikes `-35%`、energy `-39%`。  
相对 10d，11aa ep19：AEE `+0.065`、AAE `+0.33°`，spikes `-27%`、energy `-29%`。

**解读**：11aa 替换法则（Q/K + downsample 三值，`all_non_qk` 二值）在稀疏/功耗维度成功，精度仍差 ~0.06 AEE。`all_non_qk` 全铺二值过激，需在**不改两神经元部署故事**前提下回调训练配方或微调范围（见下）。

### 替换法则适应性（2026-06-12 结论）

| 规则 | 判定 | 依据 |
|---|---|---|
| Q/K 必须三值 | 保留 | h60 输入语义 |
| sn2_q 必须二值 | 保留 | 11s 负脉冲塌缩 |
| downsample 可三值 | 保留 | 11aa Phase-4 #2，额外成本低 |
| FFN 全三值 | 待 valid825 | 11u 短测 AEE 最优，功耗待验 |
| encoder/patch 大面积三值 | 禁止 | 11y/11ab Phase-4 失败 |
| `all_non_qk` 无差别二值 | **需回调** | 11aa 省电但 AEE +0.06 vs 10d |

下一步验证轴：**11u**（FFN 三值，精度优先） vs **11aah**（11aa 范围 + 10d 式 LR/freeze 回调精度）。

### 实验类型与对比口径（报告必写）

三条线共用 **NTS-11 两神经元部署故事**（三值 ATLIF-PSN + 二值 official ATLIF-PSN，无 vanilla PSN），
差别只在 **三值铺哪些路径**，以及 **训练是 full30 还是 recipe finetune**。

| 线 | 实验类型 | 三值范围 | 训练起点 | 可比对象 |
|---|---|---|---|---|
| **11aa** | scope full30 | Q/K + downsample（3 路径） | **NB0 ep59** | 11u（scope）、10d（精度）、NB0（基线） |
| **11u** | scope full30 | Q/K + 全部 FFN（24 路径） | **NB0 ep59** | 11aa（scope trade-off） |
| **11aah** | **recipe-only finetune** | **与 11aa 完全相同** | **11aa ep19**（非 NB0） | 首要 vs **11aa ep19**；次要 vs 10d |

**为何 11aah 从 11aa ep19 续训、不从 NB0？**

1. 11aa ep19 已证明该 scope 的稀疏/功耗上限（28.8G spikes）；欠的是 ~0.06 AEE，更像 **LR/warmup/freeze 配方** 问题，而非 scope 选错。
2. 从 NB0 重训会把「学新神经元布局」和「调配方」混在一起，无法隔离变量。
3. 对标 10d 的成功配方（warm720 + freeze1224 + 慢 LR），在 **已收敛的 11aa 权重** 上做 15-epoch 抛光，算力更省、假设更清晰。

**写进表格/论文时的标注建议**

- 11aa / 11u：标注 `NB0→full30`；横比 AEE / spikes / energy（同 valid825 协议）。
- 11aah：标注 `11aa-ep19→finetune-15ep` 或 **11aa+recipe**；**不要**与 11aa 混为同一训练轨迹，也**不要**称为新 scope / 新 paradigm。
- 11aah vs 10d：可报最终 AEE，但需脚注「10d 从 NB0 全训 30ep，11aah 从 11aa 微调 15ep，训练轨迹不同」。
- 11aah vs 11aa ep19：**同 scope 同权重起点**后的配方收益，是最干净的对比。

---

## NTS-11aa 精度回调：超参 / LR 策略（2026-06-12）

**动机**：11aa ep19 已验证稀疏/功耗优势（28.8G spikes / 22.9k uJ），但 AEE 仍比 10d 高约 **+0.065**。  
在**不改神经元替换范围**（Q/K + downsample 三值，sn2_q + `all_non_qk` 二值）前提下，仅回调训练配方。

### 配方对比

| 项 | 11aa 原配方（Phase-4 full30） | 11aah 回调 | 11aai 消融（未开跑） |
|---|---|---|---|
| LR warmup | 200 step / start 0.1 | **720 / 0.05**（对齐 sc_mu） | 同 11aah |
| threshold freeze | 816 step | **1224 step**（对齐 09e/10d） | 同 11aah |
| neuron_lr / backbone_lr | 5e-5 / 2e-6 | **3e-5 / 1e-6**（10d 风格） | 5e-5 / 2e-6（Phase-4 fast） |
| 续训起点 | NB0 ep59 | **11aa ep19** | 同 11aah |
| 续训长度 | 30 epoch | **15 epoch finetune** | 15 epoch |

- 配置生成：`entrypoints/make_nts11aa_tune_configs.py`
- 11aah 配置：`configs/generated/nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15.yml`
- 11aai 配置：`configs/generated/nts11aai_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_fastlr_ft15.yml`

### NTS-11aah finetune（已完成 15 epoch）

- 目录：`results/nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15_bs8_20260612_194020_setsid`
- resume：`nts11aa_..._setsid/checkpoint_epoch19.pth`（**非 NB0**）
- 训练：`2026-06-12 19:40` → `2026-06-13 01:12`（ft ep0–14，共 15 epoch）
- 已存 checkpoint：ep0 / 4 / 9 / 14（`force_save_epochs`）

短 valid loss（**非 valid825**，仅供选 epoch；勿与 11aa valid825 直接数值对比）：

| ft epoch | val loss | ft epoch | val loss |
|---:|---:|---:|---:|
| 0 | 1.1995 | 8 | 1.196 |
| 4 | 1.1517 | 9 | 1.1024 |
| 5 | 1.1995 | 11 | **1.1323** |
| 6 | 1.1881 | 14 | 1.1431 |

短 valid 最优 ft ep9 = 1.102，但 **valid825 排名与短 valid 不一致**（见下）。

### valid825（已完成，ep0/4/9/14）

| rank | ft epoch | AEE | AAE | total_spikes(G) | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | **0** | **1.5160** | 10.01° | 29.78 | 6.43% | 23652 |
| 2 | 14 | 1.5695 | 10.13° | 30.15 | 6.51% | 24114 |
| 3 | 9 | 1.5666 | 10.52° | 31.54 | 6.81% | 25249 |
| 4 | 4 | 1.5775 | 10.24° | 29.53 | 6.37% | 23556 |

**推荐 checkpoint：ft ep0**（`checkpoint_epoch0.pth`）。

### 与 11aa ep19 / 10d 对比（同 valid825 协议）

| 线 | 类型 | AEE | Δ vs 11aa | spikes | Δ vs 11aa |
|---|---|---:|---:|---:|---:|
| 11aa ep19 | scope full30 | 1.5426 | — | 28.83G | — |
| **11aah ft ep0** | recipe finetune | **1.5160** | **−0.027** | 29.78G | +3% |
| 10d ep29 | 成熟 H9 | 1.4781 | — | 39.30G | — |

**结论（recipe-only finetune）**：

- 从 11aa ep19 换 warm720/freeze1224/stdlr 后 **仅 1 个 finetune epoch（ft ep0）** 即收回 **0.027 AEE**，功耗基本持平（+3% spikes/energy）。
- 继续 finetune 至 ep4/9/14 **valid825 反而变差**，与短 valid 误导相反；后续若再调配方，可考虑 **更早停** 或只存 ft ep0–2。
- 相对 10d 仍差 **+0.038 AEE**，但 spikes 低 **24%**（29.8G vs 39.3G）、energy 低 **27%**（23.7k vs 32.3k uJ）。

目录：`standard_valid825/`、`profile_ranking_valid825.md`。

---

## NTS-11u FFN 全三值 full30（副服 / 本机）

- 配置：`configs/nts11u_hw_h60_s23_scope_ffn_all_ternary_scope_full30_20260612_130819.yml`
- 目录：`results/nts11u_hw_h60_s23_scope_ffn_all_ternary_scope_full30_bs8_20260612_130819_setsid`
- 范围：Q/K + 全部 FFN mlp 三值（48 路径）+ sn2_q 二值 + 其余 `all_non_qk` 二值
- 配方：Phase-4 默认 fastlr（5e-5 / 2e-6）+ warmup 200/0.1 + freeze 816

### Checkpoint 清单（截至 2026-06-12 20:20）

| epoch | 文件 | train val loss |
|---:|---|---:|
| 0–18 | `checkpoint_epoch{0..18}.pth` | 见下 |
| 19+ | **未完成** | ep19 训练至 ~20% 时被 11aah 抢占 GPU 中断 |

训练 val loss 走势（短 valid，非 valid825）：

| epoch | val loss | epoch | val loss |
|---:|---:|---:|---:|
| 9 | 1.355 | 14 | 1.268 |
| 10 | 1.363 | 15 | 1.280 |
| 11 | 1.344 | 16 | 1.329 |
| 12 | 1.298 | 17 | 1.289 |
| 13 | 1.308 | 18 | **1.268** |

ep18 val loss 已回到 ep14 水平，后期 valid825 有望改善。

### valid825（标准化推理，`run_h9_standard_valid825_eval.py`）

**已完成**（ep9 / 14 / 15）：

| epoch | AEE | AAE | total_spikes(G) | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|
| 9 | 1.6752 | 11.08 | 49.26 | 10.63% | 40432 |
| 14 | 1.6794 | 10.81 | 49.02 | 10.58% | 40450 |
| 15 | 1.7060 | 11.08 | 50.51 | 10.90% | 41656 |

相对 11aa ep19（AEE 1.543 / spikes 28.8G）：**11u 在 ep≤15 时精度更差、功耗更高**——FFN 全三值早期未收敛。

**valid825 ep16+**：改由**副服务器**执行；本机 `run_nts11u_resume_and_valid825_queue.sh` 已停止。

### 11u vs 11aa / 11aah 定位

| 线 | 范围策略 | 精度轴 | 功耗轴 |
|---|---|---|---|
| **11aa** | downsample 三值 + 其余二值 | valid825 已评，AEE 1.543 | **最优**（28.8G） |
| **11aah** | 同 11aa，只调 LR | **valid825 ft ep0 AEE 1.516** | ~29.8G（+3% vs 11aa） |
| **11u** | FFN 全三值 | 短测 AEE 最优，full30 后期待验 | 早期 ~49G，偏高 |

---

## NTS-11 软硬件协同审计 + HW-friendly scope（2026-06-13 夜间接管）

**动机**：11aa 全局 firing 6.22% 好看，但 **downsample 三值层 firing 31–53%**（10d 仅 25–29%），硅片 2-bit 编码热点 + synops_logic 13.2%（NB0 5.7%）。DATE 叙事需「软件可改、硬件可固化」。

**硬件难点 → 软件修复**（详见 `hw_autoresearch_nts07/docs/16_hw_sw_pain_points_and_software_fixes.md`）：

| 优先级 | 痛点 | 软件线 | verify 三值层数 |
|---|---|---|---|
| P1 | downsample 三值热点 | **11aw**（sn2q scope，去掉 downsample 三值） | **24** ternary |
| P2 | S0/S1 无效 2-bit Q/K | **11ax**（仅 s23 Q/K 三值） | **16** ternary |
| P3 | 精度 −0.06 vs 10d | **11az**（11aah ep0 + 11aw scope finetune 5ep） | 24 |
| P5 | TTB mask 未导出 | `export_token_mask_from_profile.py` → `hw_masks/` | — |

**短测结果**（`nts11_hw_friendly_short_20260613_022912`，valid10，1224 step，自 NB0）：

| rank | 线 | AEE | AAE | SOPs(G) | firing |
|---:|---|---:|---:|---:|---:|
| 1 | **11aw** | 3.889 | 72.44 | 1.522 | 3.60% |
| 2 | 11ay（=11aw） | 3.889 | 72.44 | 1.522 | 3.60% |
| 3 | **11ax** | 3.917 | 74.17 | **1.259** | **2.98%** |

短测选优：**11aw**（综合分最低）。11ax SOPs 更低，full30 后作备选对照。

**autopilot 状态**（2026-06-13 13:36 重启）：
- 已修：`save_path` 须为 `run_dir/checkpoint_epoch{}.pth` + `run_dir.mkdir`（此前误传目录名，30 epoch 权重被 optimizer state 覆盖）
- 上轮 full30 训练至 ep28 val loss ~1.40，但 checkpoint 不可恢复 → 已从 NB0 重训
- **进行中**：`nts11aw_..._bs8_20260613_133609_setsid`（Epoch 0，~10h full30+valid825）
- 日志：`results/nts11_hw_friendly_autopilot_overnight.log`；训练：`...133609_setsid/train.log`
- 流程：→ **11aw full30** → valid825（ep 9/14/19/24/28/29）

**验收**（相对 11aa ep19）：AEE ≤ 1.52；downsample max firing < 30%；effective_G < 90。

---

## 后续任务（2026-06-14 更新）

1. ~~**11aah valid825**~~：已完成，推荐 **ft ep0**（AEE 1.516，−0.027 vs 11aa ep19）
2. **11u valid825 + 续训**（副服）：本机不跑
3. ~~**NTS-11bd unified autopilot**~~：full30 + valid825 完成，最优 ep19 AEE 1.561（不及 11aqa）
4. ~~**11aqa 抛光 finetune**~~：已完成，**当前全局最优 11aa-scope：ft ep5 AEE 1.497**
5. ~~**NTS-11 Phase-5 自动管线**~~：已完成，最优 **11aq ft ep2** AEE 1.506 / spikes 28.76G
5. **NTS-11 Phase-5 自动管线**（存档）：
   - 配置生成：`entrypoints/make_nts11_phase5_configs.py`（13 组短测）
   - 一键管线：`entrypoints/run_nts11_phase5_autopilot.py`
   - 监控：`tail -f results/nts11_phase5_autopilot_*/status.log`
   - 扫参轴（两神经元故事不变）：
     - **scope**：downsample / +ffn_s0 / +ffn_s2 / ffn_s2-only
     - **recipe**：warm720+freeze1224 × std/fast/slow LR
     - **attention**：bipolar_mu、alpha0
     - **resume**：NB0 full30 | 11aa ep19 finetune 3ep | 11aah ft ep0 polish 2ep
   - 流程：valid10 短测（1224 step）→ 选优 → full30 或短 finetune → valid825 → 写 RUNS.md

## NTS-11 Phase-5 短测 → 全量（自动追加）

- 时间：`2026-06-13T08:43:31`
- 驱动目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts11_phase5_autopilot_20260613_022152`
- 短测最优：`nts11aq_hw_h60_s23_ds_w720_fastlr_ftaa19`（valid10 AEE `1.2372`、track `finetune`、resume `checkpoint_epoch19.pth`）
- scope：`downsample_ternary` | recipe：`w720_fastlr`
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts11aq_hw_h60_s23_ds_w720_fastlr_ftaa19_full_20260613_070741.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts11aq_hw_h60_s23_ds_w720_fastlr_ftaa19_full_20260613_070741_bs8_20260613_070741_setsid`

### valid825

| epoch | AEE | AAE | total_spikes(G) | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.5477 | 10.1476 | 29.3836 | 6.3426% | 23376.20 |
| 1 | 1.5833 | 10.3918 | 30.3560 | 6.5525% | 24117.97 |
| 2 | 1.5057 | 10.0152 | 28.7560 | 6.2071% | 22879.13 |

Phase-5 当前最优：epoch2 AEE `1.5057`。

### Phase-5 解读（vs 11aa / 11aah / 10d）

| 线 | 类型 | valid825 最优 | AEE | spikes | vs 11aa ep19 |
|---|---|---:|---:|---:|---:|
| 11aa ep19 | scope full30 | ep19 | 1.5426 | 28.83G | — |
| 11aah ft ep0 | recipe finetune (stdlr) | ft ep0 | 1.5160 | 29.78G | −0.027 |
| **11aq ft ep2** | recipe finetune (**fastlr**) | ft ep2 | **1.5057** | **28.76G** | **−0.037** |
| 10d ep29 | 成熟 H9 | ep29 | 1.4781 | 39.30G | −0.064（但高功耗） |

短测结论：

- **NB0 + warm720 全训 9 组全部落选**（valid10 AEE 2.7–3.4）：新配方不能从 NB0 直接 1224 step 冷启动，必须走 **11aa ep19 续训**。
- **finetune 赛道包揽短测前 4**：11aq（fastlr）> 11as（aah0 抛光）> 11ap（stdlr）> 11ar（扩 scope 到 ffn_s2）。
- 全量 3ep finetune 后 **ep2 最优**（与 11aah「ep0 最优」不同），说明 fastlr 可多训 2–3 epoch；stdlr 仍宜早停。

推荐新 checkpoint：**`nts11aq_..._setsid/checkpoint_epoch2.pth`**（11aa scope + warm720/fastlr，从 11aa ep19 finetune）。

## NTS-11aqa 抛光 finetune（11aq ep2 → 5ep）

- 时间：`2026-06-13T12:50` → `2026-06-13T15:26`（train + valid825）
- 类型：**recipe-only finetune**（scope 与 11aa 相同）
- 续训起点：**11aq ep2**（非 NB0）
- 配置：`configs/generated/nts11aqa_hw_h60_s23_ds_w720_fastlr_ftaq2_ft5.yml`
- 目录：`results/nts11aqa_hw_h60_s23_ds_w720_fastlr_ftaq2_ft5_bs8_20260613_125039_setsid`

### valid825

| epoch | AEE | AAE | spikes(G) | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|
| 3 | 1.5484 | 10.1656 | 29.2481 | 6.3133% | 23350.00 |
| 4 | 1.5874 | 10.4612 | 30.2551 | 6.5307% | 24141.85 |
| **5** | **1.4969** | **9.9314** | **28.8696** | **6.2317%** | **23101.57** |
| 6 | 1.5723 | 10.2584 | 30.9886 | 6.6890% | 24725.15 |
| 7 | 1.5138 | 9.7777 | 29.0952 | 6.2803% | 23377.71 |

**11aa-scope 家族当前最优：ft ep5 AEE `1.4969`**（spikes 28.87G，距 10d ep29 仅 +0.019 AEE）。

推荐 checkpoint：**`nts11aqa_..._setsid/checkpoint_epoch5.pth`**。ep5 后精度回落，不宜继续长训。

### 横向对比（valid825）

| 线 | 类型 | 最优 ep | AEE | spikes | vs 11aa ep19 |
|---|---|---:|---:|---:|---:|
| 11aa ep19 | scope full30 | 19 | 1.5426 | 28.83G | — |
| 11aah ft ep0 | finetune stdlr | 0 | 1.5160 | 29.78G | −0.027 |
| 11aq ft ep2 | finetune fastlr | 2 | 1.5057 | 28.76G | −0.037 |
| **11aqa ft ep5** | finetune fastlr+ | 5 | **1.4969** | **28.87G** | **−0.046** |
| 10d ep29 | 成熟 H9 | 29 | 1.4781 | 39.30G | −0.064（高功耗） |

## NTS-11bd 统一注意力 短测 → 全量（两版配置）

共同设定：**h60 Shiftmax 全 12 block**（`target_blocks` S0–S3 共 12 个，无 Legacy s23）、warm720 + freeze1224 + fastlr、**NB0 ep59 → full30**。

| 版 | 代号 | scope 三值范围 | 短测 AEE | 全量目录 |
|---|---|---|---:|---|
| **v1** | `u12_dsffn2` | downsample(3) + **S2 FFN sn1/sn2(12)** | 2.528 | `.../nts11bd_u12_dsffn2_w720_fastlr_full30_20260613_212628_bs8_...` |
| **v2** | `u12_ds` | **仅 downsample(3)**（同 11aa scope） | 2.633 | `.../nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_...` |

- v1 驱动：unified autopilot `20260613_163756`（`2026-06-14 08:19` 完成）
- v2 驱动：rank2 launcher `run_nts11bd_rank2_full30_valid825.sh`（`2026-06-14` 完成；`225226` 为重复启动已废弃）

### valid825

首轮 autopilot 评 ep9/14/19/24/28/29；补评 ep15/20/26（按 val loss 低点）。

| epoch | AEE | AAE | spikes(G) | firing | 备注 |
|---:|---:|---:|---:|---:|---|
| 9 | 1.6659 | 11.0034 | 35.28 | 7.62% | |
| 14 | 1.6476 | 10.5908 | 35.67 | 7.71% | |
| 15 | 1.6477 | 10.6261 | 36.75 | 7.94% | 补评 |
| **19** | **1.5606** | **9.9941** | **35.62** | **7.70%** | **综合最优** |
| 20 | 1.6094 | 10.3087 | 36.90 | 7.97% | 补评 |
| 24 | 1.5748 | 10.1392 | 35.98 | 7.78% | |
| **26** | **1.5577** | **9.9829** | **36.70** | **7.93%** | 补评，**AEE 最低** |
| 28 | 1.5930 | 10.4246 | 38.46 | 8.31% | |
| 29 | 1.5745 | 9.9444 | 36.20 | 7.82% | |

11bd 推荐：**`checkpoint_epoch19.pth`**（AEE 1.561，spikes 35.6G）。ep26 AEE 略低（1.558）但 spikes 更高。

### 全指标横比（valid825 最优 checkpoint，同协议）

| 线 | ep | AEE | AAE | PE1 | PE2 | outlier | spikes | firing | effective | sparsity | energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **10d** | 29 | **1.478** | **9.69** | 0.512 | 0.190 | 0.087 | 39.30G | 8.48% | 104.3G | 91.5% | 32301 |
| **11aqa** | 5 | 1.497 | 9.93 | 0.520 | 0.193 | 0.089 | **28.87G** | **6.23%** | **76.6G** | **93.8%** | **23102** |
| 11aa | 19 | 1.543 | 10.02 | 0.527 | 0.204 | 0.096 | 28.83G | 6.22% | 76.5G | 93.8% | 22893 |
| **11bd-v2 ds** | 19 | 1.565 | 9.92 | 0.531 | 0.210 | 0.102 | 29.17G | 6.30% | 77.5G | 93.7% | 23109 |
| 11bd-v1 dsffn2 | 19 | 1.561 | 9.99 | 0.527 | 0.205 | 0.099 | 35.62G | 7.70% | 94.6G | 92.3% | 28853 |
| 11bd-v1 dsffn2 | 26 | 1.558 | 9.98 | 0.523 | 0.203 | 0.097 | 36.70G | 7.93% | 97.5G | 92.1% | 29761 |

Δ vs 10d（AEE 越低越好）：

| 线 | Δ AEE | Δ spikes | 解读 |
|---|---:|---:|---|
| 11aqa | **+0.019** | **−27%** | 精度最接近 10d，功耗最优族 |
| 11aa | +0.065 | −27% | scope 成功、Legacy s23 attn |
| 11bd-v2 | +0.087 | −26% | **功耗≈11aa，但 U12 attn 多欠 ~0.02 AEE** |
| 11bd-v1 | +0.083 | −9% | 扩 scope 到 S2 FFN **功耗崩**（+24% spikes vs v2） |

### 11bd 两版互比 + 后续精度优化方向

**v2 ds 优于 v1 dsffn2**：同 U12 attn 下，S2 FFN 三值只增加 ~7G spikes、未换精度收益；**应保留 downsample-only scope**。

**相对 11aa 的缺口（~+0.022 AEE）** 主要来自 **注意力从 s23→U12 全 12 block**（scope 相同、训练轨迹同为 NB0 full30）。S0/S1 也走 h60 Shiftmax 可能扰动浅层特征。

**推荐优化路径（按优先级）**：

1. **11bd-v2 ep19 → recipe finetune**（最省算力）：从 `checkpoint_epoch19.pth` 走 11aqa 同款 warm720/fastlr 3–5ep；验证 U12 权重能否用 finetune 收回 0.02–0.04 AEE。
2. **注意力消融 `u12_ds` + s23 blocks**：scope 保持 v2，仅 `target_blocks` 改回 S2+S3（8 block，对齐 11aa/10d）；隔离 U12 是否精度瓶颈。
3. **finetune 起点改为 11aa ep19 或 11aqa ep5**（非 NB0）：在已收敛的 downsample-scope 权重上换 U12 attn 或继续抛光，避免冷启动混学 layout+attn。
4. **暂不推进 v1 dsffn2 / 更大 scope**：valid825 已证 FFN 三值扩 scope 伤功耗、不助 AEE。
5. **可选补评 v2 ep26/ep28**（val loss 1.17–1.19）：当前 valid825 仅 ep9/14/19/24/28/29，短 val 最优点或未完全覆盖。

精度/功耗标杆仍为 **11aqa ft ep5**（AEE 1.497，距 10d +0.019）。
