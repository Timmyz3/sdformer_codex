# H9 Run Log

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
