# H8 FFN Block Search

H8 is the first experiment family after fixing the high-level story:

- Attention Q/K uses PSN + ATLIF adaptive threshold + ternary output.
- FFN and other high-SOP non-attention targets use PSN + ATLIF adaptive
  threshold + binary output.
- The searched variable is only which FFN blocks are replaced.

This keeps the mechanism interpretable: attention gets signed ternary
expressiveness, while FFN/downsample layers get non-negative sparse binary
spikes for hardware-friendly SOP reduction.

## Fixed Core

All H8 configs start from the PSN baseline checkpoint:

`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`

Every config keeps:

| part | setting |
|---|---|
| attention | all stages `attn.sn_q` + `attn.sn_k`, `output_mode: ternary` |
| always-on FFN target | stage0 FFN, `output_mode: binary` |
| always-on high-SOP target | stage0 + stage2 downsample, `output_mode: binary` |
| trainable mode | `all` |
| optimizer | AdamW, lr `2e-5`, AMP on |
| short probe length | `max_train_steps: 120` |

Compared with H7, H8 deliberately uses weaker FFN sparse pressure and
full-network training. H7 showed that frozen-backbone FFN expansion could reduce
SOPs, but accuracy degraded before the model adapted.

## Search Configs

| config | searched FFN target | purpose |
|---|---|---|
| `h8a_stage1_block0_all_120.yml` | stage1 block0 FFN | isolate the first stage1 block |
| `h8b_stage1_block1_all_120.yml` | stage1 block1 FFN | isolate the second stage1 block |
| `h8c_stage1_all_all_120.yml` | stage1 block0+1 FFN | combine the two stage1 blocks |
| `h8d_stage2_late_all_120.yml` | stage2 block4+5 FFN | test late stage2 blocks without replacing all stage2 FFN |

Expanded block-search configs:

| config | searched FFN target | purpose |
|---|---|---|
| `h8e_stage0_block0_all_120.yml` | stage0 block0 FFN only | isolate stage0 block0 without stage0 block1 |
| `h8f_stage0_block1_all_120.yml` | stage0 block1 FFN only | isolate stage0 block1 without stage0 block0 |
| `h8g_stage2_block0_all_120.yml` | stage2 block0 FFN | stage2 per-block scan |
| `h8h_stage2_block1_all_120.yml` | stage2 block1 FFN | stage2 per-block scan |
| `h8i_stage2_block2_all_120.yml` | stage2 block2 FFN | stage2 per-block scan |
| `h8j_stage2_block3_all_120.yml` | stage2 block3 FFN | stage2 per-block scan |
| `h8k_stage2_block4_all_120.yml` | stage2 block4 FFN | stage2 per-block scan, split from `h8d` |
| `h8l_stage2_block5_all_120.yml` | stage2 block5 FFN | stage2 per-block scan, split from `h8d` |
| `h8m_stage3_block0_all_120.yml` | stage3 block0 FFN | late low-resolution block scan |
| `h8n_stage3_block1_all_120.yml` | stage3 block1 FFN | late low-resolution block scan |
| `h8o_stage3_all_all_120.yml` | stage3 block0+1 FFN | full stage3 FFN replacement |
| `h8p_stage1b0_stage2b4_all_120.yml` | stage1 block0 + stage2 block4 FFN | combine current best block with a deeper candidate |
| `h8q_stage1b0_stage3b0_all_120.yml` | stage1 block0 + stage3 block0 FFN | combine current best block with a late candidate |
| `h8r_stage2_mid_all_120.yml` | stage2 block2+3 FFN | middle-stage2 pair scan |

Notes:

- `h8e` and `h8f` are not H6a-core extensions. They deliberately isolate the
  two stage0 FFN blocks to test whether H6a needs both.
- `h8g` through `h8o` keep the H6a core and add exactly one stage2/stage3
  candidate, except the explicit pair/full-stage variants.
- `h8p` and `h8q` are second-round combinations around the current best short
  probe, `h8a`.

Promotion rule for full training:

- Prefer a config with AEE close to H6a/H4h short-probe reference and SOPs
  below H6a.
- If two configs are close in AEE, choose the lower-SOP one.
- If all configs lose too much AEE, keep the fixed attention ternary core and
  lower FFN `activity_eta` before expanding further.

## Commands

Short train:

```bash
SDFORMER_USE_MLFLOW=0 python neuron_experiments/H8_ffn_block_search/entrypoints/train.py \
  --config neuron_experiments/H8_ffn_block_search/configs/<config>.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H8_ffn_block_search/results/<name>_checkpoint_epoch{}.pth
```

Profile:

```bash
python neuron_experiments/H8_ffn_block_search/entrypoints/profile_sops.py \
  --config neuron_experiments/H8_ffn_block_search/configs/<config>.yml \
  --checkpoint neuron_experiments/H8_ffn_block_search/results/<checkpoint>.pth \
  --output-dir neuron_experiments/H8_ffn_block_search/results/<profile_dir> \
  --split valid \
  --num-samples 10 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

Run the expanded queue:

```bash
neuron_experiments/H8_ffn_block_search/entrypoints/run_block_search_queue.sh
```

To let the queue wait for another training process first:

```bash
WAIT_PID=<pid> neuron_experiments/H8_ffn_block_search/entrypoints/run_block_search_queue.sh
```

Run the expanded queue and then automatically promote the best effective short
probe to a full run:

```bash
WAIT_PID=<pid> neuron_experiments/H8_ffn_block_search/entrypoints/run_block_search_then_promote.sh
```

The default promotion rule is conservative: `AEE <= 1.07`, `AAE <= 6.35`,
`SOPs <= 3.60G`, and `TOP_K=1`. The historical `h8a` short probe is included
as a fallback comparison via `EXTRA_STAMP=20260511_165537`.

## Results

Short probe date: 2026-05-12 UTC. Run timestamp:
`20260511_165537`. All successful rows use the epoch0 checkpoint after
`max_train_steps: 120`, then valid10 profiling.

Reference rows from earlier valid10 probes:

| run | firing | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|
| H4h Q/K ATLIF-PSN reference | 0.087457 | 3.7283G | 1.027874 | 6.088951 |
| H6a Q/K ternary + stage0 FFN/downsample binary | 0.086604 | 3.6919G | 1.049870 | 6.106936 |
| H7 stage0+1 frozen probe | 0.083854 | 3.5747G | 1.074384 | 6.388056 |

H8 short probes:

| run | searched FFN target | status | train step sec | max GPU GiB | firing | SOPs | AEE | AAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `h8a_stage1_block0_all_120` | stage1 block0 | ok | 1.8909 | 75.655 | 0.082783 | 3.5290G | 1.050845 | 6.160332 |
| `h8b_stage1_block1_all_120` | stage1 block1 | ok | 1.8859 | 75.656 | 0.081692 | 3.4825G | 1.103242 | 6.463592 |
| `h8c_stage1_all_all_120` | stage1 block0+1 | OOM at step 8 | n/a | ~78.6 | n/a | n/a | n/a | n/a |
| `h8d_stage2_late_all_120` | stage2 block4+5 | ok | 1.8750 | 75.658 | 0.081611 | 3.4791G | 1.113702 | 6.219929 |

Layer-group firing rates:

| run | Q/K | stage0 FFN | stage1 FFN | stage2 FFN | downsample |
|---|---:|---:|---:|---:|---:|
| `h8a_stage1_block0_all_120` | 0.036314 | 0.127954 | 0.070374 | 0.102682 | 0.190799 |
| `h8b_stage1_block1_all_120` | 0.036257 | 0.126271 | 0.070081 | 0.102892 | 0.190073 |
| `h8d_stage2_late_all_120` | 0.036531 | 0.125290 | 0.079035 | 0.097926 | 0.191367 |

Current read:

- `h8a_stage1_block0_all_120` is the best candidate so far. It keeps AEE almost
  tied with H6a valid10 (`1.050845` vs `1.049870`) while reducing SOPs from
  `3.6919G` to `3.5290G`.
- `h8b` and `h8d` buy slightly lower SOPs, but the accuracy loss is too large
  for the current story.
- `h8c` should be rerun only with lower memory settings, for example batch size
  12 or gradient accumulation, because full-network training with both stage1
  blocks OOMs at batch size 16.

Recommended promotion path: use `h8a` as the next full-run candidate, or run a
slightly longer short probe of `h8a` before full training to verify that the
precision/SOP tradeoff holds beyond 120 steps.
