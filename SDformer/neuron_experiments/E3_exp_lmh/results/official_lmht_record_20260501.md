# E3 official LMHT run record - 2026-05-01

## Source alignment

- Official source repo: `/root/private_data/work/optimization_sources/neuron_optimization/LMH_LMHT_SNN`
- Remote: `https://github.com/hzc1208/LMHT_SNN`
- Commit: `d9e0db3ce917c4c93acc46d8a63e4d4919e7eb2c`
- Official files checked:
  - `modules.py`: `LMHTNeuron`, `LMHT_Inference_Neuron`, `IFNeuron`, `QCFS`
  - `utils.py`: `replace_activation_by_LMHT(model, L, T, init_mem=0.)`
- Official training replacement rule for `L > 1`: `LMHTNeuron(L, T, 2./L, init_mem)`.
- E3 uses `L=2`, therefore `spiking_neuron.v_th = 1.0`.

## Experiment-local files

- Configs:
  - `neuron_experiments/E3_exp_lmh/configs/smoke.yml`
  - `neuron_experiments/E3_exp_lmh/configs/full.yml`
- Entrypoint:
  - `neuron_experiments/E3_exp_lmh/entrypoints/train.py`
- Overlay modules:
  - `neuron_experiments/E3_exp_lmh/overlay/models/STSwinNet_SNN/Spiking_modules.py`
  - `neuron_experiments/E3_exp_lmh/overlay/models/STSwinNet_SNN/experimental_neurons/factory.py`
  - `neuron_experiments/E3_exp_lmh/overlay/models/STSwinNet_SNN/experimental_neurons/single/lmh.py`

Baseline `third_party/SDformerFlow` was not edited.

## Verification before full run

- Py compile passed for E3 entrypoint, `Spiking_modules.py`, factory, and `single/lmh.py`.
- Minimal `LMHNode(T=10, v_threshold=1.0, levels=2)` forward passed:
  - output shape `(10, 2, 3, 4, 4)`
  - `mask` shape `(10, 10, 1, 1, 1, 1)`
  - `mask_linear` shape `(10, 10, 1, 1, 1)`
  - `alpha.requires_grad=True`
  - `v_threshold.requires_grad=False`

## ATLIF freeze decision

Freeze ATLIF was not continued to full training.

| checkpoint | AEE | AAE | firing | SOPs |
|---|---:|---:|---:|---:|
| E2 ATLIF epoch59 | 2.5128 | 12.5417 | 0.12212 | 5.2062G |
| Freeze epoch2 | 2.5498 | 13.4973 | 0.11745 | 5.0068G |
| Freeze epoch4 | 2.5837 | 13.5899 | 0.11563 | 4.9292G |
| PSN baseline epoch59 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |

Reason: freeze reduced ATLIF firing/SOPs slightly, but accuracy remained much worse than PSN baseline and training loss rebounded in epoch3/4. It was judged not worth a full ATLIF continuation.

## Speed tests

All E3 tests load PSN baseline checkpoint:

`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`

| test | config | batch | workers | status | epoch time | samples/s | max mem |
|---|---|---:|---:|---|---:|---:|---:|
| smoke | `configs/smoke.yml` | 8 | 4 | pass | 38.83s | 3.2963 | 65.467 GiB |
| speed | temp `/tmp/e3_smoke_bs10.yml` | 10 | 4 | OOM | n/a | n/a | 78.81 GiB process use |
| speed | temp `/tmp/e3_smoke_bs9.yml` | 9 | 4 | pass | 37.59s | 3.3521 | 73.420 GiB |

Selected full setting: batch 9, workers 8, AMP enabled, `pin_memory=false`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Full training

Started:

```bash
setsid env SDFORMER_USE_MLFLOW=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /opt/conda/envs/sdformerflow/bin/python neuron_experiments/E3_exp_lmh/entrypoints/train.py \
  --config /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E3_exp_lmh/configs/full.yml \
  --prev_runid /root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E3_exp_lmh/results/e3_official_lmht_full_bs9w8_amp_pinfalse_20260501_035012_checkpoint_epoch{}.pth
```

- PID: `1626911`
- Log: `neuron_experiments/E3_exp_lmh/results/e3_official_lmht_full_bs9w8_amp_pinfalse_20260501_035012.log`
- PID file: `neuron_experiments/E3_exp_lmh/results/e3_official_lmht_full_bs9w8_amp_pinfalse_20260501_035012.pid`
- Checkpoint pattern: `neuron_experiments/E3_exp_lmh/results/e3_official_lmht_full_bs9w8_amp_pinfalse_20260501_035012_checkpoint_epoch{}.pth`

Startup check:

- PID alive at `2026-05-01 03:50 UTC`
- GPU memory used: about `69771 MiB`
- GPU utilization: `97%`
- Log reached `Epoch 0`

Epoch0 check:

- Epoch0 train loss: `7.345396297177895`
- Epoch0 validation loss: `5.816481989622116`
- Epoch time: `2130.62s`
- Train step time: `2.6079s`
- Train throughput: `3.4511 samples/s`
- Max GPU memory: `76.367 GiB`
- Checkpoint: `neuron_experiments/E3_exp_lmh/results/e3_official_lmht_full_bs9w8_amp_pinfalse_20260501_035012_checkpoint_epoch0.pth`
- Continued to `Epoch 1`; no OOM observed.

## Storage cleanup - 2026-05-01

The `/root/private_data` mount reached 100% usage while E3 full training was running.

Cleanup performed:

- Deleted E3 full run checkpoint epochs `< 20`; retained epoch20+ full `.pth` checkpoints at the time of cleanup.
- Deleted E3 smoke/speed checkpoint `.pth` files; logs were retained.
- Deleted redundant E3 `*_state_dict.pth` companions; full `.pth` checkpoints are sufficient for eval/resume.
- Deleted redundant E2 ATLIF intermediate checkpoints from old full/short/benchmark runs.
- Retained E2 key evaluated anchors:
  - `full_pretrained...checkpoint_epoch45.pth`
  - `full_pretrained...checkpoint_epoch59.pth`
  - `freeze_threshold_only...checkpoint_epoch2.pth`
  - `freeze_threshold_only...checkpoint_epoch4.pth`
  - final/selected old E2 full anchors where present.

Disk status after cleanup:

- `/root/private_data`: `297G` free, `51%` used.

## Full training completion and inference - 2026-05-02

Training completed through epoch59.

Training/validation highlights:

| epoch | train loss | validation loss | note |
|---:|---:|---:|---|
| 0 | 7.3454 | 5.8165 | first checkpoint |
| 20 | 2.0887 | 3.0968 | after lr drop |
| 30 | 1.8759 | 2.6965 | validation improving |
| 40 | 1.7312 | 2.6784 | stable |
| 50 | 1.7041 | 2.6574 | stable |
| 55 | 1.7122 | 2.4156 | best validation loss, but no full checkpoint saved by train-loss based saver |
| 59 | 1.6761 | n/a | final checkpoint |

Inference/profile command pattern:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E3_exp_lmh/configs/full.yml \
  --checkpoint <checkpoint> \
  --output-dir <profile_dir> \
  --split valid \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 0 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE
```

Profile outputs:

- Epoch54: `neuron_experiments/E3_exp_lmh/results/profile_sops_official_lmht_epoch54_valid40_20260502_182138`
- Epoch59: `neuron_experiments/E3_exp_lmh/results/profile_sops_official_lmht_epoch59_valid40_20260502_182214`

| run | checkpoint | AEE | AAE | firing | SOPs |
|---|---|---:|---:|---:|---:|
| PSN baseline | epoch59 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| E2 ATLIF official-copy low-SOP | epoch49 | 3.8743 | 20.7177 | 0.06626 | 2.8245G |
| E2 ATLIF official-copy low-SOP | epoch59 | 3.7574 | 18.6163 | 0.06730 | 2.8692G |
| E2 ATLIF official-copy low-SOP | epoch30 | 3.6035 | 19.4891 | 0.07051 | 3.0059G |
| E2 ATLIF official-copy | epoch59 | 2.5128 | 12.5417 | 0.12212 | 5.2062G |
| E2 ATLIF Plan A continued | epoch10 | 5.6600 | 27.6559 | 0.16096 | 6.8619G |
| E2 ATLIF Plan A continued | epoch19 | 5.6760 | 29.2109 | 0.16163 | 6.8903G |
| E2 ATLIF freeze-threshold continued | epoch2 | 2.5498 | 13.4973 | 0.11745 | 5.0068G |
| E2 ATLIF freeze-threshold continued | epoch4 | 2.5837 | 13.5899 | 0.11563 | 4.9292G |
| E3 official LMHT | epoch54 | 2.5621 | 9.6492 | 0.22770 | 9.7070G |
| E3 official LMHT | epoch59 | 2.7290 | 10.1696 | 0.23083 | 9.8404G |

Assessment:

- E3 official LMHT trained without runtime failure, but final inference is not better than baseline.
- Epoch54 is better than epoch59 for inference, so final epochs slightly overfit/degrade on validation metrics.
- Compared with E2 ATLIF, E3 has somewhat better AAE but worse AEE and much worse firing/SOPs.
- Compared with PSN baseline, E3 is worse on both accuracy and sparsity.
- Earlier E2 ATLIF official-copy low-SOP profiles did beat PSN baseline on SOPs/firing, but their AEE/AAE were much worse, so they are not useful accuracy-sparsity tradeoffs as-is.
- E2 ATLIF freeze-threshold continuation improved sparsity versus E2 ATLIF epoch59, but did not recover accuracy and still remained worse than PSN baseline.

## E3 official-source compliance review - 2026-05-02

Official source checked:

- Repo: `/root/private_data/work/optimization_sources/neuron_optimization/LMH_LMHT_SNN`
- Remote: `https://github.com/hzc1208/LMHT_SNN`
- Commit: `d9e0db3ce917c4c93acc46d8a63e4d4919e7eb2c`
- Files: `modules.py`, `utils.py`, `main.py`

What matches the official implementation:

- Core neuron copied from official `modules.py`:
  - `TwoLevelFunction.forward/backward`
  - `FourLevelFunction.forward/backward`
  - `LMHTNeuron.alpha`
  - `LMHTNeuron.mask`
  - `LMHTNeuron.mask_linear`
  - `LMHTNeuron.forward`
  - `IFNeuron`
  - `QCFS`
  - `LMHT_Inference_Neuron`
- Threshold rule follows official `utils.replace_activation_by_LMHT`:
  - Official uses `LMHTNeuron(L, T, 2./L, init_mem)` when `L > 1`.
  - E3 config uses `levels: 2`, therefore `v_th: 1.0`.
- Training parameter status follows official:
  - `alpha.requires_grad=True`
  - `mask.requires_grad=True`
  - `mask_linear.requires_grad=True`
  - `v_threshold.requires_grad=False`
- Optimizer is compatible with an official path:
  - Official `main.py` uses AdamW for DVS data or Spikformer.
  - E3 uses AdamW, AMP, all model parameters trainable.
- Runtime logs confirmed all experimental spiking modules were replaced with `LMHNode`.
- Runtime logs confirmed learned temporal masks were instantiated:
  - `45` modules with `mask` shape `[10, 10, 1, 1, 1, 1]`
  - `60` modules with `mask` shape `[2, 2, 1, 1, 1, 1]`

Experiment-local adaptations:

- `LMHNode` is a thin wrapper over `LMHTNeuron` so SDFormerFlow can call it through the existing `Spiking_neuron` factory.
- `factory.py` passes SDFormerFlow config fields `levels` and `initial_mem` into `LMHNode`.
- `Spiking_modules.py` accepts `levels/initial_mem` in the existing `spiking_kwargs` path.
- `entrypoints/train.py` only patches baseline launch-time behavior for experiment overlays and `pin_memory`; baseline files under `third_party/SDformerFlow` are not edited.

Important deviation from official inference protocol:

- Official `main.py --direct_inference` calls:
  - `replace_by_LMHT_Inference(model, L, T)`
  - `replace_layer_bias(model, L)`
  - expands runtime steps to `L * T`
- Current E3 profile uses the trained `LMHTNeuron/LMHNode` directly; it does not convert to `LMHT_Inference_Neuron`.
- Reason: SDFormerFlow's event input and model are wired around `num_steps=10` / 10 voxel bins. Official direct-inference would expand the neuron runtime to `L*T=20` for `L=2,T=10`, which is not directly compatible with the current fixed 10-bin input path.
- Therefore, E3 is correct as a training-time LMHT replacement experiment, but it is not a full reproduction of the official post-training direct-inference reparameterization path.

Conclusion:

- The poor E3 metrics are not explained by an obvious wrong core-neuron transcription or wrong threshold setting.
- A remaining methodological mismatch is the missing official direct-inference reparameterization; supporting it would require a separate E3 inference-adapter experiment that handles `L*T` temporal expansion and layer-bias rescaling.

Storage note:

- Redundant E3 `*_state_dict.pth` files generated after the earlier cleanup were deleted after profiling.
- `/root/private_data` after cleanup: `220G` free, `64%` used.
