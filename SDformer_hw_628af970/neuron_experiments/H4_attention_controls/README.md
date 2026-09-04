# H4 Attention Controls

H4 contains ablation-only controls for attention sparsity. It does not modify
the SDFormerFlow baseline. The first control replaces all attention `sn_q` and
`sn_k` inner spiking neurons with a zero-output module after checkpoint loading,
then runs the normal SOPs/metric profile.

## Q/K-Off Profile

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H4_attention_controls/entrypoints/profile_sops.py \
  --config neuron_experiments/H4_attention_controls/configs/baseline_qk_off.yml \
  --checkpoint experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --output-dir neuron_experiments/H4_attention_controls/results/profile_baseline_qk_off_epoch59_valid40_20260510 \
  --split valid \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

Result on 2026-05-10:

| experiment | samples | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 40 | 0.042556 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| PSN baseline with Q/K-off | 40 | 0.000000 | 0.076818 | 3.2748G | 1.622467 | 7.875712 |

Interpretation:

- Directly zeroing Q/K lowers the global SOPs proxy by about `9.6%`.
- Accuracy drops, so the baseline does use Q/K information.
- This supports the H3 story: ATLIF-PSN is not merely deleting Q/K at inference;
  it gives the rest of the model a chance to adapt to a sparse/near-off Q/K path.
