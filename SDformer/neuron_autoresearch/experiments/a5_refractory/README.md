# A5: Refractory-Period Pruning

**Hypothesis**: After emitting a spike, a neuron carries little new information
for 2-3 timesteps. Hardware-enforced refractory period reduces temporal firing
density with zero ALU overhead (2-bit counter per neuron, AND gate on output).

**Paper backing**: Activity Pruning AT-LIF (NeurIPS 2024/2025) — achieves
0.06 avg firing rate on ImageNet via adaptive threshold + current-based output.

## Training Protocol (Full Model — 60 epochs)

Unlike gate-only experiments, A5 wraps all neurons with RefractoryNeuron,
changing the forward-pass behavior. Full model retraining from baseline is required.

```bash
cd /root/private_data/work/SDformer && \
SDFORMER_USE_MLFLOW=0 python neuron_autoresearch/entrypoints/train.py \
  --config neuron_autoresearch/experiments/a5_refractory/configs/full.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth
```

- **Init**: PSN baseline epoch59
- **Epochs**: 60
- **Batch**: 4
- **LR**: 0.0001 (fine-tuning level)
- **Milestones**: [20, 30, 40, 50]

## Hardware Mapping
- 2-bit saturating counter per neuron → zero ALU overhead
- counter=0 → active, counter>0 → output forced to zero
- Per-neuron area: ~20 µm² in 28nm (2 flip-flops + 1 AND gate)
