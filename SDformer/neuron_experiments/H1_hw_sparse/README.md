# H1: Hardware-Friendly Adaptive Sparse Neuron

Extends G1 (6 layer0 nodes) to all encoder stages with HardwareSparseNeuron.

## Key Components

- **HardwareSparseNeuron**: fused BN-gate-spike primitive with running firing-rate tracking
- **Multi-stage targets**: `layer0_only` / `all_stages_proj` / `all_stages_full`
- **ATLIF-style threshold regularization**: soft penalty pushing firing rate toward target

## Config Variations

| Config | Epochs | BS | `stage_selection` | `hw_activity_scale` | Purpose |
|--------|--------|----|--------------------|----------------------|---------|
| `smoke.yml` | 1 | 1 | `all_stages_proj` | 0.0 | Linkage check |
| `short_gate_only.yml` | 5 | 8 | `all_stages_proj` | 0.0 | Trend validation |
| `full.yml` | 20 | 16 | `all_stages_proj` | 0.001 | Full gate+threshold reg |

## Run

```bash
# Smoke test
cd /root/private_data/work/SDformer
python neuron_experiments/H1_hw_sparse/entrypoints/train.py \
  --config neuron_experiments/H1_hw_sparse/configs/smoke.yml \
  --prev_runid <PSN_CHECKPOINT_DIR>

# Profile
python neuron_experiments/H1_hw_sparse/entrypoints/profile_sops.py \
  --config neuron_experiments/H1_hw_sparse/configs/smoke.yml \
  --checkpoint <CHECKPOINT.pth> \
  --num-samples 40
```
