# Autoresearch: Hardware-Friendly Sparse Pipeline Optimization

## Objective
Find optimal combinations of preprocessing sparsity that maximize SOPs reduction (energy proxy) while keeping AEE degradation within 20% of the PSN baseline (AEE < 1.9).

**Story**: Hardware-friendly sparse energy saving — fewer synaptic operations = less energy on chip, with minimal accuracy loss. Complementary to user's neuron operator improvements.

## Metrics
- **Primary**: `sops` (Synaptic Operations, lower is better) — proxy for energy consumption
- **Secondary**: `aee` (Average Endpoint Error, lower is better), `firing_rate` (global spike firing rate)
- **Constraint**: aee must stay below 1.9 for a result to be "kept"

## How to Run
```bash
# Eval-only profiling (current):
bash autoresearch_sparsity/autoresearch.sh <config> [checkpoint] [num_samples] [batch_size] [num_workers]

# Training (when GPU free):
python -m autoresearch_sparsity.entrypoints.train --config <config> --prev-runid <checkpoint>
```

Outputs `METRIC name=number` lines.

## Files in Scope
- `autoresearch_sparsity/configs/*.yml` — NEW experiment configs (upstream format)
- `autoresearch_sparsity/entrypoints/profile_sparse.py` — Adapter-aware profiler (attempted)
- `autoresearch_sparsity/autoresearch.sh` — Benchmark runner (uses profile_sops.py)
- `tools/profile_sops.py` — READ-ONLY, reliable upstream profiler
- `third_party/SDformerFlow/` — READ-ONLY upstream source

## Off Limits
- `third_party/SDformerFlow/` — Do not modify upstream source
- `src/` — Do not modify existing adapter code
- `neuron_experiments/` — Do not touch user's experiments
- `neuron_autoresearch/` — Do not touch user's running experiments

## Constraints
- Must not interfere with user's running GPU training processes (currently a5_refractory)
- New configs go in `autoresearch_sparsity/configs/`
- New code goes in `autoresearch_sparsity/entrypoints/`
- No new pip/conda dependencies

## What's Been Tried

### Eval-only experiments (no training)
1. **num_bins=8**: CRASH — PSN weights are [T,T]-shaped, incompatible with changed T
2. **spike_th=0.1**: DISCARD — -40% SOPs but +330% AEE, too aggressive
3. **spike_th=0.02**: DISCARD — counterintuitively increased SOPs (+36%), AEE still terrible
4. **norm_input=std**: DISCARD — +44% SOPs, catastrophic AEE (8.07)

### Adapter-based profiling
- `profile_sparse.py` (custom profiler through the src/ adapter): ABANDONED
- The adapter (SDFormerFlowAdapter) has bugs:
  - spiking_neuron not placed in model dict (needs monkey-patch)
  - B-first vs T-first ordering mismatch with upstream model
  - It's under-tested (user's experiments bypass it via upstream entrypoints)

### Key finding
Eval-only input preprocessing changes are a dead end. The PSN model is tightly coupled to its training input distribution. **Meaningful sparsity improvements require training.**

## Strategy Going Forward

When GPU becomes free (user's a5_refractory completes):
1. Create training entrypoint (similar to neuron_autoresearch approach)
2. Train with sparse preprocessing enabled (low sparsity initially)
3. Profile and compare
4. Iteratively increase sparsity

## Training Experiment Plan (ready when GPU free)

### Phase 1: Single-module sparsity
| Config | Module | Key params | Expected SOPs impact |
|--------|--------|------------|---------------------|
| train_timestep_02 | timestep_budget | threshold=0.02 | Low (drops ~20% timesteps) |
| train_token_09 | structured_token | keep_ratio=0.9 | Low (keeps 90% tokens) |
| train_token_08 | structured_token | keep_ratio=0.8 | Medium |
| train_window_09 | window_topk | keep_ratio=0.9 | Low |
| train_window_08 | window_topk | keep_ratio=0.8 | Medium |

### Phase 2: Multi-module combinations
| Config | Combination | Expected |
|--------|-------------|----------|
| train_ts_token | timestep_budget + structured_token | Higher sparsity |
| train_window_token | window_topk + structured_token | Multi-scale sparsity |
| train_triple | timestep + window + token | Maximum sparsity |

### Phase 3: External inspiration modules
| Config | Module | Description |
|--------|--------|-------------|
| train_graph_08 | graph_token_pruner | Training-free importance-based pruning |
| train_activity_sched | activity_window_scheduler | Window scheduling with hysteresis |

## Current Status
- GPU: BUSY (user's a5_refractory training, ~19GB/80GB, 38% util, 8 python processes)
- State: WAITING for GPU availability
- Baseline: AEE=1.5848, SOPs=3.6219G (PSN baseline, confirmed)
