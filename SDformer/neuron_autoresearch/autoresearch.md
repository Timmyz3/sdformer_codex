# Autoresearch: NSC Line — SC Attention Optimization for SDformerFlow

## Objective

Close the AEE gap between SC (signed consensus) and TX (ternary alpha-XNOR) attention:
- SC: AEE=1.77, AAE=11.42, spikes=32.3G
- TX: AEE=1.53, AAE=10.29, spikes=34.6G
- **Target**: AEE<1.60, spikes<35G, AAE<10

## Baseline (stride split)

| Metric | Value | Source |
|--------|-------|--------|
| Model | MS_SpikingformerFlowNet_en4 (PSN) | upstream train_flow_parallel_supervised_SNN.py |
| AEE | 1.489 | valid825, cupy, epoch59 |
| AAE | 9.923 | valid825 |
| total_spikes | 44.05G | valid825 |
| energy | 37.6 mJ | valid825 |
| Checkpoint | `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` | |

## Best TX Result

| Metric | Value | Source |
|--------|-------|--------|
| Experiment | NTX-01 (TX V2, ternary_alpha_xnor_shiftmax, S02, β=0.25) | |
| AEE | 1.534 | valid825, epoch28 |
| total_spikes | 34.61G | (-21% vs baseline) |
| energy | 29.7 mJ | (-21%) |

## Best SC Result

| Metric | Value | Source |
|--------|-------|--------|
| Experiment | NSC-01 (signed_consensus_shiftmax, S012, ang=0.02) | |
| AEE | 1.771 | valid825, epoch29 |
| total_spikes | 32.30G | (-27% vs baseline) |
| energy | 28.23 mJ | (-25%) |

## Metrics

- **Primary**: AEE (closer to baseline 1.489 is better)
- **Secondary**: total_spikes (lower is better), AAE (lower is better), energy (lower is better)
- **Constraint**: spikes must stay < 35G

## Available Knobs

| Category | Knob | Current SC | TX (working) | Notes |
|----------|------|-----------|--------------|-------|
| Attention | mode | signed_consensus_shiftmax | ternary_alpha_xnor_shiftmax | can also try sc_agree_disagree_shiftmax |
| Attention | consensus_score_norm | head_dim | head_dim | |
| Attention | score_scale | 1.0 | 1.0 | |
| ATLIF | threshold_mode | symmetric_target_rate | symmetric_bsa_tsn | Key difference! |
| ATLIF | target_rate | 0.05 | null | SC uses target_rate, TX doesn't |
| ATLIF | activity_eta | 1.5 | 0.0 | |
| FFN | range | S012 | S02 | SC replaces more stages |
| FFN | mode | official_atlif | official_atlif | |
| Optimizer | neuron_lr | 2e-5 | 3e-5 | TX uses higher neuron_lr |
| Optimizer | backbone_lr | 2e-7 | 1e-6 | TX allows more backbone adaptation |
| Optimizer | warmup | 300 steps | none | SC uses warmup, TX doesn't |

## What's Been Tried (NSC Series)

| # | Experiment | Key Config | Result |
|---|-----------|-----------|--------|
| NSC-01 | SC S012C baseline | symmetric_target_rate, tr=0.05 | AEE=1.77 ✅ |
| NSC-04d | SC blended μ=0.5 λ=0.6 | high mu, failed to converge | ❌ valid loss ~10 |
| NSC-09d | SC μ=0.05 all stages | symmetric_bsa_tsn, no tr | 🔄 running |

## Key Hypothesis

**SC's main weakness is token-level aggregation (O(N) vs TX's O(N²))**. The signed consensus computes Σ sign(Q)×sign(K) per token, losing pairwise token interactions. TX preserves N² interactions through matrix multiplication. To close the gap, SC needs either richer token representations (ReLU K, agree/disagree channels) or better training (higher LR, aligned ATLIF with TX).

## Experiment Queue (Phase 1: Align SC with TX's working setup)

1. **NSC-10a — SC + TX ATLIF**: symmetric_bsa_tsn, no target_rate, neuron_lr=3e-5, backbone_lr=1e-6, 30ep. (NSC-09d is close to this)
2. **NSC-10b — SC + ReLU K**: Keep Q ternary, K as continuous ReLU. Richer K signal. 30ep.
3. **NSC-10c — SC agree/disagree λ sweep**: λ=[0.3, 0.5, 0.8, 1.2], 360-step short screening.
4. **NSC-10d — SC S02 only**: Reduce FFN replacement from S012 to S02 (match TX range). 30ep.
