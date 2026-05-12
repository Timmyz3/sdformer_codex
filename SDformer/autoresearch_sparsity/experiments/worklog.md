# Sparsity Autoresearch Worklog

Session start: 2026-05-08 15:45

## Setup
- **Goal**: Hardware-friendly sparse energy saving pipeline for SDFormerFlow
- **Primary metric**: SOPs (G, lower is better) — proxy for energy
- **Constraint**: AEE < 1.9 (20% above baseline 1.5848)
- **Checkpoint**: `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`
- **User GPU**: A800 80GB, ~19GB used by user's neuron training (a5_refractory)

---

### Run 1: PSN Baseline — aee=1.5848, sops=3.6219G (KEEP)
- Timestamp: 2026-05-08 20:20
- What changed: Baseline evaluation using upstream profile_sops.py
- Result: AEE=1.5848, AAE=7.5012, SOPs=3.6219G, firing_rate=0.08496 — matches user's E0 results
- Insight: Profile tool works correctly with upstream configs
- Next: Test input-level sparsity (spike_th, norm_input)

### Run 2: num_bins=8 — CRASH
- Timestamp: 2026-05-08 21:30
- What changed: Reduced num_bins from 10 to 8 (fewer temporal bins = less compute)
- Result: CRASH — PSN neuron weights are [10,10] linear layers, checkpoint incompatible with T=8 model
- Insight: Cannot change temporal dimension (num_bins, num_steps) without retraining. PSN weights are T-dependent
- Next: Try input-level parameters that don't change model architecture

### Run 3: spike_th=0.1 — aee=6.8188, sops=2.1649G (DISCARD)
- Timestamp: 2026-05-08 22:04
- What changed: Set spike_th=0.1 (binarize input: voxel values >0.1 → 1, <0.1 → 0)
- Result: SOPs -40.2% (3.62→2.16G), but AEE +330% (1.58→6.82). Too aggressive
- Insight: Input binarization destroys graded event information. Need gentler approach
- Next: Try lower spike_th (0.02)

### Run 4: spike_th=0.02 — aee=6.5880, sops=4.9255G (DISCARD)
- Timestamp: 2026-05-08 23:44
- What changed: spike_th=0.02 (gentler binarization)
- Result: Counterintuitive — SOPs INCREASED (+36% to 4.93G), AEE still terrible (6.59)
- Insight: Binarization creates sharp 0/1 transitions that INCREASE downstream spike activity. The model was trained on continuous minmax-normalized input
- Next: Try different normalization (std)

### Run 5: norm_input=std — aee=8.0746, sops=5.2294G (DISCARD)
- Timestamp: 2026-05-08 23:47
- What changed: Changed input normalization from minmax to z-score (std)
- Result: SOPs increased +44% (5.23G), AEE catastrophic (8.07)
- Insight: The model is tightly coupled to minmax-normalized input distribution. Any input distribution change at eval time breaks both accuracy AND increases spike activity
- Next: Pivot strategy — eval-only input changes are a dead end

---

## Key Insights

1. **Eval-only input preprocessing is a dead end**: The PSN model is tightly coupled to its training input distribution (minmax-normalized continuous voxels). Any change to input statistics (binarization, different normalization) destroys accuracy AND often increases spike activity.

2. **Cannot change temporal dimensions without retraining**: The PSN neuron has [T,T]-shaped linear weights. Changing num_bins/num_steps changes model architecture, making the checkpoint incompatible.

3. **Meaningful sparsity requires training**: The user's G1 result (-25% SOPs, +1.3% AEE) was achieved through training with sparse gates. Eval-only configuration changes cannot achieve comparable results.

4. **The adapter layer (SDFormerFlowAdapter) is under-tested**: It has a bug where spiking_neuron is not placed inside the model dict (the upstream YAMLParser handles this but the adapter skips it). The adapter also has B-first vs T-first ordering issues with the upstream model.

5. **profile_sops.py is the reliable profiling tool**: It works directly with upstream configs and correctly handles all data formatting.

## Strategy Going Forward

**Phase 1 (when GPU free)**: Train sparse variants from PSN baseline checkpoint
- Start with simple sparse preprocessing (timestep_budget with low threshold)
- Train for 10-20 epochs to adapt to new input distribution
- Profile and compare

**Phase 2**: Combine multiple sparse modules
- timestep_budget + structured_token at various keep ratios
- window scheduling + graph_token_pruner

**Phase 3**: Multi-module Pareto frontier search

## Next Ideas
1. Train with spike_th enabled (allow model to adapt to binarized input)
2. Combine window_topk pruning (sparse attention) with structured_token pruning
3. Explore activity_window_scheduler for temporal attention sparsity
4. Test graph_token_pruner as a training-free pre-filter
