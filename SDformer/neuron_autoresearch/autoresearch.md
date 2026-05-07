# Autoresearch: Hardware-Friendly Sparse Neuron Operators for SDformerFlow

## Objective

Design novel spiking neuron operators that are jointly optimized for:
1. **Sparsity** — minimize SOPs (synaptic operations) via structured spike gating
2. **Energy efficiency** — hardware-mappable primitives (clock-gating, AND-popcount, no multipliers)
3. **Hardware co-design** — each neuron innovation should have a clear hardware accelerator mapping
4. **Accuracy retention** — keep AEE within <10% of PSN baseline (1.5848), ideally <5%

**Story**: sparse, energy-efficient, hardware-friendly SNN for event-based optical flow.
**Not**: SOTA accuracy chasing.

## Baseline

| Metric | Value | Source |
|--------|-------|--------|
| Neuron | PSN (Parallel Spiking Neuron) | `third_party/SDformerFlow` |
| AEE | 1.5848 | valid40, epoch59 |
| AAE | 7.5012 | valid40, epoch59 |
| Firing rate | 0.08496 | valid40 |
| SOPs | 3.6219G | valid40 |
| Checkpoint | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` | |

## Best Known Sparse Result

| Metric | Value | Source |
|--------|-------|--------|
| Experiment | G1 partial sparse gate (6 layer0 nodes) | |
| AEE | 1.6056 (+1.3% vs PSN) | valid40, smoke epoch0 |
| SOPs | 2.7134G (-25.1% vs PSN) | valid40 |

## Metrics

- **Primary**: SOPs (lower is better) — estimated_total_sops from `tools/profile_sops.py`
- **Secondary**: AEE (lower is better), AAE (lower is better), firing_rate (lower is better)
- **Hardware**: gate_open_count, mean_gate_prob, spike_datapath_width

## How to Run

```bash
# Evaluation only (no training, fast):
cd /root/private_data/work/SDformer
python tools/profile_sops.py \
  --config <experiment_config.yml> \
  --checkpoint <checkpoint.pth> \
  --num-samples 40 \
  --metrics AEE \
  --output-dir <results_dir>

# Training (wait for GPU to be free):
python neuron_autoresearch/entrypoints/train.py \
  --config <config.yml> \
  --prev_runid <baseline_checkpoint>
```

## Files in Scope

- `neuron_autoresearch/` — new experiment code (this directory)
- `src/models/modules/spiking_neurons/` — neuron implementations (read reference, create variants)
- `neuron_experiments/` — existing experiment results (read only for reference)
- `tools/profile_sops.py` — evaluation harness (use as-is)

## Off Limits

- `third_party/SDformerFlow/` — baseline, READ ONLY
- `neuron_experiments/E*/`, `F*/`, `G*/`, `H*/` — existing experiments, READ ONLY
- Any running training process — do NOT kill or interfere
- Original dataset files

## Constraints

- New code goes in `neuron_autoresearch/` only
- Don't modify baseline PSN files
- Don't start training while GPU is occupied (>50% memory used)
- Training must use the experiment overlay pattern (source-patching, not editing baseline)
- Every new neuron must have a documented hardware mapping
- Prefer simplicity — 2 comparators better than 8, AND gates better than multipliers

## What's Been Tried (from neuron_experiments/)

### Blanket neuron replacements (E-series)
- **E1 SN** (Simple Spiking Neuron): smoke only, worse than PSN
- **E2 ATLIF**: adaptive threshold works for sparsity but destroys accuracy; best low-SOP run had AEE=3.75
- **E3 LMHT**: trains cleanly but SOPs too high (9.7G), missing inference reparameterization
- **E4 TS-LIF**: closest full replacement after PSN (AEE=2.18, SOPs=4.01G), but still worse
- **E5b TSN** (Ternary Spike): failed completely (AEE=29.77), paradigm mismatch with event bins
- **E6 ASN** (NASN): second-best accuracy among replacements (AEE=2.17), but 9.2x PSN SOPs

### Fused approaches (F-series)
- **F1-F5**: smoke-tested scaffolds only (fused adaptive PSN, LMH+ATLIF, adaptive TS-LIF, LMH+TS-LIF, signed hybrid)
- None have full-run results yet

### Partial gating (G/H-series)
- **G1**: 6 layer0 nodes with HardSparseGate — **best sparse/accuracy tradeoff** (25% SOP reduction, 1.3% AEE increase)
- **H1** (running now): extends G1 to all 36 encoder nodes with HardwareSparseNeuron (GTCN = gate + ATLIF threshold)

### Key architectural insight
Blanket neuron replacement (replacing all PSN neurons with a new type) consistently fails — either accuracy collapses or SOPs explode. The winning strategy is **targeted/partial insertion**: apply novel mechanisms only to specific high-impact nodes while keeping PSN elsewhere.

## Experiment Queue (Phase 1: Neuron Operators)

### Queue Priority

1. **A1 — FSN multi-level on G1 nodes**: Take G1's 6 winning nodes, upgrade from HardSparseGate to FusedSparseNeuron with num_levels=2 (ternary-style for optical flow polarity). Hypothesis: multi-level spikes carry more information per spike → can close more gates without accuracy loss.

2. **A2 — Leakage-as-gate**: Use the PSN's decay parameter as a dynamic gate signal instead of a fixed parameter. Neurons with high decay (forgetful) get lower gate probability. Hardware: decay threshold detector replaces learned gate_logit.

3. **A3 — Hierarchical shared gates**: Group neurons by layer stage, share one gate across all neurons in a group. Dramatically reduces gate parameter count (36 → 4) and hardware control wires. Test with stage-level shared gates.

4. **A4 — Spike-timing-dependent gate**: Gate probability depends on spike timing within the 10-bin sequence. Early bins (noise-dominated) get lower gate probability. Late bins (signal) get higher. Hardware: bin-counter + LUT.

5. **A5 — Refractory-period pruning**: After a neuron spikes, enforce a hardware-enforced refractory period (skip N timesteps). Reduces temporal firing density without affecting spatial pattern. Hardware: simple counter per neuron.

### Ideas Backlog (autoresearch.ideas.md)
See `autoresearch.ideas.md` for detailed experiment designs and lower-priority ideas.
