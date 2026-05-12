# H2 Code Audit Against Official ATLIF and TSN

Date: 2026-05-09

## Scope

This audit checks why `H2_adaptive_ternary_psn` produced worse accuracy and
higher SOPs than the PSN baseline. It compares the local H2 implementation
against:

- `optimization_sources/neuron_optimization/ATLIF_Activity-Pruning-SNN`
- `optimization_sources/neuron_optimization/TSN_Ternary-Spike`
- local `E2_exp_atlif` ATLIF copy

## Findings

### 1. H2 is not an official ATLIF port

Official ATLIF has a specific threshold pathway:

1. `Surrogate.forward()` emits `out * thre` and computes `thre_updates`.
2. Each ATLIF module accumulates `self.update_value += thre_updates / T`.
3. Training calls `threshold_update(model, lr)` after `optimizer.step()`.
4. `threshold_update()` manually increases `module.thresh.data`.

Official source:

- `ATLIF_Activity-Pruning-SNN/models/submodules/layers.py`
- `ATLIF_Activity-Pruning-SNN/utils/utils.py`

Local E2 copies this mechanism in:

- `E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py`
- `E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/training.py`
- `E2_exp_atlif/entrypoints/train.py`

H2 does not use this mechanism. H2 has `raw_theta` optimized only through the
flow loss. There is no `update_value`, no official `Surrogate`, and no
`threshold_update()` call.

### 2. H2 full training had no actual sparsity objective

The full config used:

```yaml
adaptive_ternary_psn:
  target_rate: 0.0
  reg_lambda: 0.0
```

Therefore the full H2 run had no explicit sparsity pressure.

### 3. H2 regularization path is non-differentiable

Even if `reg_lambda` were enabled, current H2 regularization is based on
`running_activity`, which is updated inside `torch.no_grad()`. A direct check
showed:

```text
penalty_requires_grad False
penalty_backward_error element 0 of tensors does not require grad and does not have a grad_fn
```

So current H2 activity regularization cannot train `raw_theta`.

### 4. Ternary sign output explains the high firing rate

Official TSN uses:

```python
out_s = torch.sign(x)
out_s[torch.abs(x) < 0.5] = 0
out_bp = torch.clamp(x, -1, 1)
```

H2 copied this activation shape, but combining it with attention Q/K PSN means
large negative activations become valid nonzero spikes. The trained checkpoint
shows:

```text
activity_mean=0.8703
pos_mean=0.0152
neg_mean=0.8551
```

This means most H2 activity is negative nonzero ternary output, which raises the
profiled firing rate instead of pruning it.

### 5. E2 entrypoint had a path handling risk

`E2_exp_atlif/entrypoints/train.py` changed into the baseline directory before
running the baseline script, but did not absolutize `--prev_runid`. If a future
run passes a relative checkpoint path that is valid from the SDFormer root but
not from `third_party/SDformerFlow`, it can load the wrong path or fail. This
has been fixed by adding `--prev_runid` to the path flags.

## Conclusion

The H2 result should be treated as invalid for judging ATLIF-style sparsity. It
is a hand-built fusion, not a faithful ATLIF training范式. The failure is
consistent with code behavior:

- no official ATLIF threshold update;
- no sparsity objective in the full config;
- non-differentiable activity regularization;
- ternary negative spikes counted as dense nonzero activity.

## Next Safe Implementation Direction

Do not continue H2 as written.

The safer next experiment should be a new folder, not a patch over H2:

`H3_official_atlif_tsn_adapter`

Rules:

1. Copy official ATLIF `Surrogate`, `ATLIF`, `regularize_spike`, and
   `threshold_update` mechanics directly.
2. Copy official TSN `spike_activation` and `mem_update` directly.
3. Keep ATLIF threshold update as the only adaptive sparsity mechanism.
4. Start with a local replacement only on sensitive attention Q/K layers, not
   all Q/K blocks.
5. Add a diagnostic entrypoint before training that proves:
   - thresholds are in optimizer or are manually updated;
   - `update_value` is nonzero after forward;
   - `threshold_update()` changes thresholds after one optimizer step;
   - activity/firing is measured without double counting.
