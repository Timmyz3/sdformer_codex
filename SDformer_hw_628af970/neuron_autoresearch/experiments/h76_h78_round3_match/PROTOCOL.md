# H76-H78 Round3 Unified Match-Code Protocol

## Scope

This protocol registers three independent DSEC full30 candidates. All candidates keep the frozen
TTX network boundary: 105 one-sided binary ATLIF wrappers, the same 12 encoder attention blocks,
no native QKFormer carrier, no TX/SC stage mixing, and no architecture change outside the
attention block. Each attention output is produced by a learned static per-head codebook, never by
dynamic `gate * K` value transport.

The candidates are alternatives and must not be stacked in this round:

| ID | Mode | Descriptor | New state per attention |
|---|---|---:|---|
| H76 | `binary_pc9_patch_match_code` | 9 | one static `9 x 32` codebook per head |
| H77 | `binary_lc4_match_code` | 9 | one static `9 x 32` codebook and four coefficients per head |
| H78 | `binary_g4_match_code` | 36 | one static `36 x 32` codebook per head |

## Frozen tensor semantics

For every block, `T=2`, window `H=W=9`, token count `N=162`, and head dimension `D=32`.
`Omega9` is the fixed row-major `3 x 3` displacement set. Query `i=(t,y,x)` only compares against
the opposite time `1-t`. Out-of-window displacement candidates are masked before Shiftmax.

Let `n11`, `n10`, `n01`, and `n00` be the four binary contingency popcounts over a Q/K pair.
The common static readout is

```text
descriptor a_i -> Y_i[h,d] = sum_r a_i[h,r] * C[h,r,d]
```

where `C` is a learned candidate-local codebook. There is no K tensor in this readout.

### H76 PC9

First form nine alpha-XNOR score planes:

```text
m_delta(i) = (n11(i,delta) + n00(i,delta)/64) / 32
```

Then apply the same fixed dyadic corresponding-patch kernel to every displacement plane:

```text
w(center)=4, w(axis)=2, w(corner)=1
Z(i,delta) = sum_epsilon w(epsilon) * valid(i+epsilon,delta)
p_delta(i) = sum_epsilon w(epsilon) * valid(i+epsilon,delta)
             * m_delta(i+epsilon) / Z(i,delta)
a_i = Shiftmax_Omega9(p_i)
```

`Z` is recomputed from coordinate validity, so edges and corners are exactly normalized rather
than zero padded. The nine base score planes are generated once; the patch operator does not issue
81 new Q/K comparisons.

### H77 LC4

For every `delta in Omega9`:

```text
r_delta = beta11*n11 + beta10*n10 + beta01*n01 + beta00*n00
a_i = Shiftmax_Omega9(r_i / 32)
```

The per-head initialization is exactly `[1, 0, 0, 1/64]`, making step zero equivalent to the
plain Omega9 alpha-XNOR score. Coefficients use STE quantization on a signed `1/64` grid and are
clipped to `[-1,1]` during the forward deployment path. The same four-coefficient formula is used
for all nine displacements and all 12 blocks.

### H78 G4

The 32 head lanes are split into four fixed contiguous 8-bit groups. For each group:

```text
s[g,delta] = (n11[g,delta] + n00[g,delta]/64) / 8
a[g,:] = Shiftmax_Omega9(s[g,:])
descriptor = concat(a[0,:], a[1,:], a[2,:], a[3,:])  # 36 lanes
```

There is no learned router or runtime permutation. A head dimension other than 32 is rejected.

## Full30 training contract

Every candidate starts independently from the frozen checkpoint:

```text
neuron_experiments/H9_bipolar_self_attention/results/
date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/
checkpoint_epoch2.pth
```

The common contract is DSEC `288 x 384`, batch 8, workers 8, AMP, cupy, 30 epochs, warmup 720
steps, and milestones 20/25. Checkpoints are saved at epochs `0,4,9,14,19,24,28,29`. Standard
`valid825` evaluates those same epochs and reports AEE, AAE, PE/outlier metrics, spikes, and energy.

A short smoke run may only establish finite forward/backward execution. It must not rank, reject,
or alter any candidate; all three pre-registered candidates require full30.

Generate the configs and manifest with:

```bash
python neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h76_h78_round3_match_configs.py
```

The manifest is:

```text
neuron_experiments/H9_bipolar_self_attention/configs/generated/
h76_h78_round3_match_full30_manifest.json
```

## Loading-chain acceptance

Before training, run:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_round3_match_chain.py \
  --config <H76.yml> --config <H77.yml> --config <H78.yml>
```

The frozen TTX warm-start must satisfy:

- `ATLIFTernaryPSN=105`, attention modules `=12`, candidate modules `=12`.
- `checkpoint_overlay_keys=210`, `unexpected=0`.
- H76/H78: exactly 12 missing codebook tensors and no other missing key.
- H77: exactly 12 missing codebooks plus 12 LC4 coefficient tensors and no other missing key.
- A same-mode registered state dict reloads with `strict=True`, missing 0, unexpected 0.

After training, the saved candidate checkpoint must be loaded using:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_round3_match_chain.py \
  --trained <candidate.yml> <checkpoint_epoch29.pth> --output <strict_audit.json>
```

This is a strict load. Any missing or unexpected key invalidates evaluation.

## Serial queue

`run_round3_match_after_h75.py` waits for `ALL COMPLETE MATCH-CODE:` in
`results/match_code_after_h66_status.log`. It then runs H76, H77, and H78 in that order. For each
candidate it performs direct full30 training, audits the warm-start log, strictly reloads epoch 29,
runs standard valid825, and appends the resulting ranking table to the redesign document. It is
restart-safe at completed epoch-29 and ranking artifacts.

```bash
nohup python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_round3_match_after_h75.py \
  > neuron_experiments/H9_bipolar_self_attention/results/round3_match_after_h75_watcher.log 2>&1 &
```

The queue status file is
`neuron_experiments/H9_bipolar_self_attention/results/round3_match_after_h75_status.log`.
