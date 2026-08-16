# H79-H80 Round4 Unified Assignment Protocol

## Scope

H79 and H80 are independent DSEC full30 candidates on the frozen TTX boundary. Both retain 105
one-sided binary ATLIF wrappers and replace all 12 attention blocks with one identical formula.
Their output is a learned static per-head codebook readout; neither path transports a native K/V
carrier. The modes are optional and disabled by default. They must not be stacked or stage mixed.

| ID | Mode | Assignment | Stored state per attention |
|---|---|---|---|
| H79 | `binary_cf10_match_code` | row Shiftmax over 9 offsets plus null | `9 x 32` codebook and 2 dyadic coefficients per head |
| H80 | `binary_dn9_match_code` | row and destination Shiftmax product | `9 x 32` codebook per head |

## Common score and static readout

Each block uses `T=2`, a `9 x 9` spatial window, `N=162`, head dimension `D=32`, and fixed
row-major `Omega9`. Query `i=(t,y,x)` compares with the opposite-time token
`j(i,delta)=(1-t,y+dy,x+dx)`. Invalid boundary edges are masked before normalization.

```text
n11(i,delta) = popcount(q_i & k_j)
n00(i,delta) = popcount(~q_i & ~k_j)
s(i,delta) = (n11(i,delta) + n00(i,delta)/64) / 32
Y_i[h,d] = sum_delta a(i,delta) * C[h,delta,d]
```

`C` is static after training. The output equation does not read K or V.

## H79 CF10

For each query, take the two largest valid local scores and binary query activity:

```text
s1 = max_delta s(i,delta)
s2 = secondmax_delta s(i,delta)
rho = popcount(q_i) / 32
e = s1 - 1 + Q_1/64(beta_m)*(s1-s2) + Q_1/64(beta_q)*(rho-1/2)
p = Shiftmax10([s(i,delta_1), ..., s(i,delta_9), e])
Y_i[h,d] = sum_delta=1..9 p(i,delta) * C[h,delta,d]
C[h,null,d] = 0 exactly
```

`beta_m` and `beta_q` are per-head, initialized to zero, clipped to `[-1,1]`, and STE-quantized
on a `1/64` grid. Only nine codebook rows are registered. The effective tenth row is constructed
as a hard zero tensor every forward, so the null codeword cannot drift or receive gradients.

## H80 DN9

Each valid local edge `e=(i,delta)` terminates at `j(i,delta)`. Define the exact incoming set

```text
E_in(j) = {(i',delta') | j(i',delta')=j and the local edge is valid}
r(i,delta) = Shiftmax_{delta in Omega9}(s(i,delta))
c(i,delta) = Shiftmax_{e' in E_in(j(i,delta))}(s(e')) evaluated at e
a(i,delta) = Q1.7_unsigned(r(i,delta) * c(i,delta))
```

There is no final row renormalization. Incoming sets contain 4, 6, or 9 edges at spatial corners,
edges, or interior positions. The implementation precomputes fixed `162 x 9` source/destination
edge indices and never materializes an `N x N` attention matrix. The Q1.7 product uses an STE in
training and a `1/128` forward grid.

## Full30 contract

Both candidates independently warm-start from:

```text
neuron_experiments/H9_bipolar_self_attention/results/
date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/
checkpoint_epoch2.pth
```

The frozen protocol is DSEC crop `288 x 384`, batch 8, workers 8, AMP, cupy, 30 epochs, warmup
720 steps, milestones 20/25, and saved/evaluated epochs `0,4,9,14,19,24,28,29`. Standard
`valid825` reports AEE, AAE, PE/outlier metrics, total spikes, and energy. A short smoke run may
only check finite forward/backward execution and cannot eliminate either candidate.

Generate the configs and manifest without starting training:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h79_h80_round4_assignment_configs.py
```

Generated artifacts:

```text
neuron_experiments/H9_bipolar_self_attention/configs/generated/
  h79_allbinary_all12_cf10_match_code_w720_fastlr_full30.yml
  h80_allbinary_all12_dn9_match_code_w720_fastlr_full30.yml
  h79_h80_round4_assignment_full30_manifest.json
```

## Loading-chain acceptance

Run the frozen warm-start audit before queueing:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_round4_assignment_chain.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/h79_allbinary_all12_cf10_match_code_w720_fastlr_full30.yml \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/h80_allbinary_all12_dn9_match_code_w720_fastlr_full30.yml
```

Acceptance criteria:

- `ATLIFTernaryPSN=105`, attention modules `=12`, candidate modules `=12`.
- Frozen TTX checkpoint overlay keys `=210`; unexpected keys `=0`.
- H79 missing allowlist is exactly 12 codebooks plus 12 `beta_m/beta_q` tensors.
- H80 missing allowlist is exactly 12 codebooks.
- H79 audits 9 stored rows, 10 effective rows, and a fixed-zero null row in all 12 blocks.
- A newly registered same-mode state reloads with `strict=True`, missing 0, unexpected 0.

After training, audit the candidate checkpoint strictly:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_round4_assignment_chain.py \
  --trained <candidate.yml> <checkpoint_epoch29.pth> --output <strict_audit.json>
```

Any non-candidate warm-start missing key or any trained-checkpoint missing/unexpected key invalidates
the run.

## Queue

The idempotent runner is:

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/
run_round4_assignment_after_h78.py
```

It waits for `ALL COMPLETE ROUND3 MATCH-CODE`, regenerates and audits both configs, then executes
H79 followed by H80. Each candidate must finish full30, trained-checkpoint strict loading, and the
eight pre-registered valid825 evaluations before the next one starts. Its terminal marker is
`ALL COMPLETE ROUND4 ASSIGNMENT`; TTB-v2 and the unified 19-candidate deploy/operation audits wait
for that marker.

The frozen warm-start audit has been run successfully: both candidates install ATLIF105,
attention12 and candidate12 with checkpoint overlay210 and unexpected0. H79 has exactly 24
candidate-only missing keys (12 codebooks plus 12 beta tensors); H80 has exactly 12 codebooks.
Both registered same-mode states reload with strict missing0/unexpected0.
