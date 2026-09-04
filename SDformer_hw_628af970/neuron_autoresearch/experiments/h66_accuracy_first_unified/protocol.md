# H66 Accuracy-First Unified Attention Protocol

## Frozen constraints

- All 105 wrappers use one-sided binary official ATLIF. One-sided firing does not require a negative threshold.
- All 12 encoder attention blocks use one formula. No partial or TX/SC mixed deployment.
- `gate*K` and `weights@K` are allowed. The prohibited carrier is native QKFormer `K*sn2_q(sum(Q))` retained before a second gate.
- Screen on DSEC; preserve all old modules, configs, checkpoints, and results.

## Candidate paradigms

| id | paradigm | formula-level description | hardware implication |
|---|---|---|---|
| TTX reference | factorized alpha-XNOR selector | one score/token, Shiftmax, then `gate*K` | linear token path; current RTL |
| H66a | full alpha-XNOR matrix | pairwise binary Q/K alpha-XNOR, row Shiftmax, then `weights@K` | window score matrix and value accumulation |
| H66b | Hamming linear attention | binary-to-bipolar Q/K, then `Q(K^T K)` | no token matrix; `D x D` accumulators |
| H66c | TP-TTX | self/paired-time alpha-XNOR, 2-way Shiftmax, weighted K | one temporal K buffer; two comparisons/token |
| H66d | LR-TTX | self/up/down/left/right alpha-XNOR, 5-way Shiftmax, weighted K | spatial line buffers; five comparisons/token |
| H66e conditional | TP-TTX self-prior | H66c plus dyadic `+1` self-score bias, giving an approximately 2:1 prior | one score increment; run only if TP valid825 AAE remains high |
| hold | STAtten temporal block | combine temporal/spatial tokens in fixed chunks | independent V, temporal buffer, `D x D` state |
| hold | lateral inhibition | feed-forward selector plus feedback suppression | feedback state/control; not a scalar penalty |

## Accuracy-first gates

1. Train 120 steps from TTX epoch2 and evaluate valid10. Record activity without using it as an early stop.
2. Continue to 360 steps/valid40 when AEE is at most 2.2 or the accuracy trend is strongly improving.
3. Promote a valid40 candidate only when AEE is at most 1.65 with a stable downward trend.
4. Final evidence uses valid825 and requires AEE within about 5% of NB0 and at least 20% fewer spikes.

## Reproduction

```bash
cd /root/private_data/work/sdformer_codex/SDformer
python neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h66_accuracy_first_unified_configs.py
python neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py \
  --config generated/h66a_allbinary_all12_axnor_matrix_shiftmax_s120.yml \
  --steps 120 \
  --prev-runid neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth \
  --valid-samples 10 --no-promote-valid40 --tag h66a_accuracy_first
```

The profile log must report 105 ATLIF wrappers, 12 patched attention modules, and no missing/unexpected overlay-owned checkpoint keys.

For another server, the wrapper below regenerates configs and uses the same TTX checkpoint/load protocol:

```bash
bash neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h66_accuracy_first.sh \
  h66c_allbinary_all12_tp_ttx_s120 120
```
