# Autoresearch Dashboard: Sparse Pipeline

**Runs:** 5 | **Kept:** 1 | **Discarded:** 3 | **Crashed:** 1
**Baseline:** sops: 3.6219G (#1)
**Best:** sops: 3.6219G (#1, baseline)

| # | commit | sops | aee | status | description |
|---|--------|------|-----|--------|-------------|
| 1 | baseline | 3.6219G | 1.5848 | keep | PSN baseline (40 samples, upstream config) |
| 2 | baseline | 0 | 0 | crash | Reduce num_bins to 8 (T-dependent PSN weights incompatible) |
| 3 | baseline | 2.1649G (-40.2%) | 6.8188 | discard | spike_th=0.1: aggressive binarization destroys accuracy |
| 4 | baseline | 4.9255G (+36.0%) | 6.5880 | discard | spike_th=0.02: counterintuitively INCREASED spikes |
| 5 | baseline | 5.2294G (+44.4%) | 8.0746 | discard | norm_input=std: z-score norm incompatible with minmax-trained model |

**Conclusion so far**: Eval-only input preprocessing changes cannot achieve useful sparsity-accuracy tradeoffs. Need training-based approach.
