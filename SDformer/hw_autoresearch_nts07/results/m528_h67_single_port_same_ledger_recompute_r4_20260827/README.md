# M528 same-ledger recompute

`m528_sample_major_distribution_r1.csv` is the only distribution with the frozen continuous four-operator pipeline and per-sample commit. `m528_operator_isolated_distribution_r1.csv` restarts the pipeline for every operator and omits commit; its rows are diagnostic and must not be summed.

All cycle results remain CPU-model, four-bottleneck-Conv, one-sequence values. They are not RTL, Synopsys PPA, energy, full-network speedup, or a DATE headline.
