# DATE full-network system simulator

This directory converts the frozen H67 ep35 profile into a full-network
transaction ledger and a first-order cycle/traffic model. The immediate goal
is to replace the old attention-only Amdahl sensitivity with a reproducible
operator share.

The v0 model is intentionally not called cycle-accurate:

- all `Linear`, `Conv2d`, and `Conv3d` transactions come from the real
  100-sample checkpoint profile;
- ATLIF state-matrix work is reconstructed from the real module records;
- the attention normalization-to-projection slice uses the sealed RTL-calibrated
  per-stage frame model;
- residual/elementwise scheduling, SRAM conflicts, DMA overlap, CACTI, and
  DRAMsim3 are not calibrated yet;
- the materialize-all traffic number is an upper proxy, not paper energy.

Run from the repository root:

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_h67_ep35_full_network_ledger.py \
  --config hw_autoresearch_nts07/system_simulator/configs/h67_system_v0.json \
  --output hw_autoresearch_nts07/results/h67_ep35_full_network_ledger_20260821
```

The output supplies `operator_transactions.csv`, `atlif_transactions.csv`,
`activation_objects.csv`, `system_summary.json`, and a paper-boundary report.
These are the inputs for the subsequent CACTI/DRAMsim3 and baseline adapters.
