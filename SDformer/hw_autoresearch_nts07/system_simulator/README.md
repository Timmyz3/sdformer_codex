# DATE full-network system simulator

This directory converts the frozen H67 ep35 profile into a full-network
transaction ledger and a first-order cycle/traffic model. The immediate goal
is to replace the old attention-only Amdahl sensitivity with a reproducible
operator share.

The v0 model is intentionally not called cycle-accurate:

- all `Linear`, `Conv2d`, and `Conv3d` transactions come from the real
  100-sample checkpoint profile;
- ATLIF temporal-matrix work is reconstructed from the real module records;
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

## DSE envelope v1

After building v0, run:

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_h67_system_dse_envelope.py \
  --ledger hw_autoresearch_nts07/results/h67_ep35_full_network_ledger_20260821 \
  --config hw_autoresearch_nts07/system_simulator/configs/h67_system_v1_envelope.json \
  --output hw_autoresearch_nts07/results/h67_ep35_system_dse_envelope_20260821
```

v1 reports a non-attention parallelism sensitivity, the scale required for
target end-to-end speedups, object-fit SRAM residency envelopes, and external
memory-model requests. These are analytical bounds. They do not add
cycle-accurate scheduling, bank conflicts, CACTI results, or DRAMsim3 timing.

ATLIF is a `T x T` temporal matrix operator. The ledger therefore counts
`output elements x T` MACs. Its memory rows deliberately distinguish a
full-temporal-output buffer from the minimum one-output-row streaming
accumulator. Neither row is called membrane state, and neither implies that
the payload is read from and written to SRAM on every frame.

## Fair baseline registry

The baseline registry is fail-closed. It binds native Fixed2S and RQTB2S to
the same v0 ledger while leaving Prosperity and Phi-like rows blocked until
they have matched operator, frequency, memory, and operation-count adapters.
It does not import published performance numbers.

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_h67_baseline_registry.py \
  --ledger hw_autoresearch_nts07/results/h67_ep35_full_network_ledger_20260821 \
  --config hw_autoresearch_nts07/system_simulator/configs/h67_baseline_registry_v0.json \
  --output hw_autoresearch_nts07/results/h67_ep35_baseline_registry_20260821
```
