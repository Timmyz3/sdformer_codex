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

The optional `h67_system_v2_multisample_vcs.json` config replaces the original
sample0 attention anchor with the fresh 10-sample VCS mean for each stage. It
still expands one selected T450 window per block and therefore remains an
analytical frame envelope, not full-frame RTL.

Run from the repository root:

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_h67_ep35_full_network_ledger.py \
  --config hw_autoresearch_nts07/system_simulator/configs/h67_system_v0.json \
  --output hw_autoresearch_nts07/results/h67_ep35_full_network_ledger_20260821
```

For the stronger multisample attention anchor, use:

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_h67_ep35_full_network_ledger.py \
  --config hw_autoresearch_nts07/system_simulator/configs/h67_system_v2_multisample_vcs.json \
  --output hw_autoresearch_nts07/results/h67_ep35_full_network_ledger_v2_multisample_20260821
```

The output supplies `operator_transactions.csv`, `atlif_transactions.csv`,
`activation_objects.csv`, `system_summary.json`, and a paper-boundary report.
These are the inputs for the subsequent CACTI/DRAMsim3 and baseline adapters.

## Local + Motion full-system contract

The dual-line contract routes ordinary binary `Linear`/`Conv2d` tiles to two
exact arithmetic choices.  The Local path accumulates selected weight columns
for the current nonzero sources.  The Motion path retains the previous output
and applies signed weight columns only for `0->1` and `1->0` input transitions.
The selector chooses the lower-work path and periodically refreshes through
Local.  Attention Q/K projections remain Local-eligible, but are Motion-
ineligible after window partitioning moves the temporal axis.  Real s1/s10
ordered traces are now available; they measure operation work, while cycle
speedup still requires the datapath schedule and memory model.

The synthesizable chain now consists of
`qfit_dual_line_tile_selector.sv`, `qfit_dual_line_source_streamer.sv`,
`qfit_dual_line_tile_executor.sv`, and
`qfit_dual_line_stateful_tile_top.sv`.  The stateful top derives counts and
tokens from the same valid-masked bitmap, keys the exact prior bitmap/Acc32 by
stream and tile identity, and falls back to Local at boundary, refresh, shape,
seed, or protocol mismatch.  The default 256-bit/16-lane configuration has
independent and integrated VCS/SVA regressions plus paired DC/Formality.  That
M1 diagnostic uses flop-inferred weight storage and is not paper PPA.

M2B adds `qfit_local_banked_multisource_engine.sv`: a per-bank 8-word hierarchical
frontier, modulo-banked p1/p2/p4/p8 source issue, an external synchronous
weight request/response interface, and a per-lane Acc32 adder tree.  On the same
20,000 real checkpoint bitmaps its exact bank-conflict issue-beat ratios are
1.000x/1.766x/2.953x/4.646x.  All four variants pass Synopsys VCS/SVA, 3 ns
premacro logic DC, and Formality.  DC excludes external weight SRAM macro area,
so these ratios remain engine evidence rather than system speedup.  Exact
command-to-output-fire ratios, including randomized output backpressure, are
1.000x/1.730x/2.806x/4.224x.

The real-tile memory ledger v2 reports Motion with both shared output/activation
state and explicit-copy state.  It never allocates Motion state in Local-only.
Shared state gives only a small p1/p2 sampled benefit, while explicit copying
costs cycles; neither reaches the Motion survival gate.  The earlier result that
charged both output and state Acc32 writes is withdrawn.

The ordered s10 source-issue envelope is generated with:

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_dual_line_source_issue_envelope.py \
  --identity H67 <h67-dual-trace.csv> <h67-operator-runtime.csv> \
  --identity Local5 <local-dual-trace.csv> <local-operator-runtime.csv> \
  --h67-ledger <h67-ledger-directory> \
  --output <output-directory>
```

It covers 31 Motion-comparable operators plus 24 Local-only attention Q/K
operators.  Its resident-weight/no-conflict ratios are scheduling envelopes,
not measured system latency.

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/build_h67_dual_line_full_system.py \
  --ledger hw_autoresearch_nts07/results/h67_ep35_full_network_ledger_v2_multisample_20260821 \
  --config hw_autoresearch_nts07/system_simulator/configs/h67_dual_line_full_system_v0.json \
  --output hw_autoresearch_nts07/results/h67_dual_line_full_system_v0_20260821
```

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
