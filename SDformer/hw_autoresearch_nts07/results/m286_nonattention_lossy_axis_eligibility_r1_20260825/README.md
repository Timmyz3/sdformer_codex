# M286 non-attention lossy-axis eligibility audit

This frozen-trace screen rejects two proposed main axes before spending GPU or
RTL time:

- G7 official-ATLIF amplitude gating: all 105 official modules are binary and
  emit either zero or their scalar threshold.  The threshold range is
  0.9999649525--1.0, so every precommitted theta through 0.125 produces exactly
  zero additional event sparsity.
- G8 whole-temporal-token FFN bypass: 100 exact FC1 payloads contain 5,520,000
  per-timestep tokens and 112,213,979 source events.  Per-timestep empty tokens
  are 8.6646%, but that work is already absent in the bit-sparse engine.  After
  grouping all ten timesteps, the N<=8 population is only 0.00308% of tokens
  and 0.0000971% of source events.  It is not a useful whole-FFN main axis.

The next measured candidate is source-by-destination-block bounded
contribution gating: a static keep bit may suppress one complete 96-lane
update while an explicit accumulator-error ledger tracks every omitted block.
Beta zero is the exact-engine subset.  ATLIF exact remaining-budget early stop
is the second candidate.

This is an eligibility audit only.  It does not admit accuracy, cycles, RTL,
system speedup, power, energy, physical SRAM, or a headline result.  The
operator shares in the JSON are activity-weighted MAC proxies, not the frozen
620,868,243-cycle envelope.

Replay with the pinned contract:

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python \
  hw_autoresearch_nts07/system_simulator/scripts/analyze_m286_nonattention_lossy_axis_eligibility.py \
  --contract hw_autoresearch_nts07/contracts/m286_nonattention_lossy_axis_eligibility_contract_r1_20260825.json \
  --output-dir /tmp/m286_replay
```

