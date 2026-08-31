# M181 H67 FC2 hardware-simple pairing and K8 screen

Status: `PASS_EXACT_PAYLOAD_HARDWARE_SIMPLE_PAIRING_AND_K8_SCREEN_RTL_OPEN`.

This exact-payload screen uses the same 120 frozen H67 ep35 FC2 records and
the same finite dual-window recurrence as M179.  It asks two hardware questions:

1. Can four fixed two-bank lanes replace M180's global top-four-bank selector?
2. Does a fixed lane for each of the eight banks justify a K8 implementation?

## Result

The fixed-pair K4 simplification is rejected.  The predeclared XOR-4 wiring
`(0,4),(1,5),(2,6),(3,7)` needs 148,298,103 cycles.  That is 16.2382% slower
than M179's global-top4 K4 point and 2.7995% slower than the D1 K4 schedule.
Even the best of all seven XOR matchings is the same XOR-4 point, so fixed
pairing does not preserve the cross-entry gain.

The same-depth K8 screen is positive: one fixed lane per bank needs
97,607,807 cycles.  This is 1.307080x faster than the K4 top-four point,
1.476793x faster than D1 K4, and 4.344534x faster than the independently
optimized analytic K1 baseline.  The last number is an exact frozen-payload
schedule ratio, not physical or system speedup.

K8 currently reuses the K4-selected depths `D={2,4,8,8}`.  It still needs an
independent depth DSE, eight-lane RTL/VCS, matched Synopsys area/timing, eight
weight-bank responses and eight accumulator lanes.  Native descriptor and
token-directory generation are also excluded.

## Reproduction

From `hw_autoresearch_nts07` with the verified M51 payload extracted at the
given payload root:

```bash
python3 system_simulator/scripts/analyze_m181_h67_fc2_xor_pair_reservoir_dse.py \
  --manifest results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json \
  --payload-root /tmp/m176_payload.QWZFzA/hw_autoresearch_nts07/results/m51_h67_ep35_binary_input_trace_r2_gpu_20260823 \
  --m172-analyzer system_simulator/scripts/analyze_m172_h67_fc2_group_replay_cycles.py \
  --m179-analyzer system_simulator/scripts/analyze_m179_h67_fc2_dual_window_reservoir_dse.py \
  --m179-result results/m179_h67_fc2_dual_window_reservoir_exact_payload_dse_r1_20260824/m179_h67_fc2_dual_window_reservoir_exact_payload_dse.json \
  --docs359 docs/359_DATE终局冻结_20260813.md \
  --output /ABSOLUTE/NEW/OUTPUT/m181_h67_fc2_hardware_simple_pairing_and_k8_screen_exact_payload_dse.json
```

The analyzer refuses to overwrite an existing output and pins the payload
manifest, M172 analyzer, M179 analyzer/result and `docs/359` identities.
