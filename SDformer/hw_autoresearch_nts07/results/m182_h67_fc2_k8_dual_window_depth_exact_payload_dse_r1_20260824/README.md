# M182 H67 FC2 K8 dual-window depth DSE

Status: `PASS_EXACT_PAYLOAD_K8_DEPTH_DSE_RTL_OPEN`.

M182 evaluates D={1,2,4,8,16,32} for an eight-fixed-bank-lane FC2 frontend
over all 120 frozen H67 ep35 FC2 payload records.  Each group can issue one
source from every nonempty bank, so its drain length is the maximum bank
population and it does not need M180's global top-four bank sorter.

## Exact result

The in-sample per-stage minimum is D={2,4,16,32}, totaling 95,410,406 cycles.
This is 4.444593x relative to the independently optimized analytic K1 schedule
and 1.337183x relative to M179's selected K4 schedule.

The bounded D={2,4,8,8} point totals 97,607,807 cycles: only 2.3031% slower,
while its maximum two-buffer bitmap payload is 1,536 bits rather than 6,144
bits for D32.  It remains the recommended first RTL point and retains a
4.344534x K1/K8 analytic schedule ratio.

The selected depths are not holdout-validated.  Stage 2 has an exact D16/D32
tie; the lower-storage D16 point wins the deterministic tie break.  No native
descriptor producer, token-directory generation, weight-bank response,
eight-lane accumulator, power or complete-FC2 physical measurement is present.

## Reproduction

From `hw_autoresearch_nts07`, run the pinned analyzer with a new output path:

```bash
python3 system_simulator/scripts/analyze_m182_h67_fc2_k8_dual_window_depth_dse.py \
  --manifest results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json \
  --payload-root /tmp/m176_payload.QWZFzA/hw_autoresearch_nts07/results/m51_h67_ep35_binary_input_trace_r2_gpu_20260823 \
  --m172-analyzer system_simulator/scripts/analyze_m172_h67_fc2_group_replay_cycles.py \
  --m179-analyzer system_simulator/scripts/analyze_m179_h67_fc2_dual_window_reservoir_dse.py \
  --m179-result results/m179_h67_fc2_dual_window_reservoir_exact_payload_dse_r1_20260824/m179_h67_fc2_dual_window_reservoir_exact_payload_dse.json \
  --docs359 docs/359_DATE终局冻结_20260813.md \
  --output /ABSOLUTE/NEW/OUTPUT/m182_h67_fc2_k8_dual_window_depth_exact_payload_dse.json
```
