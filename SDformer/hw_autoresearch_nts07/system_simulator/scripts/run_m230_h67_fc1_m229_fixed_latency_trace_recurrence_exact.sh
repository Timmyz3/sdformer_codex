#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825"
test ! -e "$OUT"

python3 "$ROOT/system_simulator/scripts/analyze_m230_h67_fc1_m229_fixed_latency_trace_recurrence.py" \
  --manifest "$ROOT/results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json" \
  --payload-root "$ROOT/system_handoff/incoming/m51_capture_bundle_r2_20260823" \
  --m225-result "$ROOT/results/m225_h67_fc1_held_weight_context_multicast_screen_r1_20260825/m225_h67_fc1_held_weight_context_multicast_screen_r1.json" \
  --m225-seal "$ROOT/results/m225_h67_fc1_held_weight_context_multicast_screen_r1_20260825/SHA256SUMS" \
  --m226-seal "$ROOT/results/m226_m225_capacity_matched_reference_correction_r1_20260825/SHA256SUMS" \
  --m226-review-seal "$ROOT/results/m226_independent_hammer_review_r1_20260825/SHA256SUMS" \
  --m229-vcs-seal "$ROOT/results/m229_fc1_dual_held_prefetch_replay_directed_vcs_r1_exact_20260825/SHA256SUMS" \
  --m229-dc-seal "$ROOT/dc_handoff/runs/m229_fc1_dual_held_prefetch_replay_matched_dc_3p000ns_r1_20260825/evidence_manifest.sha256" \
  --m229-dc-run "$ROOT/dc_handoff/runs/m229_fc1_dual_held_prefetch_replay_matched_dc_3p000ns_r1_20260825" \
  --docs359 "$ROOT/docs/359_DATE终局冻结_20260813.md" \
  --output-dir "$OUT"

(
  cd "$OUT"
  sha256sum \
    ../../contracts/m230_h67_fc1_m229_fixed_latency_trace_recurrence_contract_r1_20260825.json \
    ../../system_simulator/scripts/analyze_m230_h67_fc1_m229_fixed_latency_trace_recurrence.py \
    ../../system_simulator/scripts/run_m230_h67_fc1_m229_fixed_latency_trace_recurrence_exact.sh \
    README.md \
    m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json \
    > SHA256SUMS
  sha256sum -c SHA256SUMS
)

echo "PASS M230 exact trace recurrence sealed at $OUT"
