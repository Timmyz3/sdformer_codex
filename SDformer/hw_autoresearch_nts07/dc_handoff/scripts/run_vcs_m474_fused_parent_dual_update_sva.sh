#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m474_fused_parent_dual_update_vcs_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M474 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m474_fused_parent_dual_update_directed_vcs_contract_r1_20260826.json 16bb571a17fd1531eac3fde7d8797da5e3794eb34a49b6f5d385be83d4e9150a
check_sha rtl_m474/m474_fused_parent_dual_update_pipeline.sv 30fdf778e5baea959c793c7b2f9d9e332364b84717f9ffd2f8ad74d85280d57c
check_sha verif_m474/m474_fused_parent_dual_update_assertions.sv ee039ba832f0a3b62035543e64253ffa932a18690e86161e39102ead9995695b
check_sha tb_m474/tb_m474_fused_parent_dual_update_pipeline.sv b9e2edbbcbc16b557ed7fab52066c6834931df0366f0ba6734ec308b4b3bd1da
check_sha dc_handoff/filelists/date_m474_fused_parent_dual_update_vcs.f 5443d7b5281a34266f9003034b1238b6e65c99015a98a643326a3cd48bf2d6a1
check_sha reviews/m474_fused_pipeline_preflight_peer_review_r1_20260826.md 2b24a42b02853df7feb4bd1fb6b1afb26fa857cdf6924faa8afa1aa4f0ffd5cb
check_sha results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json a415f8474f3a351d123670c2d3691a6414f620e3d60848a9c51242802a6956e5
check_sha results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_receipt_r1.json 7aff38ecc53d76cbf79bfc7d12561beba10258ea373a1de8c27b8568aa2d4d54
check_sha results/m473_h67_online_subset_live_pwp_r3_20260826/SHA256SUMS 8d6a6ba39c78896c0595c2f9c5211d46bca59f120ce2fee1af5101fc511839e1
check_sha results/m473_h67_online_subset_live_pwp_r3_20260826/SHA256SUMS.seal.sha256 3d5c37492180e273f70db7361ce5735d036ba0c941f86f646fdceaf20bde386a
check_sha results/m41_h67_ep35_bottleneck_int8_independent_hammer_review_r1_20260823/m41_independent_numeric_audit.json 3034d7f887139eadca4847af9fc7e080278942b93650eec956a2827961f0d0cd
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

python3 - <<'PY'
import json
from pathlib import Path
root = Path('.')
m473 = json.loads((root/'results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json').read_text())
m41 = json.loads((root/'results/m41_h67_ep35_bottleneck_int8_independent_hammer_review_r1_20260823/m41_independent_numeric_audit.json').read_text())
assert m473['status'] == 'PASS_M473_CPU_DSE_NO_GO'
assert m473['admission']['performance_admitted'] is False
assert m41['status'] == 'PASS_INDEPENDENT_CHECKPOINT_REEXPORT_ALL_WEIGHT_LAYOUT_MULTICAST_AND_S10_RAW_CONV_RECOMPUTE'
assert m41['accumulator_width']['checkpoint_tight_signed_bits'] == 19
PY

(cd results/m473_h67_online_subset_live_pwp_r3_20260826 &&
  sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)

test ! -e "$RUN_DIR" || {
  echo "M474 result directory exists: $RUN_DIR" >&2
  exit 3
}
mkdir -p "$RUN_DIR"
cp contracts/m474_fused_parent_dual_update_directed_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m474_fused_parent_dual_update_directed_vcs_contract_r1_20260826.json \
  rtl_m474/m474_fused_parent_dual_update_pipeline.sv \
  verif_m474/m474_fused_parent_dual_update_assertions.sv \
  tb_m474/tb_m474_fused_parent_dual_update_pipeline.sv \
  dc_handoff/filelists/date_m474_fused_parent_dual_update_vcs.f \
  reviews/m474_fused_pipeline_preflight_peer_review_r1_20260826.md \
  results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json \
  results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_receipt_r1.json \
  results/m473_h67_online_subset_live_pwp_r3_20260826/SHA256SUMS \
  results/m473_h67_online_subset_live_pwp_r3_20260826/SHA256SUMS.seal.sha256 \
  results/m41_h67_ep35_bottleneck_int8_independent_hammer_review_r1_20260823/m41_independent_numeric_audit.json \
  docs/359_DATE终局冻结_20260813.md > "$RUN_DIR/input_sha256.txt"

vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -top tb_m474_fused_parent_dual_update_pipeline \
  -f dc_handoff/filelists/date_m474_fused_parent_dual_update_vcs.f \
  -o "$RUN_DIR/simv" -Mdir="$RUN_DIR/csrc" \
  2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"

python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, re, sys
r = pathlib.Path(sys.argv[1])
log = (r/'sim.log').read_text()
match = re.search(
    r'^PASS M474 directed issues=(\d+) rows=(\d+) forward=(\d+) '
    r'reads=(\d+) stalls=(\d+) b2b=(\d+) oneahead=(\d+) exact=(\d+) '
    r'partialbeats=(\d+) overflow_attacks=(\d+)$', log, re.M)
if not match:
    raise SystemExit('missing M474 PASS line')
values = list(map(int, match.groups()))
expected = [6, 5, 2, 2, 5, 2, 1, 2, 2, 1]
if values != expected:
    raise SystemExit(f'M474 directed count mismatch {values} != {expected}')
if re.search(r'Error-|Assertion failed|Fatal:', log):
    raise SystemExit('M474 compile/assertion/fatal marker found')
covers = [
    'cp_forward', 'cp_macro_read', 'cp_exact_parent', 'cp_partial_parent',
    'cp_output_stall', 'cp_overflow_atomic_block', 'cp_stall_counter',
    'cp_back_to_back_completion', 'cp_one_ahead_macro_read'
]
cover_matches = {}
for name in covers:
    found = re.search(rf'\.sva\.{name}, .*? (\d+) match', log)
    if not found or int(found.group(1)) < 1:
        raise SystemExit(f'vacuous or missing cover {name}')
    cover_matches[name] = int(found.group(1))

(r/'RUN_COMPLETE.txt').write_text(
    'PASS_M474_EXACT_SHA_FUSED_PARENT_DUAL_UPDATE_SYNOPSYS_VCS\n')
receipt = {
    'schema': 'm474_fused_parent_dual_update_vcs_receipt_v1',
    'status': 'PASS_EXACT_SHA_SYNOPSYS_VCS_MICRO_FUNCTIONAL_ONLY',
    'tool': 'Synopsys VCS V-2023.12-SP1',
    'directed_counts': dict(zip(
        ['issue_accepts','row_completions','forward_hits','scratch_reads',
         'stall_cycles','back_to_back_completions','one_ahead_reads',
         'exact_parent_beats','partial_parent_beats','overflow_attacks'],
        values)),
    'sva_cover_matches': cover_matches,
    'numeric_domain': {
        'normal_beat': 'sign-extended INT8 on signed12 port',
        'maximum_unique_sources_per_row': 16,
        'row_final_bits': 12,
        'psum_final_bits': 19,
        'overflow_atomic_block_covered': True
    },
    'finding': {
        'one_ahead_registered_q_direct_consume': True,
        'same_address_raw_forward': True,
        'consume_plus_nonmatching_prefetch': True,
        'final_scratch_psum_dual_write_same_issue': True,
        'extra_completion_or_parent_read_bubble_in_directed_micro': 0
    },
    'admission': {
        'standalone_micro_rtl_functional': True,
        'cycle_per_residual_issue_if_environment_contract_holds': True,
        'm473_full_controller_rtl': False,
        'm473_performance_admitted': False,
        'physical_scratch_macro': False,
        'dc_sta': False, 'power': False, 'ppa': False,
        'full_network': False, 'system_speedup': False,
        'date_headline': False
    },
    'required_next_gate': (
        'Independent post-run hammer, then 3.0 ns TSMC28 pre-macro DC/STA; '
        'target 144-byte 1R1W macro timing/energy remains external.'
    )
}
(r/'m474_fused_parent_dual_update_vcs_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
files = [p for p in sorted(r.iterdir()) if p.is_file() and
         p.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}]
(r/'SHA256SUMS').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
(r/'SHA256SUMS.seal.sha256').write_text(
    f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY
