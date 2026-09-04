#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m478_m476r2_full_wrapper_regression_vcs_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M478 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m478_m476r2_full_wrapper_regression_vcs_contract_r1_20260826.json 66c4edeaa7c3723cbb24261be224df10e7a99099d4cfee9960ea167f28a76338
check_sha rtl_m476/m476_dual_slot_parent_queue_pipeline.sv c5aa9d0cceb4e353c2457afb6b554403d333720d84dec6fe1b0a982769893c55
check_sha rtl_m476r2/m476r2_backpressure_safe_parent_queue_pipeline.sv 4620d4666b44843be17306c984006a4423f43ad97103fd2a419aa8d901ccc37c
check_sha verif_m476/m476_dual_slot_parent_queue_assertions.sv a4a30988c0321624caaf5776995a783378a00c6a49ac3babc2dc4191afb9e0f0
check_sha verif_m476r2/m476r2_backpressure_safe_assertions.sv ea8327e07b2793cad36324d52b064b5e079b8dec3a07ad0339fb5534d87fa5e8
check_sha tb_m478/tb_m478_m476r2_full_regression.sv 6cfe1fff291bd21a0480616d4822fa3a40a695bd2dd5522e5ff2a52a4fe306e9
check_sha dc_handoff/filelists/date_m478_m476r2_full_regression_vcs.f c2890cfd175dd4a92219142e6174100434ca45b32f711b0b493552d2b1e8e1e2
check_sha results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json 36e99e859e77f0f61e12eb238360a7828d794854bab8934ea12810020c558e5c
check_sha results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/SHA256SUMS c9b4860c4b2a6b72ffbe0394824ad9b3efedb748a5d74f697d43e1ae31ed1cf6
check_sha results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/SHA256SUMS.seal.sha256 25b5a356a00fadf620d2d45dade5482e664387d4569aad9ab7ea83d682776fc0
check_sha results/m476r2_independent_hammer_review_r1_20260826/m476r2_independent_hammer_review_r1.json 1ec68d48bded366763a4bf7a3307ce153332be45ef7b61b1005e38b9a923bda1
check_sha results/m476r2_independent_hammer_review_r1_20260826/SHA256SUMS 9f8134efd1e7079fe5d94928d60dad55896f1bb7959b5812a9d20490aba69d06
check_sha results/m476r2_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 b7bc89490a96b103c737889764677c8313c32c07263eef3e1f654158d08896c0
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
check_sha /opt/synopsys/vcs/V-2023.12-SP1/bin/vcs 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287

(cd results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826 &&
  sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
(cd results/m476r2_independent_hammer_review_r1_20260826 &&
  sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)

python3 - <<'PY'
import json
from pathlib import Path
targeted = json.loads(Path('results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json').read_text())
hammer = json.loads(Path('results/m476r2_independent_hammer_review_r1_20260826/m476r2_independent_hammer_review_r1.json').read_text())
assert targeted['status'] == 'PASS_M476R2_EXACT_SHA_STALE_RAW_P0_CLOSED_MICRO_ONLY'
assert targeted['sva_cover_matches']['cp_stalled_same_address_prefetch'] >= 1
assert targeted['sva_cover_matches']['cp_release_to_new_value_forward'] >= 1
assert targeted['targeted_counts']['scratch_reads'] == 0
assert targeted['targeted_counts']['stale_mismatches'] == 0
assert hammer['status'] == 'PASS_M476R2_INDEPENDENT_HAMMER_P0_CLOSED_WITH_P1'
assert hammer['p0_findings'] == []
assert hammer['admission']['same_constraint_three_ns_dc_compare_allowed'] is True
assert hammer['admission']['m473_performance_admitted'] is False
PY

test ! -e "$RUN_DIR" || {
  echo "M478 result directory exists: $RUN_DIR" >&2
  exit 3
}
mkdir -p "$RUN_DIR"
cp contracts/m478_m476r2_full_wrapper_regression_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m478_m476r2_full_wrapper_regression_vcs_contract_r1_20260826.json \
  rtl_m476/m476_dual_slot_parent_queue_pipeline.sv \
  rtl_m476r2/m476r2_backpressure_safe_parent_queue_pipeline.sv \
  verif_m476/m476_dual_slot_parent_queue_assertions.sv \
  verif_m476r2/m476r2_backpressure_safe_assertions.sv \
  tb_m478/tb_m478_m476r2_full_regression.sv \
  dc_handoff/filelists/date_m478_m476r2_full_regression_vcs.f \
  dc_handoff/scripts/run_vcs_m478_m476r2_full_wrapper_regression_sva.sh \
  results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json \
  results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/SHA256SUMS \
  results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/SHA256SUMS.seal.sha256 \
  results/m476r2_independent_hammer_review_r1_20260826/m476r2_independent_hammer_review_r1.json \
  results/m476r2_independent_hammer_review_r1_20260826/SHA256SUMS \
  results/m476r2_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 \
  docs/359_DATE终局冻结_20260813.md \
  /opt/synopsys/vcs/V-2023.12-SP1/bin/vcs > "$RUN_DIR/input_sha256.txt"

vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -top tb_m478_m476r2_full_regression \
  -f dc_handoff/filelists/date_m478_m476r2_full_regression_vcs.f \
  -o "$RUN_DIR/simv" -Mdir="$RUN_DIR/csrc" \
  2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"

python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, re, sys
r = pathlib.Path(sys.argv[1])
compile_log = (r/'compile.log').read_text()
sim_log = (r/'sim.log').read_text()
match = re.search(
    r'^PASS M478 M476r2 full issues=(\d+) rows=(\d+) forward=(\d+) '
    r'reads=(\d+) responses=(\d+) dual_enqueue=(\d+) full=(\d+) '
    r'fullconsume=(\d+) stalls=(\d+) b2b=(\d+) exact=(\d+) '
    r'partialbeats=(\d+) id_attacks=(\d+) overflow_attacks=(\d+)$',
    sim_log, re.M)
if not match:
    raise SystemExit('missing M478 PASS line')
names = ['issues','rows','forward','reads','responses','dual_enqueue','full',
         'fullconsume','stalls','b2b','exact','partialbeats','id_attacks',
         'overflow_attacks']
values = list(map(int, match.groups()))
expected = [6,5,1,4,4,1,2,2,9,2,2,2,1,1]
if values != expected:
    raise SystemExit(f'M478 full-regression count mismatch {values} != {expected}')
if re.search(r'Error-|Assertion failed|Fatal:', compile_log + '\n' + sim_log):
    raise SystemExit('M478 compile/assertion/fatal marker found')

base_covers = [
    'cp_forward', 'cp_macro_read', 'cp_read_response', 'cp_dual_enqueue',
    'cp_queue_full', 'cp_full_consume_no_prefetch_credit',
    'cp_back_to_back_completion', 'cp_output_stall',
    'cp_overflow_atomic_block'
]
base_cover_matches = {}
for name in base_covers:
    found = re.search(rf'\.sva\.base\.{name}, .*? (\d+) match', sim_log)
    if not found or int(found.group(1)) < 1:
        raise SystemExit(f'vacuous or missing full-suite base cover {name}')
    base_cover_matches[name] = int(found.group(1))

targeted_absent = {}
for name in ['cp_stalled_same_address_prefetch',
             'cp_release_to_new_value_forward']:
    found = re.search(rf'\.sva\.{name}, .*? (\d+) match', sim_log)
    if not found or int(found.group(1)) != 0:
        raise SystemExit(f'unexpected full-suite targeted-cover identity {name}')
    targeted_absent[name] = 0

targeted = json.loads(pathlib.Path(
    'results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826/'
    'm476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json').read_text())
targeted_covers = {
    name: targeted['sva_cover_matches'][name]
    for name in ['cp_stalled_same_address_prefetch',
                 'cp_release_to_new_value_forward']
}

(r/'RUN_COMPLETE.txt').write_text(
    'PASS_M478_EXACT_SHA_M476R2_FULL_WRAPPER_REGRESSION_SYNOPSYS_VCS\n')
receipt = {
    'schema': 'm478_m476r2_full_wrapper_regression_vcs_receipt_v1',
    'status': 'PASS_M478_EXACT_SHA_M476R2_FULL_WRAPPER_REGRESSION',
    'tool': 'Synopsys VCS V-2023.12-SP1',
    'full_suite_counts': dict(zip(names, values)),
    'full_suite_base_cover_matches': base_cover_matches,
    'full_suite_targeted_cover_matches_expected_zero': targeted_absent,
    'sealed_targeted_r2_cover_matches': targeted_covers,
    'coverage_composition': {
        'full_r1_concurrency_suite_through_r2_wrapper': True,
        'separate_stalled_raw_p0_suite': True,
        'combined_required_covers_nonvacuous': True
    },
    'finding': {
        'r2_wrapper_transparent_for_complete_r1_suite': True,
        'macro_read_response_reexecuted': True,
        'dual_enqueue_reexecuted': True,
        'queue_full_and_full_consume_reexecuted': True,
        'overflow_and_id_fail_closed_reexecuted': True,
        'independent_review_p1_targeted_regression_breadth_closed': True
    },
    'admission': {
        'r2_full_wrapper_regression': True,
        'r2_targeted_p0_regression': True,
        'r2_dc_sta': False,
        'r2_formality': False,
        'physical_scratch_or_psum_macro': False,
        'm473_performance_admitted': False,
        'power': False, 'energy': False, 'paper_ppa_ready': False,
        'full_network': False, 'system_speedup': False,
        'date_headline': False
    },
    'required_next_gate': (
        'Independent receipt-blind M478 hammer. DC cost comparison remains '
        'separate; exact-SHA Formality and physical macro/PrimeTime gates remain.'
    )
}
(r/'m478_m476r2_full_wrapper_regression_vcs_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
files = [p for p in sorted(r.iterdir()) if p.is_file() and
         p.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}]
(r/'SHA256SUMS').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
(r/'SHA256SUMS.seal.sha256').write_text(
    f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY

(cd "$RUN_DIR" && sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)
echo "PASS_M478_M476R2_FULL_WRAPPER_REGRESSION_VCS run=$RUN_DIR"
