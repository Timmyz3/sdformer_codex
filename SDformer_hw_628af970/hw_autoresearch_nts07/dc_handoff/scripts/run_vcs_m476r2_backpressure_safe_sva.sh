#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m476r2_backpressure_safe_parent_queue_vcs_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M476r2 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m476r2_backpressure_safe_parent_queue_vcs_contract_r1_20260826.json 3f6feec03ed76b635066fccbec07f1b275d138e3b85b21100e7f3351fa90179a
check_sha rtl_m476/m476_dual_slot_parent_queue_pipeline.sv c5aa9d0cceb4e353c2457afb6b554403d333720d84dec6fe1b0a982769893c55
check_sha rtl_m476r2/m476r2_backpressure_safe_parent_queue_pipeline.sv 4620d4666b44843be17306c984006a4423f43ad97103fd2a419aa8d901ccc37c
check_sha verif_m476/m476_dual_slot_parent_queue_assertions.sv a4a30988c0321624caaf5776995a783378a00c6a49ac3babc2dc4191afb9e0f0
check_sha verif_m476r2/m476r2_backpressure_safe_assertions.sv ea8327e07b2793cad36324d52b064b5e079b8dec3a07ad0339fb5534d87fa5e8
check_sha tb_m476r2/tb_m476r2_backpressure_safe_parent_queue.sv ccf9a63ae411aa78af32554d31533b249b8250ec971f3c93791dadff457f5e41
check_sha dc_handoff/filelists/date_m476r2_backpressure_safe_vcs.f 551aa56ae5aa3ac2d5c27199793d92f8271fb068b1fb46abc2ad39cf12bd8a27
check_sha results/m476_dual_slot_parent_queue_vcs_r1_20260826/m476_dual_slot_parent_queue_vcs_receipt_r1.json 71316ad06010f98fd5b23481acaffc20d84541fc1aa3c31bd923ff665fd7bf08
check_sha results/m476_dual_slot_parent_queue_vcs_r1_20260826/SHA256SUMS 2358767569000940c24f3c743120c9b840b47d52046bdc8494a95a256fc3055c
check_sha results/m476_dual_slot_parent_queue_vcs_r1_20260826/SHA256SUMS.seal.sha256 637bbc9bf2b8aabfed787e8a126b016ec73c3151538ec8e4d566501890e193ae
check_sha results/m476_independent_hammer_review_r1_20260826/m476_independent_hammer_review_r1.json f90201ff6d829c7236cf220a86bc04b98b73eeaae8e5bcf5ac2e5a59e60d924b
check_sha results/m476_independent_hammer_review_r1_20260826/m476_stale_same_address_prefetch_attack_receipt_r1.json ef02f1b3804ed9698de96666791a49bb5f54900c89d0244b30a4f34a2cb647d7
check_sha results/m476_independent_hammer_review_r1_20260826/SHA256SUMS 6de80a511acf8a93309e678fad0d9a7b93009fe196a20d7d6a7125fa537879ea
check_sha results/m476_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 aa938da2e447d9d34681dd75d0fb63956458e9d93f11b24704ae7b96338fc0a8
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
check_sha /opt/synopsys/vcs/V-2023.12-SP1/bin/vcs 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287

(cd results/m476_dual_slot_parent_queue_vcs_r1_20260826 &&
  sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)
(cd results/m476_independent_hammer_review_r1_20260826 &&
  sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)

python3 - <<'PY'
import json
from pathlib import Path
root = Path('.')
r1 = json.loads((root/'results/m476_dual_slot_parent_queue_vcs_r1_20260826/m476_dual_slot_parent_queue_vcs_receipt_r1.json').read_text())
hammer = json.loads((root/'results/m476_independent_hammer_review_r1_20260826/m476_independent_hammer_review_r1.json').read_text())
p0 = json.loads((root/'results/m476_independent_hammer_review_r1_20260826/m476_stale_same_address_prefetch_attack_receipt_r1.json').read_text())
assert r1['status'] == 'PASS_EXACT_SHA_SYNOPSYS_VCS_DUAL_SLOT_MICRO_FUNCTIONAL_ONLY'
assert hammer['status'] == 'FAIL_M476_INDEPENDENT_HAMMER_P0_STALE_PARENT'
assert hammer['verdict'] == 'REVISE_NO_SAME_CONSTRAINT_DC_COMPARE'
assert hammer['admission']['same_constraint_three_ns_dc_compare_allowed'] is False
assert p0['status'] == 'REPRODUCED_M476_STALE_SAME_ADDRESS_PREFETCH_P0'
assert p0['counterexample']['queued_old_lane0'] == 5
assert p0['counterexample']['committed_new_lane0'] == 1
PY

test ! -e "$RUN_DIR" || {
  echo "M476r2 result directory exists: $RUN_DIR" >&2
  exit 3
}
mkdir -p "$RUN_DIR"
cp contracts/m476r2_backpressure_safe_parent_queue_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m476r2_backpressure_safe_parent_queue_vcs_contract_r1_20260826.json \
  rtl_m476/m476_dual_slot_parent_queue_pipeline.sv \
  rtl_m476r2/m476r2_backpressure_safe_parent_queue_pipeline.sv \
  verif_m476/m476_dual_slot_parent_queue_assertions.sv \
  verif_m476r2/m476r2_backpressure_safe_assertions.sv \
  tb_m476r2/tb_m476r2_backpressure_safe_parent_queue.sv \
  dc_handoff/filelists/date_m476r2_backpressure_safe_vcs.f \
  dc_handoff/scripts/run_vcs_m476r2_backpressure_safe_sva.sh \
  results/m476_dual_slot_parent_queue_vcs_r1_20260826/m476_dual_slot_parent_queue_vcs_receipt_r1.json \
  results/m476_dual_slot_parent_queue_vcs_r1_20260826/SHA256SUMS \
  results/m476_dual_slot_parent_queue_vcs_r1_20260826/SHA256SUMS.seal.sha256 \
  results/m476_independent_hammer_review_r1_20260826/m476_independent_hammer_review_r1.json \
  results/m476_independent_hammer_review_r1_20260826/m476_stale_same_address_prefetch_attack_receipt_r1.json \
  results/m476_independent_hammer_review_r1_20260826/SHA256SUMS \
  results/m476_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 \
  docs/359_DATE终局冻结_20260813.md \
  /opt/synopsys/vcs/V-2023.12-SP1/bin/vcs > "$RUN_DIR/input_sha256.txt"

vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -top tb_m476r2_backpressure_safe_parent_queue \
  -f dc_handoff/filelists/date_m476r2_backpressure_safe_vcs.f \
  -o "$RUN_DIR/simv" -Mdir="$RUN_DIR/csrc" \
  2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"

python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, re, sys
r = pathlib.Path(sys.argv[1])
compile_log = (r/'compile.log').read_text()
sim_log = (r/'sim.log').read_text()
match = re.search(
    r'^PASS M476r2 stalled_raw_guard stalled=(\d+) reads=(\d+) '
    r'forward=(\d+) writes=(\d+) child_checks=(\d+) '
    r'stale_mismatches=(\d+) old=(\d+) new=(\d+)$', sim_log, re.M)
if not match:
    raise SystemExit('missing M476r2 PASS line')
names = ['stalled_cycles','scratch_reads','forward_events','row_writes',
         'child_value_checks','stale_mismatches','old_value','new_value']
values = list(map(int, match.groups()))
expected = [3,0,1,2,96,0,5,1]
if values != expected:
    raise SystemExit(f'M476r2 evidence mismatch {values} != {expected}')
if re.search(r'Error-|Assertion failed|Fatal:', compile_log + '\n' + sim_log):
    raise SystemExit('M476r2 compile/assertion/fatal marker found')

required_covers = [
    'cp_stalled_same_address_prefetch',
    'cp_release_to_new_value_forward',
    'cp_forward', 'cp_output_stall', 'cp_back_to_back_completion'
]
cover_matches = {}
for name in required_covers:
    found = re.search(rf'\.(?:base\.)?{name}, .*? (\d+) match', sim_log)
    if not found or int(found.group(1)) < 1:
        raise SystemExit(f'vacuous or missing cover {name}')
    cover_matches[name] = int(found.group(1))

(r/'RUN_COMPLETE.txt').write_text(
    'PASS_M476R2_EXACT_SHA_BACKPRESSURE_SAFE_SYNOPSYS_VCS\n')
receipt = {
    'schema': 'm476r2_backpressure_safe_parent_queue_vcs_receipt_v1',
    'status': 'PASS_M476R2_EXACT_SHA_STALE_RAW_P0_CLOSED_MICRO_ONLY',
    'tool': 'Synopsys VCS V-2023.12-SP1',
    'targeted_counts': dict(zip(names, values)),
    'sva_cover_matches': cover_matches,
    'finding': {
        'frozen_r1_p0_reproduced_and_bound': True,
        'stalled_same_address_prefetch_blocked': True,
        'release_uses_new_final_value_raw_forward': True,
        'ordinary_scratch_reads_in_attack_window': 0,
        'child_lanes_matching_new_value': 96,
        'stale_child_lanes': 0,
        'sealed_r1_directed_regression_inherited': True
    },
    'claim_boundary': {
        'r1_p0_closed_in_targeted_r2_micro': True,
        'r2_targeted_micro_functional': True,
        'independent_r2_hammer': False,
        'same_constraint_three_ns_dc_compare_allowed': False,
        'r2_dc_sta': False,
        'r2_formality': False,
        'physical_scratch_or_psum_macro': False,
        'm473_performance_admitted': False,
        'power': False, 'energy': False, 'paper_ppa_ready': False,
        'full_network': False, 'system_speedup': False,
        'date_headline': False
    },
    'required_next_gate': (
        'Independent receipt-blind rerun of the original old=5/new=1 '
        'P0 attack against M476r2. No DC is allowed before that verdict.'
    )
}
(r/'m476r2_backpressure_safe_parent_queue_vcs_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
files = [p for p in sorted(r.iterdir()) if p.is_file() and
         p.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}]
(r/'SHA256SUMS').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
(r/'SHA256SUMS.seal.sha256').write_text(
    f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY

(cd "$RUN_DIR" && sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)
echo "PASS_M476R2_BACKPRESSURE_SAFE_VCS run=$RUN_DIR"
