#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m476_dual_slot_parent_queue_vcs_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M476 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m476_dual_slot_parent_queue_directed_vcs_contract_r1_20260826.json 6ee6d26eaec097698a668b4eda1868c188f5bbabfa05c5e7ceb87c3f03fcfe63
check_sha rtl_m476/m476_dual_slot_parent_queue_pipeline.sv c5aa9d0cceb4e353c2457afb6b554403d333720d84dec6fe1b0a982769893c55
check_sha verif_m476/m476_dual_slot_parent_queue_assertions.sv a4a30988c0321624caaf5776995a783378a00c6a49ac3babc2dc4191afb9e0f0
check_sha tb_m476/tb_m476_dual_slot_parent_queue_pipeline.sv e5a8eb47f97e8aaa61cb86639bf310f1410a0838ff24f0f2819ae865c196fcae
check_sha dc_handoff/filelists/date_m476_dual_slot_parent_queue_vcs.f 46ccd4dc579057dc649dbf394ea12173ba50af6cc997327fe670555cb190e833
check_sha results/m475_independent_hammer_review_r1_20260826/m475_independent_hammer_review_r1.json 459079052437bb2b260c087c35e64cfc75b1066a8cc156892e77d15fdf2b8dfe
check_sha results/m475_independent_hammer_review_r1_20260826/SHA256SUMS 916f5352ed70e9ddb74d276fd1d2d806c36f6b3359270d6fdcb73ab87a8961a9
check_sha results/m475_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 d9d8bad437fc2b3ee3aaf2ff1e9b8acdf104a9ad2ebe88e4eeee1a0d1e32389c
check_sha results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json a415f8474f3a351d123670c2d3691a6414f620e3d60848a9c51242802a6956e5
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
check_sha /opt/synopsys/vcs/V-2023.12-SP1/bin/vcs 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287

(cd results/m475_independent_hammer_review_r1_20260826 &&
  sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)

python3 - <<'PY'
import json
from pathlib import Path
root = Path('.')
m475 = json.loads((root/'results/m475_independent_hammer_review_r1_20260826/m475_independent_hammer_review_r1.json').read_text())
m473 = json.loads((root/'results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json').read_text())
assert m475['status'] == 'PASS_M475_INDEPENDENT_RECEIPT_BLIND_HAMMER_WITH_P1'
assert m475['verdict'] == 'CONDITIONAL_GO_MACRO_FEASIBILITY_PRIMETIME_FORMALITY_ONLY'
assert m475['p0_findings'] == []
assert m473['status'] == 'PASS_M473_CPU_DSE_NO_GO'
assert m473['admission']['performance_admitted'] is False
assert m473['claim_boundary']['system_speedup'] is False
PY

test ! -e "$RUN_DIR" || {
  echo "M476 result directory exists: $RUN_DIR" >&2
  exit 3
}
mkdir -p "$RUN_DIR"
cp contracts/m476_dual_slot_parent_queue_directed_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m476_dual_slot_parent_queue_directed_vcs_contract_r1_20260826.json \
  rtl_m476/m476_dual_slot_parent_queue_pipeline.sv \
  verif_m476/m476_dual_slot_parent_queue_assertions.sv \
  tb_m476/tb_m476_dual_slot_parent_queue_pipeline.sv \
  dc_handoff/filelists/date_m476_dual_slot_parent_queue_vcs.f \
  dc_handoff/scripts/run_vcs_m476_dual_slot_parent_queue_sva.sh \
  results/m475_independent_hammer_review_r1_20260826/m475_independent_hammer_review_r1.json \
  results/m475_independent_hammer_review_r1_20260826/SHA256SUMS \
  results/m475_independent_hammer_review_r1_20260826/SHA256SUMS.seal.sha256 \
  results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json \
  docs/359_DATE终局冻结_20260813.md \
  /opt/synopsys/vcs/V-2023.12-SP1/bin/vcs > "$RUN_DIR/input_sha256.txt"

vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -top tb_m476_dual_slot_parent_queue_pipeline \
  -f dc_handoff/filelists/date_m476_dual_slot_parent_queue_vcs.f \
  -o "$RUN_DIR/simv" -Mdir="$RUN_DIR/csrc" \
  2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"

python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, re, sys
r = pathlib.Path(sys.argv[1])
compile_log = (r/'compile.log').read_text()
sim_log = (r/'sim.log').read_text()
match = re.search(
    r'^PASS M476 directed issues=(\d+) rows=(\d+) forward=(\d+) '
    r'reads=(\d+) responses=(\d+) dual_enqueue=(\d+) full=(\d+) '
    r'fullconsume=(\d+) stalls=(\d+) b2b=(\d+) exact=(\d+) '
    r'partialbeats=(\d+) id_attacks=(\d+) overflow_attacks=(\d+)$',
    sim_log, re.M)
if not match:
    raise SystemExit('missing M476 PASS line')
names = ['issues','rows','forward','reads','responses','dual_enqueue','full',
         'fullconsume','stalls','b2b','exact','partialbeats','id_attacks',
         'overflow_attacks']
values = list(map(int, match.groups()))
expected = [6,5,1,4,4,1,2,2,9,2,2,2,1,1]
if values != expected:
    raise SystemExit(f'M476 directed count mismatch {values} != {expected}')
if re.search(r'Error-|Assertion failed|Fatal:', compile_log + '\n' + sim_log):
    raise SystemExit('M476 compile/assertion/fatal marker found')
covers = [
    'cp_forward', 'cp_macro_read', 'cp_read_response', 'cp_dual_enqueue',
    'cp_queue_full', 'cp_full_consume_no_prefetch_credit',
    'cp_back_to_back_completion', 'cp_output_stall',
    'cp_overflow_atomic_block'
]
cover_matches = {}
for name in covers:
    found = re.search(rf'\.sva\.{name}, .*? (\d+) match', sim_log)
    if not found or int(found.group(1)) < 1:
        raise SystemExit(f'vacuous or missing cover {name}')
    cover_matches[name] = int(found.group(1))

(r/'RUN_COMPLETE.txt').write_text(
    'PASS_M476_EXACT_SHA_DUAL_SLOT_PARENT_QUEUE_SYNOPSYS_VCS\n')
receipt = {
    'schema': 'm476_dual_slot_parent_queue_vcs_receipt_v1',
    'status': 'PASS_EXACT_SHA_SYNOPSYS_VCS_DUAL_SLOT_MICRO_FUNCTIONAL_ONLY',
    'tool': 'Synopsys VCS V-2023.12-SP1',
    'directed_counts': dict(zip(names, values)),
    'sva_cover_matches': cover_matches,
    'finding': {
        'two_slot_compacted_parent_queue': True,
        'same_cycle_response_plus_forward_dual_enqueue': True,
        'full_queue_prefetch_does_not_credit_same_cycle_consume': True,
        'held_prefetch_accepts_after_head_consume': True,
        'head_compaction_back_to_back_completion': True,
        'parent_id_mismatch_atomic_fail_closed': True,
        'overflow_atomic_fail_closed': True,
        'cumulative_debug_counters_in_production_rtl': False
    },
    'memory_geometry': {
        'external_parent_scratch': '64x1152b synchronous 1R1W = 9 KiB, excluded',
        'internal_response_slots': '2x1152b = 288 B',
        'external_resident_psum_if_64_rows': '64x1824b = 14.25 KiB, excluded'
    },
    'admission': {
        'm476_micro_functional': True,
        'm476_dc_sta': False,
        'm476_formality': False,
        'physical_scratch_or_psum_macro': False,
        'm473_performance_admitted': False,
        'power': False, 'energy': False, 'paper_ppa_ready': False,
        'full_network': False, 'system_speedup': False,
        'date_headline': False
    },
    'required_next_gate': (
        'Independent post-run hammer, then same-constraint M474-vs-M476 '
        '3.0 ns pre-macro DC/STA. Macro banking/PrimeTime/Formality remain '
        'separate mandatory gates.'
    )
}
(r/'m476_dual_slot_parent_queue_vcs_receipt_r1.json').write_text(
    json.dumps(receipt, indent=2) + '\n')
files = [p for p in sorted(r.iterdir()) if p.is_file() and
         p.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}]
(r/'SHA256SUMS').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
(r/'SHA256SUMS.seal.sha256').write_text(
    f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY

(cd "$RUN_DIR" && sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256)
echo "PASS_M476_DUAL_SLOT_PARENT_QUEUE_VCS run=$RUN_DIR"
