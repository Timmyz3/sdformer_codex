#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
DESIGN=m1880_c2_tsbg_b4_real_channel_signed_frontend
FILELIST="${HW_ROOT}/dc_handoff/filelists/iscas_m1992_c2_tsbg_b4_matched_two_axis_logic_only_dc.f"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC="${HW_ROOT}/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M1990_DIR="${HW_ROOT}/reviews/m1990_m1986_c2_tsbg_b4_parseable_vcs_result_hammer_r1_20260902"
M1990="${M1990_DIR}/review.json"
M1866_DIR="${HW_ROOT}/reviews/m1866_tsbg_ep34_same_io_b2_b4_b8_quickkill_independent_hammer_r1_20260902"
M1866="${M1866_DIR}/review.json"
SOURCE_REVIEW_DIR="${HW_ROOT}/reviews/m1993_m1992_c2_tsbg_b4_matched_dc_source_hammer_r1_20260902"
SOURCE_REVIEW="${SOURCE_REVIEW_DIR}/review.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
DC_SHELL=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
SLOW_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
FAST_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
RESULT="${HW_ROOT}/dc_handoff/runs/m1992_m1990_c2_tsbg_b4_matched_two_axis_logic_only_dc_r1_20260902"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m1992_m1990_c2_tsbg_b4_matched_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m1992_m1990_c2_tsbg_b4_matched_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m1992_m1990_c2_tsbg_b4_matched_dc_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 3; }
  [[ "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: SHA mismatch ${path}" >&2; exit 3;
  }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || exit 3
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null) || exit 3
}
seal_dir() {
  local dir="$1"
  (cd -- "${dir}" &&
    find -P . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
      -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- >SHA256SUMS &&
    sha256sum -- SHA256SUMS >SHA256SUMS.seal.sha256 &&
    sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
}
validate_dc_log() {
  local log="$1" receipt="$2" error_line start end block_sha error_count filtered
  local -a error_lines=()
  error_line='Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl'
  [[ -s "${log}" ]] || return 1
  mapfile -t error_lines < <(grep -nE '^Error:|^Fatal:' "${log}" || true)
  [[ "${#error_lines[@]}" -eq 1 ]] || return 1
  [[ "${error_lines[0]#*:}" == "${error_line}" ]] || return 1
  start=${error_lines[0]%%:*}
  [[ "${start}" =~ ^[0-9]+$ && "${start}" -ge 2 && "${start}" -le 64 ]] || return 1
  end=$((start + 15))
  [[ "$(sed -n "$((start - 1))p" "${log}")" == Initializing... ]] || return 1
  [[ "$(sed -n "$((end + 1))p" "${log}")" == "Current time:"* ]] || return 1
  block_sha="$(sed -n "${start},${end}p" "${log}" | sha256sum | awk '{print $1}')"
  [[ "${block_sha}" == 3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1 ]] || return 1
  error_count="$(grep -iEc 'error:|fatal:' "${log}" || true)"
  [[ "${error_count}" -eq 1 ]] || return 1
  filtered="${receipt}.filtered.tmp"
  awk -v start="${start}" -v end="${end}" \
    'NR < start || NR > end { print }' "${log}" >"${filtered}"
  if grep -Eq '^Error:|^Fatal:|^(Warning|Information):.*\((TIM-209|OPT-150)\)' \
      "${filtered}"; then
    rm -f -- "${filtered}"
    return 1
  fi
  rm -f -- "${filtered}"
  printf 'status=PASS_EXACT_SINGLE_BOOTSTRAP_BLOCK_WHITELIST\nblock_start_line=%s\nblock_end_line=%s\nblock_sha256=%s\nother_error_fatal_tim209_opt150_count=0\n' \
    "${start}" "${end}" "${block_sha}" >"${receipt}"
}
on_exit() {
  local rc=$?
  set +e
  if [[ ${rc} -ne 0 && ${WORK_ACTIVE} -eq 1 && -d "${WORK}" && ! -L "${WORK}" ]]; then
    printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=%s\nretry=false\n' "${rc}" \
      >"${WORK}/RUN_FAILED_OR_INCOMPLETE.txt"
    seal_dir "${WORK}" || true
    mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine" || true
  fi
  rmdir -- "${LOCK}" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

sha_exact e50027edea9470bda92e5f34f590c1c13f236e6f46b836ef4b5028465fe94f4c "${FILELIST}"
sha_exact c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe "${TCL}"
sha_exact 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5 "${SDC}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
sha_exact 8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05 \
  "${HW_ROOT}/rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 \
  "${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
sha_exact e2935ed23f2e2b24798ea6b6ab1f098fcd356e1969e31279793a063c9b07b80c "${M1990}"
sha_exact 6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830 "${M1866}"
[[ "$(wc -l <"${FILELIST}")" -eq 2 ]] || exit 3
verify_dir_seal "${M1990_DIR}"
verify_dir_seal "${M1866_DIR}"
verify_dir_seal "${SOURCE_REVIEW_DIR}"

[[ -n "${M1992_EXPECTED_RUNNER_SHA256:-}" &&
   "$(sha_file "${RUNNER}")" == "${M1992_EXPECTED_RUNNER_SHA256}" ]] || {
  echo "ERROR: caller must pin exact M1992 runner SHA" >&2; exit 3;
}
[[ -n "${M1992_EXPECTED_REVIEW_SHA256:-}" &&
   "$(sha_file "${SOURCE_REVIEW}")" == "${M1992_EXPECTED_REVIEW_SHA256}" ]] || {
  echo "ERROR: caller must pin exact M1993 review SHA" >&2; exit 3;
}
/usr/libexec/platform-python3.6 -I - "${SOURCE_REVIEW}" "${RUNNER}" "${FILELIST}" "${M1990}" "${M1866}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path
review, runner, filelist, m1990, m1866 = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads(review.read_text())
v = json.loads(m1990.read_text())
assert r['status'].startswith('PASS_M1993')
assert r['score_over_100'] >= 95
assert r['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
assert r['identity']['runner_sha256'] == sha(runner)
assert r['identity']['filelist_sha256'] == sha(filelist)
assert r['authorization'] == {'dc_shell_runs': 2, 'all_other_eda_runs': 0}
assert v['status'].startswith('PASS_M1990')
assert json.loads(m1866.read_text())['status'].startswith('PASS_INDEPENDENT_REPLAY')
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || {
  echo "ERROR: M1992 namespace is not fresh" >&2; exit 4;
}
/usr/libexec/platform-python3.6 -I - <<'PY'
from __future__ import print_function
import os
from pathlib import Path
blocked = {'dc_shell', 'dc_shell-t', 'common_shell_exec', 'common_shell_exe'}
hits = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid != os.getuid():
            continue
        comm = (p / 'comm').read_text().strip()
        argv = {Path(x.decode(errors='replace')).name
                for x in (p / 'cmdline').read_bytes().split(b'\0') if x}
    except Exception:
        continue
    if comm in blocked or blocked & argv:
        hits.append((p.name, comm))
if hits:
    raise SystemExit('same-UID DC collision: %r' % hits)
PY
mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
commit_limit="$(awk '/^CommitLimit:/ {print $2}' /proc/meminfo)"
committed="$(awk '/^Committed_AS:/ {print $2}' /proc/meminfo)"
[[ "${mem_available}" -ge 67108864 && $((commit_limit-committed)) -ge 33554432 ]] || {
  echo "ERROR: memory/commit gate" >&2; exit 4;
}

mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M1992_ATTEMPT_CONSUMED\ndc_shell_runs=2\naxes=ordinary_lru4,tsbg_b4\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler \
  >"${WORK}/license_preflight.log" 2>&1
cp -- "${FILELIST}" "${WORK}/input_filelist.f"

axis_names=(ordinary_lru4 tsbg_b4)
axis_modes=(0 1)
for index in 0 1; do
  axis="${axis_names[$index]}"
  mode="${axis_modes[$index]}"
  axis_dir="${WORK}/${axis}"
  mkdir -- "${axis_dir}"
  /usr/bin/timeout --signal=TERM --kill-after=60s 21600s \
  env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
    SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE="${LICENSE_FILE}" \
    DESIGN_NAME="${DESIGN}" HW_ROOT="${HW_ROOT}" RTL_FILELIST="${FILELIST}" \
    LIB_DB="${SLOW_DB}" MIN_LIB_DB="${FAST_DB}" SDC_FILE="${SDC}" \
    OUTPUT_DIR="${axis_dir}" OPERATING_CONDITION=ssg0p9v125c \
    CLOCK_PERIOD_NS=3.000 \
    ELAB_PARAMETERS="SCHEDULE_MODE=${mode}" \
    "${DC_SHELL}" -f "${TCL}" >"${axis_dir}/dc.log" 2>&1
  printf '0\n' >"${axis_dir}/dc.rc"
  validate_dc_log "${axis_dir}/dc.log" \
    "${axis_dir}/bootstrap_log_whitelist_receipt.txt"
  for artifact in TCL_PASS_TERMINAL.txt reports/area.rpt reports/qor.rpt \
      reports/timing_setup.rpt reports/timing_hold_diagnostic.rpt \
      reports/constraint_setup.rpt reports/precompile_loop_gate.rpt \
      reports/constraint_max_capacitance.rpt \
      reports/constraint_max_transition.rpt \
      reports/constraint_max_fanout.rpt \
      reports/port_count.txt "netlist/${DESIGN}_mapped.v" \
      "netlist/${DESIGN}_mapped.sdc" "netlist/${DESIGN}.ddc" \
      "netlist/${DESIGN}.svf"; do
    [[ -s "${axis_dir}/${artifact}" && ! -L "${axis_dir}/${artifact}" ]] || {
      echo "ERROR: missing ${axis}/${artifact}" >&2; exit 6;
    }
  done
  grep -Fxq 'TIM-209=0' "${axis_dir}/reports/precompile_loop_gate.rpt"
  grep -Fxq 'OPT-150=0' "${axis_dir}/reports/precompile_loop_gate.rpt"
  grep -Fq 'This design has no violated constraints.' \
    "${axis_dir}/reports/constraint_max_capacitance.rpt"
  grep -Fq 'This design has no violated constraints.' \
    "${axis_dir}/reports/constraint_max_transition.rpt"
  grep -Fq 'This design has no violated constraints.' \
    "${axis_dir}/reports/constraint_max_fanout.rpt"
done

/usr/libexec/platform-python3.6 -I - "${WORK}" "${RUNNER}" "${SOURCE_REVIEW}" \
    "${FILELIST}" "${M1990}" "${M1866}" <<'PY'
from __future__ import print_function
import hashlib, json, re, sys
from pathlib import Path
root, runner, review, filelist, m1990, m1866 = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def last_slack(path):
    text = path.read_text(errors='replace')
    matches = re.findall(r'slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)', text)
    if not matches:
        raise SystemExit('missing slack in %s' % path)
    return min(float(value) for value in matches)

axes = {}
for name, mode in [('ordinary_lru4', 0), ('tsbg_b4', 1)]:
    d = root / name
    area_text = (d / 'reports/area.rpt').read_text(errors='replace')
    area_match = re.search(r'Total cell area:\s*([0-9.]+)', area_text)
    if not area_match:
        raise SystemExit('missing total cell area: ' + name)
    axes[name] = {
        'schedule_mode': mode,
        'area_um2': float(area_match.group(1)),
        'setup_wns_ns': last_slack(d / 'reports/timing_setup.rpt'),
        'hold_diagnostic_wns_ns': last_slack(d / 'reports/timing_hold_diagnostic.rpt'),
        'public_port_count': int((d / 'reports/port_count.txt').read_text().strip()),
        'dc_log_sha256': sha(d / 'dc.log'),
        'mapped_netlist_sha256': sha(d / 'netlist/m1880_c2_tsbg_b4_real_channel_signed_frontend_mapped.v')
    }
base = axes['ordinary_lru4']
tsbg = axes['tsbg_b4']
area_ratio = tsbg['area_um2'] / base['area_um2']
port_equal = base['public_port_count'] == tsbg['public_port_count']
setup_met = base['setup_wns_ns'] >= 0.0 and tsbg['setup_wns_ns'] >= 0.0
receipt = {
    'schema': 'm1992_c2_tsbg_b4_matched_two_axis_logic_only_dc_receipt_r1_v1',
    'status': 'PASS_RAW_M1992_C2_TSBG_B4_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW',
    'axes': axes,
    'comparison': {
        'tsbg_over_ordinary_logic_area_ratio': area_ratio,
        'tsbg_logic_area_overhead_fraction': area_ratio - 1.0,
        'public_port_count_equal': port_equal,
        'both_setup_met': setup_met,
        'hold_is_diagnostic_not_closed': True,
        'directed_weight_bundle_reduction_fraction_from_m1990': 0.75,
        'directed_scalar_bank_request_reduction_fraction_from_m1990': 0.75,
        'cpu_premodel_speedup_not_upgraded_to_rtl': True
    },
    'candidate_gate': {
        'area_overhead_at_most_10pct': area_ratio <= 1.10,
        'public_ports_equal': port_equal,
        'both_setup_met': setup_met,
        'candidate_go': area_ratio <= 1.10 and port_equal and setup_met
    },
    'identity': {
        'runner_sha256': sha(runner),
        'source_review_sha256': sha(review),
        'filelist_sha256': sha(filelist),
        'm1990_review_sha256': sha(m1990),
        'm1866_cpu_premodel_review_sha256': sha(m1866)
    },
    'execution': {'dc_shell_runs': 2, 'retry': False, 'all_other_eda_runs': 0},
    'claim_boundary': {
        'logic_only_pre_macro': True,
        'ideal_clock': True,
        'wireload': 'ZeroWireload',
        'hold_closed': False,
        'exact_rtl_cycle_speedup': False,
        'power': False,
        'energy': False,
        'same_area': False,
        'system_speedup': False,
        'paper_ppa_ready': False,
        'both_axes_own_bundle4_candidate_state': True,
        'physical_schedule_ablation_not_conventional_baseline_ppa': True,
        'full_conventional_baseline_area_priced': False,
        'state_arrays_synthesized_as_standard_cells': True,
        'layer_private_cache_domain': True,
        'weight_domain_transition_requires_reset_or_rebind': True,
        'cross_layer_flush_or_rebind_implemented': False,
        'production_g48_dynamically_verified': False,
        'exact_cycle_ratio': False
    }
}
(root / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
(root / 'RUN_COMPLETE.txt').write_text(
    'status=PASS_RAW_M1992_C2_TSBG_B4_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW\n'
    'dc_shell_runs=2\nretry=false\nindependent_result_review_required=true\n')
PY

seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'PASS_RAW_M1992_C2_TSBG_B4_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW\n'
