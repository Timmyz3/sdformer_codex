#!/usr/bin/env bash
set -euo pipefail
umask 002

[[ $# -eq 0 ]] || { echo "ERROR: no arguments accepted" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HW_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd -- "${HW_ROOT}/.." && pwd -P)"
RUNNER="$(readlink -f -- "${BASH_SOURCE[0]}")"
DESIGN=m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend
FILELIST="${HW_ROOT}/dc_handoff/filelists/iscas_m2027_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc.f"
RTL="${HW_ROOT}/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803="${HW_ROOT}/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
TCL="${HW_ROOT}/dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC="${HW_ROOT}/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M2026_DIR="${HW_ROOT}/reviews/m2026_m2025_m2018_c2_tsbg_b4_divfree_directed_vcs_result_hammer_r1_20260902"
M2026="${M2026_DIR}/review.json"
M1866_DIR="${HW_ROOT}/reviews/m1866_tsbg_ep34_same_io_b2_b4_b8_quickkill_independent_hammer_r1_20260902"
M1866="${M1866_DIR}/review.json"
SOURCE_REVIEW_DIR="${HW_ROOT}/reviews/m2028_m2027_m2018_c2_tsbg_b4_divfree_matched_dc_source_hammer_r1_20260902"
SOURCE_REVIEW="${SOURCE_REVIEW_DIR}/review.json"
DOCS359="${HW_ROOT}/docs/359_DATE终局冻结_20260813.md"
DC_SHELL=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
LMUTIL=/opt/synopsys/scl/2025.03/linux64/bin/lmutil
LICENSE_FILE=/opt/synopsys/Synopsys.dat
LICENSE_SERVER=27030@ic.ismd-nemo
SLOW_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db
FAST_DB=/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db
RESULT="${HW_ROOT}/dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
ATTEMPT="${HW_ROOT}/dc_handoff/runs/.m2029_m2018_c2_tsbg_b4_divfree_matched_dc_attempt_consumed"
WORK="${HW_ROOT}/dc_handoff/runs/.m2029_m2018_c2_tsbg_b4_divfree_matched_dc_work.$$"
LOCK="${HW_ROOT}/dc_handoff/runs/.m2029_m2018_c2_tsbg_b4_divfree_matched_dc_launch_lock"
WORK_ACTIVE=0

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_exact() {
  local expected="$1" path="$2"
  [[ -f "${path}" && ! -L "${path}" && "$(sha_file "${path}")" == "${expected}" ]] || {
    echo "ERROR: identity ${path}" >&2; exit 3;
  }
}
verify_dir_seal() {
  local dir="$1"
  [[ -d "${dir}" && ! -L "${dir}" ]] || return 1
  (cd -- "${dir}" && sha256sum -c SHA256SUMS >/dev/null &&
    sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
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
  local log="$1" receipt="$2" line start end block_sha count filtered
  local expected='Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl'
  local -a hits=()
  [[ -s "${log}" ]] || return 1
  mapfile -t hits < <(grep -nE '^Error:|^Fatal:' "${log}" || true)
  [[ "${#hits[@]}" -eq 1 && "${hits[0]#*:}" == "${expected}" ]] || return 1
  start=${hits[0]%%:*}; end=$((start + 15))
  [[ "${start}" -ge 2 && "${start}" -le 64 ]] || return 1
  [[ "$(sed -n "$((start - 1))p" "${log}")" == Initializing... ]] || return 1
  [[ "$(sed -n "$((end + 1))p" "${log}")" == "Current time:"* ]] || return 1
  block_sha="$(sed -n "${start},${end}p" "${log}" | sha256sum | awk '{print $1}')"
  [[ "${block_sha}" == 3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1 ]] || return 1
  count="$(grep -iEc 'error:|fatal:' "${log}" || true)"
  [[ "${count}" -eq 1 ]] || return 1
  filtered="${receipt}.filtered.tmp"
  awk -v start="${start}" -v end="${end}" 'NR < start || NR > end {print}' \
    "${log}" >"${filtered}"
  if grep -Eq '^Error:|^Fatal:|^(Warning|Information):.*\((TIM-209|OPT-150)\)' \
      "${filtered}"; then
    rm -f -- "${filtered}"; return 1
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

sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21 "${RTL}"
sha_exact cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156 "${M803}"
sha_exact c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe "${TCL}"
sha_exact 808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5 "${SDC}"
sha_exact 6033a37048dbba8e5d4ed555da9c1e81748330c657ca5a4bc080c1924bc2ac47 "${M2026}"
sha_exact 6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830 "${M1866}"
sha_exact dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 "${DOCS359}"
[[ "$(wc -l <"${FILELIST}")" -eq 2 ]] || exit 3
verify_dir_seal "${M2026_DIR}"
verify_dir_seal "${M1866_DIR}"
verify_dir_seal "${SOURCE_REVIEW_DIR}"

[[ -n "${M2029_EXPECTED_RUNNER_SHA256:-}" &&
   "$(sha_file "${RUNNER}")" == "${M2029_EXPECTED_RUNNER_SHA256}" ]] || exit 3
[[ -n "${M2029_EXPECTED_SOURCE_REVIEW_SHA256:-}" &&
   "$(sha_file "${SOURCE_REVIEW}")" == "${M2029_EXPECTED_SOURCE_REVIEW_SHA256}" ]] || exit 3
/usr/libexec/platform-python3.6 -I - "${SOURCE_REVIEW}" "${RUNNER}" "${FILELIST}" \
  "${RTL}" "${M803}" "${M2026}" "${M1866}" "${DOCS359}" <<'PY'
from __future__ import print_function
import hashlib, json, sys
from pathlib import Path
review, runner, filelist, rtl, m803, m2026, m1866, docs359 = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads(review.read_text())
assert r['status'].startswith('PASS_M2028')
assert r['score_over_100'] >= 95
assert r['severity_counts'] == {'p0': 0, 'p1': 0, 'p2': 0}
for key, path in (('runner_sha256', runner), ('filelist_sha256', filelist),
                  ('m2018_rtl_sha256', rtl), ('m803_rtl_sha256', m803),
                  ('m2026_review_sha256', m2026), ('m1866_review_sha256', m1866),
                  ('docs359_sha256', docs359)):
    assert r['identity'][key] == sha(path), (key, r['identity'].get(key), sha(path))
assert r['authorization'] == {'license_queries': 1, 'dc_shell_runs': 2,
                              'all_other_eda_runs': 0, 'automatic_retry': False}
PY

[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" && ! -e "${LOCK}" ]] || exit 4
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
[[ "${mem_available}" -ge 50331648 && $((commit_limit-committed)) -ge 33554432 ]] || exit 4

cd -- "${REPO_ROOT}"
mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"
printf 'status=M2029_ATTEMPT_CONSUMED\nlicense_queries=1\ndc_shell_runs=2\naxes=ordinary_lru4,tsbg_b4\nretry=false\n' \
  >"${ATTEMPT}/ATTEMPT_CONSUMED.txt"
seal_dir "${ATTEMPT}"
WORK_ACTIVE=1
"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f Design-Compiler \
  >"${WORK}/license_preflight.log" 2>&1
cp -- "${FILELIST}" "${WORK}/input_filelist.f"

axis_names=(ordinary_lru4 tsbg_b4)
axis_modes=(0 1)
for index in 0 1; do
  axis="${axis_names[$index]}"; mode="${axis_modes[$index]}"
  axis_dir="${WORK}/${axis}"
  mkdir -- "${axis_dir}"
  /usr/bin/timeout --signal=TERM --kill-after=60s 21600s \
    env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TMPDIR=/tmp \
      SNPSLMD_LICENSE_FILE="${LICENSE_SERVER}" LM_LICENSE_FILE="${LICENSE_FILE}" \
      DESIGN_NAME="${DESIGN}" HW_ROOT="${HW_ROOT}" RTL_FILELIST="${FILELIST}" \
      LIB_DB="${SLOW_DB}" MIN_LIB_DB="${FAST_DB}" SDC_FILE="${SDC}" \
      OUTPUT_DIR="${axis_dir}" OPERATING_CONDITION=ssg0p9v125c \
      CLOCK_PERIOD_NS=3.000 ELAB_PARAMETERS="SCHEDULE_MODE=${mode}" \
      "${DC_SHELL}" -f "${TCL}" >"${axis_dir}/dc.log" 2>&1
  printf '0\n' >"${axis_dir}/dc.rc"
  validate_dc_log "${axis_dir}/dc.log" "${axis_dir}/bootstrap_log_whitelist_receipt.txt"
  for artifact in TCL_PASS_TERMINAL.txt reports/area.rpt reports/qor.rpt \
      reports/timing_setup.rpt reports/timing_hold_diagnostic.rpt \
      reports/constraint_setup.rpt reports/precompile_loop_gate.rpt \
      reports/constraint_max_capacitance.rpt reports/constraint_max_transition.rpt \
      reports/constraint_max_fanout.rpt reports/port_count.txt \
      "netlist/${DESIGN}_mapped.v" "netlist/${DESIGN}_mapped.sdc" \
      "netlist/${DESIGN}.ddc" "netlist/${DESIGN}.svf"; do
    [[ -s "${axis_dir}/${artifact}" && ! -L "${axis_dir}/${artifact}" ]] || exit 6
  done
  grep -Fxq 'TIM-209=0' "${axis_dir}/reports/precompile_loop_gate.rpt"
  grep -Fxq 'OPT-150=0' "${axis_dir}/reports/precompile_loop_gate.rpt"
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_max_capacitance.rpt"
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_max_transition.rpt"
  grep -Fq 'This design has no violated constraints.' "${axis_dir}/reports/constraint_max_fanout.rpt"
done

/usr/libexec/platform-python3.6 -I - "${WORK}" "${RUNNER}" "${SOURCE_REVIEW}" \
  "${FILELIST}" "${RTL}" "${M803}" "${M2026}" "${M1866}" <<'PY'
from __future__ import print_function
import hashlib, json, re, sys
from pathlib import Path
root, runner, review, filelist, rtl, m803, m2026, m1866 = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def last_slack(path):
    text = path.read_text(errors='replace')
    values = re.findall(r'slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)', text)
    if not values:
        raise SystemExit('missing slack in %s' % path)
    return min(float(value) for value in values)
axes = {}
for name, mode in [('ordinary_lru4', 0), ('tsbg_b4', 1)]:
    d = root / name
    text = (d / 'reports/area.rpt').read_text(errors='replace')
    match = re.search(r'Total cell area:\s*([0-9.]+)', text)
    if not match:
        raise SystemExit('missing area: ' + name)
    axes[name] = {
        'schedule_mode': mode,
        'area_um2': float(match.group(1)),
        'setup_wns_ns': last_slack(d / 'reports/timing_setup.rpt'),
        'hold_diagnostic_wns_ns': last_slack(d / 'reports/timing_hold_diagnostic.rpt'),
        'public_port_count': int((d / 'reports/port_count.txt').read_text().strip()),
        'mapped_netlist_sha256': sha(d / 'netlist' / 'm2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v'),
        'dc_log_sha256': sha(d / 'dc.log')}
base, cand = axes['ordinary_lru4'], axes['tsbg_b4']
ratio = cand['area_um2'] / base['area_um2']
receipt = {
  'schema': 'm2029_m2018_tsbg_divfree_matched_dc_receipt_r1_v1',
  'status': 'PASS_RAW_M2029_M2018_TSBG_DIVFREE_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW',
  'axes': axes,
  'comparison': {
    'tsbg_over_ordinary_logic_area_ratio': ratio,
    'tsbg_logic_area_overhead_fraction': ratio - 1.0,
    'both_setup_met': base['setup_wns_ns'] >= 0 and cand['setup_wns_ns'] >= 0,
    'public_port_count_equal': base['public_port_count'] == cand['public_port_count'],
    'm2026_directed_bundle_request_reduction_fraction': 0.75,
    'm2026_directed_scalar_request_reduction_fraction': 0.75,
    'm1866_cpu_premodel_speedup_not_upgraded_to_rtl': True},
  'candidate_gate': {
    'absolute_area_delta_at_most_10pct': abs(ratio - 1.0) <= 0.10,
    'both_setup_met': base['setup_wns_ns'] >= 0 and cand['setup_wns_ns'] >= 0,
    'public_ports_equal': base['public_port_count'] == cand['public_port_count']},
  'identity': {
    'runner_sha256': sha(runner), 'source_review_sha256': sha(review),
    'filelist_sha256': sha(filelist), 'm2018_rtl_sha256': sha(rtl),
    'm803_rtl_sha256': sha(m803), 'm2026_review_sha256': sha(m2026),
    'm1866_review_sha256': sha(m1866)},
  'execution': {'license_queries': 1, 'dc_shell_runs': 2, 'automatic_retry': False},
  'claim_boundary': {
    'logic_only_pre_macro': True, 'ideal_clock': True, 'wireload': 'ZeroWireload',
    'hold_closed': False, 'power': False, 'energy': False,
    'exact_rtl_cycle_speedup': False, 'same_area': False,
    'system_speedup': False, 'paper_ppa_ready': False,
    'production_g48_dynamically_verified': False,
    'cpu_premodel_2p533808x_upgraded_by_dc': False,
    'physical_schedule_ablation_not_full_conventional_baseline': True,
    'state_arrays_synthesized_as_standard_cells': True}}
if not all(receipt['candidate_gate'].values()):
    raise SystemExit('candidate gate failed: %r' % receipt['candidate_gate'])
(root / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
PY

printf 'RAW_PASS_M2029_M2018_TSBG_DIVFREE_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW\n' \
  >"${WORK}/RUN_COMPLETE.txt"
seal_dir "${WORK}"
mv -T -- "${WORK}" "${RESULT}"
WORK_ACTIVE=0
rmdir -- "${LOCK}"
trap - EXIT INT TERM HUP
printf 'RAW_PASS_M2029_M2018_TSBG_DIVFREE_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW\n'

