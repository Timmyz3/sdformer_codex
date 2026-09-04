#!/usr/bin/env bash
set -euo pipefail

m425_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m425_hw="$(cd "${m425_dc_root}/.." && pwd)"
m425_runner="$(realpath "${BASH_SOURCE[0]}")"
m425_run="${M425_SAIF_RUN_DIR:-${m425_hw}/results/m425r4_h67_balanced_selected_slice_direct_saif_r4_20260826}"
m425_vcs="/opt/synopsys/vcs/V-2023.12-SP1"
m425_subset="${m425_hw}/results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826"
m425_contract="contracts/m425r4_h67_balanced_selected_slice_direct_saif_contract_r4_20260826.json"
m425_filelist="dc_handoff/filelists/date_m425_h67_balanced_selected_slice_direct_saif_vcs.f"
m425_ucli="dc_handoff/scripts/m425_balanced_selected_slice_saif.ucli.tcl"
m425_saif="${m425_run}/m405_q32_elastic_selected_slice_rtl.saif"

m425_sha() { sha256sum "$1" | awk '{print $1}'; }
m425_expect() {
    local m425_path=$1
    local m425_expected=$2
    [[ -f "${m425_path}" ]] || exit 3
    [[ "$(m425_sha "${m425_path}")" == "${m425_expected}" ]] || exit 3
}

[[ ! -e "${m425_run}" ]] || exit 2
mkdir -p "${m425_run}/csrc"
m425_complete=0
trap 'm425_rc=$?; if [[ ${m425_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m425_rc}" >"${m425_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m425_hw}"

declare -A m425_expected=(
    ["${m425_contract}"]="46ba7fd6f811968b2897ef6daed873a42504e3d14f63428a9e369fefabe4cef4"
    ["contracts/m425r3_h67_balanced_selected_slice_direct_saif_contract_r3_20260826.json"]="be012271f975268f44e38c25a5fe1e99377248b24f7c6207c25924a058d6e1b0"
    ["contracts/m425r2_h67_balanced_selected_slice_direct_saif_contract_r2_20260826.json"]="dba1ba4d7ffb78ad942cf6bdf428d1e07cfd4a3675b2760a0b89d0c7606f1f8e"
    ["contracts/m425_h67_balanced_selected_slice_direct_saif_contract_r1_20260826.json"]="a94aeeaed00a9e0d07f6115647a17aa5212b25278c226f88013087ed08326fbe"
    ["results/m425_h67_balanced_selected_slice_direct_saif_r1_20260826/RUN_FAILED_OR_INCOMPLETE.txt"]="69f54defb365a7f6594d18a5f67d3c277fdaa852358c20dd914fb9d19b3d3a20"
    ["results/m425_h67_balanced_selected_slice_direct_saif_r1_20260826/sim.log"]="1d1a0dd3c5cdb7b542fdc2633ea0b2ba21b7853f0acba0d01d74531582280dbb"
    ["results/m425_h67_balanced_selected_slice_direct_saif_r1_20260826/compile.log"]="96d0becc146d82ca44bcb5c3b5f689c27c8470ba703753270f8d3a749d4b472c"
    ["results/m425r2_h67_balanced_selected_slice_direct_saif_r2_20260826/RUN_FAILED_OR_INCOMPLETE.txt"]="5eb4f7466faae3828d8bc726a5515e1e7f881094358d2af4cbfcbca1a96563f2"
    ["results/m425r2_h67_balanced_selected_slice_direct_saif_r2_20260826/sim.log"]="6f40c3cdd16ac144dd43f8c27240576cad3dc8844bad1a49d4d78ba8d787ca22"
    ["results/m425r3_h67_balanced_selected_slice_direct_saif_r3_20260826/RUN_FAILED_OR_INCOMPLETE.txt"]="1a7f3e36ad45c04edd603c56e89f363f03403b500b503993fadb9ea715cfd213"
    ["results/m425r3_h67_balanced_selected_slice_direct_saif_r3_20260826/sim.log"]="e3c8ce9d5a592fa059abb43422bafa8623eecbd72e63e0163a4e1f7ad42023cc"
    ["results/m425r3_h67_balanced_selected_slice_direct_saif_r3_20260826/m405_q32_elastic_selected_slice_rtl.saif"]="a513775bf5f7fae92b7b2d000483a4564dde333ff88f833b24ebe446d5614164"
    ["contracts/m425_h67_balanced_selected_slice_saif_subset_contract_r1_20260826.json"]="a0256ba6093e066ae57d10d2153ae102f35f5a16e3e9c9fddacc6ea2debb7ad5"
    ["results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826/m425_h67_saif_subset_manifest_r1.json"]="19ace81d1e7a0cec98ddf56c3169aa4da1c659c0592d22bf2f3ee6346390a582"
    ["results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826/m425_h67_phase_config_768.memh"]="08c03c014290a709bffe461d1cd77dfe42d02f02bb9b7756cce06996f18cb1de"
    ["results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826/m425_h67_runtime_rows_32.memh"]="666312a60d33b1ee0579b05bb3f9e6f9ddf1c86be8ddf5795af2348772f1780b"
    ["results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826/m425_h67_static_pwp_1281.memh"]="cd6ef528ba76ed4d470e14bd688d8b2c3b48b38fe15ff7d5d3bb8969ae0506cf"
    ["results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826/SHA256SUMS"]="80939e9a98f5ea29aa829f2945cd82881baaf2aaa3b60aef0631b7e3fb2cd032"
    ["results/m425_h67_balanced_selected_slice_saif_subset_r1_20260826/SHA256SUMS.seal.sha256"]="56964e3d7c8d424dc0d720da803e105a63aeb334b819d993c4c4bd36ee65601a"
    ["rtl_m414/m414_q32_balanced16_zero_stop_controller.sv"]="a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f"
    ["rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv"]="819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744"
    ["rtl_m405/m405_q32_elastic_selected_slice.sv"]="91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd"
    ["verif_m405/m405_q32_elastic_selected_slice_assertions.sv"]="71a190e373ec0016cc09314276d03f3b40d7e7731c108b3734bc29c384abfa4b"
    ["tb_m405/tb_m425_h67_balanced_selected_slice_direct_saif.sv"]="1ee804adc1ddaca965d9b0d4395d8f084e98009855b4f91ac456a0432871c12d"
    ["${m425_filelist}"]="68cc02403bb4ddbd1baa93337c3609cedd1ad0d459e9451829df6e69f6c23420"
    ["${m425_ucli}"]="6582e26df6c2965e1f6db9339f2695922ea3d90133dad8132b9d4abaac1bbaa3"
    ["dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/m416_m414_balanced_selected_slice_dc_receipt_r1.json"]="bedb903268d3e94c858e8177a383a46f35427cd9a1bdad3ad9ad398b4bc85c02"
    ["dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256"]="40fc119b1b6342f4473f5a0c1d12855b4944b1f932124f324ef69ed9c7576a79"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
m425_expect "${m425_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
: >"${m425_run}/preflight_sha_checks.txt"
for m425_path in "${!m425_expected[@]}"; do
    m425_observed="$(m425_sha "${m425_path}")"
    printf 'path=%s expected=%s observed=%s\n' "${m425_path}" \
        "${m425_expected[${m425_path}]}" "${m425_observed}" \
        >>"${m425_run}/preflight_sha_checks.txt"
    [[ "${m425_observed}" == "${m425_expected[${m425_path}]}" ]] || exit 10
done

(cd "${m425_subset}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256) \
    >"${m425_run}/subset_double_seal_check.log" 2>&1
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS \
    >"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS.seal.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m410r2_h67_q32_full_runtime_vcs_r2_20260826/RUN_MANIFEST.seal.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m408_h67_q32_static_codec_vcs_stimulus_r1_20260826/SHA256SUMS.seal.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m408_h67_q32_static_codec_full_vcs_r1_20260826/RUN_MANIFEST.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c results/m408_h67_q32_static_codec_full_vcs_r1_20260826/RUN_MANIFEST.seal.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
(cd results/m411_m410r2_full_runtime_vcs_independent_hammer_r1_20260826 && \
    sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.seal.sha256) \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1
sha256sum -c dc_handoff/runs/m416_m414_balanced_selected_slice_dc_3p000ns_r1_20260826/evidence_manifest.seal.sha256 \
    >>"${m425_run}/upstream_double_seal_check.log" 2>&1

sha256sum "${!m425_expected[@]}" "${m425_vcs}/bin/vcs" \
    >"${m425_run}/input_sha256.txt"
cp "${m425_contract}" "${m425_run}/contract.json"
export VCS_HOME="${m425_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
export M425_SAIF_FILE="${m425_saif}"

set +e
"${m425_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -debug_access+all \
    -cm assert -Mdir="${m425_run}/csrc" -f "${m425_filelist}" \
    -top tb_m425_h67_balanced_selected_slice_direct_saif \
    -o "${m425_run}/simv" >"${m425_run}/compile.log" 2>&1
m425_rc=$?
set -e
printf '%s\n' "${m425_rc}" >"${m425_run}/compile.rc"
[[ ${m425_rc} -eq 0 && -x "${m425_run}/simv" ]] || exit 20
if grep -Eiq 'Error-\[|^Error:|^Fatal:' "${m425_run}/compile.log"; then
    exit 21
fi

m425_start="$(date +%s)"
set +e
"${m425_run}/simv" -no_save -lca +M425_UCLI_SAIF_STOP \
    +ntb_random_seed=42520260826 \
    "+M425_CONFIG=${m425_subset}/m425_h67_phase_config_768.memh" \
    "+M425_ROWS=${m425_subset}/m425_h67_runtime_rows_32.memh" \
    "+M425_PWP=${m425_subset}/m425_h67_static_pwp_1281.memh" \
    -ucli -do "${m425_hw}/${m425_ucli}" -cm assert \
    -assert "report=${m425_run}/assert.report" \
    >"${m425_run}/sim.log" 2>&1
m425_rc=$?
set -e
m425_end="$(date +%s)"
printf '%s\n' "${m425_rc}" >"${m425_run}/sim.rc"
printf '%s\n' "$((m425_end-m425_start))" >"${m425_run}/sim_wall_seconds.txt"
[[ ${m425_rc} -eq 0 ]] || exit 22
[[ -s "${m425_saif}" ]] || exit 23
if grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]|protocol_error=1' \
        "${m425_run}/sim.log" "${m425_run}/assert.report"; then
    exit 24
fi
grep -Eq 'PASS M425 H67 balanced selected-slice direct-SAIF activity phases=64 rows=192000 pass0=192000 pass1=61285 early=11923 zero=93037 pop1=25755 pwp_rows=63067 low=504536 high=416630 narrow=87906 wide=416630 contributions=921166 reconstructed_lanes=48435456 metadata_mismatches=0 matcher_arithmetic_mismatches=0 codec_arithmetic_mismatches=0 reconstruction_mismatches=0 bitmap_mismatches=0 unknown_transactions=0 protocol_error=0 balanced_m414=true exploratory_pre_macro_power_activity=true paper_power_eligible=false power=false energy=false system_speedup=false headline=false measurement_cycles=[0-9]+' \
    "${m425_run}/sim.log" || exit 25

python3 - "${m425_run}" "${m425_saif}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
saif = Path(sys.argv[2])
sim = (run / "sim.log").read_text(errors="replace")
pattern = (
    r"PASS M425 H67 balanced selected-slice direct-SAIF activity "
    r"phases=(\d+) rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) "
    r"zero=(\d+) pop1=(\d+) pwp_rows=(\d+) low=(\d+) high=(\d+) "
    r"narrow=(\d+) wide=(\d+) contributions=(\d+) "
    r"reconstructed_lanes=(\d+).*measurement_cycles=(\d+)")
match = re.search(pattern, sim)
if not match:
    raise SystemExit("missing M425 PASS ledger")
values = [int(value) for value in match.groups()]
expected = [64,192000,192000,61285,11923,93037,25755,63067,
            504536,416630,87906,416630,921166,48435456]
if values[:14] != expected or values[14] <= 0:
    raise SystemExit("M425 PASS population drift")

text = saif.read_text(errors="strict")
duration_match = re.search(r"\(DURATION\s+(\d+(?:\.\d+)?)\)", text)
if not duration_match or float(duration_match.group(1)) <= 0:
    raise SystemExit("SAIF duration missing/nonpositive")
entry = re.compile(
    r"\(([^()\s]+)\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+"
    r"\(TX\s+(\d+)\)\s+\(TC\s+(\d+)\)", re.MULTILINE)
entries = [(name, int(t0), int(t1), int(tx), int(tc))
           for name,t0,t1,tx,tc in entry.findall(text)]
if len(entries) < 100:
    raise SystemExit("SAIF signal population unexpectedly small")
tx_entries = [item for item in entries if item[3] != 0]
allowed_tx_names = {"fifo_tile_q\\[1\\]", "fifo_narrow_q\\[1\\]"}
if {item[0] for item in tx_entries} - allowed_tx_names:
    raise SystemExit("SAIF contains unexpected nonzero-TX signals")
nonzero_entries = sum(item[4] > 0 for item in entries)
if nonzero_entries < 50:
    raise SystemExit("SAIF nonzero-toggle population unexpectedly small")
required_families = [
    "clk_core", "config_valid", "row_valid", "row_original",
    "result_valid", "result_center_id", "pwp_low_valid",
    "pwp_low_data", "pwp_high_valid", "contribution_valid",
    "contribution_data", "busy"]
families = {}
for family in required_families:
    matching = [item for item in entries if family in item[0]]
    families[family] = {
        "signals": len(matching),
        "signals_with_nonzero_tc": sum(item[4] > 0 for item in matching),
        "maximum_tc": max([item[4] for item in matching] or [0]),
    }
    if not matching or families[family]["maximum_tc"] <= 0:
        raise SystemExit("SAIF key family inactive/missing: " + family)
protocol = [item for item in entries if "protocol_error" in item[0]]
if not protocol or any(item[3] != 0 or item[4] != 0 for item in protocol):
    raise SystemExit("SAIF protocol_error missing, X, or toggled")

digest = hashlib.sha256(saif.read_bytes()).hexdigest()
receipt = {
    "schema": "m425r4_h67_balanced_selected_slice_direct_saif_receipt_v4",
    "status": "PASS_M425R4_H67_BALANCED_SELECTED_SLICE_SYNOPSYS_VCS_DIRECT_RTL_SAIF",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "dut": "m405_q32_elastic_selected_slice",
    "matcher": "M414 balanced16 exact tree",
    "population": {
        "phases": values[0], "source_rows": values[1],
        "pass0_tasks": values[2], "pass1_tasks": values[3],
        "early_stops": values[4], "zero_rows": values[5],
        "pop1_rows": values[6], "pwp_rows": values[7],
        "low_accepts": values[8], "high_accepts": values[9],
        "narrow_blocks": values[10], "wide_blocks": values[11],
        "contributions": values[12],
        "reconstructed_lanes": values[13],
        "measurement_cycles": values[14],
    },
    "mismatches": {
        "metadata": 0, "matcher_arithmetic": 0,
        "codec_arithmetic": 0, "reconstruction": 0,
        "bitmap": 0, "unknown_transactions": 0,
        "protocol": 0, "assertion": 0,
    },
    "saif": {
        "path": saif.name, "bytes": saif.stat().st_size,
        "sha256": digest, "duration_raw": float(duration_match.group(1)),
        "time_unit_seconds": 1e-9, "parsed_signal_entries": len(entries),
        "nonzero_toggle_signal_entries": nonzero_entries,
        "total_tx_duration_raw": sum(item[3] for item in tx_entries),
        "nonzero_tx_entries": [
            {"name": item[0], "tx_duration_raw": item[3],
             "toggle_count": item[4]} for item in tx_entries],
        "accepted_transaction_unknowns": 0,
        "key_families": families,
        "protocol_error_toggle_count": 0,
    },
    "ptpx_handoff": {
        "saif_scope":
            "tb_m425_h67_balanced_selected_slice_direct_saif.dut",
        "read_saif_strip_path":
            "tb_m425_h67_balanced_selected_slice_direct_saif/dut",
        "mapped_top": "m405_q32_elastic_selected_slice",
        "minimum_annotation_coverage_fraction": 0.95,
        "risk_summary": [
            "flattened M416 internal hierarchy and generated arrays",
            "M414 balanced implementation behind compatibility module name",
            "320 unobservable RTL debug bits removed by M416 DC",
        ],
    },
    "sim_wall_seconds": int((run / "sim_wall_seconds.txt").read_text()),
    "claim_boundary": {
        "exploratory_pre_macro_rtl_power_activity": True,
        "stratified_scope":
            "sample0, four operators, sixteen equidistant partitions each",
        "paper_power_eligible": False,
        "mapped_annotation_coverage": False,
        "power": False, "energy": False, "system_speedup": False,
        "paper_ppa_ready": False, "headline": False,
    },
}
(run / "m425r4_h67_balanced_selected_slice_direct_saif_receipt_r4.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m425_runner}" >"${m425_run}/runner_sha256.txt"
printf '%s\n' PASS_M425R4_H67_BALANCED_SELECTED_SLICE_SYNOPSYS_VCS_DIRECT_RTL_SAIF \
    >"${m425_run}/RUN_COMPLETE.txt"
find "${m425_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m425_run}/RUN_MANIFEST.sha256"
sha256sum "${m425_run}/RUN_MANIFEST.sha256" \
    >"${m425_run}/RUN_MANIFEST.seal.sha256"
m425_complete=1
echo "PASS M425 direct RTL SAIF sealed at ${m425_run}"
