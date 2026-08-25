#!/usr/bin/env bash
set -euo pipefail

# Bootstrap creates an immutable exact-SHA snapshot.  DC, fresh DDC-based STA,
# Formality, audits and receipt generation run only from that snapshot.

m43_repo=/home/zhumd/work/sdformer_codex/SDformer
m43_live_runner="$m43_repo/hw_autoresearch_nts07/dc_handoff/scripts/run_m43_r3_exact_sha_synopsys.sh"
m43_run=${M43_RUN_PATH:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m43_parent_delta_p8_l96_r3_exact_sha_synopsys_3p000ns_r2_20260823}
m43_dc_launcher=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m43_dc_resolved=/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
m43_fm_launcher=/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell
m43_fm_resolved=/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell
m43_dc_sha=23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m43_fm_sha=aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m43_finalized=0

m43_sha() { sha256sum "$1" | awk '{print $1}'; }

m43_synopsys_active() {
    pgrep -f '^/opt/synopsys/.*/(common_shell_exec|dc_shell|dc_shell-t|fm_shell|fm_shell_exec)( |$)' \
        >/dev/null 2>&1
}

m43_partial_seal() {
    local m43_rc=$?
    if [[ -d "$m43_run" && "$m43_finalized" -eq 0 ]]; then
        set +e
        {
            echo "status=FAIL_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$m43_rc"
            echo "candidate_changed=false"
            echo "paper_ppa_ready=false"
            echo "system_speedup_admitted=false"
            echo "power_or_energy_admitted=false"
            date -u +"sealed_utc=%Y-%m-%dT%H:%M:%SZ"
        } > "$m43_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
        (
            cd "$m43_run" || exit 1
            find . -type f ! -name PARTIAL_EVIDENCE.sha256 \
                ! -name PARTIAL_EVIDENCE_CHECK.raw.log -print0 \
                | sort -z | xargs -0 sha256sum > PARTIAL_EVIDENCE.sha256
            sha256sum --strict -c PARTIAL_EVIDENCE.sha256 \
                > PARTIAL_EVIDENCE_CHECK.raw.log 2>&1
        )
        find "$m43_run" -type f -exec chmod 0444 {} +
        find "$m43_run" -type d -exec chmod 0555 {} +
        set -e
    fi
    exit "$m43_rc"
}

if [[ "${1:-}" == "--bootstrap" ]]; then
    [[ "$#" -eq 3 ]] || { echo "usage: $0 --bootstrap MANIFEST EXPECTED_SHA" >&2; exit 2; }
    m43_manifest="$2"
    m43_manifest_expected="$3"
    [[ ! -e "$m43_run" ]] || { echo "refusing to overwrite fixed run: $m43_run" >&2; exit 5; }
    [[ -f "$m43_manifest" && ! -L "$m43_manifest" ]] || { echo "launch manifest missing/symlink" >&2; exit 3; }
    [[ "$(m43_sha "$m43_manifest")" == "$m43_manifest_expected" ]] || { echo "launch manifest SHA mismatch" >&2; exit 3; }
    for m43_tool in /usr/bin/python3.6 sha256sum awk find xargs cp chmod pgrep readlink; do
        [[ -x "$m43_tool" ]] || command -v "$m43_tool" >/dev/null 2>&1 || { echo "missing tool: $m43_tool" >&2; exit 2; }
    done
    if m43_synopsys_active; then
        echo "refusing concurrent Synopsys invocation" >&2
        exit 4
    fi
    mkdir -p "$m43_run/snapshot/inputs"
    trap m43_partial_seal EXIT
    /usr/bin/python3.6 - "$m43_manifest" "$m43_manifest_expected" \
            "$m43_run/snapshot/inputs" <<'PY'
from __future__ import print_function
import hashlib
import json
import pathlib
import shutil
import sys

manifest_path = pathlib.Path(sys.argv[1])
expected_manifest = sys.argv[2]
snapshot = pathlib.Path(sys.argv[3])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result

if sha(manifest_path) != expected_manifest:
    raise ValueError("manifest changed after bootstrap precheck")
manifest = json.loads(
    manifest_path.read_text(encoding="utf-8"),
    object_pairs_hook=pairs,
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("invalid JSON constant: " + value)))
if set(manifest) != {"schema", "entries"} or manifest["schema"] != "m43_r3_exact_sha_synopsys_launch_manifest_v2":
    raise ValueError("manifest schema/key drift")
seen = set()
for entry in manifest["entries"]:
    if set(entry) != {"source", "snapshot", "sha256"}:
        raise ValueError("entry schema drift")
    source = pathlib.Path(entry["source"])
    relative = pathlib.Path(entry["snapshot"])
    if (not source.is_absolute() or source.is_symlink() or not source.is_file()
            or relative.is_absolute() or ".." in relative.parts
            or str(relative) in seen):
        raise ValueError("unsafe manifest entry: {}".format(entry))
    seen.add(str(relative))
    if sha(source) != entry["sha256"]:
        raise ValueError("source SHA drift: " + str(source))
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(target))
    if target.is_symlink() or sha(target) != entry["sha256"]:
        raise ValueError("snapshot copy drift: " + str(relative))
shutil.copyfile(str(manifest_path), str(snapshot / "launch_manifest.json"))
if sha(snapshot / "launch_manifest.json") != expected_manifest:
    raise ValueError("snapshot manifest drift")
PY
    echo "$m43_manifest_expected" > "$m43_run/launch_manifest.sha256"
    {
        echo "launcher_sha256=${M43_LAUNCHER_SHA256:-UNSET}"
        echo "manifest_sha256=$m43_manifest_expected"
    } > "$m43_run/external_launch_receipt.txt"
    (
        cd "$m43_run/snapshot/inputs"
        find . -type f -print0 | sort -z | xargs -0 sha256sum \
            > "$m43_run/snapshot_input_sha256.txt"
        sha256sum --strict -c "$m43_run/snapshot_input_sha256.txt" \
            > "$m43_run/snapshot_input_check.raw.log" 2>&1
    )
    find "$m43_run/snapshot/inputs" -type f -exec chmod 0444 {} +
    find "$m43_run/snapshot/inputs" -type d -exec chmod 0555 {} +
    trap - EXIT
    exec /usr/bin/env bash \
        "$m43_run/snapshot/inputs/hw_autoresearch_nts07/dc_handoff/scripts/run_m43_r3_exact_sha_synopsys.sh" \
        --inside-snapshot "$m43_manifest_expected"
fi

[[ "${1:-}" == "--inside-snapshot" && "$#" -eq 2 ]] || {
    echo "runner must be entered through the pinned bootstrap launcher" >&2
    exit 2
}
m43_manifest_expected="$2"
m43_snapshot="$m43_run/snapshot/inputs"
m43_hw="$m43_snapshot/hw_autoresearch_nts07"
m43_runner="$m43_hw/dc_handoff/scripts/run_m43_r3_exact_sha_synopsys.sh"
m43_manifest="$m43_snapshot/launch_manifest.json"
m43_filelist="$m43_hw/dc_handoff/filelists/date_m43_r3_parent_delta_p8_l96_dc.f"
m43_sdc="$m43_hw/dc_handoff/constraints/date_m43_r3_parent_delta_p8_l96_3ns.sdc"
m43_dc_tcl="$m43_hw/dc_handoff/scripts/run_dc_m43_r3_exact_sha.tcl"
m43_sta_tcl="$m43_hw/dc_handoff/scripts/run_sta_m43_r3_exact_sha.tcl"
m43_fm_tcl="$m43_hw/dc_handoff/scripts/run_formality_m43_r3_exact_sha.tcl"
m43_auditor="$m43_hw/dc_handoff/scripts/audit_m43_r3_structural.py"
m43_builder="$m43_hw/dc_handoff/scripts/build_m43_r3_synopsys_receipt.py"
m43_validator="$m43_hw/dc_handoff/scripts/validate_m43_r3_exact_sha_synopsys.py"
m43_lib_slow="$m43_snapshot/libraries/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m43_lib_fast="$m43_snapshot/libraries/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m43_dc_snapshot="$m43_snapshot/tools/dc_resolved_binary"
m43_fm_snapshot="$m43_snapshot/tools/formality_resolved_binary"
m43_design=qfit_parent_delta_p8_l96_multicontext

trap m43_partial_seal EXIT
[[ "$(m43_sha "$m43_manifest")" == "$m43_manifest_expected" ]]
[[ "$(readlink -f "$m43_dc_launcher")" == "$m43_dc_resolved" ]]
[[ "$(readlink -f "$m43_fm_launcher")" == "$m43_fm_resolved" ]]
[[ -f "$m43_dc_resolved" && ! -L "$m43_dc_resolved" ]]
[[ -f "$m43_fm_resolved" && ! -L "$m43_fm_resolved" ]]
[[ "$(m43_sha "$m43_dc_resolved")" == "$m43_dc_sha" ]]
[[ "$(m43_sha "$m43_fm_resolved")" == "$m43_fm_sha" ]]
[[ "$(m43_sha "$m43_dc_snapshot")" == "$m43_dc_sha" ]]
[[ "$(m43_sha "$m43_fm_snapshot")" == "$m43_fm_sha" ]]
(
    cd "$m43_snapshot"
    sha256sum --strict -c "$m43_run/snapshot_input_sha256.txt"
) > "$m43_run/snapshot_recheck_before_tools.raw.log" 2>&1
if m43_synopsys_active; then
    echo "refusing concurrent Synopsys invocation after snapshot" >&2
    exit 4
fi

mkdir -p "$m43_run/reports" "$m43_run/netlist" \
    "$m43_run/work/dc" "$m43_run/work/sta" "$m43_run/work/formality"
{
    echo "dc_launcher=$m43_dc_launcher"
    echo "dc_resolved=$m43_dc_resolved"
    echo "dc_resolved_sha256=$m43_dc_sha"
    echo "formality_launcher=$m43_fm_launcher"
    echo "formality_resolved=$m43_fm_resolved"
    echo "formality_resolved_sha256=$m43_fm_sha"
} > "$m43_run/external_tool_identity.txt"
{
    echo "status=RUNNING_NOT_CITABLE"
    echo "candidate_sha256=e70239b1ec9a7d4541b0ae8d0a8f55e252fa6c804b364ab126d8201e108e0deb"
    echo "candidate_changed=false"
    echo "clock_period_ns=3.000"
    echo "sequence=dc_then_fresh_ddc_sta_then_formality_foreground"
    echo "scope=standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro"
} > "$m43_run/RUN_IN_PROGRESS.txt"

set +e
"$m43_dc_launcher" -version > "$m43_run/dc.version.raw.log" 2>&1
m43_version_rc=$?
set -e
echo "$m43_version_rc" > "$m43_run/dc.version.rc"
grep -q '^dc_shell version[[:space:]]*-  V-2023.12-SP3$' \
    "$m43_run/dc.version.raw.log"

export DESIGN_NAME="$m43_design"
export SNAPSHOT_ROOT="$m43_hw"
export RTL_FILELIST="$m43_filelist"
export SDC_FILE="$m43_sdc"
export OUTPUT_DIR="$m43_run"
export CLOCK_PERIOD_NS=3.000
export LIB_DB="$m43_lib_slow"
export MIN_LIB_DB="$m43_lib_fast"
export OPERATING_CONDITION=ssg0p9v125c

echo "$m43_dc_launcher -f $m43_dc_tcl" > "$m43_run/dc.command.txt"
set +e
(cd "$m43_run/work/dc" && "$m43_dc_launcher" -f "$m43_dc_tcl") \
    > "$m43_run/dc.raw.log" 2>&1
m43_rc=$?
set -e
echo "$m43_rc" > "$m43_run/dc.rc"
[[ "$m43_rc" -eq 0 ]]
[[ "$(grep -xc 'M43_R3_DC_INTERNAL_COMPLETE=PASS' "$m43_run/DC_INTERNAL_COMPLETE.txt")" -eq 1 ]]
! grep -Eq '^(Error|Fatal):' "$m43_run/dc.raw.log"
for m43_required in reports/constraint_contract_precompile.rpt \
        reports/constraint_contract_postcompile.rpt reports/qor.rpt \
        reports/area.rpt reports/timing_setup.rpt reports/timing_hold.rpt \
        reports/resources_precompile.rpt reports/resources_postcompile.rpt \
        reports/references_precompile.rpt reports/references_postcompile.rpt \
        reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt \
        reports/clocks.rpt "netlist/${m43_design}_mapped.v" \
        "netlist/${m43_design}_mapped.sdc" "netlist/${m43_design}.ddc" \
        "netlist/${m43_design}.svf"; do
    [[ -s "$m43_run/$m43_required" ]] || {
        echo "missing DC output: $m43_required" >&2
        exit 10
    }
done
grep -qx 'physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO' \
    "$m43_run/reports/constraint_contract_postcompile.rpt"
grep -Eq '^Name[[:space:]]+:[[:space:]]+ZeroWireload[[:space:]]*$' \
    "$m43_run/reports/constraint_contract_postcompile.rpt"
/usr/bin/python3.6 - "$m43_run/reports/clocks.rpt" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"^core_clk\s+([0-9.]+)\s+\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$",
    data, re.MULTILINE)
if not match or abs(float(match.group(1)) - 3.0) > 1e-12 or "p" in match.group(2):
    raise SystemExit("3ns ideal/unpropagated core_clk contract failed")
PY

set +e
/usr/bin/python3.6 "$m43_auditor" \
    --dc-log "$m43_run/dc.raw.log" \
    --resources-pre "$m43_run/reports/resources_precompile.rpt" \
    --resources-post "$m43_run/reports/resources_postcompile.rpt" \
    --references-pre "$m43_run/reports/references_precompile.rpt" \
    --references-post "$m43_run/reports/references_postcompile.rpt" \
    --mapped-netlist "$m43_run/netlist/${m43_design}_mapped.v" \
    --area "$m43_run/reports/area.rpt" \
    --check-design "$m43_run/reports/check_design_postcompile.rpt" \
    --report "$m43_run/reports/m43_r3_structural_audit.rpt" \
    > "$m43_run/structural_audit.raw.log" \
    2> "$m43_run/structural_audit.stderr.raw.log"
m43_rc=$?
set -e
echo "$m43_rc" > "$m43_run/structural_audit.rc"
[[ "$m43_rc" -eq 0 ]]
[[ ! -s "$m43_run/structural_audit.stderr.raw.log" ]]

[[ "$(m43_sha "$m43_dc_resolved")" == "$m43_dc_sha" ]]
export DDC_FILE="$m43_run/netlist/${m43_design}.ddc"
echo "$m43_dc_launcher -f $m43_sta_tcl" > "$m43_run/sta.command.txt"
set +e
(cd "$m43_run/work/sta" && "$m43_dc_launcher" -f "$m43_sta_tcl") \
    > "$m43_run/sta.raw.log" 2>&1
m43_rc=$?
set -e
echo "$m43_rc" > "$m43_run/sta.rc"
[[ "$m43_rc" -eq 0 ]]
[[ "$(grep -xc 'M43_R3_STA_INTERNAL_COMPLETE=PASS' "$m43_run/STA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
! grep -Eq '^(Error|Fatal):' "$m43_run/sta.raw.log"

[[ "$(m43_sha "$m43_fm_resolved")" == "$m43_fm_sha" ]]
export MAPPED_NETLIST="$m43_run/netlist/${m43_design}_mapped.v"
export SVF_FILE="$m43_run/netlist/${m43_design}.svf"
echo "$m43_fm_launcher -f $m43_fm_tcl" > "$m43_run/formality.command.txt"
set +e
(cd "$m43_run/work/formality" && "$m43_fm_launcher" -f "$m43_fm_tcl") \
    > "$m43_run/formality.raw.log" 2>&1
m43_rc=$?
set -e
echo "$m43_rc" > "$m43_run/formality.rc"
[[ "$m43_rc" -eq 0 ]]
[[ "$(grep -xc 'M43_R3_FORMALITY_INTERNAL_COMPLETE=PASS' "$m43_run/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
! grep -Eq '^(Error|Fatal):' "$m43_run/formality.raw.log"

set +e
/usr/bin/python3.6 "$m43_builder" --run "$m43_run" \
    --snapshot "$m43_snapshot" \
    --output "$m43_run/m43_r3_synopsys_receipt.json" \
    > "$m43_run/receipt_builder.raw.log" \
    2> "$m43_run/receipt_builder.stderr.raw.log"
m43_build_rc=$?
set -e
echo "$m43_build_rc" > "$m43_run/receipt_builder.rc"
[[ "$m43_build_rc" -eq 0 ]]
[[ ! -s "$m43_run/receipt_builder.stderr.raw.log" ]]

set +e
/usr/bin/python3.6 "$m43_validator" --run "$m43_run" \
    > "$m43_run/validation.raw.log" \
    2> "$m43_run/validation.stderr.raw.log"
m43_validation_rc=$?
set -e
echo "$m43_validation_rc" > "$m43_run/validation.rc"
[[ "$m43_validation_rc" -eq 0 ]]
[[ ! -s "$m43_run/validation.stderr.raw.log" ]]

(
    cd "$m43_run"
    mv RUN_IN_PROGRESS.txt RUN_BOOTSTRAP_RECORD.txt
    find . -type f ! -path './snapshot/*' ! -path './work/*' \
        ! -name output_sha256.txt ! -name output_manifest_check.raw.log \
        ! -name output_manifest_check.rc ! -name completion_seal.sha256 \
        ! -name completion_seal_check.raw.log ! -name RUN_IN_PROGRESS.txt \
        ! -name RUN_COMPLETE.txt -print0 \
        | sort -z | xargs -0 sha256sum > output_sha256.txt
    set +e
    sha256sum --strict -c output_sha256.txt > output_manifest_check.raw.log 2>&1
    m43_output_rc=$?
    set -e
    echo "$m43_output_rc" > output_manifest_check.rc
    [[ "$m43_output_rc" -eq 0 ]]
    {
        echo "status=PASS_EXACT_SHA_FRESH_M43_R3_DC_STA_FORMALITY"
        echo "candidate_sha256=e70239b1ec9a7d4541b0ae8d0a8f55e252fa6c804b364ab126d8201e108e0deb"
        echo "candidate_changed=false"
        echo "clock_period_ns=3.000"
        echo "scope=standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro"
        echo "paper_ppa_ready=false"
        echo "system_speedup_admitted=false"
        echo "power_or_energy_admitted=false"
    } > RUN_COMPLETE.txt
    sha256sum launch_manifest.sha256 snapshot_input_sha256.txt output_sha256.txt \
        m43_r3_synopsys_receipt.json RUN_COMPLETE.txt > completion_seal.sha256
    sha256sum --strict -c completion_seal.sha256 \
        > completion_seal_check.raw.log 2>&1
)

m43_finalized=1
trap - EXIT
find "$m43_run" -type f -exec chmod 0444 {} +
find "$m43_run" -type d -exec chmod 0555 {} +
echo "M43_R3_EXACT_SHA_SYNOPSYS=PASS run=$m43_run"
