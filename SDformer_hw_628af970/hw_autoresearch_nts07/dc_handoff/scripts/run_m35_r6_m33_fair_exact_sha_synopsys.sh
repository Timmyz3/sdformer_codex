#!/usr/bin/env bash
set -euo pipefail

# Bootstrap mode only validates the externally pinned launch manifest, creates
# the complete immutable input snapshot, then re-executes this runner from that
# snapshot.  All EDA and postprocessing work occurs in --inside-snapshot mode.

m35_repo=/home/zhumd/work/sdformer_codex/SDformer
m35_live_runner="$m35_repo/hw_autoresearch_nts07/dc_handoff/scripts/run_m35_r6_m33_fair_exact_sha_synopsys.sh"
m35_run=${M35_RUN_PATH:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m35_r6_m33_fair_exact_sha_synopsys_3p000ns_r6_20260823}
m35_dc_bin=/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell
m35_fm_bin=/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell
m35_dc_sha=23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m35_fm_sha=aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b
m35_finalized=0

m35_sha() { sha256sum "$1" | awk '{print $1}'; }

m35_synopsys_active() {
    pgrep -f '^/opt/synopsys/.*/(common_shell_exec|dc_shell|dc_shell-t|fm_shell|fm_shell_exec)( |$)' \
        >/dev/null 2>&1
}

m35_partial_seal() {
    local m35_rc=$?
    if [[ -d "$m35_run" && "$m35_finalized" -eq 0 ]]; then
        set +e
        {
            echo "status=FAIL_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$m35_rc"
            echo "candidate_changed=false"
            echo "paper_ppa_ready=false"
            echo "system_speedup_admitted=false"
            date -u +"sealed_utc=%Y-%m-%dT%H:%M:%SZ"
        } > "$m35_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
        (
            cd "$m35_run" || exit 1
            find . -type f ! -name PARTIAL_EVIDENCE.sha256 \
                ! -name PARTIAL_EVIDENCE_CHECK.raw.log -print0 \
                | sort -z | xargs -0 sha256sum > PARTIAL_EVIDENCE.sha256
            sha256sum --strict -c PARTIAL_EVIDENCE.sha256 \
                > PARTIAL_EVIDENCE_CHECK.raw.log 2>&1
        )
        find "$m35_run" -type f -exec chmod 0444 {} +
        find "$m35_run" -type d -exec chmod 0555 {} +
        set -e
    fi
    exit "$m35_rc"
}

if [[ "${1:-}" == "--bootstrap" ]]; then
    [[ "$#" -eq 3 ]] || { echo "usage: $0 --bootstrap MANIFEST EXPECTED_SHA" >&2; exit 2; }
    m35_manifest="$2"
    m35_manifest_expected="$3"
    [[ ! -e "$m35_run" ]] || { echo "refusing to overwrite fixed run: $m35_run" >&2; exit 5; }
    [[ -f "$m35_manifest" && ! -L "$m35_manifest" ]] || { echo "launch manifest missing/symlink" >&2; exit 3; }
    [[ "$(m35_sha "$m35_manifest")" == "$m35_manifest_expected" ]] || { echo "launch manifest SHA mismatch" >&2; exit 3; }
    for m35_tool in /usr/bin/python3.6 sha256sum awk find xargs cp chmod pgrep; do
        [[ -x "$m35_tool" ]] || command -v "$m35_tool" >/dev/null 2>&1 || { echo "missing tool: $m35_tool" >&2; exit 2; }
    done
    if m35_synopsys_active; then
        echo "refusing concurrent Synopsys invocation" >&2
        exit 4
    fi
    mkdir -p "$m35_run/snapshot/inputs"
    trap m35_partial_seal EXIT
    /usr/bin/python3.6 - "$m35_manifest" "$m35_manifest_expected" \
            "$m35_run/snapshot/inputs" <<'PY'
from __future__ import print_function
import hashlib
import json
import os
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

def pairs(pairs_list):
    result = {}
    for key, value in pairs_list:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result

if sha(manifest_path) != expected_manifest:
    raise ValueError("manifest changed after bootstrap precheck")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("invalid JSON constant: " + value)))
if set(manifest) != {"schema", "entries"} or manifest["schema"] != "m35_r6_m33_fair_launch_manifest_v1":
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
    echo "$m35_manifest_expected" > "$m35_run/launch_manifest.sha256"
    echo "launcher_sha256=${M35_LAUNCHER_SHA256:-UNSET}" > "$m35_run/external_launch_receipt.txt"
    echo "manifest_sha256=$m35_manifest_expected" >> "$m35_run/external_launch_receipt.txt"
    (
        cd "$m35_run/snapshot/inputs"
        find . -type f -print0 | sort -z | xargs -0 sha256sum > "$m35_run/snapshot_input_sha256.txt"
        sha256sum --strict -c "$m35_run/snapshot_input_sha256.txt" \
            > "$m35_run/snapshot_input_check.raw.log" 2>&1
    )
    find "$m35_run/snapshot/inputs" -type f -exec chmod 0444 {} +
    find "$m35_run/snapshot/inputs" -type d -exec chmod 0555 {} +
    trap - EXIT
    exec /usr/bin/env bash \
        "$m35_run/snapshot/inputs/hw_autoresearch_nts07/dc_handoff/scripts/run_m35_r6_m33_fair_exact_sha_synopsys.sh" \
        --inside-snapshot "$m35_manifest_expected"
fi

[[ "${1:-}" == "--inside-snapshot" && "$#" -eq 2 ]] || {
    echo "runner must be entered through the pinned bootstrap launcher" >&2
    exit 2
}
m35_manifest_expected="$2"
m35_snapshot="$m35_run/snapshot/inputs"
m35_hw="$m35_snapshot/hw_autoresearch_nts07"
m35_runner="$m35_hw/dc_handoff/scripts/run_m35_r6_m33_fair_exact_sha_synopsys.sh"
m35_manifest="$m35_snapshot/launch_manifest.json"
m35_dc_tcl="$m35_hw/dc_handoff/scripts/run_dc_m35_r6_m33_fair_exact_sha.tcl"
m35_sta_tcl="$m35_hw/dc_handoff/scripts/run_sta_m35_r6_m33_fair_exact_sha.tcl"
m35_fm_tcl="$m35_hw/dc_handoff/scripts/run_formality_m35_r6_m33_fair_exact_sha.tcl"
m35_sdc="$m35_hw/dc_handoff/constraints/date_m35_r6_m33_fair_3ns.sdc"
m35_auditor="$m35_hw/dc_handoff/scripts/audit_m35_r6_zero_multiplier.py"
m35_builder="$m35_hw/dc_handoff/scripts/build_m35_r6_m33_fair_receipt.py"
m35_validator="$m35_hw/dc_handoff/scripts/validate_m35_r6_m33_fair_exact_sha_synopsys.py"
m35_lib_slow="$m35_snapshot/libraries/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m35_lib_fast="$m35_snapshot/libraries/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"

trap m35_partial_seal EXIT
[[ "$(m35_sha "$m35_manifest")" == "$m35_manifest_expected" ]]
[[ "$(m35_sha "$m35_dc_bin")" == "$m35_dc_sha" ]]
[[ "$(m35_sha "$m35_fm_bin")" == "$m35_fm_sha" ]]
(
    cd "$m35_snapshot"
    sha256sum --strict -c "$m35_run/snapshot_input_sha256.txt"
) > "$m35_run/snapshot_recheck_before_tools.raw.log" 2>&1
if m35_synopsys_active; then
    echo "refusing concurrent Synopsys invocation after snapshot" >&2
    exit 4
fi
mkdir -p "$m35_run/m35" "$m35_run/m33"
{
    echo "status=RUNNING_NOT_CITABLE"
    echo "m35_candidate_sha256=84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854"
    echo "m33_source_sha256=2df1c28c0d22cd5a1c38a78a5838101b23bb13beec9e3e5e60ac8f84aba16c4c"
    echo "clock_period_ns=3.000"
    echo "sequence=m35_then_m33_foreground"
} > "$m35_run/RUN_IN_PROGRESS.txt"
set +e
"$m35_dc_bin" -version > "$m35_run/dc.version.raw.log" 2>&1
m35_version_rc=$?
set -e
echo "$m35_version_rc" > "$m35_run/dc.version.rc"
grep -q '^dc_shell version[[:space:]]*-  V-2023.12-SP3$' \
    "$m35_run/dc.version.raw.log"

m35_run_design() {
    local m35_key="$1"
    local m35_design="$2"
    local m35_filelist="$3"
    local m35_out="$m35_run/$m35_key"
    mkdir -p "$m35_out/reports" "$m35_out/netlist" \
        "$m35_out/work/dc" "$m35_out/work/sta" "$m35_out/work/formality"
    export DESIGN_NAME="$m35_design"
    export SNAPSHOT_ROOT="$m35_hw"
    export RTL_FILELIST="$m35_filelist"
    export SDC_FILE="$m35_sdc"
    export OUTPUT_DIR="$m35_out"
    export CLOCK_PERIOD_NS=3.000
    export LIB_DB="$m35_lib_slow"
    export MIN_LIB_DB="$m35_lib_fast"
    export OPERATING_CONDITION=ssg0p9v125c

    echo "$m35_dc_bin -f $m35_dc_tcl" > "$m35_out/dc.command.txt"
    set +e
    (cd "$m35_out/work/dc" && "$m35_dc_bin" -f "$m35_dc_tcl") \
        > "$m35_out/dc.raw.log" 2>&1
    local m35_rc=$?
    set -e
    echo "$m35_rc" > "$m35_out/dc.rc"
    [[ "$m35_rc" -eq 0 ]]
    [[ "$(grep -xc 'M35_R6_M33_FAIR_DC_INTERNAL_COMPLETE=PASS' "$m35_out/DC_INTERNAL_COMPLETE.txt")" -eq 1 ]]
    ! grep -Eq '^(Error|Fatal):' "$m35_out/dc.raw.log"
    local m35_required
    for m35_required in reports/constraint_contract_precompile.rpt \
            reports/constraint_contract_postcompile.rpt reports/qor.rpt \
            reports/area.rpt reports/timing_setup.rpt reports/timing_hold.rpt \
            reports/resources_precompile.rpt reports/resources_postcompile.rpt \
            reports/references_precompile.rpt reports/references_postcompile.rpt \
            reports/check_design_postcompile.rpt reports/check_timing_postcompile.rpt \
            reports/clocks.rpt "netlist/${m35_design}_mapped.v" \
            "netlist/${m35_design}_mapped.sdc" "netlist/${m35_design}.ddc" \
            "netlist/${m35_design}.svf"; do
        [[ -s "$m35_out/$m35_required" ]] || { echo "missing DC output $m35_key/$m35_required" >&2; return 10; }
    done
    grep -qx 'physical_contract=ZERO_WIRELOAD_IDEAL_CLOCK_NO_SRAM_MACRO' \
        "$m35_out/reports/constraint_contract_postcompile.rpt"
    /usr/bin/python3.6 - "$m35_out/reports/clocks.rpt" <<'PY'
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(r"^core_clk\s+([0-9.]+)\s+\{[^}]+\}\s+(\S+)\s+\{clk_core\}\s*$", data, re.MULTILINE)
if not match or abs(float(match.group(1)) - 3.0) > 1e-9 or "p" in match.group(2):
    raise SystemExit("3ns ideal/unpropagated core_clk contract failed")
PY
    if [[ "$m35_key" == m35 ]]; then
        set +e
        /usr/bin/python3.6 "$m35_auditor" \
            --dc-log "$m35_out/dc.raw.log" \
            --resources-pre "$m35_out/reports/resources_precompile.rpt" \
            --resources-post "$m35_out/reports/resources_postcompile.rpt" \
            --references-pre "$m35_out/reports/references_precompile.rpt" \
            --references-post "$m35_out/reports/references_postcompile.rpt" \
            --mapped-netlist "$m35_out/netlist/${m35_design}_mapped.v" \
            --report "$m35_out/reports/m35_r6_zero_multiplier_audit.rpt" \
            > "$m35_out/structural_audit.raw.log" 2> "$m35_out/structural_audit.stderr.raw.log"
        m35_rc=$?
        set -e
        echo "$m35_rc" > "$m35_out/structural_audit.rc"
        [[ "$m35_rc" -eq 0 ]]
    fi

    export DDC_FILE="$m35_out/netlist/${m35_design}.ddc"
    echo "$m35_dc_bin -f $m35_sta_tcl" > "$m35_out/sta.command.txt"
    set +e
    (cd "$m35_out/work/sta" && "$m35_dc_bin" -f "$m35_sta_tcl") \
        > "$m35_out/sta.raw.log" 2>&1
    m35_rc=$?
    set -e
    echo "$m35_rc" > "$m35_out/sta.rc"
    [[ "$m35_rc" -eq 0 ]]
    [[ "$(grep -xc 'M35_R6_M33_FAIR_STA_INTERNAL_COMPLETE=PASS' "$m35_out/STA_INTERNAL_COMPLETE.txt")" -eq 1 ]]
    ! grep -Eq '^(Error|Fatal):' "$m35_out/sta.raw.log"

    export MAPPED_NETLIST="$m35_out/netlist/${m35_design}_mapped.v"
    export SVF_FILE="$m35_out/netlist/${m35_design}.svf"
    echo "$m35_fm_bin -f $m35_fm_tcl" > "$m35_out/formality.command.txt"
    set +e
    (cd "$m35_out/work/formality" && "$m35_fm_bin" -f "$m35_fm_tcl") \
        > "$m35_out/formality.raw.log" 2>&1
    m35_rc=$?
    set -e
    echo "$m35_rc" > "$m35_out/formality.rc"
    [[ "$m35_rc" -eq 0 ]]
    [[ "$(grep -xc 'M35_R6_M33_FAIR_FORMALITY_INTERNAL_COMPLETE=PASS' "$m35_out/FORMALITY_INTERNAL_COMPLETE.txt")" -eq 1 ]]
    ! grep -Eq '^(Error|Fatal):' "$m35_out/formality.raw.log"
}

# Exact sequential order: candidate first, fair baseline second.
m35_run_design m35 qfit_complement_csd8_canonical \
    "$m35_hw/dc_handoff/filelists/date_m35_r6_canonical_dc.f"
[[ "$(m35_sha "$m35_dc_bin")" == "$m35_dc_sha" ]]
[[ "$(m35_sha "$m35_fm_bin")" == "$m35_fm_sha" ]]
m35_run_design m33 qfit_threshold_late_scale_uq0p24_radix20x4 \
    "$m35_hw/dc_handoff/filelists/date_m33_r6_fair_dc.f"

set +e
/usr/bin/python3.6 "$m35_builder" --run "$m35_run" \
    --output "$m35_run/m35_r6_m33_fair_receipt.json" \
    > "$m35_run/receipt_builder.raw.log" 2> "$m35_run/receipt_builder.stderr.raw.log"
m35_build_rc=$?
set -e
echo "$m35_build_rc" > "$m35_run/receipt_builder.rc"
[[ "$m35_build_rc" -eq 0 ]]
[[ ! -s "$m35_run/receipt_builder.stderr.raw.log" ]]
set +e
/usr/bin/python3.6 "$m35_validator" --run "$m35_run" \
    > "$m35_run/validation.raw.log" 2> "$m35_run/validation.stderr.raw.log"
m35_validation_rc=$?
set -e
echo "$m35_validation_rc" > "$m35_run/validation.rc"
[[ "$m35_validation_rc" -eq 0 ]]
[[ ! -s "$m35_run/validation.stderr.raw.log" ]]

(
    cd "$m35_run"
    mv RUN_IN_PROGRESS.txt RUN_BOOTSTRAP_RECORD.txt
    find . -type f ! -path './snapshot/*' ! -path './m35/work/*' \
        ! -path './m33/work/*' ! -name output_sha256.txt \
        ! -name output_manifest_check.raw.log ! -name output_manifest_check.rc \
        ! -name completion_seal.sha256 ! -name completion_seal_check.raw.log \
        ! -name RUN_IN_PROGRESS.txt ! -name RUN_COMPLETE.txt -print0 \
        | sort -z | xargs -0 sha256sum > output_sha256.txt
    set +e
    sha256sum --strict -c output_sha256.txt > output_manifest_check.raw.log 2>&1
    m35_output_rc=$?
    set -e
    echo "$m35_output_rc" > output_manifest_check.rc
    [[ "$m35_output_rc" -eq 0 ]]
    {
        echo "status=PASS_EXACT_SHA_FRESH_M35_AND_M33_DC_STA_FORMALITY"
        echo "m35_candidate_sha256=84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854"
        echo "candidate_changed=false"
        echo "clock_period_ns=3.000"
        echo "scope=standalone_logic_only_zero_wireload_ideal_clock_no_sram_macro"
        echo "paper_ppa_ready=false"
        echo "system_speedup_admitted=false"
    } > RUN_COMPLETE.txt
    sha256sum launch_manifest.sha256 snapshot_input_sha256.txt output_sha256.txt \
        m35_r6_m33_fair_receipt.json RUN_COMPLETE.txt > completion_seal.sha256
    sha256sum --strict -c completion_seal.sha256 > completion_seal_check.raw.log 2>&1
)

m35_finalized=1
trap - EXIT
find "$m35_run" -type f -exec chmod 0444 {} +
find "$m35_run" -type d -exec chmod 0555 {} +
echo "M35_R6_M33_FAIR_EXACT_SHA_SYNOPSYS=PASS run=$m35_run"
