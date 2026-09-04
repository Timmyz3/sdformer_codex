#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "usage: $0 RUN_DIR HW_ROOT VCS_R8_DIR OLD_DC_R1_DIR SNAPSHOT_TAG" >&2
    exit 2
fi

m37_snapshot_run_dir="$(realpath "$1")"
m37_snapshot_hw_root="$(realpath "$2")"
m37_snapshot_vcs_dir="$(realpath "$3")"
m37_snapshot_old_dir="$(realpath "$4")"
m37_snapshot_tag="$5"
if [[ ! "$m37_snapshot_tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "M37 snapshot tag contains unsafe characters" >&2
    exit 3
fi

m37_snapshot_name="sealed_snapshot_${m37_snapshot_tag}"
m37_snapshot_final="$m37_snapshot_run_dir/$m37_snapshot_name"
m37_snapshot_ledger="$m37_snapshot_run_dir/${m37_snapshot_name}.sha256"
if [[ -e "$m37_snapshot_final" || -e "$m37_snapshot_ledger" ]]; then
    echo "refusing to overwrite M37 sealed snapshot: $m37_snapshot_final" >&2
    exit 4
fi

(cd "$m37_snapshot_hw_root" && sha256sum --strict -c \
    "$m37_snapshot_run_dir/input_sha256.txt")
(cd "$m37_snapshot_run_dir" && sha256sum --strict -c dc_output_sha256.txt)
(cd "$m37_snapshot_run_dir" && sha256sum --strict -c dc_live_seal.sha256)
(cd "$m37_snapshot_run_dir" && sha256sum --strict -c formality_evidence.sha256)
(cd "$m37_snapshot_run_dir" && sha256sum --strict -c formality_live_seal.sha256)

m37_snapshot_tmp="$(mktemp -d "$m37_snapshot_run_dir/.m37_snapshot.XXXXXX")"
mkdir -p "$m37_snapshot_tmp/repository" "$m37_snapshot_tmp/vcs_r8" \
    "$m37_snapshot_tmp/old_dc_r1" "$m37_snapshot_tmp/fresh_run"

m37_repository_files=(
    rtl_m37/qfit_atlif_csd_reconstruct_t10.sv
    contracts/m37_output_receipt_r3_20260822.json
    contracts/m37_csd_reconstruct_t10_vcs_contract_r3_20260822.json
    dc_handoff/filelists/date_m37_csd_reconstruct_t10_dc.f
    dc_handoff/constraints/date_m37_csd_reconstruct_t10.sdc
    dc_handoff/scripts/audit_m37_r8_source_intent.py
    dc_handoff/scripts/audit_m37_dc_evidence.py
    dc_handoff/scripts/run_dc_m37_csd_reconstruct_t10.sh
    dc_handoff/scripts/run_dc_m37_csd_reconstruct_t10.tcl
    dc_handoff/scripts/run_formality_m37_csd_reconstruct_t10.tcl
    dc_handoff/scripts/seal_m37_r8_dc_formality_snapshot.sh
)
(
    cd "$m37_snapshot_hw_root"
    cp --parents -- "${m37_repository_files[@]}" \
        "$m37_snapshot_tmp/repository"
)

cp -- "$m37_snapshot_vcs_dir/input_sha256.txt" \
    "$m37_snapshot_vcs_dir/output_sha256.txt" \
    "$m37_snapshot_vcs_dir/run_local_seal.sha256" \
    "$m37_snapshot_vcs_dir/compile.log" \
    "$m37_snapshot_vcs_dir/sim.log" \
    "$m37_snapshot_vcs_dir/vectors.txt" \
    "$m37_snapshot_vcs_dir/rtl_multiplier_intent_audit.txt" \
    "$m37_snapshot_vcs_dir/runner_status.txt" \
    "$m37_snapshot_tmp/vcs_r8"
cp -- "$m37_snapshot_old_dir/FAILED_RESOURCE_AUDIT_DO_NOT_CITE.txt" \
    "$m37_snapshot_tmp/old_dc_r1"
cp -- "$m37_snapshot_old_dir/reports/m37_multiplier_identifier_hits.rpt" \
    "$m37_snapshot_tmp/old_dc_r1"

cp -a -- "$m37_snapshot_run_dir/reports" \
    "$m37_snapshot_run_dir/netlist" "$m37_snapshot_tmp/fresh_run"
m37_live_files=(
    dc.log
    dc.exit_status
    admission.txt
    input_sha256.txt
    dc_output_sha256.txt
    dc_runner_status.txt
    dc_live_seal.sha256
    formality.log
    formality.exit_status
    formality_admission.txt
    formality_evidence.sha256
    formality_live_seal.sha256
)
for m37_live_file in "${m37_live_files[@]}"; do
    cp -- "$m37_snapshot_run_dir/$m37_live_file" \
        "$m37_snapshot_tmp/fresh_run/$m37_live_file"
done

{
    echo "scope=STANDALONE_M37_R8_DC_3P000NS_AND_FORMALITY_EVIDENCE"
    echo "snapshot_tag=$m37_snapshot_tag"
    echo "library_binaries_embedded=false"
    echo "library_binary_hashes_are_preserved_in=fresh_run/input_sha256.txt"
    echo "system_claim_admitted=false"
    echo "power_energy_claim_admitted=false"
    echo "paper_ppa_headline_admitted=false"
} > "$m37_snapshot_tmp/snapshot_scope.txt"
(
    cd "$m37_snapshot_tmp"
    find . -type f ! -name snapshot_contents.sha256 -print0 \
        | sort -z | xargs -0 sha256sum > snapshot_contents.sha256
    sha256sum --strict -c snapshot_contents.sha256
)
mv -- "$m37_snapshot_tmp" "$m37_snapshot_final"
(
    cd "$m37_snapshot_run_dir"
    find "$m37_snapshot_name" -type f -print0 \
        | sort -z | xargs -0 sha256sum > "${m37_snapshot_name}.sha256"
    sha256sum --strict -c "${m37_snapshot_name}.sha256"
)
echo "M37_SELF_CONTAINED_EVIDENCE_SNAPSHOT=PASS path=$m37_snapshot_final"
