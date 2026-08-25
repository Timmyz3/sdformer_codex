#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 RUN_DIR HW_ROOT" >&2
    exit 2
fi

seal_run_dir="$(realpath "$1")"
seal_hw_root="$(realpath "$2")"
seal_input_ledger="$seal_run_dir/evidence.sha256"
seal_dir="$seal_run_dir/sealed_dc"

if [[ ! -d "$seal_run_dir" || ! -d "$seal_hw_root" \
        || ! -s "$seal_input_ledger" ]]; then
    echo "DC sealer requires an existing run, HW root, and evidence.sha256" >&2
    exit 3
fi
if [[ -e "$seal_dir" || -e "$seal_run_dir/sealed_dc_evidence.sha256" ]]; then
    echo "refusing to overwrite an existing DC evidence seal" >&2
    exit 4
fi

# The live-source ledger must be clean at the instant of sealing.  Afterwards,
# the self-contained manifest below points at immutable run-local snapshots.
sha256sum -c "$seal_input_ledger"

mkdir -p "$seal_dir/inputs"
seal_map="$seal_dir/source_map.tsv"
seal_libraries="$seal_dir/library_identity.sha256"
: > "$seal_map"
: > "$seal_libraries"

while read -r seal_expected seal_path; do
    [[ -n "${seal_expected:-}" && -n "${seal_path:-}" ]] || continue
    if [[ "$seal_path" == "$seal_hw_root/"* ]]; then
        seal_relative="${seal_path#"$seal_hw_root/"}"
        seal_snapshot="$seal_dir/inputs/$seal_relative"
        mkdir -p "$(dirname "$seal_snapshot")"
        cp --preserve=mode,timestamps "$seal_path" "$seal_snapshot"
        printf '%s\t%s\t%s\n' "$seal_expected" "$seal_path" \
            "sealed_dc/inputs/$seal_relative" >> "$seal_map"
    elif [[ "$seal_path" == /opt/* ]]; then
        printf '%s  %s\n' "$seal_expected" "$seal_path" >> "$seal_libraries"
    elif [[ "$seal_path" != "$seal_run_dir/"* ]]; then
        echo "DC evidence contains an unclassified external path: $seal_path" >&2
        exit 5
    fi
done < "$seal_input_ledger"

cp --preserve=mode,timestamps "$0" "$seal_dir/seal_dc_evidence.sh"

seal_tmp="$(mktemp "$seal_run_dir/.sealed_dc_evidence.XXXXXX")"
trap 'rm -f "$seal_tmp"' EXIT
(
    cd "$seal_run_dir"
    find sealed_dc -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum evidence.sha256
    while read -r seal_expected seal_path; do
        [[ -n "${seal_expected:-}" && -n "${seal_path:-}" ]] || continue
        if [[ "$seal_path" == "$seal_run_dir/"* ]]; then
            seal_relative="${seal_path#"$seal_run_dir/"}"
            sha256sum "$seal_relative"
        fi
    done < evidence.sha256
) > "$seal_tmp"
mv "$seal_tmp" "$seal_run_dir/sealed_dc_evidence.sha256"
trap - EXIT

(
    cd "$seal_run_dir"
    sha256sum -c sealed_dc_evidence.sha256
)
echo "DC_EVIDENCE_SEALED run=$seal_run_dir"
