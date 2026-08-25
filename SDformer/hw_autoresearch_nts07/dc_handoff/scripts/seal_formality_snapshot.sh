#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 RUN_DIR ATTEMPT_TAG HW_ROOT" >&2
    exit 2
fi

snapshot_run_dir="$(realpath "$1")"
snapshot_attempt="$2"
snapshot_hw_root="$(realpath "$3")"
snapshot_live_ledger="$snapshot_run_dir/formality_evidence_${snapshot_attempt}.sha256"
snapshot_dir="$snapshot_run_dir/sealed_formality_${snapshot_attempt}"
snapshot_ledger="$snapshot_run_dir/sealed_formality_evidence_${snapshot_attempt}.sha256"

if [[ ! "$snapshot_attempt" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Formality snapshot attempt tag contains unsafe characters" >&2
    exit 3
fi
if [[ ! -d "$snapshot_run_dir" || ! -d "$snapshot_hw_root" \
        || ! -s "$snapshot_live_ledger" ]]; then
    echo "Formality snapshot requires a run, HW root, and live evidence ledger" >&2
    exit 4
fi
if [[ -e "$snapshot_dir" || -e "$snapshot_ledger" ]]; then
    echo "refusing to overwrite an existing Formality snapshot" >&2
    exit 5
fi

# Prove that the live inputs still match the completed attempt before taking
# any copy.  The generated snapshot below remains valid after later wrapper
# extensions for new designs.
sha256sum -c "$snapshot_live_ledger"

mkdir -p "$snapshot_dir/inputs"
snapshot_map="$snapshot_dir/source_map.tsv"
snapshot_external="$snapshot_dir/external_identity.sha256"
: > "$snapshot_map"
: > "$snapshot_external"

while read -r snapshot_expected snapshot_path; do
    [[ -n "${snapshot_expected:-}" && -n "${snapshot_path:-}" ]] || continue
    if [[ "$snapshot_path" == "$snapshot_hw_root/"* ]]; then
        snapshot_relative="${snapshot_path#"$snapshot_hw_root/"}"
        snapshot_copy="$snapshot_dir/inputs/$snapshot_relative"
        mkdir -p "$(dirname "$snapshot_copy")"
        cp --preserve=mode,timestamps "$snapshot_path" "$snapshot_copy"
        printf '%s\t%s\t%s\n' "$snapshot_expected" "$snapshot_path" \
            "sealed_formality_${snapshot_attempt}/inputs/$snapshot_relative" \
            >> "$snapshot_map"
    elif [[ "$snapshot_path" == "$snapshot_run_dir/"* ]]; then
        :
    elif [[ "$snapshot_path" == /opt/* ]]; then
        printf '%s  %s\n' "$snapshot_expected" "$snapshot_path" \
            >> "$snapshot_external"
    else
        echo "Formality evidence contains an unclassified path: $snapshot_path" >&2
        exit 6
    fi
done < "$snapshot_live_ledger"

cp --preserve=mode,timestamps "$0" \
    "$snapshot_dir/seal_formality_snapshot.sh"

snapshot_tmp="$(mktemp "$snapshot_run_dir/.sealed_formality_evidence.XXXXXX")"
trap 'rm -f "$snapshot_tmp"' EXIT
(
    cd "$snapshot_run_dir"
    find "sealed_formality_${snapshot_attempt}" -type f -print0 \
        | sort -z | xargs -0 sha256sum
    sha256sum "formality_evidence_${snapshot_attempt}.sha256"
    while read -r snapshot_expected snapshot_path; do
        [[ -n "${snapshot_expected:-}" && -n "${snapshot_path:-}" ]] || continue
        if [[ "$snapshot_path" == "$snapshot_run_dir/"* ]]; then
            snapshot_relative="${snapshot_path#"$snapshot_run_dir/"}"
            sha256sum "$snapshot_relative"
        fi
    done < "formality_evidence_${snapshot_attempt}.sha256"
) > "$snapshot_tmp"
mv "$snapshot_tmp" "$snapshot_ledger"
trap - EXIT

(
    cd "$snapshot_run_dir"
    sha256sum -c "$(basename "$snapshot_ledger")"
)
echo "FORMALITY_SNAPSHOT_SEALED run=$snapshot_run_dir attempt=$snapshot_attempt"
