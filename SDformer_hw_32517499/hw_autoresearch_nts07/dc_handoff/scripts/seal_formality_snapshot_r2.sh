#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 RUN_DIR FORMALITY_ATTEMPT SNAPSHOT_TAG HW_ROOT" >&2
    exit 2
fi

snapshot_run_dir="$(realpath -e -- "$1")"
snapshot_attempt="$2"
snapshot_tag="$3"
snapshot_hw_root="$(realpath -e -- "$4")"
snapshot_python="${PYTHON_BIN:-python3}"
snapshot_live_ledger="$snapshot_run_dir/formality_evidence_${snapshot_attempt}.sha256"
snapshot_manifest="$snapshot_run_dir/formality_run_manifest.json"
snapshot_dir="$snapshot_run_dir/sealed_formality_${snapshot_tag}"
snapshot_ledger="$snapshot_run_dir/sealed_formality_evidence_${snapshot_tag}.sha256"

for snapshot_name in "$snapshot_attempt" "$snapshot_tag"; do
    if [[ ! "$snapshot_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "Formality snapshot tag contains unsafe characters" >&2
        exit 3
    fi
done
if [[ ! -d "$snapshot_run_dir" || ! -d "$snapshot_hw_root" \
        || ! -s "$snapshot_live_ledger" || ! -s "$snapshot_manifest" ]]; then
    echo "Formality snapshot requires the sealed run and its manifest" >&2
    exit 4
fi
if ! command -v "$snapshot_python" >/dev/null 2>&1; then
    echo "Formality snapshot Python interpreter is unavailable" >&2
    exit 6
fi
if [[ -e "$snapshot_dir" || -e "$snapshot_ledger" ]]; then
    echo "refusing to overwrite an existing Formality snapshot" >&2
    exit 5
fi

# First revalidate the completed Formality evidence.  The r2 snapshot then
# expands the manifest/filelist and preserves every RTL byte, the mapped
# netlist, SVF, reports, and wrapper inputs.  Foundry libraries remain external
# identities because copying a licensed .db into a handoff is inappropriate.
sha256sum -c "$snapshot_live_ledger"

snapshot_metadata="$("$snapshot_python" - "$snapshot_manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], "r"))
if manifest.get("mode") != "formality":
    raise SystemExit("snapshot manifest mode is not formality")
required = {"LIB_DB", "RTL_FILELIST", "MAPPED_NETLIST"}
if set(manifest.get("paths", {})) != required:
    raise SystemExit("snapshot manifest path population drift")
for name in sorted(required):
    item = manifest["paths"][name]
    if set(item) != {"path", "sha256"}:
        raise SystemExit("snapshot manifest path schema drift for " + name)
    print("{}\t{}\t{}".format(name, item["sha256"], item["path"]))
print("DESIGN\t-\t{}".format(manifest["design_name"]))
PY
)"

snapshot_design="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "DESIGN" {print $3}')"
snapshot_filelist_raw="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "RTL_FILELIST" {print $3}')"
snapshot_filelist_sha="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "RTL_FILELIST" {print $2}')"
snapshot_netlist_raw="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "MAPPED_NETLIST" {print $3}')"
snapshot_netlist_sha="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "MAPPED_NETLIST" {print $2}')"
snapshot_lib_raw="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "LIB_DB" {print $3}')"
snapshot_lib_sha="$(printf '%s\n' "$snapshot_metadata" \
    | awk -F '\t' '$1 == "LIB_DB" {print $2}')"
snapshot_canonical_existing() {
    local raw_path="$1"
    local label="$2"
    local canonical_path
    if ! canonical_path="$(realpath -e -- "$raw_path")"; then
        echo "Formality snapshot $label is missing: $raw_path" >&2
        exit 6
    fi
    if [[ ! -s "$canonical_path" ]]; then
        echo "Formality snapshot $label is empty: $canonical_path" >&2
        exit 6
    fi
    printf '%s\n' "$canonical_path"
}

snapshot_require_contained() {
    local canonical_path="$1"
    local canonical_root="$2"
    local label="$3"
    if [[ "$canonical_path" != "$canonical_root/"* ]]; then
        echo "Formality snapshot $label escapes canonical root: $canonical_path" >&2
        exit 8
    fi
}

snapshot_filelist="$(snapshot_canonical_existing \
    "$snapshot_filelist_raw" "RTL filelist")"
snapshot_netlist="$(snapshot_canonical_existing \
    "$snapshot_netlist_raw" "mapped netlist")"
snapshot_lib="$(snapshot_canonical_existing "$snapshot_lib_raw" "library")"
snapshot_svf="$(snapshot_canonical_existing \
    "$snapshot_run_dir/netlist/${snapshot_design}.svf" "SVF")"
snapshot_require_contained "$snapshot_filelist" "$snapshot_hw_root" \
    "RTL filelist"
snapshot_require_contained "$snapshot_netlist" "$snapshot_run_dir" \
    "mapped netlist"
snapshot_require_contained "$snapshot_svf" "$snapshot_run_dir" "SVF"

test "$(sha256sum "$snapshot_filelist" | awk '{print $1}')" \
    = "$snapshot_filelist_sha"
test "$(sha256sum "$snapshot_netlist" | awk '{print $1}')" \
    = "$snapshot_netlist_sha"
test "$(sha256sum "$snapshot_lib" | awk '{print $1}')" = "$snapshot_lib_sha"

snapshot_tmp_dir="$(mktemp -d "$snapshot_run_dir/.sealed_formality_r2.XXXXXX")"
trap 'rm -rf "$snapshot_tmp_dir"' EXIT
mkdir -p "$snapshot_tmp_dir/inputs/hw_root" \
    "$snapshot_tmp_dir/inputs/run" "$snapshot_tmp_dir/outputs"
snapshot_map="$snapshot_tmp_dir/source_map.tsv"
snapshot_external="$snapshot_tmp_dir/external_identity.sha256"
: > "$snapshot_map"
printf '%s  %s\n' "$snapshot_lib_sha" "$snapshot_lib" \
    > "$snapshot_external"

declare -A snapshot_target_registry=()

snapshot_normalize_target() {
    local relative_path="$1"
    local target_path
    if [[ -z "$relative_path" || "$relative_path" == /* ]]; then
        echo "Formality snapshot target must be a nonempty relative path" >&2
        exit 10
    fi
    target_path="$(realpath -m -- "$snapshot_tmp_dir/$relative_path")"
    if [[ "$target_path" != "$snapshot_tmp_dir/"* ]]; then
        echo "Formality snapshot target escapes staging root: $relative_path" >&2
        exit 10
    fi
    printf '%s\n' "$target_path"
}

snapshot_copy_one() {
    local source_path="$1"
    local expected_sha="$2"
    local relative_path="$3"
    local canonical_source
    local target_path
    local actual_sha
    canonical_source="$(snapshot_canonical_existing \
        "$source_path" "copy source")"
    target_path="$(snapshot_normalize_target "$relative_path")"
    if [[ -n "${snapshot_target_registry[$target_path]+present}" \
            || -e "$target_path" || -L "$target_path" ]]; then
        echo "Formality snapshot target collision: $relative_path" >&2
        exit 11
    fi
    snapshot_target_registry[$target_path]="$canonical_source"
    actual_sha="$(sha256sum "$canonical_source" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "Formality snapshot source drift: $canonical_source" >&2
        exit 7
    fi
    mkdir -p "$(dirname "$target_path")"
    cp --no-clobber --preserve=mode,timestamps -- \
        "$canonical_source" "$target_path"
    test "$(sha256sum "$target_path" | awk '{print $1}')" = "$expected_sha"
    printf '%s\t%s\t%s\n' "$expected_sha" "$canonical_source" "$relative_path" \
        >> "$snapshot_map"
}

snapshot_copy_one "$snapshot_filelist" "$snapshot_filelist_sha" \
    "inputs/hw_root/${snapshot_filelist#"$snapshot_hw_root/"}"
snapshot_copy_one "$snapshot_netlist" "$snapshot_netlist_sha" \
    "inputs/run/${snapshot_netlist#"$snapshot_run_dir/"}"
snapshot_copy_one "$snapshot_svf" \
    "$(sha256sum "$snapshot_svf" | awk '{print $1}')" \
    "inputs/run/${snapshot_svf#"$snapshot_run_dir/"}"

while IFS= read -r snapshot_line; do
    snapshot_line="${snapshot_line%%#*}"
    snapshot_line="${snapshot_line#"${snapshot_line%%[![:space:]]*}"}"
    snapshot_line="${snapshot_line%"${snapshot_line##*[![:space:]]}"}"
    [[ -n "$snapshot_line" ]] || continue
    if [[ "$snapshot_line" == /* ]]; then
        echo "Formality filelist entry must be relative: $snapshot_line" >&2
        exit 8
    fi
    snapshot_rtl="$(snapshot_canonical_existing \
        "$snapshot_hw_root/$snapshot_line" "filelist RTL")"
    snapshot_require_contained "$snapshot_rtl" "$snapshot_hw_root" \
        "filelist RTL"
    snapshot_copy_one "$snapshot_rtl" \
        "$(sha256sum "$snapshot_rtl" | awk '{print $1}')" \
        "inputs/hw_root/${snapshot_rtl#"$snapshot_hw_root/"}"
done < "$snapshot_filelist"

while read -r snapshot_expected snapshot_path; do
    [[ -n "${snapshot_expected:-}" && -n "${snapshot_path:-}" ]] || continue
    if [[ ! "$snapshot_expected" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Formality evidence contains a malformed SHA-256" >&2
        exit 9
    fi
    snapshot_path="${snapshot_path#\*}"
    snapshot_source="$(snapshot_canonical_existing \
        "$snapshot_path" "live-ledger source")"
    if [[ "$snapshot_source" == "$snapshot_run_dir/"* ]]; then
        snapshot_copy_one "$snapshot_source" "$snapshot_expected" \
            "outputs/${snapshot_source#"$snapshot_run_dir/"}"
    elif [[ "$snapshot_source" == "$snapshot_hw_root/"* ]]; then
        snapshot_copy_one "$snapshot_source" "$snapshot_expected" \
            "inputs/hw_root/${snapshot_source#"$snapshot_hw_root/"}"
    elif [[ "$snapshot_source" == /opt/* ]]; then
        if [[ "$(sha256sum "$snapshot_source" | awk '{print $1}')" \
                != "$snapshot_expected" ]]; then
            echo "Formality external identity source drift: $snapshot_source" >&2
            exit 7
        fi
        printf '%s  %s\n' "$snapshot_expected" "$snapshot_source" \
            >> "$snapshot_external"
    else
        echo "Formality evidence contains an unclassified canonical path: $snapshot_source" >&2
        exit 9
    fi
done < "$snapshot_live_ledger"

snapshot_copy_one "$0" "$(sha256sum "$0" | awk '{print $1}')" \
    "seal_formality_snapshot_r2.sh"
snapshot_copy_one "$snapshot_manifest" \
    "$(sha256sum "$snapshot_manifest" | awk '{print $1}')" \
    "formality_run_manifest.json"
snapshot_copy_one "$snapshot_live_ledger" \
    "$(sha256sum "$snapshot_live_ledger" | awk '{print $1}')" \
    "formality_live_evidence.sha256"
sort -u -o "$snapshot_external" "$snapshot_external"
sort -u -o "$snapshot_map" "$snapshot_map"

mv "$snapshot_tmp_dir" "$snapshot_dir"
trap - EXIT

snapshot_tmp_ledger="$(mktemp "$snapshot_run_dir/.sealed_formality_evidence_r2.XXXXXX")"
trap 'rm -f "$snapshot_tmp_ledger"' EXIT
(
    cd "$snapshot_run_dir"
    find "sealed_formality_${snapshot_tag}" -type f -print0 \
        | sort -z | xargs -0 sha256sum
) > "$snapshot_tmp_ledger"
mv "$snapshot_tmp_ledger" "$snapshot_ledger"
trap - EXIT
(
    cd "$snapshot_run_dir"
    sha256sum -c "$(basename "$snapshot_ledger")"
)
echo "FORMALITY_SNAPSHOT_R2_SEALED run=$snapshot_run_dir attempt=$snapshot_attempt snapshot=$snapshot_tag"
