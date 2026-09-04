#!/usr/bin/env bash
set -euo pipefail

m35_repo=/home/zhumd/work/sdformer_codex/SDformer
m35_manifest="$m35_repo/hw_autoresearch_nts07/contracts/m35_r6_m33_fair_exact_sha_launch_manifest_r1_20260823.json"
m35_manifest_sha=a468eb69ca5d620a15db79037415db81e403b9456ffe036984ba2abbc35183de
m35_runner="$m35_repo/hw_autoresearch_nts07/dc_handoff/scripts/run_m35_r6_m33_fair_exact_sha_synopsys.sh"

[[ "$(sha256sum "$m35_manifest" | awk '{print $1}')" == "$m35_manifest_sha" ]] || {
    echo "sealed launch manifest SHA mismatch" >&2
    exit 3
}
m35_launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
export M35_LAUNCHER_SHA256="$m35_launcher_sha"
exec /usr/bin/env bash "$m35_runner" --bootstrap "$m35_manifest" "$m35_manifest_sha"
