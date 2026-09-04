#!/usr/bin/env bash
set -euo pipefail

m35_repo=/home/zhumd/work/sdformer_codex/SDformer
m35_manifest="$m35_repo/hw_autoresearch_nts07/contracts/m35_r7_m33_fair_exact_sha_launch_manifest_r1_20260823.json"
m35_manifest_sha=925eab6c40397511522ea5299fd88e52a3e47010cfffd2ba7624474420eaeae2
m35_runner="$m35_repo/hw_autoresearch_nts07/dc_handoff/scripts/run_m35_r6_m33_fair_exact_sha_synopsys.sh"
export M35_RUN_PATH=/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m35_r6_m33_fair_exact_sha_synopsys_3p000ns_r7_20260823

[[ "$(sha256sum "$m35_manifest" | awk '{print $1}')" == "$m35_manifest_sha" ]] || {
    echo "sealed r7 launch manifest SHA mismatch" >&2
    exit 3
}
m35_launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
export M35_LAUNCHER_SHA256="$m35_launcher_sha"
exec /usr/bin/env bash "$m35_runner" --bootstrap "$m35_manifest" "$m35_manifest_sha"
