#!/usr/bin/env bash
set -euo pipefail

m43_repo=/home/zhumd/work/sdformer_codex/SDformer
m43_manifest="$m43_repo/hw_autoresearch_nts07/contracts/m43_r3_exact_sha_synopsys_launch_manifest_r1_20260823.json"
m43_manifest_sha=82d92a133cbc67c6b986ade2517e2438cd2456bd6db707265f8fe9f8a9e83505
m43_runner="$m43_repo/hw_autoresearch_nts07/dc_handoff/scripts/run_m43_r3_exact_sha_synopsys.sh"

[[ "$(sha256sum "$m43_manifest" | awk '{print $1}')" == "$m43_manifest_sha" ]] || {
    echo "sealed launch manifest SHA mismatch" >&2
    exit 3
}
m43_launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
export M43_LAUNCHER_SHA256="$m43_launcher_sha"
exec /usr/bin/env bash "$m43_runner" --bootstrap "$m43_manifest" "$m43_manifest_sha"
