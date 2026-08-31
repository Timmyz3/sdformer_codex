#!/bin/bash -p
set -euo pipefail
PATH=/usr/bin:/bin
export PATH
readonly PATH
[[ -z "${BASH_ENV:-}" && -z "${ENV:-}" ]] || {
    echo "M632 refuses startup hooks" >&2
    exit 3
}

m632_wrapper_abs="$(readlink -f "${BASH_SOURCE[0]}")"
m632_hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m632_repo_root="$(cd "${m632_hw_root}/.." && pwd)"
m632_runner="${m632_hw_root}/system_handoff/scripts/run_m511_h67_ep35_convtranspose_binary_input_capture_r1_exact_sha.sh"
m632_runner_sha="fddf6a0fc06685fa87f94248c6f48776e59142e0111db3aee2cab38691b7f2d6"

[[ "${m632_wrapper_abs}" == \
   "${m632_hw_root}/system_handoff/scripts/run_m632_m511_local_rtx3090_capture_exact_sha.sh" ]] || {
    echo "M632 wrapper canonical path drift" >&2
    exit 3
}
[[ -n "${M632_EXPECTED_WRAPPER_SHA256:-}" && \
   "$(sha256sum "${m632_wrapper_abs}" | awk '{print $1}')" == \
   "${M632_EXPECTED_WRAPPER_SHA256}" ]] || {
    echo "M632 caller did not supply the literal independently reviewed wrapper SHA" >&2
    exit 3
}
[[ "${m632_repo_root}" == "/home/zhumd/work/sdformer_codex/SDformer" && \
   "$(/usr/bin/hostname)" == "ic.ismd-nemo" ]] || {
    echo "M632 local repository/host identity drift" >&2
    exit 3
}
[[ "$(sha256sum /usr/bin/hostname | awk '{print $1}')" == \
   "c1f8c2c26baa42a5896989353aa7330cd41693435b5fe08386a8b7aa998629dc" && \
   "$(sha256sum /usr/bin/nvidia-smi | awk '{print $1}')" == \
   "6b8be04c92bf327401faa99d6c7aa7da351b0d4aca8531b422efe2e58b456886" ]] || {
    echo "M632 host/GPU identity-tool drift" >&2
    exit 3
}
[[ -f "${m632_runner}" && ! -L "${m632_runner}" && \
   "$(sha256sum "${m632_runner}" | awk '{print $1}')" == \
   "${m632_runner_sha}" ]] || {
    echo "M632 frozen M511 runner identity drift" >&2
    exit 3
}
m632_gpu=$(/usr/bin/nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | /usr/bin/sed -n '1p')
[[ "${m632_gpu}" == \
   "NVIDIA GeForce RTX 3090, GPU-2b9bf62c-21f9-6c5e-8ace-ee867d88a037, 575.64, 24576" ]] || {
    echo "M632 local GPU/driver identity drift" >&2
    exit 3
}

exec /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    M511_EXPECTED_RUNNER_SHA256="${m632_runner_sha}" \
    M511_EXPECTED_REPO_ROOT="${m632_repo_root}" \
    M632_LAUNCH_WRAPPER_PATH="${m632_wrapper_abs}" \
    M632_EXPECTED_WRAPPER_SHA256="${M632_EXPECTED_WRAPPER_SHA256}" \
    "${m632_runner}"
