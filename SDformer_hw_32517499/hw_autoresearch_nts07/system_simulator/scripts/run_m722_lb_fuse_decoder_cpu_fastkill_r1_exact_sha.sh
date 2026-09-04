#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
hw_root="${repo_root}/hw_autoresearch_nts07"
analyzer="${hw_root}/system_simulator/scripts/analyze_m722_lb_fuse_decoder_cpu_fastkill.py"
tests="${hw_root}/system_simulator/tests/test_m722_lb_fuse_decoder_cpu_fastkill.py"
contract="${hw_root}/contracts/m722_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json"
python_bin="/opt/anaconda3/envs/pytorch310/bin/python3.10"
output="${hw_root}/results/m722_lb_fuse_decoder_cpu_fastkill_r1_20260828"

[[ "$(sha256sum "${analyzer}" | awk '{print $1}')" == \
  "3693fd1078738e8e3e0928080802cf2f276d5cb5951f72134a4482ce364077df" ]] || {
  echo "M722 analyzer SHA drift" >&2
  exit 66
}
[[ "$(sha256sum "${tests}" | awk '{print $1}')" == \
  "b1d95520c568e2ef6d677beade1190e79883145aca8f34e2f50f8f01b76839b3" ]] || {
  echo "M722 tests SHA drift" >&2
  exit 67
}
[[ "$(sha256sum "${contract}" | awk '{print $1}')" == \
  "e88cb84794a83026e4c8329ba6a93798a682a421095ce82f201b87e942879545" ]] || {
  echo "M722 contract SHA drift" >&2
  exit 68
}
[[ "$(sha256sum "${python_bin}" | awk '{print $1}')" == \
  "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115" ]] || {
  echo "M722 Python SHA drift" >&2
  exit 69
}
[[ "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" == \
  "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4" ]] || {
  echo "M722 docs359 drift" >&2
  exit 70
}
[[ ! -e "${output}" && ! -L "${output}" ]] || {
  echo "M722 canonical output already exists" >&2
  exit 71
}
if compgen -G "${hw_root}/results/.m722_lb_fuse_decoder_cpu_fastkill_r1_20260828.staging.*" >/dev/null; then
  echo "M722 stale staging exists" >&2
  exit 72
fi

clean_env=(env -i
  PATH=/opt/anaconda3/envs/pytorch310/bin:/usr/bin:/bin
  LANG=C.UTF-8
  LC_ALL=C.UTF-8
  CUDA_VISIBLE_DEVICES=
  OMP_NUM_THREADS=48
  MKL_NUM_THREADS=48
  OPENBLAS_NUM_THREADS=48
  PYTHONHASHSEED=0)

"${clean_env[@]}" "${python_bin}" -m py_compile "${analyzer}" "${tests}"
"${clean_env[@]}" PYTHONPATH="${hw_root}/system_simulator/scripts" \
  "${python_bin}" -m unittest "${tests}"
"${clean_env[@]}" "${python_bin}" "${analyzer}" --self-test
"${clean_env[@]}" "${python_bin}" "${analyzer}" \
  --repo-root "${repo_root}" --contract "${contract}" --output "${output}"

(
  cd -- "${output}"
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.seal.sha256
)
[[ "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" == \
  "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4" ]]
echo "M722_RUNNER_PASS output=${output}"
