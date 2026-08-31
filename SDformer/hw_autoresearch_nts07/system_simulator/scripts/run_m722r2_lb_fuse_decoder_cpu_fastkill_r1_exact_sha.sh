#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
hw_root="${repo_root}/hw_autoresearch_nts07"
r1="${hw_root}/system_simulator/scripts/analyze_m722_lb_fuse_decoder_cpu_fastkill.py"
r2="${hw_root}/system_simulator/scripts/analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py"
test_r1="${hw_root}/system_simulator/tests/test_m722_lb_fuse_decoder_cpu_fastkill.py"
test_r2="${hw_root}/system_simulator/tests/test_m722r2_lb_fuse_decoder_preflight_repair.py"
contract="${hw_root}/contracts/m722r2_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json"
python_bin="/opt/anaconda3/envs/pytorch310/bin/python3.10"
output="${hw_root}/results/m722r2_lb_fuse_decoder_cpu_fastkill_r1_20260828"

declare -A expected=(
  ["${r1}"]="3693fd1078738e8e3e0928080802cf2f276d5cb5951f72134a4482ce364077df"
  ["${r2}"]="ed2e1a638ffc533e8b7c9c1ca933e867d1182ca80ed589b2fef547fd39715165"
  ["${test_r1}"]="b1d95520c568e2ef6d677beade1190e79883145aca8f34e2f50f8f01b76839b3"
  ["${test_r2}"]="e034981edee595596e0a7318efeb6116f98bc600eae0fe4d739347b742d939db"
  ["${contract}"]="8fbaffd0eb2b7a1ae02298b58c3071a3a9b7ab592c890e2cb156294fd3fe8039"
  ["${python_bin}"]="9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
  ["${hw_root}/docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
for path in "${!expected[@]}"; do
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected[${path}]}" ]] || {
    echo "M722-r2 SHA drift: ${path}" >&2
    exit 66
  }
done
(
  cd -- "${hw_root}/contracts"
  sha256sum -c m722r2_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json.sha256 >/dev/null
  sha256sum -c m722r2_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json.sha256.seal.sha256 >/dev/null
)
[[ ! -e "${output}" && ! -L "${output}" ]] || {
  echo "M722-r2 canonical output already exists" >&2
  exit 67
}
if compgen -G "${hw_root}/results/.m722r2_lb_fuse_decoder_cpu_fastkill_r1_20260828.staging.*" >/dev/null; then
  echo "M722-r2 stale staging exists" >&2
  exit 68
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

"${clean_env[@]}" "${python_bin}" -m py_compile "${r1}" "${r2}" "${test_r1}" "${test_r2}"
"${clean_env[@]}" PYTHONPATH="${hw_root}/system_simulator/scripts" \
  "${python_bin}" -m unittest "${test_r1}" "${test_r2}"
"${clean_env[@]}" "${python_bin}" "${r2}" --self-test
"${clean_env[@]}" "${python_bin}" "${r2}" \
  --repo-root "${repo_root}" --contract "${contract}" --output "${output}"

(
  cd -- "${output}"
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.seal.sha256
)
[[ "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" == \
  "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4" ]]
echo "M722R2_RUNNER_PASS output=${output}"
