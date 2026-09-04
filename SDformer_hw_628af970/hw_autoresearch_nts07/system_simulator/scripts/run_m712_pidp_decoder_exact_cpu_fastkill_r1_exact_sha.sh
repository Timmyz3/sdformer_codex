#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
analyzer="${repo_root}/hw_autoresearch_nts07/system_simulator/scripts/analyze_m712_pidp_decoder_exact_cpu_fastkill.py"
contract="${repo_root}/hw_autoresearch_nts07/contracts/m712_pidp_decoder_exact_cpu_fastkill_contract_r1_20260828.json"
output="${repo_root}/hw_autoresearch_nts07/results/m712_pidp_decoder_exact_cpu_fastkill_r1_20260828"

[[ "$(sha256sum "${analyzer}" | awk '{print $1}')" == \
  "87e559a1d249a9aacec31763c692a0da9e312bd753f11c63241b765fca16dbbc" ]] || {
  echo "M712 analyzer SHA drift" >&2
  exit 66
}
[[ "$(sha256sum "${contract}" | awk '{print $1}')" == \
  "5c11add1b92dceab9fe09d22234545172ba58de74f221c29ee88688b248f3bf2" ]] || {
  echo "M712 contract SHA drift" >&2
  exit 67
}
[[ ! -e "${output}" && ! -L "${output}" ]] || {
  echo "M712 canonical output already exists" >&2
  exit 68
}

exec /usr/bin/python3 "${analyzer}" \
  --repo-root "${repo_root}" \
  --contract "${contract}" \
  --output "${output}"
