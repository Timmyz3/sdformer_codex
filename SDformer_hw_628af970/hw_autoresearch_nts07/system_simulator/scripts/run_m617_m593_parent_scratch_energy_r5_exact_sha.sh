#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
python_bin="/usr/libexec/platform-python3.6"
runner="$repo_root/hw_autoresearch_nts07/system_simulator/scripts/execute_m617_m593_parent_scratch_energy_r5.py"
expected_python_sha="9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"
expected_runner_sha="cc7a721554d3da65a98f80c93c75f5e9c26de9914a68e9991876572d0a8d6844"

require_plain_file() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'M617_R5_FAIL_CLOSED: not a plain file: %s\n' "$path" >&2
    exit 66
  }
}

require_sha() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    printf 'M617_R5_FAIL_CLOSED: SHA drift: %s\n' "$path" >&2
    exit 66
  }
}

require_plain_file "$python_bin"
require_plain_file "$runner"
require_sha "$python_bin" "$expected_python_sha"
require_sha "$runner" "$expected_runner_sha"
exec "$python_bin" -B "$runner" --shell-path "${BASH_SOURCE[0]}" "$@"
