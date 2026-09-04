#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
python_bin="/usr/libexec/platform-python3.6"
runner="$repo_root/hw_autoresearch_nts07/system_simulator/scripts/execute_m606_m593_parent_scratch_energy_r3.py"
adapter="$repo_root/hw_autoresearch_nts07/system_simulator/scripts/analyze_m606_m597_m593_parent_scratch_generated_macro_energy_r3.py"

expected_python_sha="9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"
expected_runner_sha="3896c348b809b3094396bc64f63ffc7802866b3a5034e222c8addba8b21640fa"
expected_adapter_sha="69d5c2c521b84aee589b28531574d95ec621dfdeeaf35d517cc0bb386e87782d"

require_plain_file() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'M606_FAIL_CLOSED: not a plain file: %s\n' "$path" >&2
    exit 66
  }
}

require_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    printf 'M606_FAIL_CLOSED: SHA drift: %s\n' "$path" >&2
    exit 66
  }
}

require_plain_file "$python_bin"
require_plain_file "$runner"
require_plain_file "$adapter"
require_sha "$python_bin" "$expected_python_sha"
require_sha "$runner" "$expected_runner_sha"
require_sha "$adapter" "$expected_adapter_sha"

exec "$python_bin" -B "$runner" --shell-path "${BASH_SOURCE[0]}" "$@"
