#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
python_bin="/usr/libexec/platform-python3.6"
runner="$repo_root/hw_autoresearch_nts07/system_simulator/scripts/execute_m612_m593_parent_scratch_energy_r4.py"
adapter="$repo_root/hw_autoresearch_nts07/system_simulator/scripts/analyze_m612_m597_m593_parent_scratch_generated_macro_energy_r4.py"
expected_python_sha="9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"
expected_runner_sha="82cf5a6d7d33a78246b9c88fa5a4db50be4821b4a30c8ffb198f114a59b76727"
expected_adapter_sha="65f6f006c62a5e7732eefc62106af14b76eb708567da995a3b45ad9a9d78daba"

require_plain_file() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'M612_FAIL_CLOSED: not a plain file: %s\n' "$path" >&2
    exit 66
  }
}

require_sha() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    printf 'M612_FAIL_CLOSED: SHA drift: %s\n' "$path" >&2
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
