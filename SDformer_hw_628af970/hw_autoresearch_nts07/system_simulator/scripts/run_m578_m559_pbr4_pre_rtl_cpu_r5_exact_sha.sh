#!/usr/bin/env bash
set -euo pipefail

# Immutable source-only N2 runner.  No launch is legal until the later N3--N8
# chain binds these bytes and the canonical post-authorization wrapper supplies
# its inherited read-only descriptor.

if [[ $# -ne 12 ]]; then
  echo "M578 runner requires six named argument/value pairs" >&2
  exit 64
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ANALYZER="${SCRIPT_DIR}/analyze_m578_m559_pbr4_pre_rtl_cpu_r5.py"
EXPECTED_NAMES=(
  --contract
  --m511-directory
  --m511-payload-verifier-directory
  --decoder-int8-weight-package
  --output-directory
  --authorization-descriptor
)

for index in 0 1 2 3 4 5; do
  argument_index=$((2 * index + 1))
  if [[ "${!argument_index}" != "${EXPECTED_NAMES[index]}" ]]; then
    echo "M578 runner argument order/name drift" >&2
    exit 64
  fi
done

if [[ ! -f "${ANALYZER}" || -L "${ANALYZER}" ]]; then
  echo "M578 immutable analyzer missing or symlinked" >&2
  exit 65
fi

exec /usr/bin/python3 -B "${ANALYZER}" "$@"
