#!/usr/bin/env bash
set -euo pipefail

# Immutable N2 runner.  It intentionally freezes canonical later-object paths
# but no N3..N8 SHA.  The N6 authorization and N8 terminal wrapper review bind
# this runner later; direct invocation is rejected by the analyzer parent/FD
# attestation before result or attempt creation.

if [[ $# -ne 12 ]]; then
  echo "M559 runner requires six named argument/value pairs" >&2
  exit 64
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
HW_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)
ANALYZER="${SCRIPT_DIR}/analyze_m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_r4.py"

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
    echo "M559 runner argument order/name drift" >&2
    exit 64
  fi
done

if [[ ! -f "${ANALYZER}" || -L "${ANALYZER}" ]]; then
  echo "M559 immutable analyzer missing or symlinked" >&2
  exit 65
fi

exec /usr/bin/python3 -B "${ANALYZER}" "$@"
