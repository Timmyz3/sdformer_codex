#!/usr/bin/env bash
set -euo pipefail

# Immutable M590 source-only N2 runner.  No launch is legal until N3--N8
# chain binds these bytes and the canonical post-authorization wrapper supplies
# its inherited read-only descriptor.

if [[ $# -ne 12 ]]; then
  echo "M590 runner requires six named argument/value pairs" >&2
  exit 64
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ANALYZER="${SCRIPT_DIR}/analyze_m590_m559_pbr4_pre_rtl_cpu_r6.py"
EXPECTED_ANALYZER_SHA256="5550dfb032ad2c43752137c3c1038a97228a2d5265697c6d89d54425d904ccf1"
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
    echo "M590 runner argument order/name drift" >&2
    exit 64
  fi
done

if [[ ! -f "${ANALYZER}" || -L "${ANALYZER}" ]]; then
  echo "M590 immutable analyzer missing or symlinked" >&2
  exit 65
fi

ACTUAL_ANALYZER_SHA256=$(/usr/bin/sha256sum -- "${ANALYZER}")
ACTUAL_ANALYZER_SHA256=${ACTUAL_ANALYZER_SHA256%% *}
if [[ "${ACTUAL_ANALYZER_SHA256}" != "${EXPECTED_ANALYZER_SHA256}" ]]; then
  echo "M590 immutable analyzer pre-exec SHA mismatch" >&2
  exit 66
fi

exec /usr/bin/python3 -B "${ANALYZER}" "$@"
