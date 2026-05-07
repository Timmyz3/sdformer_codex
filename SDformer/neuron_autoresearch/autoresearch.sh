#!/usr/bin/env bash
# Autoresearch evaluation script for neuron operator experiments.
# Runs profile_sops.py on a config+checkpoint pair and outputs METRIC lines.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:?Usage: $0 <config_yml> [checkpoint.pth]}"
CHECKPOINT="${2:-}"

cd "$REPO_ROOT"

PROFILE_ARGS=(
  --config "$CONFIG"
  --num-samples 40
  --metrics AEE
  --split valid
)

if [ -n "$CHECKPOINT" ] && [ "$CHECKPOINT" != "none" ]; then
  PROFILE_ARGS+=(--checkpoint "$CHECKPOINT")
fi

OUTPUT=$(python tools/profile_sops.py "${PROFILE_ARGS[@]}" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "PROFILE_ERROR exit_code=$EXIT_CODE"
  echo "$OUTPUT" | tail -20
  exit $EXIT_CODE
fi

# Parse key metrics from profile_sops output
SOPs=$(echo "$OUTPUT" | grep -oP 'estimated_total_sops:\s*\K[0-9.]+(?=G?)' || echo "0")
FIRING=$(echo "$OUTPUT" | grep -oP 'global_firing_rate:\s*\K[0-9.]+' || echo "0")

AEE_LINE=$(echo "$OUTPUT" | grep -oP 'AEE:\s*\K[0-9.]+' || echo "0")
if [ "$AEE_LINE" = "0" ]; then
  # Try metrics section
  AEE_LINE=$(echo "$OUTPUT" | grep -A5 'metrics:' | grep -oP 'AEE:\s*\K[0-9.]+' || echo "0")
fi

echo "METRIC sops=$SOPs"
echo "METRIC firing_rate=$FIRING"
echo "METRIC aee=$AEE_LINE"
echo "PROFILE_OK samples=40"
