#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="all"
if [[ $# -ge 2 && "$1" == "--dataset" ]]; then
  DATASET="$2"
elif [[ $# -ge 1 ]]; then
  DATASET="${1#--dataset=}"
fi

case "${DATASET}" in
  dsec|all)
    cat <<'EOF'
DSEC download is manual.
1. Visit https://dsec.ifi.uzh.ch/dsec-datasets/download/
2. Download the event-flow sequences and preprocessing assets.
3. Place preprocessed tensors under:
   data/Datasets/DSEC/saved_flow_data/
EOF
    ;;
esac

case "${DATASET}" in
  mvsec|all)
    cat <<'EOF'
MVSEC download is manual.
1. Visit https://daniilidis-group.github.io/mvsec/
2. Download the selected sequences.
3. Place files under:
   data/Datasets/MVSEC/
EOF
    ;;
esac
