#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVED_FLOW_ROOT="${SAVED_FLOW_ROOT:-${ROOT_DIR}/data/Datasets/DSEC/saved_flow_data}"
MLFLOW_URI="${MLFLOW_URI:-file:///root/private_data/sdformer_mlflow}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/valid_DSEC_supervised_subset.yml}"
TRAIN_LIMIT_PER_SEQ="${TRAIN_LIMIT_PER_SEQ:-200}"
VALID_LIMIT_PER_SEQ="${VALID_LIMIT_PER_SEQ:-40}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dsec_dev_subset.sh make-splits [dev3|dev5]
  scripts/run_dsec_dev_subset.sh train
  scripts/run_dsec_dev_subset.sh eval <runid>
  scripts/run_dsec_dev_subset.sh full-cycle <dev3|dev5>

Environment variables:
  SAVED_FLOW_ROOT       Default: /root/private_data/work/SDformer/data/Datasets/DSEC/saved_flow_data
  MLFLOW_URI            Default: file:///root/private_data/sdformer_mlflow
  TRAIN_LIMIT_PER_SEQ   Default: 200
  VALID_LIMIT_PER_SEQ   Default: 40
  TRAIN_CONFIG          Default: configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml
  EVAL_CONFIG           Default: configs/valid_DSEC_supervised_subset.yml
EOF
}

resolve_sequences() {
  local preset="${1:-dev5}"
  case "${preset}" in
    dev3)
      echo "zurich_city_01_a zurich_city_07_a thun_00_a"
      ;;
    dev5)
      echo "zurich_city_01_a zurich_city_07_a zurich_city_09_a zurich_city_11_a thun_00_a"
      ;;
    *)
      echo "Unknown preset: ${preset}" >&2
      exit 1
      ;;
  esac
}

make_splits() {
  local preset="${1:-dev5}"
  local seqs
  seqs="$(resolve_sequences "${preset}")"
  cd "${ROOT_DIR}"
  python tools/make_dsec_subset_splits.py \
    --root "${SAVED_FLOW_ROOT}" \
    --sequences ${seqs} \
    --train-limit-per-seq "${TRAIN_LIMIT_PER_SEQ}" \
    --valid-limit-per-seq "${VALID_LIMIT_PER_SEQ}" \
    --train-output train_subset_split_seq.csv \
    --valid-output valid_subset_split_seq.csv
}

train_subset() {
  cd "${ROOT_DIR}/third_party/SDformerFlow"
  export KMP_DUPLICATE_LIB_OK=TRUE
  python train_flow_parallel_supervised_SNN.py \
    --config "${TRAIN_CONFIG}" \
    --path_mlflow "${MLFLOW_URI}"
}

eval_subset() {
  local runid="${1:-}"
  if [[ -z "${runid}" ]]; then
    echo "eval requires <runid>" >&2
    exit 1
  fi
  cd "${ROOT_DIR}/third_party/SDformerFlow"
  export KMP_DUPLICATE_LIB_OK=TRUE
  python eval_DSEC_flow_SNN.py \
    --config "${EVAL_CONFIG}" \
    --path_mlflow "${MLFLOW_URI}" \
    --runid "${runid}"
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    make-splits)
      shift || true
      make_splits "${1:-dev5}"
      ;;
    train)
      train_subset
      ;;
    eval)
      shift || true
      eval_subset "${1:-}"
      ;;
    full-cycle)
      shift || true
      make_splits "${1:-dev5}"
      train_subset
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
