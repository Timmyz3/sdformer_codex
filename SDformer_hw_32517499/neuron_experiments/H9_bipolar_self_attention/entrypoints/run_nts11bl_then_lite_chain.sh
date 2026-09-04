#!/usr/bin/env bash
# Wait for 11bl full+valid825, then launch 11lite qkonly full30+valid825 unattended.
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
BL_RUN_DIR="${1:-${EXP}/results/nts11bl_u12_ds_w720_fastlr_w360_ftbd19_ft5_bs8_20260615_024301_setsid}"
STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_DIR="${EXP}/results/nts11bl_then_lite_chain_${STAMP}"
LOG="${CHAIN_DIR}/chain.log"

mkdir -p "${CHAIN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== bl→lite chain start $(date -Is) ==="
echo "wait_bl_run_dir=${BL_RUN_DIR}"

wait_bl_done() {
  while true; do
    if [[ -f "${BL_RUN_DIR}/pipeline.log" ]] && grep -q "=== 11bl pipeline complete ===" "${BL_RUN_DIR}/pipeline.log"; then
      echo "[chain] 11bl pipeline complete marker found $(date -Is)"
      return 0
    fi
    if pgrep -f "${BL_RUN_DIR}" >/dev/null 2>&1; then
      epoch=$(ls -1 "${BL_RUN_DIR}"/checkpoint_epoch*.pth 2>/dev/null | sed -E 's/.*epoch([0-9]+).pth/\1/' | sort -n | tail -1 || true)
      echo "[chain] 11bl still running latest_ckpt_epoch=${epoch:-none} $(date -Is)"
      sleep 120
      continue
    fi
    if [[ -f "${BL_RUN_DIR}/profile_ranking_valid825.md" ]]; then
      echo "[chain] 11bl valid825 ranking present $(date -Is)"
      return 0
    fi
    if [[ -f "${BL_RUN_DIR}/pipeline.log" ]] && grep -q "=== train done" "${BL_RUN_DIR}/pipeline.log"; then
      echo "[chain] 11bl train done, waiting valid825 $(date -Is)"
      sleep 120
      continue
    fi
    echo "[chain] waiting for 11bl run_dir activity $(date -Is)"
    sleep 120
  done
}

wait_bl_done
echo "[chain] launching 11lite qkonly full30+valid825 $(date -Is)"
bash "${EXP}/entrypoints/run_nts11lite_qkonly_full30_valid825.sh"
echo "=== bl→lite chain complete $(date -Is) ==="