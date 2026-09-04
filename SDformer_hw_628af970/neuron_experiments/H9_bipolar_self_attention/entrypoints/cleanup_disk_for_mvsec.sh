#!/usr/bin/env bash
# DEPRECATED: this script removed whole nts*/ntx* result dirs by mistake.
# Use cleanup_early_checkpoints_only.sh instead (checkpoints only, never nts/ntx dirs).
# Free disk for MVSEC download + NTS10d resume. Keeps NB0 ep59, NTS07b/09e/10d key artifacts.
set -euo pipefail

echo "[cleanup] ERROR: cleanup_disk_for_mvsec.sh is deprecated; use cleanup_early_checkpoints_only.sh"
exit 1

ROOT="/root/private_data/work/sdformer_codex/SDformer"
RES="${ROOT}/neuron_experiments/H9_bipolar_self_attention/results"
BASE="${ROOT}/experiments/baseline_stride_upstream"
LOG="${RES}/disk_cleanup_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1
echo "[cleanup] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[cleanup] before: $(df -h /root/private_data | tail -1)"

freed=0
rm_dir() {
  local d="$1"
  if [[ -d "${d}" ]]; then
    local sz
    sz=$(du -sh "${d}" | awk '{print $1}')
    echo "[cleanup] remove dir ${d} (${sz})"
    rm -rf "${d}"
  fi
}

trim_ckpts() {
  local run_dir="$1"
  shift
  local keep_epochs=("$@")
  if [[ ! -d "${run_dir}" ]]; then
    return 0
  fi
  echo "[cleanup] trim ${run_dir} keep epochs: ${keep_epochs[*]}"
  local epoch
  local keep_set=" ${keep_epochs[*]} "
  for f in "${run_dir}"/checkpoint_epoch*.pth; do
    [[ -e "${f}" ]] || continue
    local base
    base=$(basename "${f}")
    local epoch
    if [[ "${base}" == *_state_dict.pth ]]; then
      epoch=${base#checkpoint_epoch}
      epoch=${epoch%_state_dict.pth}
    else
      epoch=${base#checkpoint_epoch}
      epoch=${epoch%.pth}
    fi
    if [[ "${keep_set}" != *" ${epoch} "* ]]; then
      rm -f "${f}"
    fi
  done
}

# --- superseded / concluded full runs ---
for d in \
  "${RES}/nts09a_hw_h60_freeze816_s1224_steps1224_auto_full_bs6_20260608_210900_setsid" \
  "${RES}/ntx_h60_v2_full30_20260605_163955" \
  "${RES}/nts00b_mu010_std_s360_auto_full_bs6_20260607_031912_setsid" \
  "${RES}/faps00a_dir_nokmag_s360_auto_full_bs6_20260607_032232_setsid" \
  "${RES}/nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_bs6_20260608_160513_setsid" \
  "${RES}/nts04c_hw_mu010_mis020_s360_auto_full_bs6_20260607_233610_setsid" \
  "${RES}/nts04g_hw_sched010_w720_s360_auto_full_bs6_20260608_004746_setsid" \
  "${RES}/ntx_h60_full30_20260605_020633" \
  "${RES}/ntx_tx_kmag_full30_20260606_135707"; do
  rm_dir "${d}"
done

# --- short / rapid_screen sweeps (summary already in md) ---
for d in "${RES}"/nts_nokmag_short_* "${RES}"/nts04_hw_short_* "${RES}"/nts09_sparse_* \
  "${RES}"/nts08_qk_stab_* "${RES}"/h62_conf_nts_short_* "${RES}"/faps_short_* \
  "${RES}"/nts09_priority_* "${RES}"/nts06_floor_hw_short_* "${RES}"/nts05_weak_hw_short_* \
  "${RES}"/ntx10_local_short_* "${RES}"/nts10_blocks_20260610_* \
  "${RES}"/faps_autopilot_* "${RES}"/faps_promote_* "${RES}"/faps_resume_* \
  "${RES}"/nts_nokmag_autopilot_* "${RES}"/nts09_sparse_resume_* \
  "${RES}"/nts10_blocks_autopilot_* "${RES}"/nts10_blocks_resume_* \
  "${RES}"/nts10d_crash_resume_*; do
  rm_dir "${d}"
done

# --- early smoke / debug ---
for d in "${RES}"/h12*_smoke_* "${RES}"/h13*_smoke_* "${RES}"/debug_* \
  "${RES}"/atlif_binary_* "${RES}"/atlif_ternary_*; do
  rm_dir "${d}"
done

# --- old May full30 explorations (pre-NTS mainline) ---
for d in "${RES}"/h10_*_setsid "${RES}"/h10_*.log \
  "${RES}"/h13m_*_setsid "${RES}"/h13n_*_setsid "${RES}"/h13*_guard120_* \
  "${RES}"/h14*_setsid "${RES}"/h23*_setsid "${RES}"/h28b_*_setsid \
  "${RES}"/h37_*_setsid "${RES}"/h41_*_full30_* "${RES}"/h49_*_full30_*; do
  rm_dir "${d}"
done

# --- baseline_stride_upstream: old stride dirs + redundant ckpts ---
rm_dir "${BASE}/h41_tx_stride"
rm_dir "${BASE}/h41_tx_stride_v2"
rm_dir "${BASE}/warmrestart"
rm_dir "${BASE}/extend"
rm_dir "${BASE}/continue"

echo "[cleanup] baseline state_dict purge"
find "${BASE}" -maxdepth 1 -name 'checkpoint_epoch*_state_dict.pth' -delete
echo "[cleanup] baseline keep only epoch59 weights"
for f in "${BASE}"/checkpoint_epoch*.pth; do
  [[ -e "${f}" ]] || continue
  [[ "$(basename "${f}")" == "checkpoint_epoch59.pth" ]] && continue
  rm -f "${f}"
done

# --- trim kept runs ---
trim_ckpts "${RES}/nts09e_hw_h60_freeze1224_s1224_steps1224_auto_full_bs6_20260610_001833_setsid" 19 24 29
trim_ckpts "${RES}/nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid" 24 29
trim_ckpts "${RES}/nts10d_hw_h60_s23_freeze1224_s1224_steps1224_auto_full_bs6_20260610_151207_setsid" 11

echo "[cleanup] after: $(df -h /root/private_data | tail -1)"
echo "[cleanup] log=${LOG}"
echo "[cleanup] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"