run_id:      architecture_20260810_053302_local5_erep_g2_v1
design_name: Local5_EREP_v4_G2_preimplementation_evaluation_contract
tool:        architecture-orchestrator+OpenROAD_G2a_proxy+DC_STA_SAIF_PTPX_G2b_receipt
start_time:  2026-08-10T05:33:02Z
last_stage:  arch_signoff_contract_freeze
overall:     COMPLETE_PREIMPLEMENTATION_CONTRACT_ONLY
write_scope:
  - contracts/local5_erep_g2_preimplementation_contract_v1_20260810.json
  - docs/291_Local5_EREP_G2物理评估预实现合同_20260810.md
  - memory/architecture/local5_erep_g2_run_state.md
  - tests/test_local5_erep_g2_contract_v1.py
non_overlap:
  - 不修改核心 RTL
  - 不修改 OpenROAD 配置、runner 或 SDC
  - 不修改 Kepler 校准 RTL、测试平台或结果
current_stage: complete
evidence:
  - 合同冻结 C0-C5 同一 relation-to-Acc32 功能边界
  - G2a 冻结 5ns Nangate45、OUT_DIM32、32 个基础 SRAM 宏、outline 与 pin seed
  - 从本地实物重算 config/runner/SDC/lib/LEF/tool/ORFS SHA
  - 冻结 memory-inclusive EDP、统一 activity/idle/clock-gating 和同频淘汰规则
  - G2b 冻结运行前 fail-closed receipt schema
signoff_achieved: false
blocking_items:
  - target_library_and_pvt_missing
  - absolute_area_budget_missing
  - absolute_power_budget_missing
  - target_sram_macro_models_missing
  - dc_sta_saif_ptpx_receipt_missing
  - common_activity_stimulus_sha_missing
  - c0_c5_candidate_physical_wrappers_missing
