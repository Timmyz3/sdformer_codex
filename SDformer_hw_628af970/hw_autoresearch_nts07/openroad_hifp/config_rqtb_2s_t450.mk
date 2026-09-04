export DESIGN_NAME = h67_temporal_slot_shiftmax_sync_k_2s_top
export DESIGN_NICKNAME = h67_rqtb_2s_t450_flopmem_proxy
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = \
  $(HW_ROOT)/rtl_ttx/ttx_ceil_log2_u32.sv \
  $(HW_ROOT)/rtl_ttx/ttx_exp2_lut_q8.sv \
  $(HW_ROOT)/rtl_ttx/ttx_gate_quant_q17.sv \
  $(HW_ROOT)/rtl_h67/h67_motionxor_score_q7.sv \
  $(HW_ROOT)/rtl_h67/h67_temporal_slot_encoder.sv \
  $(HW_ROOT)/rtl_h67/h67_temporal_slot_fifo_2s.sv \
  $(HW_ROOT)/rtl_h67/h67_sync_dual_bank_k_store.sv \
  $(HW_ROOT)/rtl_h67/h67_temporal_weighted_scs_directory_2s.sv \
  $(HW_ROOT)/rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint_rqtb.sdc)
export VERILOG_TOP_PARAMS ?= QUOTIENT_ENABLE 0

# 与1S强基线保持相同约束；行为memory映射为flop，仅作开放物理比较。
export CORE_UTILIZATION = 30
export PLACE_DENSITY = 0.50
export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks
