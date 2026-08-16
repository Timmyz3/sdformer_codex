export DESIGN_NAME = h67_zkqi_row_shiftmax_physical_top
export DESIGN_NICKNAME = h67_zkqi_threeway_production_cap40_setup50ps_macro_proxy
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = \
  $(HW_ROOT)/rtl_h67/h67_motionxor_score_q7.sv \
  $(HW_ROOT)/rtl_h67/h67_sync_qk_row_store.sv \
  $(HW_ROOT)/rtl_h67/h67_fakeram45_qk_row_store.sv \
  $(HW_ROOT)/rtl_h67/h67_ttb8_metadata_builder.sv \
  $(HW_ROOT)/rtl_h67/h67_pair_bitmap_metadata_builder.sv \
  $(HW_ROOT)/rtl_h67/h67_active_bundle_fifo.sv \
  $(HW_ROOT)/rtl_h67/h67_banked_active_descriptor_store.sv \
  $(HW_ROOT)/rtl_h67/h67_temporal_weighted_scs_directory_seed_2s.sv \
  $(HW_ROOT)/rtl_h67/h67_zkqi_row_shiftmax_top.sv \
  $(HW_ROOT)/rtl_h67/h67_zkqi_row_shiftmax_physical_top.sv \
  $(HW_ROOT)/rtl_ttx/ttx_exp2_lut_q8.sv \
  $(HW_ROOT)/rtl_ttx/ttx_ceil_log2_u32.sv \
  $(HW_ROOT)/rtl_ttx/ttx_gate_quant_q17.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint_h67_zkqi_production_macro.sdc)
export VERILOG_TOP_PARAMS ?= ZK_BYPASS_ENABLE 0 BUNDLE_SKIP_ENABLE 0 ROW_MEMORY_IMPL 1 DIRECTORY_MEMORY_IMPL 1
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_256x32.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_256x32.lib

export DIE_AREA = 0 0 900 700
export CORE_AREA = 10 10 890 690
export MACRO_PLACEMENT = $(HW_ROOT)/openroad_hifp/h67_zkqi_production_macros.cfg
export MACRO_PLACE_HALO = 12 12
export MACRO_PLACE_CHANNEL = 24 24
export PLACE_DENSITY_LB_ADDON = 0.05
export CAP_MARGIN = 40
export SETUP_SLACK_MARGIN = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks -random_seed 42
