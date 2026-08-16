export DESIGN_NAME = qfit_local5_active_projection_tile
export DESIGN_NICKNAME = local5_relation_macro_t450_proxy
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = \
  $(HW_ROOT)/rtl_qfit/qfit_dual_color_word_skipper_index.sv \
  $(HW_ROOT)/rtl_qfit/qfit_sync_relation_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_fakeram45_relation_bank_450.sv \
  $(HW_ROOT)/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv \
  $(HW_ROOT)/rtl_qfit/qfit_source_multicast_term_builder.sv \
  $(HW_ROOT)/rtl_qfit/qfit_tcfm5_acc_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_tcfm5_projection_top.sv \
  $(HW_ROOT)/rtl_qfit/qfit_linear5_projection_top.sv \
  $(HW_ROOT)/rtl_qfit/qfit_local5_active_projection_tile.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint.sdc)
export VERILOG_TOP_PARAMS ?= HEIGHT 15 WIDTH 15 TIME_PLANES 2 HEAD_DIM 32 OUT_DIM 2 BACKEND_KIND 0 RELATION_READ_LATENCY 1 RELATION_MEMORY_IMPL 1

export ADDITIONAL_LEFS = \
  $(PLATFORM_DIR)/lef/fakeram45_256x16.lef \
  $(PLATFORM_DIR)/lef/fakeram45_256x32.lef
export ADDITIONAL_LIBS = \
  $(PLATFORM_DIR)/lib/fakeram45_256x16.lib \
  $(PLATFORM_DIR)/lib/fakeram45_256x32.lib

# Equal fixed outline for both backends. The 12 relation macros consume about
# 52,085 um^2. Accumulator RMW arrays remain standard-cell mapped because this
# open platform does not provide a true 1R1W macro.
export DIE_AREA = 0 0 700 550
export CORE_AREA = 10 10 690 540
export RTLMP_FLOW = True
export RTLMP_MAX_INST = 60000
export RTLMP_MIN_INST = 1000
export RTLMP_MAX_MACRO = 16
export RTLMP_MIN_MACRO = 2
export MACRO_PLACE_HALO = 8 8
export MACRO_PLACE_CHANNEL = 16 16

export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks -random_seed 42
