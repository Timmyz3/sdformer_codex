export DESIGN_NAME = qfit_local5_active_projection_tile
export DESIGN_NICKNAME = local5_active_projection_t450_proxy
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = \
  $(HW_ROOT)/rtl_qfit/qfit_dual_color_word_skipper_index.sv \
  $(HW_ROOT)/rtl_qfit/qfit_sync_relation_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv \
  $(HW_ROOT)/rtl_qfit/qfit_source_multicast_term_builder.sv \
  $(HW_ROOT)/rtl_qfit/qfit_tcfm5_acc_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_tcfm5_projection_top.sv \
  $(HW_ROOT)/rtl_qfit/qfit_linear5_projection_top.sv \
  $(HW_ROOT)/rtl_qfit/qfit_local5_active_projection_tile.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint.sdc)
export VERILOG_TOP_PARAMS ?= HEIGHT 15 WIDTH 15 TIME_PLANES 2 HEAD_DIM 32 OUT_DIM 2 BACKEND_KIND 0 RELATION_READ_LATENCY 1

# This run maps inferred memories into standard cells. It is a routing/timing
# stress proxy for the T450 control and address networks, not a memory-macro PPA.
export CORE_UTILIZATION = 35
export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks -random_seed 42
