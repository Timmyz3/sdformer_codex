export DESIGN_NAME = qfit_local5_1rw_active_projection_tile
export DESIGN_NICKNAME = local5_out32_allmacro_proxy
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = \
  $(HW_ROOT)/rtl_qfit/qfit_dual_color_word_skipper_index.sv \
  $(HW_ROOT)/rtl_qfit/qfit_sync_relation_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_fakeram45_relation_bank_450.sv \
  $(HW_ROOT)/rtl_qfit/qfit_dual_color_relation_frontier_sync.sv \
  $(HW_ROOT)/rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv \
  $(HW_ROOT)/rtl_qfit/qfit_source_multicast_term_builder.sv \
  $(HW_ROOT)/rtl_qfit/qfit_local5_color_map.sv \
  $(HW_ROOT)/rtl_qfit/qfit_fakeram45_acc_memory_90x1024.sv \
  $(HW_ROOT)/rtl_qfit/qfit_single_port_acc_memory.sv \
  $(HW_ROOT)/rtl_qfit/qfit_direct_1rw_acc_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_gasr2c_acc_bank.sv \
  $(HW_ROOT)/rtl_qfit/qfit_local5_1rw_projection_backend.sv \
  $(HW_ROOT)/rtl_qfit/qfit_local5_1rw_active_projection_tile.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint.sdc)
export VERILOG_TOP_PARAMS ?= MODE 1 GEOMETRY_SYNC_MODE 1 HEIGHT 15 WIDTH 15 TIME_PLANES 2 HEAD_DIM 32 OUT_DIM 32 GATE_W 9 W_W 8 ACC_W 32 RELATION_READ_LATENCY 1 RELATION_MEMORY_IMPL 1 ACC_MEMORY_IMPL 1

export ADDITIONAL_LEFS = \
  $(PLATFORM_DIR)/lef/fakeram45_128x256.lef \
  $(PLATFORM_DIR)/lef/fakeram45_256x16.lef \
  $(PLATFORM_DIR)/lef/fakeram45_256x32.lef
export ADDITIONAL_LIBS = \
  $(PLATFORM_DIR)/lib/fakeram45_128x256.lib \
  $(PLATFORM_DIR)/lib/fakeram45_256x16.lib \
  $(PLATFORM_DIR)/lib/fakeram45_256x32.lib

# Direct、Issue和DS共用该固定outline与32个SRAM宏。面积留白用于暴露
# bank-local宽向量寄存器、乘法器和宏间通道的真实布线代价。
export DIE_AREA = 0 0 2000 1600
export CORE_AREA = 20 20 1980 1580
export RTLMP_FLOW = True
export RTLMP_POST_HOOK = $(abspath $(dir $(DESIGN_CONFIG))/local5_acc_macro_orient.tcl)
export RTLMP_MAX_INST = 120000
export RTLMP_MIN_INST = 500
export RTLMP_MAX_MACRO = 40
export RTLMP_MIN_MACRO = 2
export MACRO_PLACE_HALO = 20 20
export MACRO_PLACE_CHANNEL = 40 40

export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
# Global routing reports five single-track overflow bins on the fixed outline.
# Keep the outline unchanged and let detailed routing decide whether these
# local overflows are physically resolvable; final DRC remains the sign-off gate.
export GLOBAL_ROUTE_ARGS = -congestion_iterations 20 -verbose -allow_congestion
# OpenROAD 547465c can assert in the intermediate antenna checker when a
# global guide contains local overflow. Detailed-route DRC/antenna remains
# mandatory; only this pre-detailed-route check is bypassed.
export SKIP_GLOBAL_ROUTE_ANTENNA = 1
# Stop the diagnostic detailed-route run before the observed M3-spacing
# oscillation and preserve markers/ODB for root-cause inspection.
export DETAILED_ROUTE_ARGS = -bottom_routing_layer metal2 -top_routing_layer metal10 -save_guide_updates -verbose 1 -droute_end_iter 16
export ABC_AREA = 0
# The optional Nangate45 extract_fa rewrite scales pathologically on this
# 1024-bit-vector top. ABC still maps the same arithmetic to the same library.
export ADDER_MAP_FILE =
# Preserve repeated bank hierarchy through Yosys. OpenROAD links the complete
# top-level netlist, while avoiding a prohibitively expensive cross-bank merge.
export SYNTH_ARGS =
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks -random_seed 42
