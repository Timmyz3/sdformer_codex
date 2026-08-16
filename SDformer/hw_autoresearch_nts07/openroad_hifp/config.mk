export DESIGN_NAME = gatestack_dctf96_banklocal_projection_top
export DESIGN_NICKNAME = hifp_projection_t6_proxy
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = \
  $(HW_ROOT)/rtl_hitflow/gatestack_decoupled_product_engine.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf32_bank_executor.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf_term_event_adapter.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf_term_event_adapter_2c.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_ppdi_token_bank.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_ppdi_term_event_adapter_2c.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf_term_fabric.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_ppdi_dctf_term_fabric.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf96_term_datapath_top.sv \
  $(HW_ROOT)/rtl_hitflow/hitflow_banked_accumulator.sv \
  $(HW_ROOT)/rtl_hitflow/hitflow_implicit_bias_finalizer_accumulator.sv \
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf96_banklocal_projection_top.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint.sdc)

# T=6 keeps the first physical pass tractable. It is a routing/timing proxy,
# not a full-resolution area result. The four modes override the final bits.
export VERILOG_TOP_PARAMS ?= TOKENS 6 ADAPTER_CONTEXTS 2 PPDI_ENABLE 0 IMPLICIT_BIAS_FINALIZE_ENABLE 0

export CORE_UTILIZATION = 30
export PLACE_DENSITY = 0.48
export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1

# The current top exposes wide SRAM-facing vectors as block ports. Keep all
# four modes under the same pin policy so relative routing remains comparable.
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks
