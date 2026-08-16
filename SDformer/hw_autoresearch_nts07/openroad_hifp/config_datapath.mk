export DESIGN_NAME = gatestack_dctf96_term_datapath_top
export DESIGN_NICKNAME = hifp_dctf96_datapath_t6
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
  $(HW_ROOT)/rtl_hitflow/gatestack_dctf96_term_datapath_top.sv

export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint.sdc)
export VERILOG_TOP_PARAMS ?= TOKENS 6 ADAPTER_CONTEXTS 2 PPDI_ENABLE 0

export CORE_UTILIZATION = 45
export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks -random_seed 42
