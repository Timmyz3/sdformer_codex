export DESIGN_NAME = hitflow_banked_accumulator
export DESIGN_NICKNAME = hifp_accumulator_t6
export PLATFORM = nangate45

HW_ROOT := $(abspath $(dir $(DESIGN_CONFIG))/..)

export VERILOG_FILES = $(HW_ROOT)/rtl_hitflow/hitflow_banked_accumulator.sv
export SDC_FILE = $(abspath $(dir $(DESIGN_CONFIG))/constraint.sdc)
export VERILOG_TOP_PARAMS ?= TOKENS 6 BANKS 2 OUT_TILE 32

export CORE_UTILIZATION = 45
export PLACE_DENSITY_LB_ADDON = 0.05
export TNS_END_PERCENT = 100
export ABC_AREA = 0
export SYNTH_REPEATABLE_BUILD = 1
export DISABLE_GUI_IMAGES = 1
export IO_PLACER_H = metal5
export IO_PLACER_V = metal6
export PLACE_PINS_ARGS = -min_distance 1 -min_distance_in_tracks -random_seed 42
