# GateStack单context执行顶层探索约束。正式DC前必须按目标工艺复核。
create_clock -name core_clk -period 2.000 -waveform {0.000 1.000} [get_ports clk_core]
set_clock_uncertainty -setup 0.200 [get_clocks core_clk]
set_clock_uncertainty -hold 0.050 [get_clocks core_clk]

set data_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_delay 0.250 -clock core_clk $data_inputs
set_input_transition 0.100 $data_inputs
set_output_delay 0.250 -clock core_clk [all_outputs]
set_load 0.010 [all_outputs]

# rst_core是同步复位；fabric abort pulse也在clk_core域内生成，不设置false path。
set_max_fanout 32 [current_design]
set_fix_multiple_port_nets -all -buffer_constants
