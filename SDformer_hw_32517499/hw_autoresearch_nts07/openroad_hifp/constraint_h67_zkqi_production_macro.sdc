set clk_name core_clock
set clk_period 5.0
set clk_port [get_ports clk_core]

create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]
set_input_delay 0.5 -clock $clk_name $non_clock_inputs
set_output_delay 0.5 -clock $clk_name [all_outputs]
set_driving_cell -lib_cell BUF_X4 -pin Z $non_clock_inputs
set_load 0.01 [all_outputs]
