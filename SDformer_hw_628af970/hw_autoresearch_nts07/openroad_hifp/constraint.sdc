set clk_name core_clock
set clk_period 5.0
set clk_port [get_ports clk_core]

create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]
set_input_delay 0.5 -clock $clk_name $non_clock_inputs
set_output_delay 0.5 -clock $clk_name [all_outputs]
set_driving_cell -lib_cell BUF_X4 -pin Z $non_clock_inputs
set_load 0.01 [all_outputs]

# DATA_W=10 的关系存储映射到 256x16 代理宏时，高 6 位在 RTL 中恒为零。
# 仅关闭这 5 role x 2 depth-bank x 6 padding pin 的 setup/hold 分析；
# 诊断脚本会检查匹配数必须恰好为 60，避免掩盖真实数据路径。
set relation_padding_pins {}
foreach pin [get_pins -hierarchical */wd_in*] {
  set pin_name [get_full_name $pin]
  if {[regexp {g_w10\.g_bank\[[01]\]\.u_macro/wd_in\[(10|11|12|13|14|15)\]$} \
      $pin_name]} {
    lappend relation_padding_pins $pin
  }
}
if {[llength $relation_padding_pins] > 0} {
  foreach pin $relation_padding_pins {
    set_disable_timing -to [get_name $pin] [get_cells -of_objects $pin]
  }
}
