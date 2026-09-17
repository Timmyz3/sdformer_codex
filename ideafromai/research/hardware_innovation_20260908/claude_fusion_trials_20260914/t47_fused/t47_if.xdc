# T47 的接口约束：把输入/输出延时钉成 0，使「端口 → 组合逻辑 → FF」与 T25/T39/T41 的
# 「reg → 组合逻辑 → reg」拿到同一条 4ns 预算（launch 0 / capture 4），WNS 才可并列。
create_clock -period 4.000 -name clk [get_ports clk]
set_input_delay  -clock clk 0.0 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0.0 [all_outputs]
