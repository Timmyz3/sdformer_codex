# T41 补充口径的接口约束：把输入/输出延时钉成 0，使「端口 → 组合逻辑 → FF」与
# T25/T39 的「reg → 组合逻辑 → reg」拿到同一条 4ns 预算（launch 0 / capture 4）。
# 不加这三行时 t41_fused_gate 全为 input→FF 路径，report_timing_summary 给 WNS=NA。
create_clock -period 4.000 -name clk [get_ports clk]
set_input_delay  -clock clk 0.0 [get_ports {in_yv in_thr_row in_valid}]
set_output_delay -clock clk 0.0 [get_ports {out_dec out_valid}]
