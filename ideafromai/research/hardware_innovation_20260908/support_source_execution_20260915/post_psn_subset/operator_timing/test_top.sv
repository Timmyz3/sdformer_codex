module test_top(
 input logic clk_core,rst_core,cfg_we,cfg_half,
 input logic[4:0] cfg_addr,input logic signed[15:0] cfg_data,
 input logic context_we,codes_we,group_go,step_go,
 input logic signed[47:0] prefix_in,tau_in,p_in,n_in,
 input logic[4:0] lo_code_in,hi_code_in,m_in,
 input logic positive_in,constant_in,constant_gate_in,
 output logic[1:0] gate_out,locked_out,lower_hit_out,upper_hit_out
);
 operator_slice #(.NORMALIZED(0)) original(.*, .gate_out(gate_out[0]),.locked_out(locked_out[0]),.lower_hit_out(lower_hit_out[0]),.upper_hit_out(upper_hit_out[0]));
 operator_slice #(.NORMALIZED(1)) normalized(.*, .gate_out(gate_out[1]),.locked_out(locked_out[1]),.lower_hit_out(lower_hit_out[1]),.upper_hit_out(upper_hit_out[1]));
endmodule

