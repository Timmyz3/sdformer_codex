`timescale 1ns/1ps
`default_nettype none

module dual_mode_tare4_composite_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic in_mode_local5,
    input logic [31:0] in_q_anchor,
    input logic [31:0] in_k_anchor,
    input logic [31:0] in_k_target,
    input logic [31:0] selected_q_target,
    input logic [9:0] selected_bias_raw16
);

    property p_local5_mode_contract;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready && in_mode_local5 |->
            selected_q_target == in_q_anchor &&
            selected_bias_raw16 == 0;
    endproperty

    property p_motion_bias_contract;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready && !in_mode_local5 |->
            selected_bias_raw16 ==
                10'($countones(in_k_anchor ^ in_k_target) << 4);
    endproperty

    assert property (p_local5_mode_contract);
    assert property (p_motion_bias_contract);

endmodule

`default_nettype wire
