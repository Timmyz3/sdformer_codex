`timescale 1ns/1ps
`default_nettype none

module local5_tare4_composite_top (
    input  logic          clk_core,
    input  logic          rst_core,

    input  logic          in_valid,
    output logic          in_ready,
    input  logic [15:0]   in_tag,
    input  logic [31:0]   in_q_self,
    input  logic [31:0]   in_k_self,
    input  logic [31:0]   in_k_neighbor,

    output logic          out_valid,
    input  logic          out_ready,
    output logic [15:0]   out_tag,
    output logic          out_mode_local5,
    output logic [1:0]    out_kind,
    output logic [5:0]    out_update_count,
    output logic [12:0]   out_raw16,
    output logic [8:0]    out_score_q7
);

    tare4_residual_composite_core core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_tag(in_tag),
        .in_q_anchor(in_q_self),
        .in_k_anchor(in_k_self),
        .in_q_target(in_q_self),
        .in_k_target(in_k_neighbor),
        .in_bias_raw16(10'd0),
        .in_mode_meta(1'b1),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_tag(out_tag),
        .out_mode_meta(out_mode_local5),
        .out_kind(out_kind),
        .out_update_count(out_update_count),
        .out_raw16(out_raw16),
        .out_score_q7(out_score_q7)
    );

endmodule

`default_nettype wire
