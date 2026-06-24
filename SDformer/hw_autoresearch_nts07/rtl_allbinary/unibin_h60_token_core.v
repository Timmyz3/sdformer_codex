`include "unibin_h60_pkg.vh"

module unibin_h60_token_core #(
    parameter integer HEAD_DIM = `UBIN_HEAD_DIM,
    parameter integer SCORE_W = `UBIN_SCORE_W,
    parameter integer DATA_W = `UBIN_DATA_W,
    parameter integer GATE_W = `UBIN_GATE_W
)(
    input  wire [HEAD_DIM-1:0]      q_bits,
    input  wire [HEAD_DIM-1:0]      k_bits,
    input  wire signed [DATA_W-1:0] k_value,
    input  wire [7:0]               mu_q8,
    input  wire [GATE_W-1:0]        gate,
    output wire                     empty_token,
    output wire [7:0]               q_active,
    output wire [7:0]               k_active,
    output wire [7:0]               overlap,
    output wire [7:0]               mismatch,
    output wire signed [SCORE_W-1:0] fused_score,
    output wire signed [DATA_W+GATE_W-1:0] gated_k
);
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [SCORE_W-1:0] tx_score_unused;
    wire signed [SCORE_W-1:0] sc_score_unused;
    /* verilator lint_on UNUSEDSIGNAL */

    binary_popcount_consensus #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W)
    ) u_score (
        .q_bits(q_bits),
        .k_bits(k_bits),
        .mu_q8(mu_q8),
        .q_active(q_active),
        .k_active(k_active),
        .overlap(overlap),
        .mismatch(mismatch),
        .tx_score(tx_score_unused),
        .sc_score(sc_score_unused),
        .fused_score(fused_score)
    );

    assign empty_token = ((q_bits | k_bits) == {HEAD_DIM{1'b0}});

    gated_k_unit #(
        .DATA_W(DATA_W),
        .GATE_W(GATE_W)
    ) u_gated_k (
        .k_event(|k_bits),
        .k_value(k_value),
        .gate(gate),
        .gated_out(gated_k)
    );
endmodule
