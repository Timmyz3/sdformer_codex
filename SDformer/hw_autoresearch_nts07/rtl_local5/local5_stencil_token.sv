`timescale 1ns/1ps
`default_nettype none

// One-token Local-5 stencil lane: 5 alpha-XNOR scores + Shiftmax5 gates.
// Candidate order: 0=self, 1=up, 2=down, 3=left, 4=right.
module local5_stencil_token #(
    parameter int HEAD_DIM = 32,
    parameter int N_CAND   = 5,
    parameter int SCORE_W  = 16,
    parameter int GATE_W   = 9
)(
    input  logic [HEAD_DIM-1:0] q_bits,
    input  logic [HEAD_DIM-1:0] k_bits [0:N_CAND-1],
    input  logic [N_CAND-1:0]         valid,
    output logic [N_CAND*SCORE_W-1:0] score_q7,
    output logic [N_CAND*GATE_W-1:0]  gate_q17
);
    genvar gi;
    generate
        for (gi = 0; gi < N_CAND; gi = gi + 1) begin : g_score
            local5_axnor_score_q7 #(
                .HEAD_DIM(HEAD_DIM),
                .SCORE_W(SCORE_W)
            ) u_score (
                .q_bits(q_bits),
                .k_bits(k_bits[gi]),
                .overlap(),
                .same_zero(),
                .score_q7(score_q7[gi*SCORE_W +: SCORE_W])
            );
        end
    endgenerate

    local5_shiftmax5_q17 #(
        .N_CAND(N_CAND),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W)
    ) u_sm (
        .score_q7(score_q7),
        .valid(valid),
        .gate_q17(gate_q17)
    );
endmodule

`default_nettype wire
