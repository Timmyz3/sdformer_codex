`default_nettype none

// Fair active-score baseline used by the TTB8-ZKQI row top: two independent
// H67 score engines evaluate both temporal positions in parallel.
module h67_direct_score_pair #(
    parameter int HEAD_DIM = 32,
    parameter int SCORE_W = 16,
    parameter int COUNT_W = $clog2(HEAD_DIM + 1)
) (
    input  logic [2*HEAD_DIM-1:0] q_pair,
    input  logic [2*HEAD_DIM-1:0] k_pair,
    output logic [COUNT_W-1:0] overlap0,
    output logic [COUNT_W-1:0] same_zero0,
    output logic [COUNT_W-1:0] overlap1,
    output logic [COUNT_W-1:0] same_zero1,
    output logic [COUNT_W-1:0] motion,
    output logic signed [SCORE_W-1:0] score0_q7,
    output logic signed [SCORE_W-1:0] score1_q7
);
    logic [COUNT_W-1:0] motion1_unused;

    h67_motionxor_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .COUNT_W(COUNT_W),
        .ENABLE_MOTION_XOR(1'b1)
    ) u_score0 (
        .q_bits(q_pair[HEAD_DIM-1:0]),
        .k_current_bits(k_pair[HEAD_DIM-1:0]),
        .k_peer_bits(k_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .overlap(overlap0),
        .same_zero(same_zero0),
        .motion_xor(motion),
        .score_q7(score0_q7)
    );

    h67_motionxor_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .COUNT_W(COUNT_W),
        .ENABLE_MOTION_XOR(1'b1)
    ) u_score1 (
        .q_bits(q_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .k_current_bits(k_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .k_peer_bits(k_pair[HEAD_DIM-1:0]),
        .overlap(overlap1),
        .same_zero(same_zero1),
        .motion_xor(motion1_unused),
        .score_q7(score1_q7)
    );
endmodule

`default_nettype wire
