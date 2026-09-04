`default_nettype none

// Conventional optimized baseline: share K0^K1 across both temporal scores,
// but retain the q-count/k-count/overlap formulation used by the original H67
// score engine. Seven independent balanced popcount trees are required.
module h67_cse7_score_pair #(
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
    localparam int RAW_W = COUNT_W + 3;
    logic [COUNT_W-1:0] q_count0;
    logic [COUNT_W-1:0] k_count0;
    logic [COUNT_W-1:0] q_count1;
    logic [COUNT_W-1:0] k_count1;

    initial begin
        if (HEAD_DIM != 32 || COUNT_W != 6)
            $error("h67_cse7_score_pair requires HEAD_DIM=32 and COUNT_W=6");
    end

    h67_balanced_popcount32 u_q0 (.bits(q_pair[31:0]), .count(q_count0));
    h67_balanced_popcount32 u_k0 (.bits(k_pair[31:0]), .count(k_count0));
    h67_balanced_popcount32 u_o0 (
        .bits(q_pair[31:0] & k_pair[31:0]), .count(overlap0)
    );
    h67_balanced_popcount32 u_q1 (.bits(q_pair[63:32]), .count(q_count1));
    h67_balanced_popcount32 u_k1 (.bits(k_pair[63:32]), .count(k_count1));
    h67_balanced_popcount32 u_o1 (
        .bits(q_pair[63:32] & k_pair[63:32]), .count(overlap1)
    );
    h67_balanced_popcount32 u_motion (
        .bits(k_pair[31:0] ^ k_pair[63:32]), .count(motion)
    );

    assign same_zero0 = COUNT_W'(HEAD_DIM) - q_count0 - k_count0 + overlap0;
    assign same_zero1 = COUNT_W'(HEAD_DIM) - q_count1 - k_count1 + overlap1;

    function automatic logic signed [SCORE_W-1:0] finalize_score(
        input logic [COUNT_W-1:0] overlap_count,
        input logic [COUNT_W-1:0] same_zero_count,
        input logic [COUNT_W-1:0] motion_count
    );
        logic [COUNT_W-1:0] silence_integer;
        logic [3:0] silence_remainder;
        logic [RAW_W-1:0] score_integer;
        logic silence_increment;
        begin
            silence_integer = same_zero_count >> 4;
            silence_remainder = same_zero_count[3:0];
            score_integer = (RAW_W'(overlap_count) << 2)
                          + RAW_W'(motion_count)
                          + RAW_W'(silence_integer);
            silence_increment = (silence_remainder > 4'd8)
                             || ((silence_remainder == 4'd8) && score_integer[0]);
            finalize_score = SCORE_W'(score_integer + RAW_W'(silence_increment));
        end
    endfunction

    assign score0_q7 = finalize_score(overlap0, same_zero0, motion);
    assign score1_q7 = finalize_score(overlap1, same_zero1, motion);
endmodule

`default_nettype wire
