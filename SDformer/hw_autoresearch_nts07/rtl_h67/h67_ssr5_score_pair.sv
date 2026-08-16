`default_nettype none

// Strong baseline for MSSB5: the same five sufficient statistics and shared
// motion count, implemented as five independent balanced popcount trees.
module h67_ssr5_score_pair #(
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
    logic [31:0] overlap_bits0;
    logic [31:0] same_zero_bits0;
    logic [31:0] overlap_bits1;
    logic [31:0] same_zero_bits1;
    logic [31:0] motion_bits;

    initial begin
        if (HEAD_DIM != 32 || COUNT_W != 6)
            $error("h67_ssr5_score_pair requires HEAD_DIM=32 and COUNT_W=6");
    end

    assign overlap_bits0 = q_pair[31:0] & k_pair[31:0];
    assign same_zero_bits0 = ~(q_pair[31:0] | k_pair[31:0]);
    assign overlap_bits1 = q_pair[63:32] & k_pair[63:32];
    assign same_zero_bits1 = ~(q_pair[63:32] | k_pair[63:32]);
    assign motion_bits = k_pair[31:0] ^ k_pair[63:32];

    h67_balanced_popcount32 u_overlap0 (.bits(overlap_bits0), .count(overlap0));
    h67_balanced_popcount32 u_same_zero0 (.bits(same_zero_bits0), .count(same_zero0));
    h67_balanced_popcount32 u_overlap1 (.bits(overlap_bits1), .count(overlap1));
    h67_balanced_popcount32 u_same_zero1 (.bits(same_zero_bits1), .count(same_zero1));
    h67_balanced_popcount32 u_motion (.bits(motion_bits), .count(motion));

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
