`default_nettype none

module ttx_zero_k_class_score_q7 #(
    parameter int HEAD_DIM   = 32,
    parameter int SCORE_W    = 16,
    parameter int SCORE_FRAC = 7,
    parameter int ALPHA0_Q8  = 5,
    parameter int COUNT_W    = $clog2(HEAD_DIM + 1)
)(
    input  logic [COUNT_W-1:0] q_active,
    output logic signed [SCORE_W-1:0] score_q7
);
    localparam int SCORE_SHIFT = 8 + $clog2(HEAD_DIM) - SCORE_FRAC;
    localparam int NUM_W = COUNT_W + 9;

    logic [COUNT_W-1:0] same_zero;
    logic [NUM_W-1:0] tx_num_q8;
    logic [NUM_W-1:0] rounded_num;

    always_comb begin
        same_zero = COUNT_W'(HEAD_DIM) - q_active;
        tx_num_q8 = NUM_W'(same_zero) * NUM_W'(ALPHA0_Q8);
        rounded_num = tx_num_q8 + (NUM_W'(1) << (SCORE_SHIFT - 1));
        score_q7 = SCORE_W'(rounded_num) >>> SCORE_SHIFT;
    end
endmodule

`default_nettype wire
