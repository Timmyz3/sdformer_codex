`default_nettype none

module ttx_tx_score_q7 #(
    parameter int HEAD_DIM   = 32,
    parameter int SCORE_W    = 16,
    parameter int SCORE_FRAC = 7,
    parameter int ALPHA0_Q8  = 5,
    parameter int COUNT_W    = $clog2(HEAD_DIM + 1)
)(
    input  logic [HEAD_DIM-1:0] q_bits,
    input  logic [HEAD_DIM-1:0] k_bits,
    output logic [COUNT_W-1:0]  q_active,
    output logic [COUNT_W-1:0]  k_active,
    output logic [COUNT_W-1:0]  overlap,
    output logic [COUNT_W-1:0]  same_zero,
    output logic signed [SCORE_W-1:0] score_q7
);
    localparam int SCORE_SHIFT = 8 + $clog2(HEAD_DIM) - SCORE_FRAC;
    localparam int NUM_W = COUNT_W + 9;

    integer bit_idx;
    logic [COUNT_W-1:0] q_count;
    logic [COUNT_W-1:0] k_count;
    logic [COUNT_W-1:0] overlap_count;
    logic [NUM_W-1:0] tx_num_q8;
    logic [NUM_W-1:0] rounded_num;

    always_comb begin
        q_count = '0;
        k_count = '0;
        overlap_count = '0;
        for (bit_idx = 0; bit_idx < HEAD_DIM; bit_idx = bit_idx + 1) begin
            q_count = q_count + COUNT_W'(q_bits[bit_idx]);
            k_count = k_count + COUNT_W'(k_bits[bit_idx]);
            overlap_count = overlap_count
                          + {{(COUNT_W-1){1'b0}}, (q_bits[bit_idx] && k_bits[bit_idx])};
        end

        q_active = q_count;
        k_active = k_count;
        overlap = overlap_count;
        same_zero = COUNT_W'(HEAD_DIM) - q_count - k_count + overlap_count;

        tx_num_q8 = (NUM_W'(overlap_count) << 8)
                  + (NUM_W'(same_zero) * NUM_W'(ALPHA0_Q8));
        rounded_num = tx_num_q8 + (NUM_W'(1) << (SCORE_SHIFT - 1));
        score_q7 = SCORE_W'(rounded_num) >>> SCORE_SHIFT;
    end
endmodule

`default_nettype wire
