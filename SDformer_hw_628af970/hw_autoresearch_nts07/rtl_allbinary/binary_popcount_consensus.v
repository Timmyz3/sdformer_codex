`include "unibin_h60_pkg.vh"

module binary_popcount_consensus #(
    parameter integer HEAD_DIM = `UBIN_HEAD_DIM,
    parameter integer SCORE_W = `UBIN_SCORE_W,
    parameter integer SCORE_FRAC = `UBIN_SCORE_FRAC,
    parameter integer ALPHA0_Q8 = `UBIN_ALPHA0_Q8
)(
    input  wire [HEAD_DIM-1:0] q_bits,
    input  wire [HEAD_DIM-1:0] k_bits,
    input  wire [7:0]          mu_q8,
    output reg  [7:0]          q_active,
    output reg  [7:0]          k_active,
    output reg  [7:0]          overlap,
    output reg  [7:0]          mismatch,
    output reg  signed [SCORE_W-1:0] tx_score,
    output reg  signed [SCORE_W-1:0] sc_score,
    output reg  signed [SCORE_W-1:0] fused_score
);
    integer i;
    reg [7:0] q_cnt;
    reg [7:0] k_cnt;
    reg [7:0] ov_cnt;
    reg [7:0] same_zero_cnt;
    reg [31:0] tx_num_q8;
    reg [31:0] sc_num_q8;
    reg [39:0] mu_sc_q16;
    reg [39:0] fused_num_q8;
    reg [39:0] tx_rounded_q7;
    reg [39:0] sc_rounded_q7;
    reg [39:0] fused_rounded_q7;
    reg [SCORE_W-1:0] tx_score_q7;
    reg [SCORE_W-1:0] sc_score_q7;
    reg [SCORE_W-1:0] fused_score_q7;
    reg [39:0] score_denominator;
    reg [39:0] half_denominator;

    always @* begin
        q_cnt = 0;
        k_cnt = 0;
        ov_cnt = 0;
        for (i = 0; i < HEAD_DIM; i = i + 1) begin
            q_cnt = q_cnt + {7'd0, q_bits[i]};
            k_cnt = k_cnt + {7'd0, k_bits[i]};
            ov_cnt = ov_cnt + {7'd0, (q_bits[i] & k_bits[i])};
        end

        q_active = q_cnt;
        k_active = k_cnt;
        overlap = ov_cnt;
        mismatch = q_cnt + k_cnt - (ov_cnt << 1);
        same_zero_cnt = HEAD_DIM[7:0] - q_cnt - k_cnt + ov_cnt;

        // Deployment raw score proxy aligned with rtl_dc before row centering:
        // TX = (overlap + alpha0 * same_zero) / head_dim
        // SC = overlap / head_dim
        // score = TX + mu * SC, quantized to Q(SCORE_FRAC).
        score_denominator = 40'(HEAD_DIM) << (8 - SCORE_FRAC);
        half_denominator = score_denominator >> 1;
        tx_num_q8 = ({24'd0, ov_cnt} << 8) + ({24'd0, same_zero_cnt} * ALPHA0_Q8[7:0]);
        sc_num_q8 = ({24'd0, ov_cnt} << 8);
        mu_sc_q16 = {8'd0, sc_num_q8} * {32'd0, mu_q8};
        fused_num_q8 = {8'd0, tx_num_q8} + (mu_sc_q16 >> 8);
        tx_rounded_q7 = {8'd0, tx_num_q8} + half_denominator;
        sc_rounded_q7 = {8'd0, sc_num_q8} + half_denominator;
        fused_rounded_q7 = fused_num_q8 + half_denominator;
        tx_score_q7 = SCORE_W'(tx_rounded_q7 / score_denominator);
        sc_score_q7 = SCORE_W'(sc_rounded_q7 / score_denominator);
        fused_score_q7 = SCORE_W'(fused_rounded_q7 / score_denominator);
        tx_score = $signed(tx_score_q7);
        sc_score = $signed(sc_score_q7);
        fused_score = $signed(fused_score_q7);
    end
endmodule
