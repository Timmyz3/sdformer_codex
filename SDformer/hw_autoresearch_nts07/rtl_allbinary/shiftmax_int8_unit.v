`include "unibin_h60_pkg.vh"

module shiftmax_int8_unit #(
    parameter integer MAX_TOKENS = 8,
    parameter integer SCORE_W = `UBIN_SCORE_W,
    parameter integer GATE_W = `UBIN_GATE_W
)(
    input  wire signed [MAX_TOKENS*SCORE_W-1:0] scores_flat,
    input  wire [7:0]                           n_tokens,
    input  wire                                 preserve_mean,
    output reg  [MAX_TOKENS*GATE_W-1:0]         gates_flat
);
    integer i;
    reg signed [SCORE_W-1:0] score_i;
    reg signed [SCORE_W-1:0] row_max;
    reg [15:0] numerator [0:MAX_TOKENS-1];
    reg [31:0] row_sum;
    reg [31:0] scaled;
    reg [GATE_W-1:0] gate_value;
    reg [7:0] norm_shift;
    reg [31:0] bit_index;
    integer b;

    function [15:0] exp2_approx_q8;
        input signed [SCORE_W-1:0] delta;
        reg [7:0] shift_amt;
        reg signed [SCORE_W-1:0] abs_delta;
        begin
            // Scores are integer proxies in this RTL scaffold. Negative deltas
            // are approximated by power-of-two right shifts.
            if (delta >= 0) begin
                exp2_approx_q8 = 16'd256;
            end else begin
                abs_delta = -delta;
                shift_amt = (abs_delta > 8) ? 8'd8 : abs_delta[7:0];
                exp2_approx_q8 = 16'd256 >> shift_amt;
            end
        end
    endfunction

    function [7:0] ceil_log2_u32;
        input [31:0] value;
        integer j;
        reg [31:0] probe;
        begin
            ceil_log2_u32 = 8'd0;
            probe = (value <= 1) ? 32'd1 : (value - 1);
            for (j = 0; j < 32; j = j + 1) begin
                if (probe[j])
                    ceil_log2_u32 = j[7:0] + 8'd1;
            end
        end
    endfunction

    always @* begin
        gates_flat = {MAX_TOKENS*GATE_W{1'b0}};
        scaled = 32'd0;
        gate_value = {GATE_W{1'b0}};
        norm_shift = 8'd0;
        bit_index = 32'd0;
        row_max = scores_flat[0 +: SCORE_W];
        for (i = 0; i < MAX_TOKENS; i = i + 1) begin
            score_i = scores_flat[i*SCORE_W +: SCORE_W];
            if (i < n_tokens && score_i > row_max)
                row_max = score_i;
        end

        row_sum = 0;
        for (i = 0; i < MAX_TOKENS; i = i + 1) begin
            score_i = scores_flat[i*SCORE_W +: SCORE_W];
            if (i < n_tokens) begin
                numerator[i] = exp2_approx_q8(score_i - row_max);
                row_sum = row_sum + {16'd0, numerator[i]};
            end else begin
                numerator[i] = 0;
            end
        end
        if (row_sum == 0)
            row_sum = 1;
        norm_shift = ceil_log2_u32(row_sum);

        for (i = 0; i < MAX_TOKENS; i = i + 1) begin
            if (i < n_tokens) begin
                scaled = numerator[i] * ((1 << GATE_W) - 1);
                if (preserve_mean)
                    scaled = scaled * n_tokens;
                gate_value = {GATE_W{1'b0}};
                for (b = 0; b < GATE_W; b = b + 1) begin
                    bit_index = {24'd0, norm_shift} + b;
                    if (bit_index < 32)
                        gate_value[b] = scaled[bit_index];
                end
                gates_flat[i*GATE_W +: GATE_W] = gate_value;
            end
        end
    end
endmodule
