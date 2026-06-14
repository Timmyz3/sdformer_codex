`include "nts07_pkg.vh"

// Shiftmax over a single token row (scores already fused TX+mu*SC).
// gate[i] = 2^(s[i]-max) / 2^ceil(log2(sum 2^(s[i]-max)))
module shiftmax_unit #(
    parameter integer MAX_TOKENS = 98,
    parameter integer SCORE_W = 6,
    parameter integer GATE_W = 8
)(
    input  wire signed [SCORE_W-1:0] scores [0:MAX_TOKENS-1],
    input  wire [6:0]                n_tokens,
    input  wire                      preserve_mean,
    output wire [GATE_W-1:0]         gates [0:MAX_TOKENS-1]
);
    integer i;
    reg signed [SCORE_W-1:0] row_max;
    reg signed [SCORE_W-1:0] shifted;
    reg [31:0] numerator [0:MAX_TOKENS-1];
    reg [31:0] row_sum;
    reg [4:0]  denom_pow;
    reg [31:0] denominator;

    function automatic [31:0] pow2_shifted(input integer s);
        begin
            if (s >= 0)
                pow2_shifted = 32'd1 << s;
            else if (s > -32)
                pow2_shifted = 32'd1 >> (-s);
            else
                pow2_shifted = 32'd0;
        end
    endfunction

    function automatic [4:0] ceil_log2(input [31:0] x);
        integer b;
        begin
            ceil_log2 = 0;
            if (x == 0) begin
                ceil_log2 = 0;
            end else begin
                for (b = 31; b >= 0; b = b - 1) begin
                    if (x > (32'd1 << b)) begin
                        ceil_log2 = b + 1;
                        b = -1;
                    end
                end
            end
        end
    endfunction

    always @* begin
        row_max = scores[0];
        for (i = 1; i < MAX_TOKENS; i = i + 1) begin
            if (i < n_tokens && scores[i] > row_max)
                row_max = scores[i];
        end

        row_sum = 0;
        for (i = 0; i < MAX_TOKENS; i = i + 1) begin
            if (i < n_tokens) begin
                shifted = scores[i] - row_max;
                numerator[i] = pow2_shifted(shifted);
                row_sum = row_sum + numerator[i];
            end else begin
                numerator[i] = 0;
            end
        end

        denom_pow = ceil_log2(row_sum);
        denominator = 32'd1 << denom_pow;
        if (denominator == 0)
            denominator = 32'd1;

        for (i = 0; i < MAX_TOKENS; i = i + 1) begin
            if (i < n_tokens) begin
                gates[i] = (numerator[i] * (preserve_mean ? n_tokens : 8'd1) * ((1<<GATE_W)-1)) / denominator;
            end else begin
                gates[i] = 0;
            end
        end
    end
endmodule