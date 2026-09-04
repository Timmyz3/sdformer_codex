`timescale 1ns/1ps
`default_nettype none

// Bit-exact gate quantizer that uses the load-time certificate only when valid.
// Certificate failure or absence follows the frozen row_sum ceil-log2 path.
module h67_certified_gate_quant_q17 #(
    parameter int EXP_W = 16,
    parameter int SUM_W = 32,
    parameter int TOKEN_W = 9,
    parameter int GATE_W = 9,
    parameter int GATE_FRAC = 7,
    parameter int GATE_MAX = 2 << GATE_FRAC
) (
    input  logic [EXP_W-1:0]   exp_q8,
    input  logic [SUM_W-1:0]   row_sum_q8,
    input  logic [TOKEN_W-1:0] n_tokens,
    input  logic               preserve_mean,
    input  logic               certificate_valid,
    input  logic               certificate_pass,
    input  logic [5:0]         certified_shift,
    output logic               used_certificate,
    output logic [GATE_W-1:0]  gate_q17
);
    logic [5:0] fallback_shift_w;
    logic [5:0] denominator_shift_w;
    logic [TOKEN_W-1:0] token_scale_w;
    logic [EXP_W+TOKEN_W-1:0] exp_token_product_w;
    logic [31:0] scaled_w;
    logic [31:0] quotient_w;
    logic [31:0] remainder_w;
    logic [31:0] half_w;
    logic [31:0] rounded_w;

    ttx_ceil_log2_u32 u_fallback_shift (
        .value(row_sum_q8),
        .shift_amount(fallback_shift_w)
    );

    assign used_certificate = certificate_valid && certificate_pass;
    assign denominator_shift_w = used_certificate
                               ? certified_shift : fallback_shift_w;

    always_comb begin
        token_scale_w = preserve_mean ? n_tokens : TOKEN_W'(1);
        exp_token_product_w = exp_q8 * token_scale_w;
        scaled_w = 32'(exp_token_product_w) << GATE_FRAC;
        quotient_w = 32'd0;
        remainder_w = 32'd0;
        half_w = 32'd0;
        rounded_w = 32'd0;

        if (row_sum_q8 != 0 || used_certificate) begin
            if (denominator_shift_w == 0) begin
                rounded_w = scaled_w;
            end else begin
                quotient_w = scaled_w >> denominator_shift_w;
                remainder_w = scaled_w
                            - (quotient_w << denominator_shift_w);
                half_w = 32'd1 << (denominator_shift_w - 1'b1);
                rounded_w = quotient_w;
                if ((remainder_w > half_w)
                    || ((remainder_w == half_w) && quotient_w[0])) begin
                    rounded_w = quotient_w + 1'b1;
                end
            end
        end

        if (rounded_w > 32'(GATE_MAX))
            gate_q17 = GATE_W'(GATE_MAX);
        else
            gate_q17 = rounded_w[GATE_W-1:0];
    end
endmodule

`default_nettype wire
