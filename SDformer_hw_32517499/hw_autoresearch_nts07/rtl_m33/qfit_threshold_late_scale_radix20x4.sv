`timescale 1ns/1ps
`default_nettype none

// M33 exact threshold late-scale engine.
//
// Four signed Acc32 values share one signed Q24 threshold.  Each accumulator
// is represented by five balanced radix-128 signed-INT8 digits, and the
// threshold by four digits.  The 20 products/output occupy 80 lanes of the
// existing 96-lane multiplier pool; the remaining 16 lanes are forced idle.
// Full signed56 products are retained here.  Q-format selection, RNE,
// saturation, and post-scale bias are deliberately outside this arithmetic
// identity block and therefore cannot be claimed from this module alone.
module qfit_threshold_late_scale_radix20x4 #(
    parameter int TAG_W = 48,
    localparam int OUTPUTS = 4,
    localparam int ACC_W = 32,
    localparam int THRESHOLD_W = 24,
    localparam int ACC_DIGITS = 5,
    localparam int THRESHOLD_DIGITS = 4,
    localparam int PRODUCTS_PER_OUTPUT = 20,
    localparam int MULTIPLIERS = 96,
    localparam int PRODUCT_W = 16,
    localparam int RESULT_W = 56,
    localparam int RECOMBINE_W = 64,
    localparam int ACC_RESIDUAL_W = 36,
    localparam int THRESHOLD_RESIDUAL_W = 29
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              input_valid,
    output logic                              input_ready,
    input  logic [TAG_W-1:0]                  input_tag,
    input  logic [OUTPUTS-1:0]                input_valid_bits,
    input  logic signed [ACC_W-1:0]           input_accumulator [0:OUTPUTS-1],
    input  logic signed [THRESHOLD_W-1:0]     input_threshold,

    output logic                              output_valid,
    input  logic                              output_ready,
    output logic [TAG_W-1:0]                  output_tag,
    output logic [OUTPUTS-1:0]                output_valid_bits,
    output logic signed [RESULT_W-1:0]        output_product [0:OUTPUTS-1],

    output logic [MULTIPLIERS-1:0]            multiplier_active_mask,
    output logic                              digit_residual_zero,
    output logic                              recombination_fits_signed56,
    output logic                              protocol_error
);
    logic signed [ACC_RESIDUAL_W-1:0] acc_residual
        [0:OUTPUTS-1][0:ACC_DIGITS];
    logic signed [THRESHOLD_RESIDUAL_W-1:0] threshold_residual
        [0:THRESHOLD_DIGITS];
    logic signed [7:0] acc_digit [0:OUTPUTS-1][0:ACC_DIGITS-1];
    logic signed [7:0] threshold_digit [0:THRESHOLD_DIGITS-1];
    logic signed [7:0] multiplier_a [0:MULTIPLIERS-1];
    logic signed [7:0] multiplier_b [0:MULTIPLIERS-1];
    wire signed [PRODUCT_W-1:0] multiplier_product [0:MULTIPLIERS-1];
    logic signed [RECOMBINE_W-1:0] recombined [0:OUTPUTS-1];
    logic input_fire;

    function automatic logic signed [RECOMBINE_W-1:0]
        widen_shift_product(
            input logic signed [PRODUCT_W-1:0] product,
            input integer shift
        );
        logic signed [RECOMBINE_W-1:0] wide_product;
        begin
            wide_product = {
                {(RECOMBINE_W-PRODUCT_W){product[PRODUCT_W-1]}}, product
            };
            widen_shift_product = wide_product <<< shift;
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (OUTPUTS != 4 || ACC_W != 32 || THRESHOLD_W != 24
            || ACC_DIGITS != 5 || THRESHOLD_DIGITS != 4
            || PRODUCTS_PER_OUTPUT != 20 || MULTIPLIERS != 96
            || PRODUCT_W != 16 || RESULT_W != 56
            || RECOMBINE_W != 64 || ACC_RESIDUAL_W < 33
            || THRESHOLD_RESIDUAL_W < 25)
            $fatal(1, "M33 arithmetic/resource contract drift");
    end
`endif

    assign input_ready = !output_valid || output_ready;
    assign input_fire = input_valid && input_ready;

    // Wider residuals are mandatory: INT32_MAX-(-1) and
    // INT24_MAX-(-1) overflow in their original signed widths.
    always_comb begin : balanced_radix_conversion
        for (int output_index = 0; output_index < OUTPUTS; output_index++) begin
            acc_residual[output_index][0] = {
                {(ACC_RESIDUAL_W-ACC_W){input_accumulator[output_index][ACC_W-1]}},
                input_accumulator[output_index]
            };
            for (int digit_index = 0; digit_index < ACC_DIGITS;
                 digit_index++) begin
                // Sign-copy bit 6 into bit 7 maps low residues 64..127 to
                // balanced digits -64..-1 without an undersized subtraction.
                acc_digit[output_index][digit_index] = {
                    acc_residual[output_index][digit_index][6],
                    acc_residual[output_index][digit_index][6:0]
                };
                acc_residual[output_index][digit_index+1]
                    = ($signed(acc_residual[output_index][digit_index])
                        - $signed({
                            {(ACC_RESIDUAL_W-8){
                                acc_digit[output_index][digit_index][7]}},
                            acc_digit[output_index][digit_index]
                        })) >>> 7;
            end
        end

        threshold_residual[0] = {
            {(THRESHOLD_RESIDUAL_W-THRESHOLD_W){
                input_threshold[THRESHOLD_W-1]}},
            input_threshold
        };
        for (int digit_index = 0; digit_index < THRESHOLD_DIGITS;
             digit_index++) begin
            threshold_digit[digit_index] = {
                threshold_residual[digit_index][6],
                threshold_residual[digit_index][6:0]
            };
            threshold_residual[digit_index+1]
                = ($signed(threshold_residual[digit_index])
                    - $signed({
                        {(THRESHOLD_RESIDUAL_W-8){
                            threshold_digit[digit_index][7]}},
                        threshold_digit[digit_index]
                    })) >>> 7;
        end
    end

    always_comb begin : select_late_scale_operands
        multiplier_active_mask = '0;
        for (int lane = 0; lane < MULTIPLIERS; lane++) begin
            multiplier_a[lane] = '0;
            multiplier_b[lane] = '0;
        end
        for (int output_index = 0; output_index < OUTPUTS; output_index++) begin
            for (int acc_digit_index = 0; acc_digit_index < ACC_DIGITS;
                 acc_digit_index++) begin
                for (int threshold_digit_index = 0;
                     threshold_digit_index < THRESHOLD_DIGITS;
                     threshold_digit_index++) begin
                    if (input_fire && input_valid_bits[output_index]) begin
                        multiplier_a[(output_index*PRODUCTS_PER_OUTPUT)
                            +(acc_digit_index*THRESHOLD_DIGITS)
                            +threshold_digit_index]
                            = acc_digit[output_index][acc_digit_index];
                        multiplier_b[(output_index*PRODUCTS_PER_OUTPUT)
                            +(acc_digit_index*THRESHOLD_DIGITS)
                            +threshold_digit_index]
                            = threshold_digit[threshold_digit_index];
                        multiplier_active_mask[
                            (output_index*PRODUCTS_PER_OUTPUT)
                            +(acc_digit_index*THRESHOLD_DIGITS)
                            +threshold_digit_index] = 1'b1;
                    end
                end
            end
        end
    end

    qfit_signed_int8_mul96_pool #(
        .MULTIPLIERS(MULTIPLIERS), .IN_W(8)
    ) u_mul_pool (
        .operand_a(multiplier_a),
        .operand_b(multiplier_b),
        .product(multiplier_product)
    );

    // Every partial product is widened before shifting; all addition is
    // explicitly signed64.  signed56 is checked before truncation.
    always_comb begin : recombine_radix_products
        for (int output_index = 0; output_index < OUTPUTS; output_index++) begin
            recombined[output_index] = '0;
            for (int acc_digit_index = 0; acc_digit_index < ACC_DIGITS;
                 acc_digit_index++) begin
                for (int threshold_digit_index = 0;
                     threshold_digit_index < THRESHOLD_DIGITS;
                     threshold_digit_index++) begin
                    recombined[output_index] = $signed(recombined[output_index])
                        + widen_shift_product(
                            multiplier_product[
                                (output_index*PRODUCTS_PER_OUTPUT)
                                +(acc_digit_index*THRESHOLD_DIGITS)
                                +threshold_digit_index],
                            7*(acc_digit_index+threshold_digit_index)
                        );
                end
            end
        end
    end

    always_comb begin : arithmetic_guards
        digit_residual_zero = threshold_residual[THRESHOLD_DIGITS] == 0;
        recombination_fits_signed56 = 1'b1;
        for (int output_index = 0; output_index < OUTPUTS; output_index++) begin
            digit_residual_zero &= acc_residual[output_index][ACC_DIGITS] == 0;
            if (input_valid_bits[output_index])
                recombination_fits_signed56 &=
                    recombined[output_index][RECOMBINE_W-1:RESULT_W]
                    == {(RECOMBINE_W-RESULT_W){
                        recombined[output_index][RESULT_W-1]}};
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            output_valid <= 1'b0;
            output_tag <= '0;
            output_valid_bits <= '0;
            protocol_error <= 1'b0;
            for (int output_index = 0; output_index < OUTPUTS; output_index++)
                output_product[output_index] <= '0;
        end else begin
            if (input_fire && (!digit_residual_zero
                    || !recombination_fits_signed56))
                protocol_error <= 1'b1;
            if (input_ready) begin
                output_valid <= input_valid;
                if (input_valid) begin
                    output_tag <= input_tag;
                    output_valid_bits <= input_valid_bits;
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++)
                        output_product[output_index]
                            <= recombined[output_index][RESULT_W-1:0];
                end
            end
        end
    end
endmodule

`default_nettype wire
