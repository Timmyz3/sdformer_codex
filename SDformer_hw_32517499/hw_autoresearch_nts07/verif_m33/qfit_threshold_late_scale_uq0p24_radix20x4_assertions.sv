`timescale 1ns/1ps
`default_nettype none

module qfit_threshold_late_scale_uq0p24_radix20x4_assertions #(
    parameter int TAG_W = 48,
    localparam int OUTPUTS = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic input_valid,
    input logic input_ready,
    input logic [TAG_W-1:0] input_tag,
    input logic [OUTPUTS-1:0] input_valid_bits,
    input logic signed [31:0] input_accumulator [0:OUTPUTS-1],
    input logic [23:0] input_threshold_uq0p24,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [OUTPUTS-1:0] output_valid_bits,
    input logic signed [55:0] output_product [0:OUTPUTS-1],
    input logic [95:0] multiplier_active_mask,
    input logic digit_residual_zero,
    input logic recombination_fits_signed56,
    input logic protocol_error
);
    logic [95:0] expected_active_mask;

    function automatic logic signed [55:0] exact_uq_product(
        input logic signed [31:0] accumulator,
        input logic [23:0] threshold
    );
        logic signed [55:0] accumulator_wide;
        logic signed [55:0] threshold_wide;
        begin
            accumulator_wide = {{24{accumulator[31]}}, accumulator};
            threshold_wide = {{32{1'b0}}, threshold};
            exact_uq_product = accumulator_wide * threshold_wide;
        end
    endfunction

    always_comb begin
        expected_active_mask = '0;
        if (input_valid && input_ready) begin
            for (int output_index = 0; output_index < OUTPUTS; output_index++)
                for (int lane = 0; lane < 20; lane++)
                    expected_active_mask[(output_index*20)+lane]
                        = input_valid_bits[output_index];
        end
    end

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M33_UQ_SVA_BOUND=1");

    assert property (input_ready == (!output_valid || output_ready));
    assert property (input_valid && !input_ready |=>
        $stable({input_valid, input_tag, input_valid_bits,
                 input_threshold_uq0p24, input_accumulator[0],
                 input_accumulator[1], input_accumulator[2],
                 input_accumulator[3]}));
    assert property (output_valid && !output_ready |=>
        $stable({output_valid, output_tag, output_valid_bits,
                 output_product[0], output_product[1],
                 output_product[2], output_product[3]}));
    assert property (input_valid && input_ready |=>
        output_valid && output_tag == $past(input_tag)
        && output_valid_bits == $past(input_valid_bits));
    assert property (input_valid |-> digit_residual_zero
        && recombination_fits_signed56);
    assert property (multiplier_active_mask == expected_active_mask);
    assert property (!protocol_error);

    for (genvar output_index = 0; output_index < OUTPUTS; output_index++) begin
        assert property (input_valid && input_ready
            && input_valid_bits[output_index] |=>
            $signed(output_product[output_index]) == exact_uq_product(
                $past(input_accumulator[output_index]),
                $past(input_threshold_uq0p24)));
    end

    cover property ((input_valid && input_ready
        && input_valid_bits == 4'hf)[*64]);
    cover property (output_valid && !output_ready
        ##3 output_valid && output_ready);
    cover property (input_valid && input_ready && input_valid_bits == 4'h0);
    cover property (input_valid && input_ready
        && input_threshold_uq0p24 == 24'hffffff);
endmodule

bind qfit_threshold_late_scale_uq0p24_radix20x4
    qfit_threshold_late_scale_uq0p24_radix20x4_assertions #(
        .TAG_W(TAG_W)
    ) m33_uq_assertions (.*);

`default_nettype wire
