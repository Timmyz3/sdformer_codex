`timescale 1ns/1ps
`default_nettype none

module qfit_threshold_late_scale_radix20x4_assertions #(
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
    input logic signed [23:0] input_threshold,
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
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M33_SVA_BOUND=1");

    assert property (input_ready == (!output_valid || output_ready));
    assert property (input_valid && !input_ready |=>
        $stable({input_valid, input_tag, input_valid_bits, input_threshold,
                 input_accumulator[0], input_accumulator[1],
                 input_accumulator[2], input_accumulator[3]}));
    assert property (output_valid && !output_ready |=>
        $stable({output_valid, output_tag, output_valid_bits,
                 output_product[0], output_product[1],
                 output_product[2], output_product[3]}));
    assert property (input_valid && input_ready |=>
        output_valid && output_tag == $past(input_tag)
        && output_valid_bits == $past(input_valid_bits));
    assert property (input_valid |-> digit_residual_zero
        && recombination_fits_signed56);
    assert property (multiplier_active_mask[95:80] == 0);
    assert property (!(input_valid && input_ready)
        |-> multiplier_active_mask == 0);
    assert property (input_valid && input_ready
        |-> $countones(multiplier_active_mask)
            == 20*$countones(input_valid_bits));
    assert property (!protocol_error);

    cover property (input_valid && input_ready && input_valid_bits == 4'hf
        ##1 input_valid && input_ready && input_valid_bits == 4'hf);
    cover property ((input_valid && input_ready
        && input_valid_bits == 4'hf)[*8]);
    cover property (output_valid && !output_ready
        ##3 output_valid && output_ready);
    cover property (input_valid && input_ready && input_valid_bits == 4'h1);
endmodule

bind qfit_threshold_late_scale_radix20x4
    qfit_threshold_late_scale_radix20x4_assertions #(
        .TAG_W(TAG_W)
    ) m33_assertions (.*);

`default_nettype wire
