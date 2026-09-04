`timescale 1ns/1ps
`default_nettype none

module qfit_complement_csd8_late_scale_assertions #(
    parameter int TAG_W = 48,
    parameter int EPOCH_W = 16,
    localparam int OUTPUTS = 8,
    localparam int TERMS = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic config_valid,
    input logic config_ready,
    input logic [EPOCH_W-1:0] config_epoch,
    input logic [9:0] config_delta,
    input logic [TERMS-1:0] config_term_valid,
    input logic [TERMS-1:0] config_term_negative,
    input logic [3:0] config_term_shift [0:TERMS-1],
    input logic config_loaded,
    input logic [EPOCH_W-1:0] loaded_epoch,
    input logic config_release_valid,
    input logic config_release_ready,
    input logic input_valid,
    input logic input_ready,
    input logic [TAG_W-1:0] input_tag,
    input logic [OUTPUTS-1:0] input_valid_bits,
    input logic signed [31:0] input_accumulator [0:OUTPUTS-1],
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [EPOCH_W-1:0] output_epoch,
    input logic [OUTPUTS-1:0] output_valid_bits,
    input logic signed [55:0] output_product [0:OUTPUTS-1],
    input logic descriptor_legal,
    input logic uses_integer_multiplier,
    input logic busy,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M35_SVA_BOUND=1");

    assert property (!uses_integer_multiplier);
    assert property (config_valid && config_ready && descriptor_legal
        |=> config_loaded && !protocol_error);
    assert property (config_valid && config_ready && !descriptor_legal
        |=> protocol_error && !config_loaded);
    assert property (config_loaded && !(config_release_valid
        && config_release_ready) |=> $stable(loaded_epoch));
    assert property (input_valid && !input_ready |=>
        $stable({input_valid, input_tag, input_valid_bits,
                 input_accumulator[0], input_accumulator[1],
                 input_accumulator[2], input_accumulator[3],
                 input_accumulator[4], input_accumulator[5],
                 input_accumulator[6], input_accumulator[7]}));
    assert property (output_valid && !output_ready |=>
        $stable({output_valid, output_tag, output_epoch, output_valid_bits,
                 output_product[0], output_product[1],
                 output_product[2], output_product[3],
                 output_product[4], output_product[5],
                 output_product[6], output_product[7]}));
    assert property (input_valid && input_ready |-> config_loaded);
    assert property (config_release_ready |-> !busy && !input_valid);
    assert property (protocol_error |=> protocol_error);

    cover property ((input_valid && input_ready
        && input_valid_bits == 8'hff)[*64]);
    cover property (output_valid && !output_ready
        ##3 output_valid && output_ready);
    cover property (config_release_valid && config_release_ready
        ##1 config_valid && config_ready);
    cover property (config_valid && config_ready && !descriptor_legal
        ##1 protocol_error && !config_loaded);
endmodule

bind qfit_complement_csd8_late_scale
    qfit_complement_csd8_late_scale_assertions #(
        .TAG_W(TAG_W), .EPOCH_W(EPOCH_W)
    ) m35_assertions (.*);

`default_nettype wire
