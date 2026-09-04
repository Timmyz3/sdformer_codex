`timescale 1ns/1ps
`default_nettype none

module qfit_complement_csd8_canonical_assertions #(
    parameter int TAG_W = 48,
    parameter int EPOCH_W = 16,
    localparam int OUTPUTS = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic config_valid,
    input logic config_ready,
    input logic [EPOCH_W-1:0] config_epoch,
    input logic [3:0] config_descriptor_id,
    input logic config_loaded,
    input logic [EPOCH_W-1:0] loaded_epoch,
    input logic [3:0] loaded_descriptor_id,
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

    initial $display("M35_R5_SVA_BOUND=1");

    // This equivalence closes the independent review's 4'hA lexical-alias
    // blind spot for every executable 4-bit candidate value.
    assert property (descriptor_legal == (config_descriptor_id <= 4'd9));
    assert property (!uses_integer_multiplier);
    assert property (config_valid && config_ready && descriptor_legal
        |=> config_loaded && !protocol_error
            && loaded_epoch == $past(config_epoch)
            && loaded_descriptor_id == $past(config_descriptor_id));
    assert property (config_valid && config_ready && !descriptor_legal
        |=> protocol_error && !config_loaded);
    assert property (config_loaded && !(config_release_valid
        && config_release_ready)
        |=> $stable({loaded_epoch, loaded_descriptor_id}));
    assert property (config_loaded |-> loaded_descriptor_id <= 4'd9);
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
    assert property (config_release_ready
        |-> config_loaded && !busy && !input_valid);
    assert property (protocol_error |=> protocol_error);

    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd0 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd1 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd2 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd3 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd4 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd5 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd6 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd7 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd8 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd9 && descriptor_legal);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd10 && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd11 && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd12 && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd13 && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd14 && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property (config_valid && config_ready
        && config_descriptor_id == 4'd15 && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property ((input_valid && input_ready
        && input_valid_bits == 8'hff)[*128]);
    cover property (output_valid && !output_ready
        ##3 output_valid && output_ready);
    cover property (config_release_valid && config_release_ready
        ##1 config_valid && config_ready);
endmodule

bind qfit_complement_csd8_canonical
    qfit_complement_csd8_canonical_assertions #(
        .TAG_W(TAG_W), .EPOCH_W(EPOCH_W)
    ) m35_r5_assertions (.*);

`default_nettype wire
