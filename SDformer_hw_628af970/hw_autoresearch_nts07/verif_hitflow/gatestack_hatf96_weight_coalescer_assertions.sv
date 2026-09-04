`timescale 1ns/1ps
`default_nettype none

module gatestack_hatf96_weight_coalescer_assertions #(
    parameter int BANK_COUNT = 3,
    parameter int LANES_PER_BANK = 32,
    parameter int WEIGHT_W = 8,
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic req_valid,
    input logic req_ready,
    input logic [BANK_COUNT-1:0] bank_req_valid,
    input logic [BANK_COUNT-1:0] bank_req_ready,
    input logic [(BANK_COUNT*TAG_W)-1:0] bank_req_tags,
    input logic [(BANK_COUNT*INPUT_CH_W)-1:0] bank_req_input_channels,
    input logic [(BANK_COUNT*OUTPUT_TILE_W)-1:0] bank_req_output_tiles,
    input logic [BANK_COUNT-1:0] bank_rsp_valid,
    input logic [BANK_COUNT-1:0] bank_rsp_ready,
    input logic rsp_valid,
    input logic rsp_ready,
    input logic [TAG_W-1:0] rsp_tag,
    input logic [INPUT_CH_W-1:0] rsp_input_channel,
    input logic [OUTPUT_TILE_W-1:0] rsp_supertile,
    input logic [(BANK_COUNT*LANES_PER_BANK*WEIGHT_W)-1:0] rsp_weights,
    input logic rsp_error,
    input logic protocol_error
);
    property p_response_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
        rsp_valid && !rsp_ready |=>
            rsp_valid && $stable({rsp_tag, rsp_input_channel,
                                  rsp_supertile, rsp_weights, rsp_error});
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    property p_error_response_has_protocol_error;
        @(posedge clk_core) disable iff (rst_core)
        rsp_valid && rsp_error |-> protocol_error;
    endproperty

    property p_no_bank_response_accept_without_request;
        @(posedge clk_core) disable iff (rst_core)
        |(bank_rsp_valid & bank_rsp_ready) |-> !req_ready;
    endproperty

    assert property (p_response_stable_under_backpressure);
    assert property (p_protocol_error_sticky);
    assert property (p_error_response_has_protocol_error);
    assert property (p_no_bank_response_accept_without_request);

    generate
        for (genvar bank = 0; bank < BANK_COUNT; bank = bank + 1) begin : g_bank
            property p_bank_request_stable_under_backpressure;
                @(posedge clk_core) disable iff (rst_core)
                bank_req_valid[bank] && !bank_req_ready[bank] |=>
                    bank_req_valid[bank] &&
                    $stable({bank_req_tags[(bank*TAG_W) +: TAG_W],
                             bank_req_input_channels[
                                 (bank*INPUT_CH_W) +: INPUT_CH_W],
                             bank_req_output_tiles[
                                 (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W]});
            endproperty
            assert property (p_bank_request_stable_under_backpressure);
        end
    endgenerate

    cover property (@(posedge clk_core) disable iff (rst_core)
                    req_valid && req_ready);
    cover property (@(posedge clk_core) disable iff (rst_core)
                    &(bank_req_valid & bank_req_ready));
    cover property (@(posedge clk_core) disable iff (rst_core)
                    &(bank_rsp_valid & bank_rsp_ready));
endmodule

`default_nettype wire
