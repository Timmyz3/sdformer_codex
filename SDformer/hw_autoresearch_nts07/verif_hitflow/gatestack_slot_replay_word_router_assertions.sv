`timescale 1ns/1ps
`default_nettype none

module gatestack_slot_replay_word_router_assertions #(
    parameter int TAG_W = 32,
    parameter int WORD_INDEX_W = 7,
    parameter int FORMAT_W = 2
) (
    input logic clk_core,
    input logic rst_core,
    input logic input_valid,
    input logic input_ready,
    input logic [63:0] input_data,
    input logic [WORD_INDEX_W-1:0] input_index,
    input logic input_last,
    input logic [TAG_W-1:0] input_payload_tag,
    input logic input_mode_is_csr,
    input logic [FORMAT_W-1:0] input_format,
    input logic resident_valid,
    input logic ipd_valid,
    input logic raw_valid,
    input logic protocol_error
);
    property p_input_stable;
        @(posedge clk_core) disable iff (rst_core)
        input_valid && !input_ready |=> input_valid &&
            $stable({input_data, input_index, input_last,
                     input_payload_tag, input_mode_is_csr, input_format});
    endproperty
    assert property (p_input_stable);

    property p_one_route;
        @(posedge clk_core) disable iff (rst_core)
        $onehot0({resident_valid, ipd_valid, raw_valid});
    endproperty
    assert property (p_one_route);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
