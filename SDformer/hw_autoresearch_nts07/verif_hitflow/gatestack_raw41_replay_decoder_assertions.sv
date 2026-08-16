`timescale 1ns/1ps
`default_nettype none

module gatestack_raw41_replay_decoder_assertions #(
    parameter int TAG_W = 32,
    parameter int TOKEN_ID_W = 8,
    parameter int LANE_ID_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic direct_valid,
    input logic direct_ready,
    input logic [8:0] direct_gate_code,
    input logic [LANE_ID_W-1:0] direct_lane_id,
    input logic [TOKEN_ID_W-1:0] direct_token_id,
    input logic direct_head_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic done_error,
    input logic protocol_error
);

    property p_direct_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        direct_valid && !direct_ready |=> direct_valid &&
            $stable({direct_gate_code, direct_lane_id,
                     direct_token_id, direct_head_last});
    endproperty

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid &&
            $stable({done_tag, done_error});
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    assert property (p_direct_stable_under_stall);
    assert property (p_done_stable_under_stall);
    assert property (p_protocol_error_sticky);

endmodule

`default_nettype wire
