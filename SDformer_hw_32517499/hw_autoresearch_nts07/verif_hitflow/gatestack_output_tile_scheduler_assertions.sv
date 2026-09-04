`timescale 1ns/1ps
`default_nettype none

module gatestack_output_tile_scheduler_assertions #(
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int HEAD_COUNT_W = 6,
    parameter int CONTEXT_ID_W = 1,
    parameter int HEAD_ID_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic tile_start_valid,
    input logic tile_start_ready,
    input logic [TAG_W-1:0] tile_start_tag,
    input logic [OUTPUT_TILE_W-1:0] tile_start_output_tile,
    input logic [HEAD_COUNT_W-1:0] tile_start_head_count,
    input logic head_issue_valid,
    input logic head_issue_ready,
    input logic [CONTEXT_ID_W-1:0] head_issue_context_id,
    input logic [TAG_W-1:0] head_issue_tag,
    input logic [HEAD_ID_W-1:0] head_issue_head_id,
    input logic [HEAD_COUNT_W-1:0] head_issue_head_index,
    input logic [INPUT_CH_W-1:0] head_issue_input_channel_base,
    input logic [OUTPUT_TILE_W-1:0] head_issue_output_tile,
    input logic head_issue_last_head,
    input logic head_issue_last_output_tile,
    input logic group_done_valid,
    input logic group_done_ready,
    input logic [TAG_W-1:0] group_done_tag,
    input logic group_done_error,
    input logic protocol_error
);
    property p_tile_start_stable;
        @(posedge clk_core) disable iff (rst_core)
        tile_start_valid && !tile_start_ready |=>
            tile_start_valid && $stable({tile_start_tag,
                tile_start_output_tile, tile_start_head_count});
    endproperty
    assert property (p_tile_start_stable);

    property p_head_issue_stable;
        @(posedge clk_core) disable iff (rst_core)
        head_issue_valid && !head_issue_ready |=>
            head_issue_valid && $stable({head_issue_context_id,
                head_issue_tag, head_issue_head_id, head_issue_head_index,
                head_issue_input_channel_base, head_issue_output_tile,
                head_issue_last_head, head_issue_last_output_tile});
    endproperty
    assert property (p_head_issue_stable);

    property p_group_done_stable;
        @(posedge clk_core) disable iff (rst_core)
        group_done_valid && !group_done_ready |=>
            group_done_valid &&
            $stable({group_done_tag, group_done_error});
    endproperty
    assert property (p_group_done_stable);

    property p_no_overlapping_requests;
        @(posedge clk_core) disable iff (rst_core)
        !(tile_start_valid && head_issue_valid);
    endproperty
    assert property (p_no_overlapping_requests);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);

    property p_protocol_error_stops_new_dispatch;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !tile_start_valid && !head_issue_valid;
    endproperty
    assert property (p_protocol_error_stops_new_dispatch);
endmodule

`default_nettype wire
