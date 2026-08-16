`timescale 1ns/1ps
`default_nettype none

module gatestack_replay_plan_builder_assertions #(
    parameter int TAG_W = 32,
    parameter int CONTEXT_ID_W = 1,
    parameter int HEAD_ID_W = 5,
    parameter int ROUTE_W = 2,
    parameter int FORMAT_W = 2,
    parameter int HEAD_COUNT_W = 6,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int WORD_INDEX_W = 7,
    parameter int EVENT_COUNT_W = 13
) (
    input logic clk_core,
    input logic rst_core,
    input logic plan_valid,
    input logic plan_ready,
    input logic [CONTEXT_ID_W-1:0] plan_context_id,
    input logic [HEAD_ID_W-1:0] plan_head_id,
    input logic [TAG_W-1:0] plan_payload_tag,
    input logic [TAG_W-1:0] plan_execution_tag,
    input logic [ROUTE_W-1:0] plan_route,
    input logic [FORMAT_W-1:0] plan_format,
    input logic [HEAD_COUNT_W-1:0] plan_head_index,
    input logic [INPUT_CH_W-1:0] plan_input_channel_base,
    input logic [OUTPUT_TILE_W-1:0] plan_output_tile,
    input logic plan_last_head,
    input logic plan_last_output_tile,
    input logic plan_cache_owned,
    input logic plan_slot_replay_required,
    input logic [WORD_INDEX_W-1:0] plan_replay_start_word,
    input logic [7:0] plan_resident_term_count,
    input logic [EVENT_COUNT_W-1:0] plan_resident_event_count,
    input logic reject_valid,
    input logic reject_ready,
    input logic [TAG_W-1:0] reject_payload_tag,
    input logic [TAG_W-1:0] reject_execution_tag,
    input logic protocol_error
);
    property p_plan_stable;
        @(posedge clk_core) disable iff (rst_core)
        plan_valid && !plan_ready |=> plan_valid &&
            $stable({plan_context_id, plan_head_id, plan_payload_tag,
                plan_execution_tag, plan_route, plan_head_index,
                plan_format,
                plan_input_channel_base, plan_output_tile, plan_last_head,
                plan_last_output_tile, plan_cache_owned,
                plan_slot_replay_required, plan_replay_start_word,
                plan_resident_term_count, plan_resident_event_count});
    endproperty

    property p_route_format_contract;
        @(posedge clk_core) disable iff (rst_core)
        plan_valid |->
            ((plan_route == ROUTE_W'(0) &&
              plan_format == FORMAT_W'(1) && plan_cache_owned) ||
             (plan_route == ROUTE_W'(1) &&
              (plan_format == FORMAT_W'(1) ||
               plan_format == FORMAT_W'(2)) && !plan_cache_owned &&
              plan_replay_start_word == '0) ||
             (plan_route == ROUTE_W'(2) &&
              plan_format == FORMAT_W'(0) && !plan_cache_owned &&
              plan_replay_start_word == '0));
    endproperty
    assert property (p_plan_stable);
    assert property (p_route_format_contract);

    property p_reject_stable;
        @(posedge clk_core) disable iff (rst_core)
        reject_valid && !reject_ready |=> reject_valid &&
            $stable({reject_payload_tag, reject_execution_tag});
    endproperty
    assert property (p_reject_stable);

    property p_one_response;
        @(posedge clk_core) disable iff (rst_core)
        !(plan_valid && reject_valid);
    endproperty
    assert property (p_one_response);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
