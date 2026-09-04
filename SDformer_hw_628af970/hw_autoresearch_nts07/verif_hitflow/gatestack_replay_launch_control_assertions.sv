`timescale 1ns/1ps
`default_nettype none

module gatestack_replay_launch_control_assertions #(
    parameter int TAG_W=32,
    parameter int EVENT_COUNT_W=13,
    parameter int WORD_INDEX_W=7
) (
    input logic clk_core, rst_core,
    input logic launch_done_valid, launch_done_ready,
    input logic [1:0] launch_done_route,
    input logic [TAG_W-1:0] launch_done_tag,
    input logic launch_done_error,
    input logic slot_replay_begin_valid, slot_replay_begin_ready,
    input logic [WORD_INDEX_W-1:0] slot_replay_start_word,
    input logic resident_start_valid, resident_start_ready,
    input logic [TAG_W-1:0] resident_start_tag,
    input logic [7:0] resident_start_term_count,
    input logic [EVENT_COUNT_W-1:0] resident_start_event_count,
    input logic ipd_start_valid, ipd_start_ready,
    input logic raw_start_valid, raw_start_ready,
    input logic [TAG_W-1:0] raw_start_tag,
    input logic route_start_valid, route_start_ready,
    input logic [1:0] route_start_select,
    input logic protocol_error
);
    property p_done_stable;
        @(posedge clk_core) disable iff(rst_core)
        launch_done_valid&&!launch_done_ready |=> launch_done_valid&&
          $stable({launch_done_route,launch_done_tag,launch_done_error});
    endproperty
    property p_slot_start_stable;
        @(posedge clk_core) disable iff(rst_core)
        slot_replay_begin_valid&&!slot_replay_begin_ready |=>
          slot_replay_begin_valid&&$stable(slot_replay_start_word);
    endproperty
    property p_resident_start_stable;
        @(posedge clk_core) disable iff(rst_core)
        resident_start_valid&&!resident_start_ready |=>resident_start_valid&&
          $stable({resident_start_tag,resident_start_term_count,
                   resident_start_event_count});
    endproperty
    property p_raw_start_stable;
        @(posedge clk_core) disable iff(rst_core)
        raw_start_valid&&!raw_start_ready |=>raw_start_valid&&
          $stable(raw_start_tag);
    endproperty
    property p_ipd_start_stable;
        @(posedge clk_core) disable iff(rst_core)
        ipd_start_valid&&!ipd_start_ready |=> ipd_start_valid;
    endproperty
    property p_route_start_stable;
        @(posedge clk_core) disable iff(rst_core)
        route_start_valid&&!route_start_ready |=>route_start_valid&&
          $stable(route_start_select);
    endproperty
    property p_one_decoder;
        @(posedge clk_core) disable iff(rst_core)
        $onehot0({resident_start_valid,ipd_start_valid,raw_start_valid});
    endproperty
    property p_error_sticky;
        @(posedge clk_core) disable iff(rst_core) protocol_error|=>protocol_error;
    endproperty
    assert property(p_done_stable);
    assert property(p_slot_start_stable);
    assert property(p_resident_start_stable);
    assert property(p_raw_start_stable);
    assert property(p_ipd_start_stable);
    assert property(p_route_start_stable);
    assert property(p_one_decoder);
    assert property(p_error_sticky);
endmodule

`default_nettype wire
