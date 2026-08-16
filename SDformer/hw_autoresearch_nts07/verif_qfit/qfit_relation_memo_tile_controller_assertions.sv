`timescale 1ns/1ps
`default_nettype none

module qfit_relation_memo_tile_controller_assertions #(
    parameter int HEAD_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic tile_start,
    input logic tile_ready,
    input logic tile_done,
    input logic fallback_taken,
    input logic use_replay,
    input logic replay_start,
    input logic replay_cmd_ready,
    input logic replay_done,
    input logic replay_miss,
    input logic recompute_request,
    input logic recompute_grant,
    input logic head_start,
    input logic head_ready,
    input logic [HEAD_W-1:0] head_index,
    input logic descriptor_stream_idle,
    input logic projection_start,
    input logic projection_close,
    input logic projection_close_ready,
    input logic protocol_error
);
    property p_start_only_when_ready;
        @(posedge clk_core) disable iff (rst_core)
            tile_start |-> tile_ready;
    endproperty

    property p_issue_paths_are_exclusive;
        @(posedge clk_core) disable iff (rst_core)
            !(replay_start && head_start);
    endproperty

    property p_replay_issue_contract;
        @(posedge clk_core) disable iff (rst_core)
            replay_start |-> use_replay && $past(replay_cmd_ready);
    endproperty

    property p_recompute_issue_contract;
        @(posedge clk_core) disable iff (rst_core)
            head_start
            |-> !use_replay
                && $past(recompute_grant)
                && $past(head_ready);
    endproperty

    property p_projection_close_contract;
        @(posedge clk_core) disable iff (rst_core)
            projection_close
            |-> $past(descriptor_stream_idle)
                && $past(projection_close_ready);
    endproperty

    property p_fallback_switches_to_live;
        @(posedge clk_core) disable iff (rst_core)
            replay_done && replay_miss |=> fallback_taken && !use_replay;
    endproperty

    property p_request_holds_head;
        @(posedge clk_core) disable iff (rst_core)
            recompute_request && !(recompute_grant && head_ready)
            |=> recompute_request && $stable(head_index);
    endproperty

    property p_control_pulses_are_single_cycle;
        @(posedge clk_core) disable iff (rst_core)
            replay_start |=> !replay_start;
    endproperty

    property p_head_start_is_single_cycle;
        @(posedge clk_core) disable iff (rst_core)
            head_start |=> !head_start;
    endproperty

    property p_projection_start_is_single_cycle;
        @(posedge clk_core) disable iff (rst_core)
            projection_start |=> !projection_start;
    endproperty

    property p_tile_done_returns_idle;
        @(posedge clk_core) disable iff (rst_core)
            tile_done |-> tile_ready;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
            $past(protocol_error) |-> protocol_error;
    endproperty

    assert property (p_start_only_when_ready);
    assert property (p_issue_paths_are_exclusive);
    assert property (p_replay_issue_contract);
    assert property (p_recompute_issue_contract);
    assert property (p_projection_close_contract);
    assert property (p_fallback_switches_to_live);
    assert property (p_request_holds_head);
    assert property (p_control_pulses_are_single_cycle);
    assert property (p_head_start_is_single_cycle);
    assert property (p_projection_start_is_single_cycle);
    assert property (p_tile_done_returns_idle);
    assert property (p_protocol_error_sticky);
endmodule

bind qfit_relation_memo_tile_controller
    qfit_relation_memo_tile_controller_assertions #(
        .HEAD_W(HEAD_W)
    ) u_qfit_relation_memo_tile_controller_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .tile_start(tile_start),
        .tile_ready(tile_ready),
        .tile_done(tile_done),
        .fallback_taken(fallback_taken),
        .use_replay(use_replay),
        .replay_start(replay_start),
        .replay_cmd_ready(replay_cmd_ready),
        .replay_done(replay_done),
        .replay_miss(replay_miss),
        .recompute_request(recompute_request),
        .recompute_grant(recompute_grant),
        .head_start(head_start),
        .head_ready(head_ready),
        .head_index(head_index),
        .descriptor_stream_idle(descriptor_stream_idle),
        .projection_start(projection_start),
        .projection_close(projection_close),
        .projection_close_ready(projection_close_ready),
        .protocol_error(protocol_error)
    );

`default_nettype wire
