`timescale 1ns/1ps
`default_nettype none

module gatestack_replay_atomic_commit_assertions #(
    parameter int FORMAT_W = 2,
    parameter int ROUTE_W = 2,
    parameter int WORD_INDEX_W = 7
) (
    input logic clk_core,
    input logic rst_core,
    input logic plan_valid,
    input logic plan_ready,
    input logic plan_slot_replay_required,
    input logic [FORMAT_W-1:0] plan_format,
    input logic [ROUTE_W-1:0] plan_route,
    input logic plan_cache_owned,
    input logic [WORD_INDEX_W-1:0] plan_replay_start_word,
    input logic [7:0] plan_resident_term_count,
    input logic projection_commit_pulse,
    input logic [FORMAT_W-1:0] projection_format,
    input logic projection_reserve_ready,
    input logic slot_commit_pulse,
    input logic slot_reserve_ready,
    input logic lifecycle_commit_pulse,
    input logic lifecycle_reserve_ready,
    input logic reject_valid,
    input logic reject_ready,
    input logic commit_pulse,
    input logic protocol_error
);
    logic reject_fire;
    assign reject_fire = reject_valid && reject_ready;

    property p_atomic_projection;
        @(posedge clk_core) disable iff (rst_core)
        projection_commit_pulse |-> lifecycle_commit_pulse &&
            projection_reserve_ready && lifecycle_reserve_ready &&
            (!plan_slot_replay_required ||
             (slot_commit_pulse && slot_reserve_ready));
    endproperty
    assert property (p_atomic_projection);

    property p_projection_format_is_atomic;
        @(posedge clk_core) disable iff (rst_core)
        projection_commit_pulse |-> projection_format == plan_format;
    endproperty
    assert property (p_projection_format_is_atomic);

    property p_committed_offset_and_ownership;
        @(posedge clk_core) disable iff (rst_core)
        projection_commit_pulse |->
            (plan_cache_owned == (plan_route == ROUTE_W'(0))) &&
            (plan_route == ROUTE_W'(0) ?
             plan_replay_start_word == WORD_INDEX_W'(
                2 + ((32'(plan_resident_term_count) + 1) >> 1)) :
             plan_replay_start_word == '0);
    endproperty
    assert property (p_committed_offset_and_ownership);

    property p_atomic_lifecycle;
        @(posedge clk_core) disable iff (rst_core)
        lifecycle_commit_pulse |-> projection_commit_pulse &&
            (!plan_slot_replay_required || slot_commit_pulse);
    endproperty
    assert property (p_atomic_lifecycle);

    property p_atomic_slot;
        @(posedge clk_core) disable iff (rst_core)
        slot_commit_pulse |-> projection_commit_pulse &&
            lifecycle_commit_pulse && plan_slot_replay_required;
    endproperty
    assert property (p_atomic_slot);

    property p_plan_has_one_outcome;
        @(posedge clk_core) disable iff (rst_core)
        plan_valid && plan_ready |->
            (projection_commit_pulse ^ reject_fire);
    endproperty
    assert property (p_plan_has_one_outcome);

    property p_commit_pulse_matches_previous_fire;
        @(posedge clk_core) disable iff (rst_core)
        commit_pulse |-> $past(projection_commit_pulse);
    endproperty
    assert property (p_commit_pulse_matches_previous_fire);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
