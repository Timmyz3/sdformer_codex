`timescale 1ns/1ps
`default_nettype none

module gatestack_replay_control_plane_assertions #(
    parameter int TAG_W = 32,
    parameter int CONTEXT_ID_W = 1,
    parameter int HEAD_ID_W = 5,
    parameter int HEAD_COUNT_W = 6
) (
    input logic clk_core,
    input logic rst_core,
    input logic head_request_ready,
    input logic request_outstanding_q,
    input logic completion_fire,
    input logic [CONTEXT_ID_W-1:0] request_context_q,
    input logic [HEAD_ID_W-1:0] request_head_q,
    input logic [HEAD_COUNT_W-1:0] request_head_index_q,
    input logic request_last_head_q,
    input logic projection_commit_pulse,
    input logic slot_commit_pulse,
    input logic lifecycle_commit_pulse,
    input logic plan_slot_replay_required,
    input logic head_complete_valid,
    input logic head_complete_ready,
    input logic [CONTEXT_ID_W-1:0] head_complete_context_id,
    input logic [HEAD_ID_W-1:0] head_complete_head_id,
    input logic [HEAD_COUNT_W-1:0] head_complete_head_index,
    input logic head_complete_last_head,
    input logic [TAG_W-1:0] head_complete_payload_tag,
    input logic [TAG_W-1:0] head_complete_execution_tag,
    input logic head_complete_error,
    input logic protocol_error
);
    property p_single_outstanding_blocks_admission;
        @(posedge clk_core) disable iff (rst_core)
        request_outstanding_q |-> !head_request_ready;
    endproperty
    assert property (p_single_outstanding_blocks_admission);

    property p_completion_requires_request;
        @(posedge clk_core) disable iff (rst_core)
        head_complete_valid |-> request_outstanding_q;
    endproperty
    assert property (p_completion_requires_request);

    property p_request_identity_stable;
        @(posedge clk_core) disable iff (rst_core)
        request_outstanding_q && !completion_fire |=>
            $stable({request_context_q, request_head_q,
                     request_head_index_q, request_last_head_q});
    endproperty
    assert property (p_request_identity_stable);

    property p_completion_stable;
        @(posedge clk_core) disable iff (rst_core)
        head_complete_valid && !head_complete_ready |=>
            head_complete_valid &&
            $stable({head_complete_context_id, head_complete_head_id,
                     head_complete_head_index, head_complete_last_head,
                     head_complete_payload_tag,
                     head_complete_execution_tag, head_complete_error});
    endproperty
    assert property (p_completion_stable);

    property p_projection_lifecycle_atomic;
        @(posedge clk_core) disable iff (rst_core)
        projection_commit_pulse |-> lifecycle_commit_pulse &&
            (!plan_slot_replay_required || slot_commit_pulse);
    endproperty
    assert property (p_projection_lifecycle_atomic);

    property p_lifecycle_projection_atomic;
        @(posedge clk_core) disable iff (rst_core)
        lifecycle_commit_pulse |-> projection_commit_pulse;
    endproperty
    assert property (p_lifecycle_projection_atomic);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
