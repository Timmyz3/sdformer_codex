`timescale 1ns/1ps
`default_nettype none

module gatestack_dualtag_replay_lifecycle_assertions #(
    parameter int TAG_W = 32,
    parameter int CONTEXT_ID_W = 1,
    parameter int HEAD_ID_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic slot_release_valid,
    input logic slot_release_ready,
    input logic [CONTEXT_ID_W-1:0] slot_release_context_id,
    input logic [HEAD_ID_W-1:0] slot_release_head_id,
    input logic cache_release_valid,
    input logic cache_release_ready,
    input logic [CONTEXT_ID_W-1:0] cache_release_context_id,
    input logic [HEAD_ID_W-1:0] cache_release_head_id,
    input logic [TAG_W-1:0] cache_release_payload_tag,
    input logic session_done_valid,
    input logic session_done_ready,
    input logic [TAG_W-1:0] session_done_payload_tag,
    input logic [TAG_W-1:0] session_done_execution_tag,
    input logic session_done_error,
    input logic protocol_error
);
    property p_slot_release_stable;
        @(posedge clk_core) disable iff (rst_core)
        slot_release_valid && !slot_release_ready |=> slot_release_valid &&
            $stable({slot_release_context_id, slot_release_head_id});
    endproperty
    assert property (p_slot_release_stable);

    property p_cache_release_stable;
        @(posedge clk_core) disable iff (rst_core)
        cache_release_valid && !cache_release_ready |=>
            cache_release_valid &&
            $stable({cache_release_context_id, cache_release_head_id,
                     cache_release_payload_tag});
    endproperty
    assert property (p_cache_release_stable);

    property p_done_stable;
        @(posedge clk_core) disable iff (rst_core)
        session_done_valid && !session_done_ready |=> session_done_valid &&
            $stable({session_done_payload_tag,
                     session_done_execution_tag, session_done_error});
    endproperty
    assert property (p_done_stable);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
