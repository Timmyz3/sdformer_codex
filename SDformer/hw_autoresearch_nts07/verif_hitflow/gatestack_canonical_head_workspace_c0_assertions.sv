`timescale 1ns/1ps
`default_nettype none

module gatestack_canonical_head_workspace_c0_assertions #(
    parameter int TOKENS        = 162,
    parameter int LANES         = 32,
    parameter int GATE_W        = 9,
    parameter int CONTEXTS      = 2,
    parameter int HEADS         = 24,
    parameter int TAG_W         = 32,
    parameter int CONTEXT_ID_W  = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W     = (HEADS <= 1) ? 1 : $clog2(HEADS)
) (
    input logic clk_core,
    input logic rst_core,
    input logic head_begin_ready,
    input logic metadata_valid,
    input logic metadata_ready,
    input logic [CONTEXT_ID_W-1:0] metadata_context_id,
    input logic [HEAD_ID_W-1:0] metadata_head_id,
    input logic [TAG_W-1:0] metadata_tag,
    input logic [3:0] metadata_active_classes,
    input logic [7:0] metadata_active_tokens,
    input logic [7:0] metadata_term_count,
    input logic [12:0] metadata_event_count,
    input logic [7:0] metadata_bitmap_term_count,
    input logic [12:0] metadata_fadc_destination_bytes,
    input logic metadata_overflow,
    input logic raw_capture_error,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [GATE_W-1:0] descriptor_gate_code,
    input logic [4:0] descriptor_lane_id,
    input logic [7:0] descriptor_destination_count,
    input logic descriptor_last,
    input logic destination_valid,
    input logic destination_ready,
    input logic [7:0] destination_token_id,
    input logic destination_last_for_term,
    input logic destination_bitmap_valid,
    input logic destination_bitmap_ready,
    input logic [TOKENS-1:0] destination_bitmap,
    input logic raw_token_valid,
    input logic raw_token_ready,
    input logic [7:0] raw_token_id,
    input logic [GATE_W-1:0] raw_gate_code,
    input logic [LANES-1:0] raw_k_bits,
    input logic emit_done_valid,
    input logic emit_done_ready,
    input logic [TAG_W-1:0] emit_done_tag,
    input logic emit_done_error,
    input logic protocol_error
);

    property p_metadata_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        metadata_valid && !metadata_ready |=> metadata_valid &&
            $stable({metadata_context_id, metadata_head_id, metadata_tag,
                     metadata_active_classes, metadata_active_tokens,
                     metadata_term_count, metadata_event_count,
                     metadata_bitmap_term_count,
                     metadata_fadc_destination_bytes, metadata_overflow,
                     raw_capture_error});
    endproperty

    property p_descriptor_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        descriptor_valid && !descriptor_ready |=> descriptor_valid &&
            $stable({descriptor_gate_code, descriptor_lane_id,
                     descriptor_destination_count, descriptor_last});
    endproperty

    property p_destination_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        destination_valid && !destination_ready |=> destination_valid &&
            $stable({destination_token_id, destination_last_for_term});
    endproperty

    property p_raw_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        raw_token_valid && !raw_token_ready |=> raw_token_valid &&
            $stable({raw_token_id, raw_gate_code, raw_k_bits});
    endproperty

    property p_destination_bitmap_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        destination_bitmap_valid && !destination_bitmap_ready |=>
            destination_bitmap_valid && $stable(destination_bitmap);
    endproperty

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        emit_done_valid && !emit_done_ready |=> emit_done_valid &&
            $stable({emit_done_tag, emit_done_error});
    endproperty

    property p_outputs_are_exclusive;
        @(posedge clk_core) disable iff (rst_core)
        $onehot0({metadata_valid, descriptor_valid, destination_valid,
                  destination_bitmap_valid, raw_token_valid, emit_done_valid});
    endproperty

    property p_descriptor_is_legal;
        @(posedge clk_core) disable iff (rst_core)
        descriptor_valid |-> descriptor_destination_count != 0 &&
            32'(descriptor_lane_id) < LANES;
    endproperty

    property p_destination_is_legal;
        @(posedge clk_core) disable iff (rst_core)
        destination_valid |-> 32'(destination_token_id) < TOKENS;
    endproperty

    property p_raw_is_legal;
        @(posedge clk_core) disable iff (rst_core)
        raw_token_valid |-> 32'(raw_token_id) < TOKENS;
    endproperty

    property p_idle_has_no_response;
        @(posedge clk_core) disable iff (rst_core)
        head_begin_ready |-> !metadata_valid && !descriptor_valid &&
            !destination_valid && !destination_bitmap_valid &&
            !raw_token_valid && !emit_done_valid;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    assert property (p_metadata_stable_under_stall);
    assert property (p_descriptor_stable_under_stall);
    assert property (p_destination_stable_under_stall);
    assert property (p_destination_bitmap_stable_under_stall);
    assert property (p_raw_stable_under_stall);
    assert property (p_done_stable_under_stall);
    assert property (p_outputs_are_exclusive);
    assert property (p_descriptor_is_legal);
    assert property (p_destination_is_legal);
    assert property (p_raw_is_legal);
    assert property (p_idle_has_no_response);
    assert property (p_protocol_error_sticky);

endmodule

`default_nettype wire
