`timescale 1ns/1ps
`default_nettype none
module m476r2_backpressure_safe_assertions #(
    parameter int LANES = 96,
    parameter int ROW_BITS = 6
) (
    input logic clk_core, reset_n, protocol_error,
    input logic prefetch_valid, prefetch_ready,
    input logic [ROW_BITS-1:0] prefetch_parent_id,
    input logic scratch_read_enable,
    input logic [ROW_BITS-1:0] scratch_read_address,
    input logic issue_valid, issue_ready, issue_first, issue_last,
    input logic issue_parent_valid,
    input logic [ROW_BITS-1:0] issue_row_id, issue_parent_id,
    input logic [LANES*12-1:0] issue_residual_data,
    input logic [LANES*19-1:0] issue_psum_prior,
    input logic scratch_write_enable,
    input logic [ROW_BITS-1:0] scratch_write_address,
    input logic [LANES*12-1:0] scratch_write_data,
    input logic psum_write_valid, psum_write_ready,
    input logic [ROW_BITS-1:0] psum_write_address,
    input logic [LANES*19-1:0] psum_write_data,
    input logic row_complete,
    input logic [1:0] parent_queue_occupancy,
    input logic parent_queue_full,
    input logic debug_forward_event, debug_scratch_read_event,
    input logic debug_read_response_event, debug_dual_enqueue_event,
    input logic debug_overflow_block_event,
    input logic debug_stalled_raw_prefetch_event
);
    m476_dual_slot_parent_queue_assertions #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) base (.*);

    ap_stalled_same_address_prefetch_blocked: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        debug_stalled_raw_prefetch_event |-> !prefetch_ready &&
        !scratch_read_enable && !debug_forward_event);
    ap_stalled_raw_has_exact_shape: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        debug_stalled_raw_prefetch_event |-> prefetch_valid && issue_valid &&
        issue_last && prefetch_parent_id == issue_row_id && !psum_write_ready);

    cp_stalled_same_address_prefetch: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        debug_stalled_raw_prefetch_event && psum_write_valid);
    cp_release_to_new_value_forward: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        debug_stalled_raw_prefetch_event ##1
        !debug_stalled_raw_prefetch_event && prefetch_valid &&
        prefetch_ready && issue_valid && issue_ready && debug_forward_event);
endmodule
`default_nettype wire
