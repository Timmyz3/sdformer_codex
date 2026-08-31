`timescale 1ns/1ps
`default_nettype none

// M476r2 preserves the sealed M476 dual-slot core and closes its independently
// reproduced stale-RAW corner.  A same-address prefetch cannot enter the core
// while the corresponding final issue is recoverably stalled by psum output
// backpressure.  Once psum_write_ready rises, both handshakes proceed and the
// sealed core's same-cycle write-to-parent forwarding supplies the new value.
//
// The guard intentionally depends on issue_row_id/prefetch_parent_id and the
// sink-ready bit, not issue_parent_id or the core's consume/capacity chain.
module m476r2_backpressure_safe_parent_queue_pipeline #(
    parameter int LANES = 96,
    parameter int ROW_BITS = 6
) (
    input  logic clk_core,
    input  logic reset_n,
    input  logic prefetch_valid,
    output logic prefetch_ready,
    input  logic [ROW_BITS-1:0] prefetch_parent_id,
    output logic scratch_read_enable,
    output logic [ROW_BITS-1:0] scratch_read_address,
    input  logic [LANES*12-1:0] scratch_read_data,
    input  logic issue_valid,
    output logic issue_ready,
    input  logic [ROW_BITS-1:0] issue_row_id,
    input  logic issue_first,
    input  logic issue_last,
    input  logic issue_parent_valid,
    input  logic [ROW_BITS-1:0] issue_parent_id,
    input  logic [LANES*12-1:0] issue_residual_data,
    input  logic [LANES*19-1:0] issue_psum_prior,
    output logic scratch_write_enable,
    output logic [ROW_BITS-1:0] scratch_write_address,
    output logic [LANES*12-1:0] scratch_write_data,
    output logic psum_write_valid,
    input  logic psum_write_ready,
    output logic [ROW_BITS-1:0] psum_write_address,
    output logic [LANES*19-1:0] psum_write_data,
    output logic row_complete,
    output logic protocol_error,
    output logic row_active,
    output logic [1:0] parent_queue_occupancy,
    output logic parent_queue_full,
    output logic debug_forward_event,
    output logic debug_scratch_read_event,
    output logic debug_read_response_event,
    output logic debug_dual_enqueue_event,
    output logic debug_overflow_block_event,
    output logic debug_stalled_raw_prefetch_event
);
    logic core_prefetch_valid;
    logic core_prefetch_ready;
    logic stalled_raw_hazard_w;

    assign stalled_raw_hazard_w = prefetch_valid && issue_valid && issue_last
        && prefetch_parent_id == issue_row_id && !psum_write_ready;
    assign core_prefetch_valid = prefetch_valid && !stalled_raw_hazard_w;
    assign prefetch_ready = core_prefetch_ready && !stalled_raw_hazard_w;
    assign debug_stalled_raw_prefetch_event = stalled_raw_hazard_w;

    m476_dual_slot_parent_queue_pipeline #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) u_core (
        .prefetch_valid(core_prefetch_valid),
        .prefetch_ready(core_prefetch_ready),
        .*
    );
endmodule

`default_nettype wire
