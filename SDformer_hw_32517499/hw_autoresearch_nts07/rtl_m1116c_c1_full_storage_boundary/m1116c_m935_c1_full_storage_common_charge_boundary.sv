`timescale 1ns/1ps
`default_nettype none

// Additive M1116C storage-boundary wrapper.
//
// The frozen M935 compute island is instantiated without modification.  Its
// nine live 128x128 1RW parent macros remain inside the measured candidate.
// The much larger psum and weight stores remain outside this logic/macro
// island and are represented by live, addressed ready/valid service ports.
// No psum, weight, metadata, reserve, padding, dummy, or tied-off area macro is
// instantiated here.  The exact byte boundary is frozen separately by the
// M1116C mapping manifest.
//
// A request is issued atomically to weight and (for the first beat of a row)
// psum services.  The external services are in-order and hold response data
// while valid until ready.  The two responses are joined before the original
// M935 issue_data interface is asserted.  This wrapper adds only three control
// bits (outstanding, first-beat, sticky boundary fault), not a payload FIFO.
module m1116c_m935_c1_full_storage_common_charge_boundary (
    input  logic          clk_core,
    input  logic          reset_n,

    input  logic          prep_valid,
    output logic          prep_ready,
    input  logic          prep_task_start,
    input  logic          prep_task_last,
    input  logic [15:0]   prep_epoch,
    input  logic [5:0]    prep_row_id,
    input  logic [15:0]   prep_mask,
    input  logic [3:0]    prep_reserved,

    output logic          issue_request_valid,
    output logic [15:0]   issue_request_epoch,
    output logic [5:0]    issue_request_row_id,
    output logic          issue_request_first,
    output logic          issue_request_last,
    output logic          issue_request_source_valid,
    output logic [3:0]    issue_request_source_index,
    output logic          issue_request_parent_valid,
    output logic [5:0]    issue_request_parent_id,

    // Live external weight/product service.  The address tuple selects the
    // frozen task/row/source beat.  The external service returns the exact
    // signed 96x12 residual payload consumed by M935.
    output logic          weight_read_request_valid,
    input  logic          weight_read_request_ready,
    output logic [15:0]   weight_read_request_epoch,
    output logic [5:0]    weight_read_request_row_id,
    output logic [3:0]    weight_read_request_source_index,
    output logic          weight_read_request_source_valid,
    input  logic          weight_read_response_valid,
    output logic          weight_read_response_ready,
    input  logic [1151:0] weight_product_residual_data,

    // Live external psum service.  A read is requested only for the first
    // source beat; every completed row writes one 96x19 payload back.
    output logic          psum_read_request_valid,
    input  logic          psum_read_request_ready,
    output logic [15:0]   psum_read_request_epoch,
    output logic [5:0]    psum_read_request_address,
    input  logic          psum_read_response_valid,
    output logic          psum_read_response_ready,
    input  logic [1823:0] psum_read_response_data,
    output logic          psum_write_valid,
    input  logic          psum_write_ready,
    output logic [5:0]    psum_write_address,
    output logic [1823:0] psum_write_data,

    output logic          row_complete_valid,
    input  logic          row_complete_ready,
    output logic [5:0]    row_complete_id,
    output logic          task_done_valid,
    output logic [15:0]   task_done_epoch,

    output logic          protocol_error,
    output logic          preprocess_busy,
    output logic          execute_busy,
    output logic          active_directory_bank,
    output logic [1:0]    parent_queue_occupancy,
    output logic [2:0]    parent_reserved_occupancy,
    output logic [63:0]   debug_parent_live_bitmap,
    output logic [63:0]   debug_written_bitmap,
    output logic          debug_scratch_read_event,
    output logic          debug_scratch_write_event,
    output logic          debug_forward_event,
    output logic          debug_read_response_event,
    output logic          debug_dual_enqueue_event,
    output logic          debug_dead_write_elision_event,
    output logic          debug_deadline_hold_event,
    output logic          debug_overflow_block_event,
    output logic          debug_stalled_raw_event,
    output logic [63:0]   count_issue_accepts,
    output logic [63:0]   count_parent_edges,
    output logic [63:0]   count_dead_write_elisions,
    output logic [63:0]   count_macro_reads,
    output logic [63:0]   count_macro_writes,
    output logic [63:0]   count_forwards,
    output logic [63:0]   count_deadline_holds,
    output logic [63:0]   count_issue_stalls,
    output logic [63:0]   count_psum_commits,
    output logic [63:0]   count_row_completions
);
    logic service_outstanding_q;
    logic service_first_q;
    logic boundary_fault_q;
    logic core_issue_data_valid;
    logic core_issue_data_ready;
    logic core_protocol_error;
    logic request_accept_w;
    logic response_accept_w;

    // Cross-service atomic request: a first-beat request cannot be accepted by
    // only one of weight/psum.  Later source beats require only weight.
    always_comb begin
        weight_read_request_valid = issue_request_valid
            && !service_outstanding_q
            && (!issue_request_first || psum_read_request_ready);
        psum_read_request_valid = issue_request_valid
            && !service_outstanding_q && issue_request_first
            && weight_read_request_ready;
        request_accept_w = weight_read_request_valid
            && weight_read_request_ready;

        weight_read_request_epoch = issue_request_epoch;
        weight_read_request_row_id = issue_request_row_id;
        weight_read_request_source_index = issue_request_source_index;
        weight_read_request_source_valid = issue_request_source_valid;
        psum_read_request_epoch = issue_request_epoch;
        psum_read_request_address = issue_request_row_id;

        core_issue_data_valid = service_outstanding_q
            && weight_read_response_valid
            && (!service_first_q || psum_read_response_valid);
        weight_read_response_ready = service_outstanding_q
            && core_issue_data_ready
            && (!service_first_q || psum_read_response_valid);
        psum_read_response_ready = service_outstanding_q
            && service_first_q && core_issue_data_ready
            && weight_read_response_valid;
        response_accept_w = core_issue_data_valid && core_issue_data_ready;
        protocol_error = core_protocol_error || boundary_fault_q;
    end

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            service_outstanding_q <= 1'b0;
            service_first_q <= 1'b0;
            boundary_fault_q <= 1'b0;
        end else begin
            if (request_accept_w) begin
                service_outstanding_q <= 1'b1;
                service_first_q <= issue_request_first;
            end
            if (response_accept_w) begin
                service_outstanding_q <= 1'b0;
                service_first_q <= 1'b0;
            end
            if ((weight_read_response_valid && !service_outstanding_q)
                    || (psum_read_response_valid
                        && (!service_outstanding_q || !service_first_q))
                    || (service_outstanding_q && !issue_request_valid))
                boundary_fault_q <= 1'b1;
        end
    end

    m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935 (
        .clk_core(clk_core),
        .reset_n(reset_n),
        .prep_valid(prep_valid),
        .prep_ready(prep_ready),
        .prep_task_start(prep_task_start),
        .prep_task_last(prep_task_last),
        .prep_epoch(prep_epoch),
        .prep_row_id(prep_row_id),
        .prep_mask(prep_mask),
        .prep_reserved(prep_reserved),
        .issue_request_valid(issue_request_valid),
        .issue_request_epoch(issue_request_epoch),
        .issue_request_row_id(issue_request_row_id),
        .issue_request_first(issue_request_first),
        .issue_request_last(issue_request_last),
        .issue_request_source_valid(issue_request_source_valid),
        .issue_request_source_index(issue_request_source_index),
        .issue_request_parent_valid(issue_request_parent_valid),
        .issue_request_parent_id(issue_request_parent_id),
        .issue_data_valid(core_issue_data_valid),
        .issue_data_ready(core_issue_data_ready),
        .issue_residual_data(weight_product_residual_data),
        .issue_psum_prior(psum_read_response_data),
        .psum_write_valid(psum_write_valid),
        .psum_write_ready(psum_write_ready),
        .psum_write_address(psum_write_address),
        .psum_write_data(psum_write_data),
        .row_complete_valid(row_complete_valid),
        .row_complete_ready(row_complete_ready),
        .row_complete_id(row_complete_id),
        .task_done_valid(task_done_valid),
        .task_done_epoch(task_done_epoch),
        .protocol_error(core_protocol_error),
        .preprocess_busy(preprocess_busy),
        .execute_busy(execute_busy),
        .active_directory_bank(active_directory_bank),
        .parent_queue_occupancy(parent_queue_occupancy),
        .parent_reserved_occupancy(parent_reserved_occupancy),
        .debug_parent_live_bitmap(debug_parent_live_bitmap),
        .debug_written_bitmap(debug_written_bitmap),
        .debug_scratch_read_event(debug_scratch_read_event),
        .debug_scratch_write_event(debug_scratch_write_event),
        .debug_forward_event(debug_forward_event),
        .debug_read_response_event(debug_read_response_event),
        .debug_dual_enqueue_event(debug_dual_enqueue_event),
        .debug_dead_write_elision_event(debug_dead_write_elision_event),
        .debug_deadline_hold_event(debug_deadline_hold_event),
        .debug_overflow_block_event(debug_overflow_block_event),
        .debug_stalled_raw_event(debug_stalled_raw_event),
        .count_issue_accepts(count_issue_accepts),
        .count_parent_edges(count_parent_edges),
        .count_dead_write_elisions(count_dead_write_elisions),
        .count_macro_reads(count_macro_reads),
        .count_macro_writes(count_macro_writes),
        .count_forwards(count_forwards),
        .count_deadline_holds(count_deadline_holds),
        .count_issue_stalls(count_issue_stalls),
        .count_psum_commits(count_psum_commits),
        .count_row_completions(count_row_completions)
    );
endmodule

`default_nettype wire
