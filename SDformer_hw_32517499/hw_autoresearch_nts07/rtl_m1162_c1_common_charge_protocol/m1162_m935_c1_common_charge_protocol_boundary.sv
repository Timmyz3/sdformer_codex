`timescale 1ns/1ps
`default_nettype none

// Additive protocol repair of the M1116C common-charge boundary.
//
// The frozen M935 compute island and its nine live parent SRAM macros are not
// modified.  The exact 214,912-byte ledger remains the M1116C mapping:
//   parent 18,432 B internal; psum 122,880 B external common charge;
//   weight 49,152 B external common charge; reserve 24,448 B model-only.
//
// Protocol contract:
// - ready may depend on valid, but neither request valid depends on ready;
// - one transaction is outstanding; its request tuple is latched once;
// - weight and first-beat psum requests accept independently, exactly once;
// - a response may arrive after its own request accepts and must hold valid
//   and payload until ready; same-cycle request/response is prohibited;
// - skewed responses are joined and consumed atomically by frozen M935;
// - reset cancels the wrapper transaction and external services must discard
//   it; any response observed after reset release without a request is spurious;
// - cancellation or mutation of M935's held request is a sticky error.
//
// With zero request stalls and one-cycle service response latency, completed
// issue-data handshakes have minimum II=2.  No prior CPU schedule or speedup is
// inherited; a matched service-aware replay is mandatory.
module m1162_m935_c1_common_charge_protocol_boundary (
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

    output logic          weight_read_request_valid,
    input  logic          weight_read_request_ready,
    output logic [15:0]   weight_read_request_epoch,
    output logic [5:0]    weight_read_request_row_id,
    output logic [3:0]    weight_read_request_source_index,
    output logic          weight_read_request_source_valid,
    input  logic          weight_read_response_valid,
    output logic          weight_read_response_ready,
    input  logic [1151:0] weight_product_residual_data,

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
    logic request_active_q;
    logic weight_request_accepted_q;
    logic psum_request_accepted_q;
    logic boundary_fault_q;

    logic [15:0] request_epoch_q;
    logic [5:0]  request_row_id_q;
    logic        request_first_q;
    logic        request_last_q;
    logic        request_source_valid_q;
    logic [3:0]  request_source_index_q;
    logic        request_parent_valid_q;
    logic [5:0]  request_parent_id_q;

    logic core_issue_data_valid;
    logic core_issue_data_ready;
    logic core_protocol_error;
    logic weight_request_fire_w;
    logic psum_request_fire_w;
    logic response_accept_w;
    logic request_tuple_mutated_w;

    always_comb begin
        // Before the transaction is latched, M935's stable held request drives
        // the service address.  Afterwards, the latched tuple is authoritative.
        weight_read_request_epoch = request_active_q
            ? request_epoch_q : issue_request_epoch;
        weight_read_request_row_id = request_active_q
            ? request_row_id_q : issue_request_row_id;
        weight_read_request_source_index = request_active_q
            ? request_source_index_q : issue_request_source_index;
        weight_read_request_source_valid = request_active_q
            ? request_source_valid_q : issue_request_source_valid;
        psum_read_request_epoch = request_active_q
            ? request_epoch_q : issue_request_epoch;
        psum_read_request_address = request_active_q
            ? request_row_id_q : issue_request_row_id;

        // Neither valid has any ready in its combinational fan-in.
        weight_read_request_valid = issue_request_valid
            && (!request_active_q || !weight_request_accepted_q);
        psum_read_request_valid = issue_request_valid
            && (request_active_q ? request_first_q : issue_request_first)
            && (!request_active_q || !psum_request_accepted_q);
        weight_request_fire_w = weight_read_request_valid
            && weight_read_request_ready;
        psum_request_fire_w = psum_read_request_valid
            && psum_read_request_ready;

        // Each service may return only after its own request was accepted.
        // A first-beat transaction is exposed to M935 only after both requests
        // and both responses are present.  The ready join consumes neither
        // response alone and therefore needs no payload FIFO.
        core_issue_data_valid = request_active_q
            && weight_request_accepted_q
            && (!request_first_q || psum_request_accepted_q)
            && weight_read_response_valid
            && (!request_first_q || psum_read_response_valid);
        weight_read_response_ready = core_issue_data_valid
            && core_issue_data_ready;
        psum_read_response_ready = core_issue_data_valid
            && core_issue_data_ready && request_first_q;
        response_accept_w = core_issue_data_valid && core_issue_data_ready;

        request_tuple_mutated_w = request_active_q && issue_request_valid
            && ((issue_request_epoch != request_epoch_q)
                || (issue_request_row_id != request_row_id_q)
                || (issue_request_first != request_first_q)
                || (issue_request_last != request_last_q)
                || (issue_request_source_valid != request_source_valid_q)
                || (issue_request_source_index != request_source_index_q)
                || (issue_request_parent_valid != request_parent_valid_q)
                || (issue_request_parent_id != request_parent_id_q));
        protocol_error = core_protocol_error || boundary_fault_q;
    end

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            request_active_q <= 1'b0;
            weight_request_accepted_q <= 1'b0;
            psum_request_accepted_q <= 1'b0;
            boundary_fault_q <= 1'b0;
            request_epoch_q <= '0;
            request_row_id_q <= '0;
            request_first_q <= 1'b0;
            request_last_q <= 1'b0;
            request_source_valid_q <= 1'b0;
            request_source_index_q <= '0;
            request_parent_valid_q <= 1'b0;
            request_parent_id_q <= '0;
        end else begin
            if (!request_active_q && issue_request_valid) begin
                request_active_q <= 1'b1;
                weight_request_accepted_q <= weight_request_fire_w;
                psum_request_accepted_q <= !issue_request_first
                    || psum_request_fire_w;
                request_epoch_q <= issue_request_epoch;
                request_row_id_q <= issue_request_row_id;
                request_first_q <= issue_request_first;
                request_last_q <= issue_request_last;
                request_source_valid_q <= issue_request_source_valid;
                request_source_index_q <= issue_request_source_index;
                request_parent_valid_q <= issue_request_parent_valid;
                request_parent_id_q <= issue_request_parent_id;
            end else if (request_active_q) begin
                if (weight_request_fire_w)
                    weight_request_accepted_q <= 1'b1;
                if (psum_request_fire_w)
                    psum_request_accepted_q <= 1'b1;
            end

            if (response_accept_w) begin
                request_active_q <= 1'b0;
                weight_request_accepted_q <= 1'b0;
                psum_request_accepted_q <= 1'b0;
            end

            // Sticky fail-closed checks.  A response simultaneous with its
            // request acceptance is early because minimum response latency is
            // one cycle.  Reset is the only error clear/cancellation event.
            if ((weight_read_response_valid
                    && (!request_active_q || !weight_request_accepted_q))
                    || (psum_read_response_valid
                        && (!request_active_q || !request_first_q
                            || !psum_request_accepted_q))
                    || (request_active_q && !issue_request_valid)
                    || request_tuple_mutated_w)
                boundary_fault_q <= 1'b1;
        end
    end

    m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935 (
        .clk_core(clk_core), .reset_n(reset_n),
        .prep_valid(prep_valid), .prep_ready(prep_ready),
        .prep_task_start(prep_task_start), .prep_task_last(prep_task_last),
        .prep_epoch(prep_epoch), .prep_row_id(prep_row_id),
        .prep_mask(prep_mask), .prep_reserved(prep_reserved),
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
        .preprocess_busy(preprocess_busy), .execute_busy(execute_busy),
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
