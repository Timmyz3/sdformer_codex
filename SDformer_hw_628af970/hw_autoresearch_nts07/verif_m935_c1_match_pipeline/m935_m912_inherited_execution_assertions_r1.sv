`timescale 1ns/1ps
`default_nettype none

// Draft SVA for the additive M912 metadata pipeline.  These properties are
// intentionally bound to real internal accept events, not only debug pins.
module m935_m912_inherited_execution_assertions_r1 (
    input logic clk_core,
    input logic reset_n,
    input logic protocol_error,
    input logic imminent_protocol_fault,
    input logic exec_active,
    input logic active_valid,
    input logic [5:0] active_row,
    input logic [15:0] active_residual,
    input logic active_parent_valid,
    input logic [5:0] active_parent,
    input logic active_first,
    input logic active_primed,
    input logic active_relation_ok,
    input logic next_valid,
    input logic [5:0] next_row,
    input logic [4:0] active_pop,
    input logic [4:0] next_pop,
    input logic next_relation_ok,
    input logic row_candidate_valid,
    input logic [5:0] row_candidate_row,
    input logic [4:0] row_candidate_pop,
    input logic row_candidate_relation_ok,
    input logic pf_valid,
    input logic [5:0] pf_consumer,
    input logic [5:0] pf_parent,
    input logic [4:0] pf_pop,
    input logic pf_candidate_valid,
    input logic [5:0] pf_candidate_consumer,
    input logic [4:0] pf_candidate_pop,
    input logic prefetch_accept,
    input logic macro_read_accept,
    input logic live_write_accept,
    input logic forward_accept,
    input logic dead_elision_accept,
    input logic deadline_hold,
    input logic stalled_raw,
    input logic read_pending,
    input logic scratch_write_enable,
    input logic [5:0] scratch_address,
    input logic [63:0] completed_bitmap,
    input logic [63:0] prefetched_bitmap,
    input logic [63:0] written_bitmap,
    input logic [1:0] queue_occupancy,
    input logic [2:0] reserved_occupancy,
    input logic queue_overflow,
    input logic issue_request_valid,
    input logic [15:0] issue_request_epoch,
    input logic [5:0] issue_request_row_id,
    input logic issue_request_first,
    input logic issue_request_last,
    input logic issue_request_source_valid,
    input logic [3:0] issue_request_source_index,
    input logic issue_request_parent_valid,
    input logic [5:0] issue_request_parent_id,
    input logic issue_data_valid,
    input logic issue_data_ready,
    input logic [1151:0] issue_residual_data,
    input logic [1823:0] issue_psum_prior,
    input logic issue_accept,
    input logic preaccept_protocol_ok,
    input logic overflow_block,
    input logic psum_write_valid,
    input logic psum_write_ready,
    input logic [5:0] psum_write_address,
    input logic [1823:0] psum_write_data,
    input logic row_complete_valid,
    input logic row_complete_ready,
    input logic [5:0] row_complete_id,
    input logic task_done_valid,
    input logic debug_scratch_read_event,
    input logic debug_scratch_write_event,
    input logic debug_forward_event,
    input logic debug_read_response_event,
    input logic debug_dual_enqueue_event,
    input logic debug_dead_write_elision_event,
    input logic debug_deadline_hold_event,
    input logic debug_overflow_block_event,
    input logic debug_stalled_raw_event,
    input logic [63:0] count_parent_edges,
    input logic [63:0] count_macro_reads,
    input logic [63:0] count_macro_writes,
    input logic [63:0] count_forwards,
    input logic [63:0] count_dead_write_elisions,
    input logic [63:0] count_psum_commits,
    input logic [63:0] count_row_completions
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n || protocol_error || imminent_protocol_fault);

    // Metadata slots are reservations.  A valid slot is exact, unique and may
    // not point at a row already architecturally completed.
    ap_active_exact: assert property (
        active_valid |-> active_relation_ok && !completed_bitmap[active_row]);
    ap_next_exact: assert property (
        next_valid |-> next_relation_ok && !completed_bitmap[next_row]);
    ap_contexts_unique: assert property (
        active_valid && next_valid |-> active_row != next_row);
    ap_stable_context_order: assert property (
        active_valid && next_valid
        |-> active_pop < next_pop
            || (active_pop == next_pop && active_row < next_row));
    ap_candidate_after_active: assert property (
        active_valid && row_candidate_valid
        |-> row_candidate_relation_ok && active_row != row_candidate_row
            && (active_pop < row_candidate_pop
                || (active_pop == row_candidate_pop
                    && active_row < row_candidate_row)));
    ap_candidate_after_next: assert property (
        next_valid && row_candidate_valid
        |-> next_row != row_candidate_row
            && (next_pop < row_candidate_pop
                || (next_pop == row_candidate_pop
                    && next_row < row_candidate_row)));

    // No metadata slot can be overwritten while held.  Active residual and
    // first-state may change only on a real accepted beat.
    ap_active_hold: assert property (
        active_valid && !issue_accept
        |=> active_valid && $stable({active_row, active_residual,
            active_parent_valid, active_parent, active_first,
            active_relation_ok, active_pop}));
    ap_priming_progress: assert property (
        active_valid && !active_primed |=> active_valid && active_primed);
    ap_primed_hold: assert property (
        active_valid && active_primed && !issue_accept
        |=> active_valid && active_primed);
    ap_next_hold: assert property (
        next_valid && !(issue_accept && issue_request_last)
        |=> next_valid && $stable({next_row, next_pop, next_relation_ok}));
    ap_pf_hold: assert property (
        pf_valid && !prefetch_accept
        |=> pf_valid && $stable({pf_consumer, pf_parent}));

    // The visible producer contract remains the original request/data
    // handshake.  There is no separate request acceptance and no payload FIFO.
    ap_request_maps_active: assert property (
        issue_request_valid |-> exec_active && active_valid && active_primed
            && issue_request_row_id == active_row
            && issue_request_first == active_first
            && issue_request_parent_valid == active_parent_valid
            && (!active_parent_valid
                || issue_request_parent_id == active_parent));
    ap_request_payload_stable: assert property (
        issue_request_valid && issue_data_valid && !issue_data_ready
        |=> issue_request_valid
            && $stable({issue_request_epoch, issue_request_row_id,
                issue_request_first, issue_request_last,
                issue_request_source_valid, issue_request_source_index,
                issue_request_parent_valid, issue_request_parent_id,
                issue_residual_data, issue_psum_prior}));
    ap_accept_requires_context: assert property (
        issue_accept |-> issue_request_valid && active_valid);
    ap_no_interrow_bubble: assert property (
        issue_accept && issue_request_last
            && (next_valid || row_candidate_valid)
        |=> issue_request_valid);
    ap_final_candidate_is_reserved: assert property (
        issue_accept && issue_request_last && row_candidate_valid
        |-> next_valid);

    // Prefetch tokens remain strict stable-order reservations; a later
    // candidate cannot bypass a held earlier token.
    ap_pf_exact_edge: assert property (
        pf_valid |-> !prefetched_bitmap[pf_consumer]);
    ap_pf_candidate_strictly_later_key: assert property (
        pf_valid && pf_candidate_valid
        |-> (pf_pop < pf_candidate_pop)
            || (pf_pop == pf_candidate_pop
                && pf_consumer < pf_candidate_consumer));
    ap_prefetch_accept_has_token: assert property (
        prefetch_accept |-> pf_valid);
    ap_macro_read_was_written: assert property (
        macro_read_accept |-> written_bitmap[pf_parent]);
    ap_one_port: assert property (
        !(macro_read_accept && live_write_accept));
    ap_forward_is_live_final: assert property (
        forward_accept |-> live_write_accept
            && pf_parent == active_row && !macro_read_accept);
    ap_deadline_real_read: assert property (
        deadline_hold |-> macro_read_accept && !live_write_accept
            && !issue_accept);
    ap_stalled_raw_blocks_access: assert property (
        stalled_raw |-> !macro_read_accept && !forward_accept);
    ap_scratch_write_identity: assert property (
        scratch_write_enable |-> live_write_accept
            && scratch_address == active_row);

    // Architectural final remains atomic and fail-closed.
    ap_completion_atomic: assert property (
        row_complete_valid == psum_write_valid);
    ap_completion_identity: assert property (
        row_complete_valid |-> row_complete_id == psum_write_address
            && row_complete_id == active_row);
    ap_final_accept_atomic: assert property (
        issue_accept && issue_request_last
        |-> psum_write_valid && psum_write_ready
            && row_complete_valid && row_complete_ready);
    ap_preaccept_fault_atomic: assert property (
        disable iff (!reset_n || protocol_error)
        issue_data_valid && issue_request_valid && !preaccept_protocol_ok
        |-> !issue_data_ready && !issue_accept && !macro_read_accept
            && !live_write_accept && !dead_elision_accept
            && !psum_write_valid && !row_complete_valid);
    ap_overflow_atomic: assert property (
        disable iff (!reset_n || protocol_error)
        overflow_block |-> !issue_data_ready && !issue_accept
            && !live_write_accept && !psum_write_valid
            && !row_complete_valid);
    ap_psum_stable: assert property (
        psum_write_valid && !(psum_write_ready && row_complete_ready)
        |=> psum_write_valid
            && $stable({psum_write_address, psum_write_data,
                row_complete_valid, row_complete_id}));

    ap_queue_bound: assert property (queue_occupancy <= 2);
    ap_reserved_bound: assert property (reserved_occupancy <= 2);
    ap_no_queue_overflow: assert property (!queue_overflow);
    ap_parent_conservation: assert property (
        count_parent_edges == count_macro_reads + count_forwards);
    ap_row_conservation: assert property (
        count_row_completions == count_psum_commits);
    ap_dead_live_conservation: assert property (
        count_row_completions
            == count_macro_writes + count_dead_write_elisions);

    // Debug pins are explicitly delayed observers.  Functional assertions use
    // the internal events above, so these properties cannot waive a real path.
    ap_debug_read_delay: assert property (
        debug_scratch_read_event == $past(macro_read_accept));
    ap_debug_write_delay: assert property (
        debug_scratch_write_event == $past(live_write_accept));
    ap_debug_forward_delay: assert property (
        debug_forward_event == $past(forward_accept));
    ap_debug_response_delay: assert property (
        debug_read_response_event == $past(read_pending));
    ap_debug_dual_delay: assert property (
        debug_dual_enqueue_event
            == $past(read_pending && forward_accept));
    ap_debug_dead_delay: assert property (
        debug_dead_write_elision_event == $past(dead_elision_accept));
    ap_debug_deadline_delay: assert property (
        debug_deadline_hold_event == $past(deadline_hold));
    ap_debug_overflow_delay: assert property (
        debug_overflow_block_event == $past(overflow_block));
    ap_debug_raw_delay: assert property (
        debug_stalled_raw_event == $past(stalled_raw));

    cp_two_cycle_initial_fill: cover property (
        exec_active && !active_valid
        ##1 active_valid && !active_primed
        ##1 active_valid && active_primed
        ##1 issue_request_valid);
    cp_promote_without_bubble: cover property (
        issue_accept && issue_request_last && next_valid
        ##1 issue_request_valid);
    cp_pf_replace_same_edge: cover property (
        prefetch_accept && pf_candidate_valid ##1 pf_valid);
    cp_dead_then_live: cover property (
        dead_elision_accept ##1 live_write_accept);
    cp_deadline_then_write: cover property (
        deadline_hold ##1 live_write_accept);
    cp_task_done: cover property (task_done_valid);
endmodule

bind m935_m912_three_stage_exact_parent_match_product_capture_island
    m935_m912_inherited_execution_assertions_r1 u_m919_assertions_r2 (
        .clk_core(clk_core),
        .reset_n(reset_n),
        .protocol_error(protocol_error),
        .imminent_protocol_fault(fault_condition_w),
        .exec_active(exec_active_q),
        .active_valid(active_ctx_valid_q),
        .active_row(active_ctx_row_q),
        .active_residual(active_ctx_residual_q),
        .active_parent_valid(active_ctx_parent_valid_q),
        .active_parent(active_ctx_parent_q),
        .active_first(active_ctx_first_q),
        .active_primed(active_ctx_primed_q),
        .active_relation_ok(active_ctx_relation_ok_q),
        .next_valid(next_ctx_valid_q),
        .next_row(next_ctx_row_q),
        .active_pop(active_ctx_original_pop_q),
        .next_pop(next_ctx_original_pop_q),
        .next_relation_ok(next_ctx_relation_ok_q),
        .row_candidate_valid(row_candidate_valid_w),
        .row_candidate_row(row_candidate_row_w),
        .row_candidate_pop(row_candidate_pop_w),
        .row_candidate_relation_ok(row_candidate_relation_ok_w),
        .pf_valid(pf_token_valid_q),
        .pf_consumer(pf_token_consumer_q),
        .pf_parent(pf_token_parent_q),
        .pf_pop(directory_q[exec_bank_q][pf_token_consumer_q][27:23]),
        .pf_candidate_valid(pf_candidate_valid_w),
        .pf_candidate_consumer(pf_candidate_consumer_w),
        .pf_candidate_pop(pf_candidate_pop_w),
        .prefetch_accept(prefetch_accept_w),
        .macro_read_accept(macro_read_accept_w),
        .live_write_accept(live_write_accept_w),
        .forward_accept(forward_accept_w),
        .dead_elision_accept(dead_elision_accept_w),
        .deadline_hold(deadline_hold_w),
        .stalled_raw(stalled_same_address_w),
        .read_pending(read_pending_q),
        .scratch_write_enable(scratch_write_enable_w),
        .scratch_address(scratch_address_w),
        .completed_bitmap(completed_bitmap_q),
        .prefetched_bitmap(prefetched_edge_bitmap_q),
        .written_bitmap(written_bitmap_q),
        .queue_occupancy(parent_queue_occupancy),
        .reserved_occupancy(parent_reserved_occupancy),
        .queue_overflow(queue_overflow_w),
        .issue_request_valid(issue_request_valid),
        .issue_request_epoch(issue_request_epoch),
        .issue_request_row_id(issue_request_row_id),
        .issue_request_first(issue_request_first),
        .issue_request_last(issue_request_last),
        .issue_request_source_valid(issue_request_source_valid),
        .issue_request_source_index(issue_request_source_index),
        .issue_request_parent_valid(issue_request_parent_valid),
        .issue_request_parent_id(issue_request_parent_id),
        .issue_data_valid(issue_data_valid),
        .issue_data_ready(issue_data_ready),
        .issue_residual_data(issue_residual_data),
        .issue_psum_prior(issue_psum_prior),
        .issue_accept(issue_accept_w),
        .preaccept_protocol_ok(preaccept_protocol_ok_w),
        .overflow_block(issue_data_valid && arithmetic_authoritative_w
            && issue_last_w && (row_overflow_w || psum_overflow_w)),
        .psum_write_valid(psum_write_valid),
        .psum_write_ready(psum_write_ready),
        .psum_write_address(psum_write_address),
        .psum_write_data(psum_write_data),
        .row_complete_valid(row_complete_valid),
        .row_complete_ready(row_complete_ready),
        .row_complete_id(row_complete_id),
        .task_done_valid(task_done_valid),
        .debug_scratch_read_event(debug_scratch_read_event),
        .debug_scratch_write_event(debug_scratch_write_event),
        .debug_forward_event(debug_forward_event),
        .debug_read_response_event(debug_read_response_event),
        .debug_dual_enqueue_event(debug_dual_enqueue_event),
        .debug_dead_write_elision_event(debug_dead_write_elision_event),
        .debug_deadline_hold_event(debug_deadline_hold_event),
        .debug_overflow_block_event(debug_overflow_block_event),
        .debug_stalled_raw_event(debug_stalled_raw_event),
        .count_parent_edges(count_parent_edges),
        .count_macro_reads(count_macro_reads),
        .count_macro_writes(count_macro_writes),
        .count_forwards(count_forwards),
        .count_dead_write_elisions(count_dead_write_elisions),
        .count_psum_commits(count_psum_commits),
        .count_row_completions(count_row_completions)
    );

`default_nettype wire
