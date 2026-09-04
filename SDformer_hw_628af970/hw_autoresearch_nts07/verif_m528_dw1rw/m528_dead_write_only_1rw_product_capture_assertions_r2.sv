`timescale 1ns/1ps
`default_nettype none

module m528_dead_write_only_1rw_product_capture_assertions_r2 (
    input logic clk_core,
    input logic reset_n,
    input logic protocol_error,
    input logic imminent_protocol_fault,
    input logic prep_valid,
    input logic prep_ready,
    input logic prep_task_start,
    input logic prep_task_last,
    input logic [15:0] prep_epoch,
    input logic [5:0] prep_row_id,
    input logic [15:0] prep_mask,
    input logic [3:0] prep_reserved,
    input logic prep_store,
    input logic prep_active,
    input logic prep_bank,
    input logic match_active,
    input logic exec_active,
    input logic exec_bank,
    input logic exec_bank_state_ok,
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
    input logic matching_parent_authoritative,
    input logic arithmetic_authoritative,
    input logic current_live,
    input logic [15:0] current_residual_mask,
    input logic current_parent_relation_ok,
    input logic psum_write_valid,
    input logic psum_write_ready,
    input logic [5:0] psum_write_address,
    input logic [1823:0] psum_write_data,
    input logic row_complete_valid,
    input logic row_complete_ready,
    input logic [5:0] row_complete_id,
    input logic task_done_valid,
    input logic [15:0] task_done_epoch,
    input logic scratch_read,
    input logic scratch_write,
    input logic [5:0] scratch_address,
    input logic [1151:0] scratch_write_data,
    input logic forward_event,
    input logic [5:0] lookahead_parent,
    input logic [5:0] lookahead_consumer,
    input logic lookahead_valid,
    input logic lookahead_immediate_next,
    input logic read_pending,
    input logic read_response_event,
    input logic dead_elision,
    input logic deadline_hold,
    input logic stalled_raw,
    input logic overflow_block,
    input logic queue_overflow,
    input logic [1:0] queue_occupancy,
    input logic [2:0] reserved_occupancy,
    input logic [63:0] written_bitmap,
    input logic [63:0] live_bitmap,
    input logic [63:0] count_issue_accepts,
    input logic [63:0] count_parent_edges,
    input logic [63:0] count_dead_write_elisions,
    input logic [63:0] count_macro_reads,
    input logic [63:0] count_macro_writes,
    input logic [63:0] count_forwards,
    input logic [63:0] count_deadline_holds,
    input logic [63:0] count_psum_commits,
    input logic [63:0] count_row_completions
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n || protocol_error || imminent_protocol_fault);

    // Single-port and exact one-cycle response contract.
    ap_read_xor_write: assert property (!(scratch_read && scratch_write));
    ap_read_was_written: assert property (
        scratch_read |-> written_bitmap[scratch_address]);
    ap_read_response_latency: assert property (scratch_read |=> read_response_event);
    ap_no_spurious_response: assert property (
        read_response_event |-> $past(scratch_read));
    ap_reserved_bound: assert property (reserved_occupancy <= 3'd2);
    ap_queue_bound: assert property (queue_occupancy <= 2'd2);
    ap_no_queue_overflow: assert property (!queue_overflow);
    ap_no_consume_credit: assert property (
        reserved_occupancy == 3'd2 |-> !(scratch_read || forward_event));

    // Dead suppresses only the scratch write.  Live always writes, including
    // same-cycle forwarding; row completion and psum commit stay atomic.
    ap_dead_has_completion: assert property (
        dead_elision |-> row_complete_valid && row_complete_ready
            && psum_write_valid && psum_write_ready && !scratch_write);
    ap_live_final_writes: assert property (
        issue_accept && issue_request_last && current_live |-> scratch_write);
    ap_forward_is_new_live_write: assert property (
        forward_event |-> scratch_write && !scratch_read
            && issue_accept && issue_request_last && current_live
            && scratch_address == issue_request_row_id
            && lookahead_parent == issue_request_row_id);
    ap_completion_atomic: assert property (
        row_complete_valid == psum_write_valid);
    ap_completion_identity: assert property (
        row_complete_valid |-> row_complete_id == psum_write_address
            && row_complete_id == issue_request_row_id);
    ap_overflow_atomic_block: assert property (
        disable iff (!reset_n || protocol_error)
        overflow_block |-> !issue_data_ready && !scratch_write
            && !row_complete_valid);
    ap_preaccept_fault_is_atomic: assert property (
        disable iff (!reset_n || protocol_error)
        issue_data_valid && issue_request_valid && !preaccept_protocol_ok
            |-> !issue_data_ready && !issue_accept && !scratch_read
                && !scratch_write && !dead_elision
                && !psum_write_valid && !row_complete_valid);
    ap_parent_only_nonzero_is_preaccept_fault: assert property (
        disable iff (!reset_n || protocol_error)
        issue_data_valid && issue_request_valid
            && !issue_request_source_valid && issue_request_parent_valid
            && issue_residual_data != 1152'b0
            |-> !preaccept_protocol_ok);
    ap_overflow_only_when_authoritative: assert property (
        disable iff (!reset_n || protocol_error)
        overflow_block |-> arithmetic_authoritative
            && matching_parent_authoritative);

    // Frozen ordering/ownership and stale-RAW safety.
    ap_one_execution_owner: assert property (exec_active |-> exec_bank_state_ok);
    ap_no_prep_overwrite_exec: assert property (
        prep_store && exec_active |-> prep_bank != exec_bank);
    ap_parent_is_live_and_exact_subset: assert property (
        issue_request_valid && issue_request_parent_valid
            |-> live_bitmap[issue_request_parent_id]
                && current_parent_relation_ok);
    ap_stalled_raw_blocks_access: assert property (
        stalled_raw |-> !scratch_read && !forward_event);
    ap_deadline_is_one_next_edge: assert property (
        deadline_hold |-> lookahead_valid && lookahead_immediate_next
            && scratch_read && !scratch_write && !issue_accept);
    ap_source_index_matches_mask: assert property (
        issue_request_valid && issue_request_source_valid
            |-> current_residual_mask[issue_request_source_index]);
    ap_parent_only_beat_is_zero: assert property (
        issue_data_valid && issue_request_valid
            && !issue_request_source_valid && issue_request_parent_valid
            |-> issue_residual_data == 1152'b0);
    for (genvar lane = 0; lane < 96; lane = lane + 1) begin : g_int8_format
        ap_signed12_is_sign_extended_int8: assert property (
            issue_data_valid && issue_request_valid
                && issue_request_source_valid
            |-> issue_residual_data[lane*12 + 8 +: 4]
                == {4{issue_residual_data[lane*12 + 7]}});
    end

    // Ready/valid stability at both architectural boundaries.
    ap_issue_request_stable: assert property (
        issue_request_valid && issue_data_valid && !issue_data_ready
        |=> issue_request_valid
            && $stable({issue_request_epoch, issue_request_row_id,
                issue_request_first, issue_request_last,
                issue_request_source_valid, issue_request_source_index,
                issue_request_parent_valid, issue_request_parent_id,
                issue_residual_data, issue_psum_prior}));
    ap_psum_stable: assert property (
        psum_write_valid && !(psum_write_ready && row_complete_ready)
        |=> psum_write_valid
            && $stable({psum_write_address, psum_write_data,
                row_complete_valid, row_complete_id}));
    ap_prep_stable: assert property (
        prep_valid && !prep_ready |=> prep_valid
            && $stable({prep_task_start, prep_task_last, prep_epoch,
                prep_row_id, prep_mask, prep_reserved}));

    // Conservation identities hold continuously because all corresponding
    // events increment on the same accepted edge.
    ap_parent_edge_conservation: assert property (
        count_parent_edges == count_macro_reads + count_forwards);
    ap_row_commit_conservation: assert property (
        count_row_completions == count_psum_commits);
    ap_dead_live_write_conservation: assert property (
        count_row_completions
            == count_macro_writes + count_dead_write_elisions);
    ap_forward_is_read_replacement: assert property (
        count_forwards <= count_parent_edges);
    ap_deadline_counter: assert property (
        deadline_hold |=> count_deadline_holds == $past(count_deadline_holds) + 1);
    ap_task_done_drained: assert property (
        task_done_valid |-> !exec_active && queue_occupancy == 0
            && reserved_occupancy == 0 && task_done_epoch == $past(issue_request_epoch));

    // Coverage obligations mirror the source-only prespec.  The future VCS
    // launch admission must require every named cover to be hit or explicitly
    // classify an attack-only cover separately.
    cp_dead_plus_read: cover property (dead_elision && scratch_read);
    cp_live_deadline_read_then_write: cover property (
        deadline_hold ##1 scratch_write);
    cp_same_address_forward: cover property (forward_event);
    cp_pending_plus_forward: cover property (read_pending && forward_event);
    cp_full_then_consume_no_credit: cover property (
        reserved_occupancy == 2 && issue_accept && issue_request_last
            && issue_request_parent_valid && !(scratch_read || forward_event));
    cp_three_dead: cover property (dead_elision ##1 dead_elision ##1 dead_elision);
    cp_alternating_dead_live: cover property (
        dead_elision ##1 scratch_write ##1 dead_elision);
    cp_exact_parent: cover property (
        issue_accept && !issue_request_source_valid
            && issue_request_parent_valid);
    cp_partial_parent_multibeat: cover property (
        issue_accept && issue_request_first && issue_request_parent_valid
            && !issue_request_last ##[1:16]
        issue_accept && issue_request_last);
    cp_back_to_back_completion: cover property (
        row_complete_valid && row_complete_ready ##1
        row_complete_valid && row_complete_ready);
    cp_stalled_same_address: cover property (stalled_raw ##[1:8] forward_event);
    cp_pingpong_overlap: cover property (prep_active && exec_active);
    cp_row_zero: cover property (row_complete_valid && row_complete_id == 0);
    cp_row_sixty_three: cover property (
        row_complete_valid && row_complete_id == 6'd63);
    cp_all_slices_nonzero: cover property (
        scratch_write
            && (&{(|scratch_write_data[127:0]),
                    (|scratch_write_data[255:128]),
                    (|scratch_write_data[383:256]),
                    (|scratch_write_data[511:384]),
                    (|scratch_write_data[639:512]),
                    (|scratch_write_data[767:640]),
                    (|scratch_write_data[895:768]),
                    (|scratch_write_data[1023:896]),
                    (|scratch_write_data[1151:1024])}));
    cp_dirty_reserved_attack: cover property (
        prep_valid && prep_ready && prep_reserved != 0 ##1 protocol_error);
    cp_overflow_attack: cover property (overflow_block ##1 protocol_error);
    cp_stale_epoch_attack: cover property (
        prep_valid && prep_ready && prep_task_start ##1 protocol_error);
endmodule

bind m528_dead_write_only_1rw_product_capture_island_r2
    m528_dead_write_only_1rw_product_capture_assertions_r2 u_m528_assertions (
        .clk_core(clk_core),
        .reset_n(reset_n),
        .protocol_error(protocol_error),
        .imminent_protocol_fault(fault_condition_w),
        .prep_valid(prep_valid),
        .prep_ready(prep_ready),
        .prep_task_start(prep_task_start),
        .prep_task_last(prep_task_last),
        .prep_epoch(prep_epoch),
        .prep_row_id(prep_row_id),
        .prep_mask(prep_mask),
        .prep_reserved(prep_reserved),
        .prep_store(prep_store_w),
        .prep_active(prep_active_q),
        .prep_bank(prep_bank_q),
        .match_active(match_active_q),
        .exec_active(exec_active_q),
        .exec_bank(exec_bank_q),
        .exec_bank_state_ok(bank_state_q[exec_bank_q] == BANK_EXEC),
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
        .matching_parent_authoritative(matching_parent_authoritative_w),
        .arithmetic_authoritative(arithmetic_authoritative_w),
        .current_live(current_live_w),
        .current_residual_mask(issue_work_mask_w),
        .current_parent_relation_ok(!current_parent_valid_w
            || (popcount16(mask_q[exec_bank_q][current_parent_id_w]) >= 1
                && ((mask_q[exec_bank_q][current_parent_id_w]
                        & current_original_mask_w)
                    == mask_q[exec_bank_q][current_parent_id_w])
                && !((mask_q[exec_bank_q][current_parent_id_w]
                        == current_original_mask_w)
                    && current_parent_id_w >= current_row_w))),
        .psum_write_valid(psum_write_valid),
        .psum_write_ready(psum_write_ready),
        .psum_write_address(psum_write_address),
        .psum_write_data(psum_write_data),
        .row_complete_valid(row_complete_valid),
        .row_complete_ready(row_complete_ready),
        .row_complete_id(row_complete_id),
        .task_done_valid(task_done_valid),
        .task_done_epoch(task_done_epoch),
        .scratch_read(macro_read_accept_w),
        .scratch_write(live_write_accept_w),
        .scratch_address(scratch_address_w),
        .scratch_write_data(row_final_packed_w),
        .forward_event(forward_accept_w),
        .lookahead_parent(lookahead_parent_w),
        .lookahead_consumer(lookahead_consumer_w),
        .lookahead_valid(lookahead_valid_w),
        .lookahead_immediate_next(lookahead_immediate_next_w),
        .read_pending(read_pending_q),
        .read_response_event(read_pending_q),
        .dead_elision(dead_elision_accept_w),
        .deadline_hold(deadline_hold_w),
        .stalled_raw(stalled_same_address_w),
        .overflow_block(debug_overflow_block_event),
        .queue_overflow(queue_overflow_w),
        .queue_occupancy(parent_queue_occupancy),
        .reserved_occupancy(parent_reserved_occupancy),
        .written_bitmap(written_bitmap_q),
        .live_bitmap(parent_live_q[exec_bank_q]),
        .count_issue_accepts(count_issue_accepts),
        .count_parent_edges(count_parent_edges),
        .count_dead_write_elisions(count_dead_write_elisions),
        .count_macro_reads(count_macro_reads),
        .count_macro_writes(count_macro_writes),
        .count_forwards(count_forwards),
        .count_deadline_holds(count_deadline_holds),
        .count_psum_commits(count_psum_commits),
        .count_row_completions(count_row_completions)
    );

`default_nettype wire
