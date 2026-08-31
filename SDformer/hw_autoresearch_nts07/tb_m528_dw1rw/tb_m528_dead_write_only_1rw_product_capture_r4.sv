`timescale 1ns/1ps
`default_nettype none

module tb_m528_dead_write_only_1rw_product_capture_r3;
    logic clk_core, reset_n;
    logic prep_valid, prep_ready, prep_task_start, prep_task_last;
    logic [15:0] prep_epoch, prep_mask;
    logic [5:0] prep_row_id;
    logic [3:0] prep_reserved;
    logic issue_request_valid, issue_request_first, issue_request_last;
    logic issue_request_source_valid, issue_request_parent_valid;
    logic [15:0] issue_request_epoch;
    logic [5:0] issue_request_row_id, issue_request_parent_id;
    logic [3:0] issue_request_source_index;
    logic issue_data_valid, issue_data_ready;
    logic [1151:0] issue_residual_data;
    logic [1823:0] issue_psum_prior;
    logic psum_write_valid, psum_write_ready;
    logic [5:0] psum_write_address;
    logic [1823:0] psum_write_data;
    logic row_complete_valid, row_complete_ready;
    logic [5:0] row_complete_id;
    logic task_done_valid;
    logic [15:0] task_done_epoch;
    logic protocol_error, preprocess_busy, execute_busy;
    logic active_directory_bank;
    logic [1:0] parent_queue_occupancy;
    logic [2:0] parent_reserved_occupancy;
    logic [63:0] debug_parent_live_bitmap, debug_written_bitmap;
    logic debug_scratch_read_event, debug_scratch_write_event;
    logic debug_forward_event, debug_read_response_event;
    logic debug_dual_enqueue_event, debug_dead_write_elision_event;
    logic debug_deadline_hold_event, debug_overflow_block_event;
    logic debug_stalled_raw_event;
    logic [63:0] count_issue_accepts, count_parent_edges;
    logic [63:0] count_dead_write_elisions, count_macro_reads;
    logic [63:0] count_macro_writes, count_forwards;
    logic [63:0] count_deadline_holds, count_issue_stalls;
    logic [63:0] count_psum_commits, count_row_completions;
    // VCS procedural force requires a static lifetime RHS.  Directed tasks
    // copy their computed value here before force; this register is TB-only.
    logic [1151:0] force_parent_data_static;

    logic [15:0] stimulus_masks [0:63];
    logic [15:0] reference_residual [0:63];
    logic [15:0] reference_mask_by_slot [0:7][0:63];
    logic [15:0] reference_residual_by_slot [0:7][0:63];
    integer reference_parent [0:63];
    integer reference_pop [0:63];
    integer reference_parent_by_slot [0:7][0:63];
    integer reference_parent_refcount [0:7][0:63];
    integer reference_pop_by_slot [0:7][0:63];
    logic [63:0] reference_live_bitmap [0:7];
    integer reference_active_rows [0:7];
    integer reference_parent_edges [0:7];
    integer expected_row [0:63][0:95];
    integer expected_row_by_slot [0:7][0:63][0:95];
    logic [15:0] score_remaining [0:7][0:63];
    logic score_started [0:7][0:63];
    integer last_commit_pop [0:7];
    integer last_commit_id [0:7];
    integer error_count, commit_count, done_count, attack_count;
    integer oracle_issue_accepts [0:7];
    integer oracle_live_writes [0:7];
    integer oracle_dead_elisions [0:7];
    integer oracle_macro_reads [0:7];
    integer oracle_forwards [0:7];
    integer oracle_deadline_holds [0:7];
    integer oracle_issue_stalls [0:7];
    integer oracle_psum_commits [0:7];
    integer oracle_row_completions [0:7];
    // Cleanroom cycle-model state.  Every field below is derived from the
    // accepted prep stimulus, the frozen matcher/order rules, and the TB's
    // externally driven sink schedule.  DUT ready/debug/directory/live/queue
    // signals are never used to generate any oracle state or expected pulse.
    integer oracle_match_countdown [0:7];
    integer oracle_load_sequence [0:7];
    logic oracle_task_ready [0:7];
    logic [15:0] oracle_epoch_by_slot [0:7];
    integer oracle_next_load_sequence;
    logic oracle_exec_active;
    integer oracle_exec_slot;
    logic [15:0] oracle_exec_epoch;
    logic [63:0] oracle_completed_bitmap;
    logic [63:0] oracle_prefetched_bitmap;
    logic [63:0] oracle_written_bitmap;
    logic oracle_row_inflight;
    integer oracle_current_row;
    logic [15:0] oracle_residual_remaining;
    logic oracle_slot0_valid, oracle_slot1_valid;
    integer oracle_slot0_parent, oracle_slot1_parent;
    integer oracle_slot0_consumer, oracle_slot1_consumer;
    logic [1151:0] oracle_slot0_data, oracle_slot1_data;
    logic oracle_read_pending;
    integer oracle_read_pending_parent, oracle_read_pending_consumer;
    logic [1151:0] oracle_read_pending_data;
    logic [1151:0] oracle_scratch_data [0:63];
    logic oracle_done_expected;
    logic [15:0] oracle_done_epoch;

    logic oracle_raw_pending;
    logic [15:0] oracle_raw_epoch;
    integer oracle_raw_consumer, oracle_raw_parent, oracle_raw_age;
    logic oracle_prev_macro_read;
    integer oracle_prev_macro_read_address;
    logic [1151:0] oracle_prev_macro_read_data;

    integer cov_dead_plus_read;
    integer cov_deadline_read_write;
    integer cov_same_address_forward;
    integer cov_pending_plus_forward;
    integer cov_full_no_credit;
    integer cov_liveness_sequences;
    integer cov_parent_modes;
    integer cov_stalled_raw_recovery;
    integer cov_pingpong_overlap;
    integer cov_endpoint_rows;
    integer cov_all_slices;
    integer cov_consecutive_distinct_reads;
    integer cov_response_identity_checks;
    integer dead_run_q;
    logic saw_three_dead_q, saw_alternating_q, previous_dead_q;
    logic completion_seen_q;
    logic saw_exact_parent_q, saw_partial_parent_q, saw_multibeat_parent_q;
    logic saw_back_to_back_completion_q, completion_last_cycle_q;
    logic saw_stalled_raw_q, saw_row_zero_q, saw_row_sixty_three_q;
    logic [8:0] slices_seen_q;
    logic deadline_last_cycle_q;

    integer attack_dirty_reserved_count;
    integer attack_stale_epoch_count;
    integer attack_overflow_count;
    integer attack_wrong_parent_count;
    integer attack_read_before_write_count;
    integer attack_parent_only_nonzero_count;
    integer unsigned lfsr_q;
    logic attack_overflow_mode;
    logic attack_parent_only_nonzero_mode;
    logic normal_score_enable;

    m528_dead_write_only_1rw_product_capture_island_r2 dut (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic integer pc16(input logic [15:0] value);
        integer result;
        begin
            result = 0;
            for (integer bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                result = result + value[bit_index];
            return result;
        end
    endfunction

    function automatic integer pc64(input logic [63:0] value);
        integer result;
        begin
            result = 0;
            for (integer bit_index = 0; bit_index < 64; bit_index = bit_index + 1)
                result = result + value[bit_index];
            return result;
        end
    endfunction

    function automatic logic signed [11:0] source_value12(
        input integer source, input integer lane
    );
        logic signed [7:0] value8;
        integer raw;
        begin
            raw = ((source * 7 + lane * 3) % 17) - 8;
            value8 = raw;
            source_value12 = {{4{value8[7]}}, value8};
        end
    endfunction

    function automatic logic [1151:0] oracle_pack_row12(
        input integer slot, input integer row
    );
        logic [1151:0] packed_row;
        begin
            packed_row = '0;
            for (integer lane = 0; lane < 96; lane = lane + 1)
                packed_row[lane*12 +: 12] =
                    expected_row_by_slot[slot][row][lane][11:0];
            return packed_row;
        end
    endfunction

    task automatic clear_drivers;
        begin
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
            prep_epoch = '0;
            prep_row_id = '0;
            prep_mask = '0;
            prep_reserved = '0;
            attack_overflow_mode = 1'b0;
            attack_parent_only_nonzero_mode = 1'b0;
        end
    endtask

    task automatic reset_dut;
        begin
            normal_score_enable = 1'b0;
            clear_drivers();
            reset_n = 1'b0;
            repeat (5) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1'b1;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic make_directed_masks;
        begin
            for (integer row = 0; row < 64; row = row + 1)
                stimulus_masks[row] = 16'b0;
            stimulus_masks[0] = 16'h0001;
            stimulus_masks[1] = 16'h0003;
            stimulus_masks[2] = 16'h0005;
            stimulus_masks[3] = 16'h0007;
            stimulus_masks[4] = 16'h0003; // equal-earlier exact parent
            stimulus_masks[5] = 16'h000f;
            stimulus_masks[6] = 16'h003f;
            stimulus_masks[7] = 16'h00ff;
            stimulus_masks[8] = 16'h0fff;
            stimulus_masks[9] = 16'hffff;
            stimulus_masks[10] = 16'h00f3;
            stimulus_masks[11] = 16'h0303;
            stimulus_masks[12] = 16'h3333;
            stimulus_masks[13] = 16'h5555;
            stimulus_masks[14] = 16'haaaa;
            stimulus_masks[15] = 16'h8000;
            stimulus_masks[16] = 16'h8001;
            stimulus_masks[31] = 16'h00ff;
            stimulus_masks[47] = 16'h0f0f;
            stimulus_masks[63] = 16'hffff;
        end
    endtask

    task automatic make_random_masks(input integer seed);
        integer unsigned state;
        begin
            state = seed;
            for (integer row = 0; row < 64; row = row + 1) begin
                state = state * 32'd1664525 + 32'd1013904223;
                if ((row % 9) == 0)
                    stimulus_masks[row] = 16'b0;
                else
                    stimulus_masks[row] = state[15:0]
                        & ((16'h0001 << ((row % 16) + 1)) - 1'b1);
            end
            stimulus_masks[0] = 16'h0001;
            stimulus_masks[1] = 16'h0003;
            stimulus_masks[2] = 16'h0007;
            stimulus_masks[63] = 16'hffff;
        end
    endtask

    // This exact pattern forces two adjacent reads of already-written, distinct
    // parents.  Row 1 is forwarded to row 2.  While row 2 consumes that slot,
    // row 3 prefetches row 0; on the next cycle row 4 prefetches row 1.  The
    // cleanroom response queue checks both identities and their unequal data.
    task automatic make_consecutive_distinct_read_masks;
        begin
            for (integer row = 0; row < 64; row = row + 1)
                stimulus_masks[row] = 16'b0;
            stimulus_masks[0] = 16'h0001;
            stimulus_masks[1] = 16'h0002;
            stimulus_masks[2] = 16'h0006;
            stimulus_masks[3] = 16'h0005;
            stimulus_masks[4] = 16'h000a;
        end
    endtask

    // Cleanroom reference repeats M504: maximum-population exact subset;
    // equal current/later patterns excluded; lowest candidate ID wins ties.
    task automatic build_reference(input logic [15:0] epoch);
        integer best, best_pop, current_pop, candidate_pop;
        integer slot;
        begin
            slot = epoch[2:0];
            reference_live_bitmap[slot] = 64'b0;
            reference_active_rows[slot] = 0;
            reference_parent_edges[slot] = 0;
            for (integer row = 0; row < 64; row = row + 1)
                reference_parent_refcount[slot][row] = 0;
            for (integer row = 0; row < 64; row = row + 1) begin
                current_pop = pc16(stimulus_masks[row]);
                reference_pop[row] = current_pop;
                best = -1;
                best_pop = 0;
                if (current_pop >= 2) begin
                    for (integer candidate = 0; candidate < 64;
                            candidate = candidate + 1) begin
                        candidate_pop = pc16(stimulus_masks[candidate]);
                        if (((stimulus_masks[candidate] & stimulus_masks[row])
                                    == stimulus_masks[candidate])
                                && candidate_pop >= 1
                                && !((stimulus_masks[candidate]
                                        == stimulus_masks[row])
                                    && candidate >= row)
                                && candidate_pop > best_pop) begin
                            best = candidate;
                            best_pop = candidate_pop;
                        end
                    end
                end
                reference_parent[row] = best;
                reference_parent_by_slot[slot][row] = best;
                reference_pop_by_slot[slot][row] = current_pop;
                reference_mask_by_slot[slot][row] = stimulus_masks[row];
                if (stimulus_masks[row] != 16'b0)
                    reference_active_rows[slot] =
                        reference_active_rows[slot] + 1;
                if (best >= 0) begin
                    reference_parent_refcount[slot][best] =
                        reference_parent_refcount[slot][best] + 1;
                    reference_parent_edges[slot] =
                        reference_parent_edges[slot] + 1;
                end
                reference_residual[row] = (best >= 0)
                    ? stimulus_masks[row] ^ stimulus_masks[best]
                    : stimulus_masks[row];
                reference_residual_by_slot[slot][row] = (best >= 0)
                    ? stimulus_masks[row] ^ stimulus_masks[best]
                    : stimulus_masks[row];
                score_remaining[slot][row] = reference_residual[row];
                score_started[slot][row] = 1'b0;
                for (integer lane = 0; lane < 96; lane = lane + 1) begin
                    expected_row[row][lane] = 0;
                    for (integer source = 0; source < 16; source = source + 1)
                        if (stimulus_masks[row][source])
                            expected_row[row][lane] = expected_row[row][lane]
                                + $signed(source_value12(source, lane));
                    expected_row_by_slot[slot][row][lane] =
                        expected_row[row][lane];
                end
            end
            for (integer row = 0; row < 64; row = row + 1)
                if (reference_parent_refcount[slot][row] > 0)
                    reference_live_bitmap[slot][row] = 1'b1;
            last_commit_pop[slot] = -1;
            last_commit_id[slot] = -1;
            oracle_issue_accepts[slot] = 0;
            oracle_live_writes[slot] = 0;
            oracle_dead_elisions[slot] = 0;
            oracle_macro_reads[slot] = 0;
            oracle_forwards[slot] = 0;
            oracle_deadline_holds[slot] = 0;
            oracle_issue_stalls[slot] = 0;
            oracle_psum_commits[slot] = 0;
            oracle_row_completions[slot] = 0;
            oracle_epoch_by_slot[slot] = epoch;
            oracle_match_countdown[slot] = -1;
            oracle_load_sequence[slot] = 32'h7fff_ffff;
            oracle_task_ready[slot] = 1'b0;
        end
    endtask

    task automatic load_task(input logic [15:0] epoch);
        begin
            for (integer row = 0; row < 64; row = row + 1) begin
                @(negedge clk_core);
                prep_valid = 1'b1;
                prep_task_start = (row == 0);
                prep_task_last = (row == 63);
                prep_epoch = epoch;
                prep_row_id = row[5:0];
                prep_mask = stimulus_masks[row];
                prep_reserved = 4'b0;
                while (!prep_ready) @(negedge clk_core);
                @(posedge clk_core);
            end
            @(negedge clk_core);
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
        end
    endtask

    task automatic wait_done(input logic [15:0] epoch);
        integer watchdog;
        begin
            watchdog = 0;
            while (!(task_done_valid && task_done_epoch == epoch)) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 20000)
                    $fatal(1, "task timeout epoch=%0d", epoch);
            end
        end
    endtask

    // Payload is a pure function of source index and lane, never of consumer
    // row.  Therefore exact parent reuse is numerically checkable.
    always_comb begin
        issue_data_valid = issue_request_valid;
        issue_residual_data = '0;
        issue_psum_prior = '0;
        for (integer lane = 0; lane < 96; lane = lane + 1) begin
            if (issue_request_source_valid)
                issue_residual_data[lane*12 +: 12] = attack_overflow_mode
                    ? {{4{1'b0}}, 8'sd127}
                    : source_value12(issue_request_source_index, lane);
            if (attack_overflow_mode)
                issue_psum_prior[lane*19 +: 19] = 19'sd262143;
        end
        if (attack_parent_only_nonzero_mode && issue_request_valid
                && issue_request_parent_valid
                && !issue_request_source_valid)
            issue_residual_data[11:0] = 12'sd1;
    end

    // Deterministic constrained backpressure.  Both sinks vary independently,
    // including same-address final stalls and queue-full consume cycles.
    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            lfsr_q <= 32'h5281_1a5d;
            psum_write_ready <= 1'b1;
            row_complete_ready <= 1'b1;
        end else begin
            lfsr_q <= {lfsr_q[30:0],
                lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            psum_write_ready <= lfsr_q[0] | lfsr_q[3];
            row_complete_ready <= lfsr_q[1] | lfsr_q[5];
        end
    end

    task automatic oracle_expect_bit(
        input string label, input logic got, input logic expected
    );
        begin
            if (got !== expected) begin
                error_count = error_count + 1;
                $error("cleanroom pulse mismatch %s got=%b expected=%b epoch=%0d",
                    label, got, expected, oracle_exec_epoch);
            end
        end
    endtask

    task automatic oracle_reset_state;
        begin
            oracle_exec_active = 1'b0;
            oracle_exec_slot = 0;
            oracle_exec_epoch = '0;
            oracle_completed_bitmap = '0;
            oracle_prefetched_bitmap = '0;
            oracle_written_bitmap = '0;
            oracle_row_inflight = 1'b0;
            oracle_current_row = 0;
            oracle_residual_remaining = '0;
            oracle_slot0_valid = 1'b0;
            oracle_slot1_valid = 1'b0;
            oracle_slot0_parent = 0;
            oracle_slot1_parent = 0;
            oracle_slot0_consumer = 0;
            oracle_slot1_consumer = 0;
            oracle_slot0_data = '0;
            oracle_slot1_data = '0;
            oracle_read_pending = 1'b0;
            oracle_read_pending_parent = 0;
            oracle_read_pending_consumer = 0;
            oracle_read_pending_data = '0;
            oracle_done_expected = 1'b0;
            oracle_done_epoch = '0;
            oracle_raw_pending = 1'b0;
            oracle_raw_epoch = '0;
            oracle_raw_consumer = 0;
            oracle_raw_parent = 0;
            oracle_raw_age = 0;
            oracle_prev_macro_read = 1'b0;
            oracle_prev_macro_read_address = 0;
            oracle_prev_macro_read_data = '0;
            oracle_next_load_sequence = 0;
            dead_run_q = 0;
            previous_dead_q = 1'b0;
            completion_seen_q = 1'b0;
            deadline_last_cycle_q = 1'b0;
            completion_last_cycle_q = 1'b0;
            saw_stalled_raw_q = 1'b0;
            for (integer slot = 0; slot < 8; slot = slot + 1) begin
                oracle_match_countdown[slot] = -1;
                oracle_load_sequence[slot] = 32'h7fff_ffff;
                oracle_task_ready[slot] = 1'b0;
                oracle_epoch_by_slot[slot] = '0;
            end
            for (integer row = 0; row < 64; row = row + 1)
                oracle_scratch_data[row] = '0;
        end
    endtask

    task automatic oracle_activate(input integer slot);
        begin
            oracle_exec_active = 1'b1;
            oracle_exec_slot = slot;
            oracle_exec_epoch = oracle_epoch_by_slot[slot];
            oracle_task_ready[slot] = 1'b0;
            oracle_completed_bitmap = '0;
            oracle_prefetched_bitmap = '0;
            oracle_written_bitmap = '0;
            oracle_row_inflight = 1'b0;
            oracle_current_row = 0;
            oracle_residual_remaining = '0;
            oracle_slot0_valid = 1'b0;
            oracle_slot1_valid = 1'b0;
            oracle_read_pending = 1'b0;
            oracle_raw_pending = 1'b0;
            oracle_prev_macro_read = 1'b0;
            oracle_issue_accepts[slot] = 0;
            oracle_live_writes[slot] = 0;
            oracle_dead_elisions[slot] = 0;
            oracle_macro_reads[slot] = 0;
            oracle_forwards[slot] = 0;
            oracle_deadline_holds[slot] = 0;
            oracle_issue_stalls[slot] = 0;
            oracle_psum_commits[slot] = 0;
            oracle_row_completions[slot] = 0;
            last_commit_pop[slot] = -1;
            last_commit_id[slot] = -1;
        end
    endtask

    task automatic oracle_compare_counters(input integer slot);
        begin
            if (count_issue_accepts !== oracle_issue_accepts[slot]
                    || count_parent_edges !==
                        oracle_macro_reads[slot] + oracle_forwards[slot]
                    || count_dead_write_elisions !== oracle_dead_elisions[slot]
                    || count_macro_reads !== oracle_macro_reads[slot]
                    || count_macro_writes !== oracle_live_writes[slot]
                    || count_forwards !== oracle_forwards[slot]
                    || count_deadline_holds !== oracle_deadline_holds[slot]
                    || count_issue_stalls !== oracle_issue_stalls[slot]
                    || count_psum_commits !== oracle_psum_commits[slot]
                    || count_row_completions !== oracle_row_completions[slot]) begin
                error_count = error_count + 1;
                $error("cleanroom counter mismatch epoch=%0d issue=%0d/%0d parent=%0d/%0d write=%0d/%0d dead=%0d/%0d read=%0d/%0d fwd=%0d/%0d hold=%0d/%0d stall=%0d/%0d",
                    oracle_exec_epoch, count_issue_accepts,
                    oracle_issue_accepts[slot], count_parent_edges,
                    oracle_macro_reads[slot] + oracle_forwards[slot],
                    count_macro_writes, oracle_live_writes[slot],
                    count_dead_write_elisions, oracle_dead_elisions[slot],
                    count_macro_reads, oracle_macro_reads[slot],
                    count_forwards, oracle_forwards[slot],
                    count_deadline_holds, oracle_deadline_holds[slot],
                    count_issue_stalls, oracle_issue_stalls[slot]);
            end
        end
    endtask

    // Full independent cycle oracle.  DUT outputs occur only in comparisons;
    // expected values and every state transition above them are generated from
    // the frozen reference tables and external stimulus schedule.
    task automatic oracle_cycle;
        integer slot, selected, selected_pop, current, next_row, next_pop;
        integer look_consumer, look_parent, look_pop, source_index;
        integer reserved, got_value, chosen_slot, chosen_sequence;
        logic current_valid, next_valid, look_valid, parent_valid, current_live;
        logic [15:0] work_mask, remaining_after;
        logic source_valid, synthetic_parent, issue_last, parent_authoritative;
        logic sinks_ready, base_ready, expected_ready, expected_accept;
        logic expected_live_write, expected_dead_elision, expected_forward;
        logic expected_macro_read, expected_deadline, expected_stalled_raw;
        logic expected_prefetch, expected_issue_stall, expected_psum_valid;
        logic expected_read_response, expected_dual_enqueue, completion_dead;
        logic just_deactivated;
        logic next_slot0_valid, next_slot1_valid;
        integer next_slot0_parent, next_slot1_parent;
        integer next_slot0_consumer, next_slot1_consumer;
        logic [1151:0] next_slot0_data, next_slot1_data;
        logic [1151:0] expected_residual_payload, expected_row_data;
        begin
            just_deactivated = 1'b0;

            // A predicted done pulse is checked before it is retired.  It is
            // generated by the prior cleanroom drained state, never by DUT done.
            oracle_expect_bit("task_done_valid", task_done_valid,
                oracle_done_expected);
            if (oracle_done_expected) begin
                if (task_done_epoch !== oracle_done_epoch) begin
                    error_count = error_count + 1;
                    $error("cleanroom done epoch mismatch got=%0d expected=%0d",
                        task_done_epoch, oracle_done_epoch);
                end
                done_count = done_count + 1;
                slot = oracle_done_epoch[2:0];
                oracle_compare_counters(slot);
                if (debug_written_bitmap !== reference_live_bitmap[slot]
                        || oracle_written_bitmap !== reference_live_bitmap[slot]
                        || oracle_macro_reads[slot] + oracle_forwards[slot]
                            != reference_parent_edges[slot]
                        || oracle_live_writes[slot]
                            != pc64(reference_live_bitmap[slot])
                        || oracle_dead_elisions[slot]
                            != reference_active_rows[slot]
                                - pc64(reference_live_bitmap[slot])
                        || oracle_row_completions[slot]
                            != oracle_psum_commits[slot]
                        || oracle_row_completions[slot]
                            != oracle_live_writes[slot]
                                + oracle_dead_elisions[slot]) begin
                    error_count = error_count + 1;
                    $error("cleanroom per-epoch closure mismatch epoch=%0d",
                        oracle_done_epoch);
                end
            end
            oracle_done_expected = 1'b0;

            oracle_expect_bit("execute_busy", execute_busy, oracle_exec_active);
            if (protocol_error) begin
                oracle_raw_pending = 1'b0; // explicit task-abort cleanup
                $fatal(1, "normal cleanroom epoch aborted by protocol_error");
            end

            if (oracle_exec_active) begin
                slot = oracle_exec_slot;
                oracle_compare_counters(slot);
                if (debug_parent_live_bitmap !== reference_live_bitmap[slot]
                        || debug_written_bitmap !== oracle_written_bitmap) begin
                    error_count = error_count + 1;
                    $error("cleanroom live/written mismatch epoch=%0d live=%h/%h written=%h/%h",
                        oracle_exec_epoch, debug_parent_live_bitmap,
                        reference_live_bitmap[slot], debug_written_bitmap,
                        oracle_written_bitmap);
                end

                selected = -1;
                selected_pop = 31;
                for (integer row = 0; row < 64; row = row + 1) begin
                    if (reference_mask_by_slot[slot][row] != 16'b0
                            && !oracle_completed_bitmap[row]
                            && (selected < 0
                                || reference_pop_by_slot[slot][row]
                                    < selected_pop)) begin
                        selected = row;
                        selected_pop = reference_pop_by_slot[slot][row];
                    end
                end
                current_valid = oracle_row_inflight || selected >= 0;
                current = oracle_row_inflight ? oracle_current_row : selected;

                next_row = -1;
                next_pop = 31;
                if (current_valid) begin
                    for (integer row = 0; row < 64; row = row + 1) begin
                        if (reference_mask_by_slot[slot][row] != 16'b0
                                && !oracle_completed_bitmap[row]
                                && row != current
                                && (next_row < 0
                                    || reference_pop_by_slot[slot][row]
                                        < next_pop)) begin
                            next_row = row;
                            next_pop = reference_pop_by_slot[slot][row];
                        end
                    end
                end
                next_valid = next_row >= 0;

                look_consumer = -1;
                look_parent = -1;
                look_pop = 31;
                for (integer row = 0; row < 64; row = row + 1) begin
                    if (reference_mask_by_slot[slot][row] != 16'b0
                            && !oracle_completed_bitmap[row]
                            && reference_parent_by_slot[slot][row] >= 0
                            && !oracle_prefetched_bitmap[row]
                            && (look_consumer < 0
                                || reference_pop_by_slot[slot][row]
                                    < look_pop)) begin
                        look_consumer = row;
                        look_parent = reference_parent_by_slot[slot][row];
                        look_pop = reference_pop_by_slot[slot][row];
                    end
                end
                look_valid = look_consumer >= 0;

                work_mask = oracle_row_inflight ? oracle_residual_remaining
                    : (current_valid
                        ? reference_residual_by_slot[slot][current] : 16'b0);
                source_index = -1;
                for (integer source = 0; source < 16; source = source + 1)
                    if (source_index < 0 && work_mask[source])
                        source_index = source;
                source_valid = source_index >= 0;
                parent_valid = current_valid
                    && reference_parent_by_slot[slot][current] >= 0;
                synthetic_parent = current_valid && !source_valid && parent_valid;
                remaining_after = work_mask;
                if (source_valid)
                    remaining_after[source_index] = 1'b0;
                issue_last = current_valid
                    && (synthetic_parent || remaining_after == 16'b0);
                parent_authoritative = !parent_valid
                    || (oracle_slot0_valid
                        && oracle_slot0_parent
                            == reference_parent_by_slot[slot][current]
                        && oracle_slot0_consumer == current);
                reserved = oracle_slot0_valid + oracle_slot1_valid
                    + oracle_read_pending;
                sinks_ready = !issue_last
                    || (psum_write_ready && row_complete_ready);
                base_ready = current_valid && parent_authoritative
                    && sinks_ready;
                expected_deadline = current_valid && base_ready && issue_last
                    && reference_live_bitmap[slot][current]
                    && look_valid && next_valid && look_consumer == next_row
                    && reserved < 2 && oracle_written_bitmap[look_parent]
                    && look_parent != current;
                expected_ready = base_ready && !expected_deadline;
                expected_accept = current_valid && expected_ready;
                expected_live_write = expected_accept && issue_last
                    && reference_live_bitmap[slot][current];
                expected_dead_elision = expected_accept && issue_last
                    && !reference_live_bitmap[slot][current];
                expected_stalled_raw = current_valid && issue_last
                    && reference_live_bitmap[slot][current] && look_valid
                    && look_parent == current && !base_ready;
                expected_forward = expected_live_write && look_valid
                    && reserved < 2 && look_parent == current;
                expected_macro_read = look_valid && reserved < 2
                    && oracle_written_bitmap[look_parent]
                    && !expected_live_write && !expected_stalled_raw;
                expected_prefetch = expected_forward || expected_macro_read;
                expected_issue_stall = current_valid && !expected_ready;
                expected_psum_valid = current_valid && parent_authoritative
                    && issue_last && sinks_ready && !expected_deadline;
                expected_read_response = oracle_read_pending;
                expected_dual_enqueue = expected_read_response
                    && expected_forward;

                expected_residual_payload = '0;
                if (source_valid)
                    for (integer lane = 0; lane < 96; lane = lane + 1)
                        expected_residual_payload[lane*12 +: 12] =
                            source_value12(source_index, lane);
                expected_row_data = current_valid
                    ? oracle_pack_row12(slot, current) : '0;

                oracle_expect_bit("issue_request_valid", issue_request_valid,
                    current_valid);
                oracle_expect_bit("issue_data_valid", issue_data_valid,
                    current_valid);
                oracle_expect_bit("issue_data_ready", issue_data_ready,
                    expected_ready);
                oracle_expect_bit("scratch_read", debug_scratch_read_event,
                    expected_macro_read);
                oracle_expect_bit("scratch_write", debug_scratch_write_event,
                    expected_live_write);
                oracle_expect_bit("forward", debug_forward_event,
                    expected_forward);
                oracle_expect_bit("read_response", debug_read_response_event,
                    expected_read_response);
                oracle_expect_bit("dual_enqueue", debug_dual_enqueue_event,
                    expected_dual_enqueue);
                oracle_expect_bit("dead_elision",
                    debug_dead_write_elision_event, expected_dead_elision);
                oracle_expect_bit("deadline_hold", debug_deadline_hold_event,
                    expected_deadline);
                oracle_expect_bit("stalled_raw", debug_stalled_raw_event,
                    expected_stalled_raw);
                oracle_expect_bit("overflow_block", debug_overflow_block_event,
                    1'b0);
                oracle_expect_bit("psum_valid", psum_write_valid,
                    expected_psum_valid);
                oracle_expect_bit("row_complete_valid", row_complete_valid,
                    expected_psum_valid);

                if (current_valid) begin
                    if (issue_request_epoch !== oracle_exec_epoch
                            || issue_request_row_id !== current[5:0]
                            || issue_request_first !== !oracle_row_inflight
                            || issue_request_last !== issue_last
                            || issue_request_source_valid !== source_valid
                            || (source_valid
                                && issue_request_source_index
                                    !== source_index[3:0])
                            || issue_request_parent_valid !== parent_valid
                            || (parent_valid
                                && issue_request_parent_id !==
                                    reference_parent_by_slot[slot][current][5:0])
                            || issue_residual_data !== expected_residual_payload
                            || issue_psum_prior !== 1824'b0) begin
                        error_count = error_count + 1;
                        $error("cleanroom request/payload mismatch epoch=%0d row=%0d",
                            oracle_exec_epoch, current);
                    end
                end
                if (expected_psum_valid) begin
                    if (psum_write_address !== current[5:0]
                            || row_complete_id !== current[5:0]) begin
                        error_count = error_count + 1;
                        $error("cleanroom architectural address mismatch row=%0d",
                            current);
                    end
                    for (integer lane = 0; lane < 96; lane = lane + 1) begin
                        got_value = $signed(psum_write_data[lane*19 +: 19]);
                        if (got_value
                                != expected_row_by_slot[slot][current][lane]) begin
                            error_count = error_count + 1;
                            $error("cleanroom psum mismatch row=%0d lane=%0d got=%0d exp=%0d",
                                current, lane, got_value,
                                expected_row_by_slot[slot][current][lane]);
                        end
                    end
                end

                // All cover points below consume only cleanroom expected events.
                if (expected_dead_elision && expected_macro_read)
                    cov_dead_plus_read = cov_dead_plus_read + 1;
                if (deadline_last_cycle_q && expected_live_write
                        && expected_read_response)
                    cov_deadline_read_write = cov_deadline_read_write + 1;
                if (expected_forward)
                    cov_same_address_forward = cov_same_address_forward + 1;
                if (expected_dual_enqueue)
                    cov_pending_plus_forward = cov_pending_plus_forward + 1;
                if (reserved == 2 && expected_accept && issue_last
                        && parent_valid && !expected_macro_read
                        && !expected_forward)
                    cov_full_no_credit = cov_full_no_credit + 1;
                if (prep_valid && prep_ready)
                    cov_pingpong_overlap = cov_pingpong_overlap + 1;

                // Causal stalled-RAW recovery: no sticky historical credit.
                if (oracle_raw_pending) begin
                    oracle_raw_age = oracle_raw_age + 1;
                    if (expected_forward) begin
                        if (oracle_raw_epoch != oracle_exec_epoch
                                || oracle_raw_consumer != look_consumer
                                || oracle_raw_parent != look_parent
                                || oracle_raw_age < 1 || oracle_raw_age > 8)
                            $fatal(1, "unrelated/cross-task RAW forward credit epoch=%0d consumer=%0d parent=%0d age=%0d",
                                oracle_exec_epoch, look_consumer, look_parent,
                                oracle_raw_age);
                        cov_stalled_raw_recovery =
                            cov_stalled_raw_recovery + 1;
                        oracle_raw_pending = 1'b0;
                    end else if (oracle_raw_age >= 8) begin
                        oracle_raw_pending = 1'b0;
                        $fatal(1, "stalled RAW timeout epoch=%0d consumer=%0d parent=%0d",
                            oracle_raw_epoch, oracle_raw_consumer,
                            oracle_raw_parent);
                    end
                end
                if (expected_stalled_raw) begin
                    saw_stalled_raw_q = 1'b1;
                    if (!oracle_raw_pending) begin
                        oracle_raw_pending = 1'b1;
                        oracle_raw_epoch = oracle_exec_epoch;
                        oracle_raw_consumer = look_consumer;
                        oracle_raw_parent = look_parent;
                        oracle_raw_age = 0;
                    end else if (oracle_raw_consumer != look_consumer
                            || oracle_raw_parent != look_parent)
                        $fatal(1, "second unrelated RAW stall before recovery");
                end

                if (expected_accept && issue_last) begin
                    completion_dead = !reference_live_bitmap[slot][current];
                    if (completion_dead) begin
                        dead_run_q = dead_run_q + 1;
                        if (dead_run_q >= 3)
                            saw_three_dead_q = 1'b1;
                    end else
                        dead_run_q = 0;
                    if (completion_seen_q && previous_dead_q != completion_dead)
                        saw_alternating_q = 1'b1;
                    previous_dead_q = completion_dead;
                    completion_seen_q = 1'b1;
                    if (completion_last_cycle_q)
                        saw_back_to_back_completion_q = 1'b1;
                    if (current == 0)
                        saw_row_zero_q = 1'b1;
                    if (current == 63)
                        saw_row_sixty_three_q = 1'b1;
                    if (reference_pop_by_slot[slot][current]
                                < last_commit_pop[slot]
                            || (reference_pop_by_slot[slot][current]
                                    == last_commit_pop[slot]
                                && current <= last_commit_id[slot])) begin
                        error_count = error_count + 1;
                        $error("cleanroom stable order mismatch row=%0d", current);
                    end
                    last_commit_pop[slot] =
                        reference_pop_by_slot[slot][current];
                    last_commit_id[slot] = current;
                    commit_count = commit_count + 1;
                end
                if (parent_valid && synthetic_parent)
                    saw_exact_parent_q = 1'b1;
                if (parent_valid && source_valid)
                    saw_partial_parent_q = 1'b1;
                if (parent_valid && !oracle_row_inflight && !issue_last)
                    saw_multibeat_parent_q = 1'b1;
                if (saw_three_dead_q && saw_alternating_q)
                    cov_liveness_sequences = 1;
                if (saw_exact_parent_q && saw_partial_parent_q
                        && saw_multibeat_parent_q
                        && saw_back_to_back_completion_q)
                    cov_parent_modes = 1;
                if (saw_row_zero_q && saw_row_sixty_three_q)
                    cov_endpoint_rows = 1;
                if (expected_live_write)
                    for (integer slice = 0; slice < 9; slice = slice + 1)
                        if (|expected_row_data[slice*128 +: 128])
                            slices_seen_q[slice] = 1'b1;
                if (&slices_seen_q)
                    cov_all_slices = 1;

                // Consecutive foundry-read identity/data strength point.
                if (expected_macro_read && oracle_prev_macro_read
                        && look_parent != oracle_prev_macro_read_address
                        && oracle_scratch_data[look_parent]
                            != oracle_prev_macro_read_data)
                    cov_consecutive_distinct_reads =
                        cov_consecutive_distinct_reads + 1;
                if (expected_read_response)
                    cov_response_identity_checks =
                        cov_response_identity_checks + 1;

                // Frozen pop -> prior response -> same-cycle forward queue order.
                next_slot0_valid = oracle_slot0_valid;
                next_slot0_parent = oracle_slot0_parent;
                next_slot0_consumer = oracle_slot0_consumer;
                next_slot0_data = oracle_slot0_data;
                next_slot1_valid = oracle_slot1_valid;
                next_slot1_parent = oracle_slot1_parent;
                next_slot1_consumer = oracle_slot1_consumer;
                next_slot1_data = oracle_slot1_data;
                if (expected_accept && issue_last && parent_valid) begin
                    next_slot0_valid = oracle_slot1_valid;
                    next_slot0_parent = oracle_slot1_parent;
                    next_slot0_consumer = oracle_slot1_consumer;
                    next_slot0_data = oracle_slot1_data;
                    next_slot1_valid = 1'b0;
                    next_slot1_parent = 0;
                    next_slot1_consumer = 0;
                    next_slot1_data = '0;
                end
                if (expected_read_response) begin
                    if (!next_slot0_valid) begin
                        next_slot0_valid = 1'b1;
                        next_slot0_parent = oracle_read_pending_parent;
                        next_slot0_consumer = oracle_read_pending_consumer;
                        next_slot0_data = oracle_read_pending_data;
                    end else if (!next_slot1_valid) begin
                        next_slot1_valid = 1'b1;
                        next_slot1_parent = oracle_read_pending_parent;
                        next_slot1_consumer = oracle_read_pending_consumer;
                        next_slot1_data = oracle_read_pending_data;
                    end else
                        $fatal(1, "cleanroom response queue overflow");
                end
                if (expected_forward) begin
                    if (!next_slot0_valid) begin
                        next_slot0_valid = 1'b1;
                        next_slot0_parent = look_parent;
                        next_slot0_consumer = look_consumer;
                        next_slot0_data = expected_row_data;
                    end else if (!next_slot1_valid) begin
                        next_slot1_valid = 1'b1;
                        next_slot1_parent = look_parent;
                        next_slot1_consumer = look_consumer;
                        next_slot1_data = expected_row_data;
                    end else
                        $fatal(1, "cleanroom forward queue overflow");
                end
                oracle_slot0_valid = next_slot0_valid;
                oracle_slot0_parent = next_slot0_parent;
                oracle_slot0_consumer = next_slot0_consumer;
                oracle_slot0_data = next_slot0_data;
                oracle_slot1_valid = next_slot1_valid;
                oracle_slot1_parent = next_slot1_parent;
                oracle_slot1_consumer = next_slot1_consumer;
                oracle_slot1_data = next_slot1_data;
                oracle_read_pending = expected_macro_read;
                if (expected_macro_read) begin
                    oracle_read_pending_parent = look_parent;
                    oracle_read_pending_consumer = look_consumer;
                    oracle_read_pending_data = oracle_scratch_data[look_parent];
                end

                if (expected_prefetch) begin
                    oracle_prefetched_bitmap[look_consumer] = 1'b1;
                    if (expected_macro_read)
                        oracle_macro_reads[slot] =
                            oracle_macro_reads[slot] + 1;
                    if (expected_forward)
                        oracle_forwards[slot] = oracle_forwards[slot] + 1;
                end
                if (expected_live_write) begin
                    oracle_written_bitmap[current] = 1'b1;
                    oracle_scratch_data[current] = expected_row_data;
                    oracle_live_writes[slot] = oracle_live_writes[slot] + 1;
                end
                if (expected_dead_elision)
                    oracle_dead_elisions[slot] =
                        oracle_dead_elisions[slot] + 1;
                if (expected_deadline)
                    oracle_deadline_holds[slot] =
                        oracle_deadline_holds[slot] + 1;
                if (expected_issue_stall)
                    oracle_issue_stalls[slot] = oracle_issue_stalls[slot] + 1;
                if (expected_accept) begin
                    oracle_issue_accepts[slot] =
                        oracle_issue_accepts[slot] + 1;
                    if (issue_last) begin
                        oracle_completed_bitmap[current] = 1'b1;
                        oracle_row_inflight = 1'b0;
                        oracle_residual_remaining = '0;
                        oracle_psum_commits[slot] =
                            oracle_psum_commits[slot] + 1;
                        oracle_row_completions[slot] =
                            oracle_row_completions[slot] + 1;
                    end else begin
                        oracle_row_inflight = 1'b1;
                        oracle_current_row = current;
                        oracle_residual_remaining = remaining_after;
                    end
                end
                deadline_last_cycle_q = expected_deadline;
                completion_last_cycle_q = expected_accept && issue_last;
                oracle_prev_macro_read = expected_macro_read;
                if (expected_macro_read) begin
                    oracle_prev_macro_read_address = look_parent;
                    oracle_prev_macro_read_data =
                        oracle_scratch_data[look_parent];
                end

                if (!current_valid && !oracle_slot0_valid
                        && !oracle_slot1_valid && !oracle_read_pending) begin
                    if (oracle_raw_pending)
                        $fatal(1, "RAW recovery escaped task epoch=%0d consumer=%0d parent=%0d age=%0d",
                            oracle_raw_epoch, oracle_raw_consumer,
                            oracle_raw_parent, oracle_raw_age);
                    oracle_exec_active = 1'b0;
                    oracle_done_expected = 1'b1;
                    oracle_done_epoch = oracle_exec_epoch;
                    just_deactivated = 1'b1;
                end
            end else begin
                oracle_expect_bit("idle_issue_request", issue_request_valid,
                    1'b0);
                oracle_expect_bit("idle_issue_ready", issue_data_ready, 1'b0);
                oracle_expect_bit("idle_scratch_read",
                    debug_scratch_read_event, 1'b0);
                oracle_expect_bit("idle_scratch_write",
                    debug_scratch_write_event, 1'b0);
                oracle_expect_bit("idle_forward", debug_forward_event, 1'b0);
                oracle_expect_bit("idle_deadline", debug_deadline_hold_event,
                    1'b0);
            end

            // Fixed preprocessing schedule: 64 matcher cycles plus the launch
            // edge after the accepted 64th row.  No DUT busy/ready/debug signal
            // is used to decide when the cleanroom execution model starts.
            for (integer scan_slot = 0; scan_slot < 8;
                    scan_slot = scan_slot + 1) begin
                if (oracle_match_countdown[scan_slot] > 0) begin
                    oracle_match_countdown[scan_slot] =
                        oracle_match_countdown[scan_slot] - 1;
                    if (oracle_match_countdown[scan_slot] == 0)
                        oracle_task_ready[scan_slot] = 1'b1;
                end
            end
            // prep_ready is consumed only as the public accepted-prep
            // handshake.  It never feeds a predicted execution microevent.
            if (prep_valid && prep_ready && prep_task_last) begin
                slot = prep_epoch[2:0];
                if (oracle_epoch_by_slot[slot] != prep_epoch)
                    $fatal(1, "prep epoch has no cleanroom reference");
                oracle_match_countdown[slot] = 65;
                oracle_load_sequence[slot] = oracle_next_load_sequence;
                oracle_next_load_sequence = oracle_next_load_sequence + 1;
            end
            if (!oracle_exec_active && !just_deactivated) begin
                chosen_slot = -1;
                chosen_sequence = 32'h7fff_ffff;
                for (integer scan_slot = 0; scan_slot < 8;
                        scan_slot = scan_slot + 1) begin
                    if (oracle_task_ready[scan_slot]
                            && oracle_load_sequence[scan_slot]
                                < chosen_sequence) begin
                        chosen_slot = scan_slot;
                        chosen_sequence = oracle_load_sequence[scan_slot];
                    end
                end
                if (chosen_slot >= 0)
                    oracle_activate(chosen_slot);
            end
        end
    endtask

    always @(posedge clk_core) begin
        if (!reset_n)
            oracle_reset_state();
        else if (normal_score_enable)
            oracle_cycle();
        else begin
            // Reset/task-abort boundary for the bounded recovery token.
            oracle_raw_pending = 1'b0;
            oracle_prev_macro_read = 1'b0;
        end
    end

    // Observation-only post-edge checker.  The expected queue and pending
    // response above are already committed before any DUT internals are read.
    // Thus these observations cannot generate or repair the oracle prediction;
    // they only prove exact foundry-response address/data identity.
    always @(negedge clk_core) begin
        if (reset_n && normal_score_enable && oracle_exec_active) begin
            if (dut.slot0_valid_q !== oracle_slot0_valid
                    || dut.slot1_valid_q !== oracle_slot1_valid
                    || dut.read_pending_q !== oracle_read_pending)
                $fatal(1, "observed queue validity differs from cleanroom model");
            if (oracle_slot0_valid
                    && (dut.slot0_parent_id_q !== oracle_slot0_parent[5:0]
                        || dut.slot0_consumer_id_q
                            !== oracle_slot0_consumer[5:0]
                        || dut.slot0_data_q !== oracle_slot0_data))
                $fatal(1, "slot0 foundry response identity/data mismatch");
            if (oracle_slot1_valid
                    && (dut.slot1_parent_id_q !== oracle_slot1_parent[5:0]
                        || dut.slot1_consumer_id_q
                            !== oracle_slot1_consumer[5:0]
                        || dut.slot1_data_q !== oracle_slot1_data))
                $fatal(1, "slot1 foundry response identity/data mismatch");
            if (oracle_read_pending
                    && (dut.read_pending_parent_q
                            !== oracle_read_pending_parent[5:0]
                        || dut.read_pending_consumer_q
                            !== oracle_read_pending_consumer[5:0]))
                $fatal(1, "pending foundry response identity mismatch");
        end
    end

    task automatic expect_fault(input string label);
        integer watchdog;
        begin
            watchdog = 0;
            while (!protocol_error && watchdog < 20) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
            end
            if (!protocol_error)
                $fatal(1, "protocol attack not detected: %s", label);
            attack_count = attack_count + 1;
        end
    endtask

    task automatic attack_dirty_reserved;
        begin
            reset_dut();
            @(negedge clk_core);
            prep_valid = 1'b1;
            prep_task_start = 1'b1;
            prep_epoch = 16'd100;
            prep_row_id = 0;
            prep_mask = 1;
            prep_reserved = 4'b0001;
            @(posedge clk_core);
            expect_fault("dirty reserved directory bits");
            attack_dirty_reserved_count = attack_dirty_reserved_count + 1;
        end
    endtask

    task automatic attack_stale_epoch;
        begin
            reset_dut();
            make_directed_masks();
            load_task(16'd200);
            wait (execute_busy);
            @(negedge clk_core);
            prep_valid = 1'b1;
            prep_task_start = 1'b1;
            prep_task_last = 1'b0;
            prep_epoch = 16'd199;
            prep_row_id = 0;
            prep_mask = 1;
            prep_reserved = 0;
            while (!prep_ready) @(negedge clk_core);
            @(posedge clk_core);
            expect_fault("stale epoch");
            attack_stale_epoch_count = attack_stale_epoch_count + 1;
        end
    endtask

    task automatic attack_overflow;
        begin
            reset_dut();
            for (integer row = 0; row < 64; row = row + 1)
                stimulus_masks[row] = 0;
            stimulus_masks[0] = 16'h0001;
            load_task(16'd300);
            attack_overflow_mode = 1'b1;
            wait (debug_overflow_block_event);
            expect_fault("signed19 overflow atomicity");
            attack_overflow_count = attack_overflow_count + 1;
        end
    endtask

    task automatic attack_wrong_parent_and_dead_live;
        begin
            reset_dut();
            make_directed_masks();
            load_task(16'd400);
            wait (execute_busy && issue_request_valid
                && issue_request_row_id == 6'd1);
            force dut.directory_q[0][1][21:16] = 6'd63;
            force dut.parent_live_q[0][63] = 1'b0;
            expect_fault("wrong parent and illegal dead-parent relation");
            attack_wrong_parent_count = attack_wrong_parent_count + 1;
            release dut.directory_q[0][1][21:16];
            release dut.parent_live_q[0][63];
        end
    endtask

    task automatic attack_read_before_write;
        begin
            reset_dut();
            make_directed_masks();
            load_task(16'd500);
            wait (execute_busy);
            @(negedge clk_core);
            force dut.lookahead_parent_w = 6'd60;
            force dut.macro_read_accept_w = 1'b1;
            expect_fault("macro read before written");
            attack_read_before_write_count =
                attack_read_before_write_count + 1;
            release dut.lookahead_parent_w;
            release dut.macro_read_accept_w;
        end
    endtask

    // A stale positive response payload is deliberately exposed while its
    // valid/identity are not authoritative.  The held final must neither
    // overflow-fault nor commit.  A later matching legal response must allow
    // the same final beat to complete without poisoning the task.
    task automatic test_held_final_stale_parent_then_legal;
        logic [63:0] before_psum, before_rows, before_writes, before_dead;
        begin
            reset_dut();
            for (integer row = 0; row < 64; row = row + 1)
                stimulus_masks[row] = 16'b0;
            stimulus_masks[0] = 16'h0001;
            stimulus_masks[1] = 16'h0003;
            build_reference(16'd600);
            load_task(16'd600);
            wait (issue_request_valid && issue_request_row_id == 6'd1
                && issue_request_parent_valid && issue_request_source_valid
                && issue_request_last);
            @(negedge clk_core);
            before_psum = count_psum_commits;
            before_rows = count_row_completions;
            before_writes = count_macro_writes;
            before_dead = count_dead_write_elisions;
            force dut.slot0_valid_q = 1'b0;
            force dut.slot1_valid_q = 1'b0;
            force dut.read_pending_q = 1'b0;
            force dut.slot0_data_q = {96{12'h7ff}};
            repeat (3) begin
                @(negedge clk_core);
                if (protocol_error || debug_overflow_block_event
                        || issue_data_ready || psum_write_valid
                        || row_complete_valid || debug_scratch_write_event
                        || debug_dead_write_elision_event
                        || count_psum_commits != before_psum
                        || count_row_completions != before_rows
                        || count_macro_writes != before_writes
                        || count_dead_write_elisions != before_dead)
                    $fatal(1, "stale nonauthoritative parent poisoned held final");
            end
            force_parent_data_static = '0;
            for (integer lane = 0; lane < 96; lane = lane + 1)
                force_parent_data_static[lane*12 +: 12] = source_value12(0, lane);
            release dut.slot0_valid_q;
            release dut.slot1_valid_q;
            release dut.read_pending_q;
            release dut.slot0_data_q;
            force dut.slot0_valid_q = 1'b1;
            force dut.slot0_parent_id_q = 6'd0;
            force dut.slot0_consumer_id_q = 6'd1;
            force dut.slot0_data_q = force_parent_data_static;
            force psum_write_ready = 1'b1;
            force row_complete_ready = 1'b1;
            @(negedge clk_core);
            if (!issue_data_ready || debug_overflow_block_event
                    || protocol_error)
                $fatal(1, "later authoritative parent did not release held final");
            @(posedge clk_core);
            release dut.slot0_valid_q;
            release dut.slot0_parent_id_q;
            release dut.slot0_consumer_id_q;
            release dut.slot0_data_q;
            release psum_write_ready;
            release row_complete_ready;
            @(negedge clk_core);
            if (protocol_error || count_row_completions != before_rows + 1)
                $fatal(1, "legal parent completion failed after stale hold");
        end
    endtask

    task automatic attack_parent_only_nonzero_atomic;
        logic [63:0] before_issue, before_parent, before_dead;
        logic [63:0] before_reads, before_writes, before_forwards;
        logic [63:0] before_holds, before_stalls, before_psum, before_rows;
        logic [15:0] legal_epoch;
        logic [5:0] legal_row, legal_parent;
        begin
            reset_dut();
            make_directed_masks();
            load_task(16'd700);
            force psum_write_ready = 1'b1;
            force row_complete_ready = 1'b1;
            wait (issue_request_valid && issue_request_parent_valid
                && !issue_request_source_valid && issue_request_last
                && issue_data_valid && issue_data_ready
                && issue_residual_data == 1152'b0);
            @(negedge clk_core);
            if (!psum_write_ready || !row_complete_ready
                    || !issue_data_ready || !psum_write_valid
                    || !row_complete_valid)
                $fatal(1, "legal-zero parent beat was not otherwise accepting");
            legal_epoch = issue_request_epoch;
            legal_row = issue_request_row_id;
            legal_parent = issue_request_parent_id;
            before_issue = count_issue_accepts;
            before_parent = count_parent_edges;
            before_dead = count_dead_write_elisions;
            before_reads = count_macro_reads;
            before_writes = count_macro_writes;
            before_forwards = count_forwards;
            before_holds = count_deadline_holds;
            before_stalls = count_issue_stalls;
            before_psum = count_psum_commits;
            before_rows = count_row_completions;
            attack_parent_only_nonzero_mode = 1'b1;
            #0;
            if (!issue_request_valid || !issue_data_valid
                    || issue_request_epoch != legal_epoch
                    || issue_request_row_id != legal_row
                    || issue_request_parent_id != legal_parent
                    || issue_request_source_valid || !issue_request_last)
                $fatal(1, "malformed attack changed more than current payload");
            if (issue_data_ready || psum_write_valid || row_complete_valid
                    || debug_scratch_read_event || debug_scratch_write_event
                    || debug_forward_event || debug_dual_enqueue_event
                    || debug_dead_write_elision_event
                    || debug_deadline_hold_event
                    || debug_overflow_block_event || debug_stalled_raw_event)
                $fatal(1, "malformed parent-only beat leaked preaccept event");
            @(posedge clk_core);
            @(negedge clk_core);
            if (!protocol_error || count_issue_accepts != before_issue
                    || count_parent_edges != before_parent
                    || count_dead_write_elisions != before_dead
                    || count_macro_reads != before_reads
                    || count_macro_writes != before_writes
                    || count_forwards != before_forwards
                    || count_deadline_holds != before_holds
                    || count_issue_stalls != before_stalls
                    || count_psum_commits != before_psum
                    || count_row_completions != before_rows)
                $fatal(1, "malformed parent-only fault was not atomically sterile");
            expect_fault("parent-only nonzero residual atomic block");
            attack_parent_only_nonzero_count =
                attack_parent_only_nonzero_count + 1;
            release psum_write_ready;
            release row_complete_ready;
        end
    endtask

    initial begin
        error_count = 0;
        commit_count = 0;
        done_count = 0;
        attack_count = 0;
        attack_dirty_reserved_count = 0;
        attack_stale_epoch_count = 0;
        attack_overflow_count = 0;
        attack_wrong_parent_count = 0;
        attack_read_before_write_count = 0;
        attack_parent_only_nonzero_count = 0;
        cov_dead_plus_read = 0;
        cov_deadline_read_write = 0;
        cov_same_address_forward = 0;
        cov_pending_plus_forward = 0;
        cov_full_no_credit = 0;
        cov_liveness_sequences = 0;
        cov_parent_modes = 0;
        cov_stalled_raw_recovery = 0;
        cov_pingpong_overlap = 0;
        cov_endpoint_rows = 0;
        cov_all_slices = 0;
        cov_consecutive_distinct_reads = 0;
        cov_response_identity_checks = 0;
        dead_run_q = 0;
        saw_three_dead_q = 1'b0;
        saw_alternating_q = 1'b0;
        previous_dead_q = 1'b0;
        completion_seen_q = 1'b0;
        saw_exact_parent_q = 1'b0;
        saw_partial_parent_q = 1'b0;
        saw_multibeat_parent_q = 1'b0;
        saw_back_to_back_completion_q = 1'b0;
        completion_last_cycle_q = 1'b0;
        saw_stalled_raw_q = 1'b0;
        saw_row_zero_q = 1'b0;
        saw_row_sixty_three_q = 1'b0;
        slices_seen_q = 9'b0;
        deadline_last_cycle_q = 1'b0;
        reset_n = 1'b0;
        clear_drivers();
        normal_score_enable = 1'b0;

        // Directed task plus ping-pong overlap with the identical task.
        reset_dut();
        make_directed_masks();
        build_reference(16'd1);
        build_reference(16'd2);
        normal_score_enable = 1'b1;
        load_task(16'd1);
        wait (execute_busy);
        fork
            begin
                repeat (12) @(posedge clk_core);
                load_task(16'd2);
            end
            begin
                wait_done(16'd1);
            end
        join
        wait_done(16'd2);

        // P2 strength task: two adjacent distinct-address/data foundry reads.
        make_consecutive_distinct_read_masks();
        build_reference(16'd3);
        load_task(16'd3);
        wait_done(16'd3);

        // Reproducible constrained-random mask populations.
        for (integer test_index = 0; test_index < 4; test_index = test_index + 1) begin
            make_random_masks(32'h5290_0000 + test_index);
            build_reference(10 + test_index);
            load_task(10 + test_index);
            wait_done(10 + test_index);
        end
        normal_score_enable = 1'b0;
        if (protocol_error)
            $fatal(1, "normal suite raised protocol_error");
        if (error_count != 0)
            $fatal(1, "normal scoreboard errors=%0d", error_count);

        if (cov_dead_plus_read < 1 || cov_deadline_read_write < 1
                || cov_same_address_forward < 1
                || cov_pending_plus_forward < 1
                || cov_full_no_credit < 1
                || cov_liveness_sequences < 1
                || cov_parent_modes < 1
                || cov_stalled_raw_recovery < 1
                || cov_pingpong_overlap < 1
                || cov_endpoint_rows < 1 || cov_all_slices < 1)
            $fatal(1, "normal coverage minima missed %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                cov_dead_plus_read, cov_deadline_read_write,
                cov_same_address_forward, cov_pending_plus_forward,
                cov_full_no_credit, cov_liveness_sequences,
                cov_parent_modes, cov_stalled_raw_recovery,
                cov_pingpong_overlap, cov_endpoint_rows, cov_all_slices);
        $display("COVERAGE_M533_M528_DW1RW_R3 dead_plus_read=%0d deadline_read_write=%0d same_address_forward=%0d pending_plus_forward=%0d full_no_credit=%0d liveness_sequences=%0d parent_modes=%0d stalled_raw_recovery=%0d pingpong_overlap=%0d endpoint_rows=%0d all_slices=%0d minima=1 normal_covers=11",
            cov_dead_plus_read, cov_deadline_read_write,
            cov_same_address_forward, cov_pending_plus_forward,
            cov_full_no_credit, cov_liveness_sequences,
            cov_parent_modes, cov_stalled_raw_recovery,
            cov_pingpong_overlap, cov_endpoint_rows, cov_all_slices);
        if (cov_consecutive_distinct_reads < 1
                || cov_response_identity_checks < 2)
            $fatal(1, "P2 foundry response strength missed pairs=%0d checks=%0d",
                cov_consecutive_distinct_reads,
                cov_response_identity_checks);
        $display("P2_STRENGTH_M533_M528_DW1RW_R3 consecutive_distinct_reads=%0d response_identity_checks=%0d minima_pairs=1 minima_responses=2",
            cov_consecutive_distinct_reads,
            cov_response_identity_checks);

        test_held_final_stale_parent_then_legal();

        attack_dirty_reserved();
        attack_stale_epoch();
        attack_overflow();
        attack_wrong_parent_and_dead_live();
        attack_read_before_write();
        attack_parent_only_nonzero_atomic();
        if (attack_count != 6 || attack_dirty_reserved_count != 1
                || attack_stale_epoch_count != 1
                || attack_overflow_count != 1
                || attack_wrong_parent_count != 1
                || attack_read_before_write_count != 1
                || attack_parent_only_nonzero_count != 1)
            $fatal(1, "attack coverage count=%0d", attack_count);

        $display("PASS_M533_M528_DW1RW_R3_DIRECTED_RANDOM_AND_ATTACKS commits=%0d done=%0d attacks=%0d dirty_reserved=%0d stale_epoch=%0d overflow=%0d wrong_parent=%0d read_before_write=%0d parent_only_nonzero=%0d functional_vcs_only=true trace_recurrence=false speedup=false ppa=false energy=false full_network=false headline=false",
            commit_count, done_count, attack_count,
            attack_dirty_reserved_count, attack_stale_epoch_count,
            attack_overflow_count, attack_wrong_parent_count,
            attack_read_before_write_count,
            attack_parent_only_nonzero_count);
        $finish;
    end

    initial begin
        #3000000;
        $fatal(1, "global watchdog expired");
    end
endmodule

`default_nettype wire
