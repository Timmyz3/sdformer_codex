`timescale 1ns/1ps
`default_nettype none

module tb_m528_dead_write_only_1rw_product_capture_r2;
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

    logic [15:0] stimulus_masks [0:63];
    logic [15:0] reference_residual [0:63];
    integer reference_parent [0:63];
    integer reference_pop [0:63];
    integer reference_parent_by_slot [0:7][0:63];
    integer reference_parent_refcount [0:7][0:63];
    integer reference_pop_by_slot [0:7][0:63];
    logic [63:0] reference_live_bitmap [0:7];
    integer reference_active_rows [0:7];
    integer reference_parent_edges [0:7];
    integer expected_row [0:63][0:95];
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
    logic execute_busy_d;

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
                score_remaining[slot][row] = reference_residual[row];
                score_started[slot][row] = 1'b0;
                for (integer lane = 0; lane < 96; lane = lane + 1) begin
                    expected_row[row][lane] = 0;
                    for (integer source = 0; source < 16; source = source + 1)
                        if (stimulus_masks[row][source])
                            expected_row[row][lane] = expected_row[row][lane]
                                + $signed(source_value12(source, lane));
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

    // Cleanroom oracle: matcher/refcounts/live map are calculated only from
    // stimulus masks.  Dynamic microevents are counted from accepted public
    // protocol and separately compared against the DUT architectural counters.
    always @(posedge clk_core) begin
        integer slot, expected_source, got_value;
        logic completion_now, completion_dead;
        if (!reset_n) begin
            execute_busy_d <= 1'b0;
            dead_run_q = 0;
            previous_dead_q = 1'b0;
            completion_seen_q = 1'b0;
            deadline_last_cycle_q = 1'b0;
            completion_last_cycle_q = 1'b0;
            saw_stalled_raw_q = 1'b0;
        end else begin
            execute_busy_d <= execute_busy;
        end

        if (reset_n && normal_score_enable && execute_busy
                && !execute_busy_d) begin
            slot = issue_request_epoch[2:0];
            if (debug_parent_live_bitmap !== reference_live_bitmap[slot]) begin
                error_count = error_count + 1;
                $error("complete live bitmap mismatch epoch=%0d got=%h exp=%h",
                    issue_request_epoch, debug_parent_live_bitmap,
                    reference_live_bitmap[slot]);
            end
        end

        if (reset_n && normal_score_enable && issue_request_valid) begin
            slot = issue_request_epoch[2:0];
            if (issue_request_parent_valid
                    != (reference_parent_by_slot[slot][issue_request_row_id] >= 0)) begin
                error_count = error_count + 1;
                $error("parent-valid mismatch row=%0d", issue_request_row_id);
            end
            if (issue_request_parent_valid
                    && issue_request_parent_id
                        != reference_parent_by_slot[slot][issue_request_row_id][5:0]) begin
                error_count = error_count + 1;
                $error("parent-id mismatch row=%0d", issue_request_row_id);
            end
            expected_source = -1;
            for (integer source = 0; source < 16; source = source + 1)
                if (expected_source < 0
                        && score_remaining[slot][issue_request_row_id][source])
                    expected_source = source;
            if ((expected_source >= 0) != issue_request_source_valid) begin
                error_count = error_count + 1;
                $error("source-valid mismatch row=%0d", issue_request_row_id);
            end
            if (expected_source >= 0
                    && issue_request_source_index != expected_source[3:0]) begin
                error_count = error_count + 1;
                $error("source order mismatch row=%0d", issue_request_row_id);
            end
            if (!score_started[slot][issue_request_row_id]
                    != issue_request_first) begin
                error_count = error_count + 1;
                $error("first mismatch row=%0d", issue_request_row_id);
            end
            if (issue_data_valid && !issue_data_ready)
                oracle_issue_stalls[slot] = oracle_issue_stalls[slot] + 1;
            if (issue_data_valid && issue_data_ready) begin
                oracle_issue_accepts[slot] = oracle_issue_accepts[slot] + 1;
                if (expected_source >= 0)
                    score_remaining[slot][issue_request_row_id][expected_source]
                        <= 1'b0;
                score_started[slot][issue_request_row_id] <= 1'b1;
                if (issue_request_last
                        != ((expected_source < 0)
                            || ((score_remaining[slot][issue_request_row_id]
                                    & ~(16'b1 << expected_source)) == 0))) begin
                    error_count = error_count + 1;
                    $error("last mismatch row=%0d", issue_request_row_id);
                end
                if (issue_request_parent_valid && !issue_request_source_valid)
                    saw_exact_parent_q = 1'b1;
                if (issue_request_parent_valid && issue_request_source_valid)
                    saw_partial_parent_q = 1'b1;
                if (issue_request_parent_valid && issue_request_first
                        && !issue_request_last)
                    saw_multibeat_parent_q = 1'b1;
                if (issue_request_last) begin
                    oracle_psum_commits[slot] = oracle_psum_commits[slot] + 1;
                    oracle_row_completions[slot] =
                        oracle_row_completions[slot] + 1;
                    if (reference_live_bitmap[slot][issue_request_row_id])
                        oracle_live_writes[slot] = oracle_live_writes[slot] + 1;
                    else
                        oracle_dead_elisions[slot] =
                            oracle_dead_elisions[slot] + 1;
                end
            end
        end

        if (reset_n && normal_score_enable && debug_scratch_read_event) begin
            slot = issue_request_epoch[2:0];
            oracle_macro_reads[slot] = oracle_macro_reads[slot] + 1;
        end
        if (reset_n && normal_score_enable && debug_forward_event) begin
            slot = issue_request_epoch[2:0];
            oracle_forwards[slot] = oracle_forwards[slot] + 1;
            cov_same_address_forward = cov_same_address_forward + 1;
        end
        if (reset_n && normal_score_enable && debug_deadline_hold_event) begin
            slot = issue_request_epoch[2:0];
            oracle_deadline_holds[slot] = oracle_deadline_holds[slot] + 1;
        end

        completion_now = reset_n && normal_score_enable && row_complete_valid
            && row_complete_ready && psum_write_ready;
        completion_dead = completion_now
            && !reference_live_bitmap[issue_request_epoch[2:0]][row_complete_id];
        if (reset_n && normal_score_enable) begin
            if (debug_dead_write_elision_event && debug_scratch_read_event)
                cov_dead_plus_read = cov_dead_plus_read + 1;
            if (deadline_last_cycle_q && debug_scratch_write_event
                    && debug_read_response_event)
                cov_deadline_read_write = cov_deadline_read_write + 1;
            if (debug_dual_enqueue_event)
                cov_pending_plus_forward = cov_pending_plus_forward + 1;
            if (parent_reserved_occupancy == 3'd2
                    && issue_data_valid && issue_data_ready
                    && issue_request_last && issue_request_parent_valid
                    && !debug_scratch_read_event && !debug_forward_event)
                cov_full_no_credit = cov_full_no_credit + 1;
            if (prep_valid && prep_ready && execute_busy)
                cov_pingpong_overlap = cov_pingpong_overlap + 1;
            if (debug_stalled_raw_event)
                saw_stalled_raw_q = 1'b1;
            if (saw_stalled_raw_q && debug_forward_event)
                cov_stalled_raw_recovery = cov_stalled_raw_recovery + 1;
            if (completion_now) begin
                if (completion_dead) begin
                    dead_run_q = dead_run_q + 1;
                    if (dead_run_q >= 3)
                        saw_three_dead_q = 1'b1;
                end else begin
                    dead_run_q = 0;
                end
                if (completion_seen_q && previous_dead_q != completion_dead)
                    saw_alternating_q = 1'b1;
                previous_dead_q = completion_dead;
                completion_seen_q = 1'b1;
                if (completion_last_cycle_q)
                    saw_back_to_back_completion_q = 1'b1;
                if (row_complete_id == 6'd0)
                    saw_row_zero_q = 1'b1;
                if (row_complete_id == 6'd63)
                    saw_row_sixty_three_q = 1'b1;
            end
            if (saw_three_dead_q && saw_alternating_q)
                cov_liveness_sequences = 1;
            if (saw_exact_parent_q && saw_partial_parent_q
                    && saw_multibeat_parent_q
                    && saw_back_to_back_completion_q)
                cov_parent_modes = 1;
            if (saw_row_zero_q && saw_row_sixty_three_q)
                cov_endpoint_rows = 1;
            if (debug_scratch_write_event) begin
                for (integer slice = 0; slice < 9; slice = slice + 1)
                    if (|dut.row_final_packed_w[slice*128 +: 128])
                        slices_seen_q[slice] = 1'b1;
            end
            if (&slices_seen_q)
                cov_all_slices = 1;
            deadline_last_cycle_q = debug_deadline_hold_event;
            completion_last_cycle_q = completion_now;
        end

        if (completion_now) begin
            slot = issue_request_epoch[2:0];
            for (integer lane = 0; lane < 96; lane = lane + 1) begin
                got_value = $signed(psum_write_data[lane*19 +: 19]);
                if (got_value != expected_row[psum_write_address][lane]) begin
                    error_count = error_count + 1;
                    $error("psum mismatch row=%0d lane=%0d got=%0d exp=%0d",
                        psum_write_address, lane, got_value,
                        expected_row[psum_write_address][lane]);
                end
            end
            if (reference_pop_by_slot[slot][psum_write_address]
                        < last_commit_pop[slot]
                    || (reference_pop_by_slot[slot][psum_write_address]
                            == last_commit_pop[slot]
                        && psum_write_address <= last_commit_id[slot])) begin
                error_count = error_count + 1;
                $error("stable order mismatch row=%0d", psum_write_address);
            end
            last_commit_pop[slot] =
                reference_pop_by_slot[slot][psum_write_address];
            last_commit_id[slot] = psum_write_address;
            commit_count = commit_count + 1;
        end

        if (reset_n && normal_score_enable && task_done_valid) begin
            slot = task_done_epoch[2:0];
            done_count = done_count + 1;
            if (debug_written_bitmap !== reference_live_bitmap[slot]
                    || count_parent_edges != reference_parent_edges[slot]
                    || count_issue_accepts != oracle_issue_accepts[slot]
                    || count_macro_writes != oracle_live_writes[slot]
                    || count_dead_write_elisions != oracle_dead_elisions[slot]
                    || count_macro_reads != oracle_macro_reads[slot]
                    || count_forwards != oracle_forwards[slot]
                    || count_deadline_holds != oracle_deadline_holds[slot]
                    || count_issue_stalls != oracle_issue_stalls[slot]
                    || count_psum_commits != oracle_psum_commits[slot]
                    || count_row_completions != oracle_row_completions[slot]
                    || oracle_live_writes[slot]
                        != pc64(reference_live_bitmap[slot])
                    || oracle_dead_elisions[slot]
                        != reference_active_rows[slot]
                            - pc64(reference_live_bitmap[slot])
                    || count_parent_edges != count_macro_reads + count_forwards
                    || count_row_completions != count_psum_commits
                    || count_row_completions
                        != count_macro_writes + count_dead_write_elisions) begin
                error_count = error_count + 1;
                $error("cleanroom task oracle mismatch epoch=%0d live=%h exp_live=%h parent=%0d/%0d writes=%0d/%0d dead=%0d/%0d reads=%0d/%0d forwards=%0d/%0d holds=%0d/%0d stalls=%0d/%0d",
                    task_done_epoch, debug_written_bitmap,
                    reference_live_bitmap[slot], count_parent_edges,
                    reference_parent_edges[slot], count_macro_writes,
                    oracle_live_writes[slot], count_dead_write_elisions,
                    oracle_dead_elisions[slot], count_macro_reads,
                    oracle_macro_reads[slot], count_forwards,
                    oracle_forwards[slot], count_deadline_holds,
                    oracle_deadline_holds[slot], count_issue_stalls,
                    oracle_issue_stalls[slot]);
            end
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
        logic [1151:0] legal_parent_data;
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
            legal_parent_data = '0;
            for (integer lane = 0; lane < 96; lane = lane + 1)
                legal_parent_data[lane*12 +: 12] = source_value12(0, lane);
            release dut.slot0_valid_q;
            release dut.slot1_valid_q;
            release dut.read_pending_q;
            release dut.slot0_data_q;
            force dut.slot0_valid_q = 1'b1;
            force dut.slot0_parent_id_q = 6'd0;
            force dut.slot0_consumer_id_q = 6'd1;
            force dut.slot0_data_q = legal_parent_data;
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
        begin
            reset_dut();
            make_directed_masks();
            load_task(16'd700);
            wait (issue_request_valid && issue_request_parent_valid
                && !issue_request_source_valid && issue_request_last);
            @(negedge clk_core);
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
            if (issue_data_ready || psum_write_valid || row_complete_valid
                    || debug_scratch_read_event || debug_scratch_write_event
                    || debug_forward_event || debug_dead_write_elision_event
                    || debug_deadline_hold_event)
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
        execute_busy_d = 1'b0;
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
        $display("COVERAGE_M530_M528_DW1RW_R2 dead_plus_read=%0d deadline_read_write=%0d same_address_forward=%0d pending_plus_forward=%0d full_no_credit=%0d liveness_sequences=%0d parent_modes=%0d stalled_raw_recovery=%0d pingpong_overlap=%0d endpoint_rows=%0d all_slices=%0d minima=1 normal_covers=11",
            cov_dead_plus_read, cov_deadline_read_write,
            cov_same_address_forward, cov_pending_plus_forward,
            cov_full_no_credit, cov_liveness_sequences,
            cov_parent_modes, cov_stalled_raw_recovery,
            cov_pingpong_overlap, cov_endpoint_rows, cov_all_slices);

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

        $display("PASS_M530_M528_DW1RW_R2_DIRECTED_RANDOM_AND_ATTACKS commits=%0d done=%0d attacks=%0d dirty_reserved=%0d stale_epoch=%0d overflow=%0d wrong_parent=%0d read_before_write=%0d parent_only_nonzero=%0d functional_vcs_only=true trace_recurrence=false speedup=false ppa=false energy=false full_network=false headline=false",
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
