`timescale 1ns/1ps
`default_nettype none

// Complete source-only M528 island.  This is the single admitted structure:
// exact dynamic subset capture, ping-pong directory ownership, stable
// popcount/row ordering, one earliest-parent lookahead, deadline-aware 1RW
// arbitration, two reserved response entries, signed reconstruction, dead-
// write-only storage, architectural completion and conservation counters.
//
// It is intentionally not a full-network scheduler.  A source producer uses
// issue_request_* to supply the requested signed12 residual vector and the
// resident signed19 psum prior.  The island owns all row/parent scheduling.
module m528_dead_write_only_1rw_product_capture_island_r2 (
    input  logic          clk_core,
    input  logic          reset_n,

    // One 64-row mask task is loaded into the non-executing ping-pong bank.
    // Every row ID 0..63 must occur exactly once.  task_start is asserted only
    // on the first accepted row; task_last only on the 64th unique row.
    input  logic          prep_valid,
    output logic          prep_ready,
    input  logic          prep_task_start,
    input  logic          prep_task_last,
    input  logic [15:0]   prep_epoch,
    input  logic [5:0]    prep_row_id,
    input  logic [15:0]   prep_mask,
    input  logic [3:0]    prep_reserved,

    // Source payload for the single internally selected issue request.
    output logic          issue_request_valid,
    output logic [15:0]   issue_request_epoch,
    output logic [5:0]    issue_request_row_id,
    output logic          issue_request_first,
    output logic          issue_request_last,
    output logic          issue_request_source_valid,
    output logic [3:0]    issue_request_source_index,
    output logic          issue_request_parent_valid,
    output logic [5:0]    issue_request_parent_id,
    input  logic          issue_data_valid,
    output logic          issue_data_ready,
    input  logic [1151:0] issue_residual_data,
    input  logic [1823:0] issue_psum_prior,

    // Architectural boundaries.  A final beat is atomic: psum and row
    // completion either both handshake or neither does.
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
    localparam logic [2:0] BANK_FREE  = 3'd0;
    localparam logic [2:0] BANK_LOAD  = 3'd1;
    localparam logic [2:0] BANK_MATCH = 3'd2;
    localparam logic [2:0] BANK_READY = 3'd3;
    localparam logic [2:0] BANK_EXEC  = 3'd4;

    function automatic logic [4:0] popcount16(input logic [15:0] value);
        logic [4:0] result;
        begin
            result = '0;
            for (int bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                result = result + value[bit_index];
            return result;
        end
    endfunction

    function automatic logic [63:0] row_bit(input logic [5:0] row_id);
        logic [63:0] result;
        begin
            result = 64'b1 << row_id;
            return result;
        end
    endfunction

    // Directory layout is frozen: residual[15:0], parent[21:16],
    // parent_valid[22], original_popcount[27:23], reserved[31:28]=0.
    logic [15:0] mask_q [0:1][0:63];
    logic [31:0] directory_q [0:1][0:63];
    logic [63:0] parent_live_q [0:1];
    logic [15:0] bank_epoch_q [0:1];
    logic [2:0] bank_state_q [0:1];

    logic prep_active_q, prep_bank_q;
    logic [15:0] prep_epoch_q;
    logic [63:0] prep_loaded_q;
    logic match_active_q, match_bank_q;
    logic [5:0] match_row_q;
    logic epoch_seen_q;
    logic [15:0] newest_epoch_q;

    logic free_bank_valid_w, free_bank_w;
    logic ready_bank_valid_w, ready_bank_w;
    logic prep_accept_w, prep_store_w, prep_semantic_ok_w;
    logic [63:0] prep_loaded_after_w;
    logic prep_last_expected_w;

    logic [15:0] match_current_mask_w;
    logic [4:0] match_current_pop_w;
    logic match_best_valid_w;
    logic [4:0] match_best_pop_w;
    logic [5:0] match_best_id_w;
    logic [15:0] match_best_mask_w;
    logic [15:0] match_residual_w;
    logic [31:0] match_directory_w;

    // Execution-bank metadata and one in-flight arithmetic row.
    logic exec_active_q, exec_bank_q;
    logic [15:0] exec_epoch_q;
    logic [63:0] completed_bitmap_q;
    logic [63:0] prefetched_edge_bitmap_q;
    logic [63:0] written_bitmap_q;
    logic row_inflight_q;
    logic [5:0] current_row_q;
    logic [15:0] residual_remaining_q;
    logic signed [12:0] row_acc_q [0:95];
    logic signed [19:0] psum_acc_q [0:95];

    logic selected_row_valid_w;
    logic [5:0] selected_row_w;
    logic [4:0] selected_pop_w;
    logic current_valid_w;
    logic [5:0] current_row_w;
    logic [31:0] current_directory_w;
    logic [15:0] current_original_mask_w;
    logic [15:0] current_residual_mask_w;
    logic current_parent_valid_w;
    logic [5:0] current_parent_id_w;
    logic [4:0] current_original_pop_w;
    logic current_live_w;

    logic next_row_valid_w;
    logic [5:0] next_row_w;
    logic [4:0] next_row_pop_w;
    logic lookahead_valid_w;
    logic [5:0] lookahead_consumer_w;
    logic [5:0] lookahead_parent_w;
    logic [4:0] lookahead_pop_w;
    logic lookahead_immediate_next_w;
    logic lookahead_written_w;

    logic [15:0] issue_work_mask_w, issue_remaining_after_w;
    logic issue_source_found_w;
    logic [3:0] issue_source_index_w;
    logic issue_synthetic_parent_w;
    logic issue_last_w;
    logic parent_ready_w;
    logic [1151:0] parent_source_data_w;

    logic signed [13:0] row_partial_w [0:95];
    logic signed [13:0] row_final_w [0:95];
    logic signed [20:0] psum_final_w [0:95];
    logic [1151:0] row_final_packed_w;
    logic row_overflow_w, psum_overflow_w;
    logic residual_int8_format_ok_w;
    logic matching_parent_authoritative_w;
    logic preaccept_protocol_ok_w;
    logic arithmetic_authoritative_w;
    logic final_sinks_ready_w, base_issue_ready_w;
    logic issue_accept_w, consume_parent_w;

    // Ordered response queue plus one synchronous response reservation.
    logic slot0_valid_q, slot1_valid_q;
    logic [5:0] slot0_parent_id_q, slot1_parent_id_q;
    logic [5:0] slot0_consumer_id_q, slot1_consumer_id_q;
    logic [1151:0] slot0_data_q, slot1_data_q;
    logic slot0_valid_n, slot1_valid_n;
    logic [5:0] slot0_parent_id_n, slot1_parent_id_n;
    logic [5:0] slot0_consumer_id_n, slot1_consumer_id_n;
    logic [1151:0] slot0_data_n, slot1_data_n;
    logic read_pending_q;
    logic [5:0] read_pending_parent_q;
    logic [5:0] read_pending_consumer_q;
    logic [2:0] queue_count_w, reserved_count_w;
    logic queue_overflow_w;

    logic deadline_hold_w, stalled_same_address_w;
    logic live_write_accept_w, dead_elision_accept_w;
    logic forward_accept_w, macro_read_accept_w, prefetch_accept_w;
    logic [5:0] scratch_address_w;
    logic [1151:0] scratch_read_data_w;
    logic [1151:0] issue_residual_effective_w;
    logic scratch_enable_w, scratch_write_enable_w;

    logic task_drained_w;
    logic fault_condition_w;
    logic fault_q;

    // The free/ready choices are deterministic; bank zero wins ties.
    always_comb begin
        free_bank_valid_w = 1'b0;
        free_bank_w = 1'b0;
        ready_bank_valid_w = 1'b0;
        ready_bank_w = 1'b0;
        for (int bank = 0; bank < 2; bank = bank + 1) begin
            if (!free_bank_valid_w && bank_state_q[bank] == BANK_FREE) begin
                free_bank_valid_w = 1'b1;
                free_bank_w = bank[0];
            end
            if (!ready_bank_valid_w && bank_state_q[bank] == BANK_READY) begin
                ready_bank_valid_w = 1'b1;
                ready_bank_w = bank[0];
            end
        end
    end

    always_comb begin
        prep_ready = !fault_q && !match_active_q
            && (prep_active_q || free_bank_valid_w);
        prep_accept_w = prep_valid && prep_ready;
        prep_loaded_after_w = prep_loaded_q | row_bit(prep_row_id);
        prep_last_expected_w = &prep_loaded_after_w;
        if (!prep_active_q) begin
            prep_semantic_ok_w = prep_task_start && free_bank_valid_w
                && prep_reserved == 4'b0
                && (!epoch_seen_q || prep_epoch > newest_epoch_q)
                && !prep_task_last;
        end else begin
            prep_semantic_ok_w = !prep_task_start
                && prep_reserved == 4'b0
                && prep_epoch == prep_epoch_q
                && !(prep_loaded_q & row_bit(prep_row_id))
                && prep_task_last == prep_last_expected_w;
        end
        prep_store_w = prep_accept_w && prep_semantic_ok_w;
    end

    // One 64-way exact-subset compare per current row.  Strict greater-than
    // replacement while scanning candidate IDs upward preserves the lowest-ID
    // candidate on equal population.  Equal patterns at current/later IDs are
    // excluded exactly as in M504; earlier equal patterns remain eligible.
    always_comb begin
        match_current_mask_w = mask_q[match_bank_q][match_row_q];
        match_current_pop_w = popcount16(match_current_mask_w);
        match_best_valid_w = 1'b0;
        match_best_pop_w = '0;
        match_best_id_w = '0;
        match_best_mask_w = '0;
        if (match_current_pop_w >= 2) begin
            for (int candidate = 0; candidate < 64; candidate = candidate + 1) begin
                if (((mask_q[match_bank_q][candidate] & match_current_mask_w)
                            == mask_q[match_bank_q][candidate])
                        && popcount16(mask_q[match_bank_q][candidate]) >= 1
                        && !((mask_q[match_bank_q][candidate]
                                == match_current_mask_w)
                            && candidate >= match_row_q)
                        && (!match_best_valid_w
                            || popcount16(mask_q[match_bank_q][candidate])
                                > match_best_pop_w)) begin
                    match_best_valid_w = 1'b1;
                    match_best_pop_w =
                        popcount16(mask_q[match_bank_q][candidate]);
                    match_best_id_w = candidate[5:0];
                    match_best_mask_w = mask_q[match_bank_q][candidate];
                end
            end
        end
        match_residual_w = match_best_valid_w
            ? (match_current_mask_w ^ match_best_mask_w)
            : match_current_mask_w;
        match_directory_w = {
            4'b0,
            match_current_pop_w,
            match_best_valid_w,
            match_best_id_w,
            match_residual_w
        };
    end

    // Stable lexicographic scanner for the current and immediate-next rows.
    always_comb begin
        selected_row_valid_w = 1'b0;
        selected_row_w = '0;
        selected_pop_w = 5'd31;
        if (exec_active_q) begin
            for (int row = 0; row < 64; row = row + 1) begin
                if (mask_q[exec_bank_q][row] != 16'b0
                        && !completed_bitmap_q[row]
                        && (!selected_row_valid_w
                            || directory_q[exec_bank_q][row][27:23]
                                < selected_pop_w)) begin
                    selected_row_valid_w = 1'b1;
                    selected_row_w = row[5:0];
                    selected_pop_w = directory_q[exec_bank_q][row][27:23];
                end
            end
        end
        current_valid_w = row_inflight_q || selected_row_valid_w;
        current_row_w = row_inflight_q ? current_row_q : selected_row_w;
        current_directory_w = current_valid_w
            ? directory_q[exec_bank_q][current_row_w] : 32'b0;
        current_original_mask_w = current_valid_w
            ? mask_q[exec_bank_q][current_row_w] : 16'b0;
        current_residual_mask_w = current_directory_w[15:0];
        current_parent_id_w = current_directory_w[21:16];
        current_parent_valid_w = current_directory_w[22];
        current_original_pop_w = current_directory_w[27:23];
        current_live_w = current_valid_w
            && parent_live_q[exec_bank_q][current_row_w];

        next_row_valid_w = 1'b0;
        next_row_w = '0;
        next_row_pop_w = 5'd31;
        if (current_valid_w) begin
            for (int row = 0; row < 64; row = row + 1) begin
                if (mask_q[exec_bank_q][row] != 16'b0
                        && !completed_bitmap_q[row]
                        && row[5:0] != current_row_w
                        && (!next_row_valid_w
                            || directory_q[exec_bank_q][row][27:23]
                                < next_row_pop_w)) begin
                    next_row_valid_w = 1'b1;
                    next_row_w = row[5:0];
                    next_row_pop_w = directory_q[exec_bank_q][row][27:23];
                end
            end
        end
    end

    // The only lookahead descriptor is the earliest unaccepted parent edge in
    // the same stable order.  No later edge can bypass it.
    always_comb begin
        lookahead_valid_w = 1'b0;
        lookahead_consumer_w = '0;
        lookahead_parent_w = '0;
        lookahead_pop_w = 5'd31;
        if (exec_active_q) begin
            for (int row = 0; row < 64; row = row + 1) begin
                if (mask_q[exec_bank_q][row] != 16'b0
                        && !completed_bitmap_q[row]
                        && directory_q[exec_bank_q][row][22]
                        && !prefetched_edge_bitmap_q[row]
                        && (!lookahead_valid_w
                            || directory_q[exec_bank_q][row][27:23]
                                < lookahead_pop_w)) begin
                    lookahead_valid_w = 1'b1;
                    lookahead_consumer_w = row[5:0];
                    lookahead_parent_w =
                        directory_q[exec_bank_q][row][21:16];
                    lookahead_pop_w =
                        directory_q[exec_bank_q][row][27:23];
                end
            end
        end
        lookahead_immediate_next_w = lookahead_valid_w && next_row_valid_w
            && lookahead_consumer_w == next_row_w;
        lookahead_written_w = lookahead_valid_w
            && written_bitmap_q[lookahead_parent_w];
    end

    always_comb begin
        issue_work_mask_w = row_inflight_q
            ? residual_remaining_q : current_residual_mask_w;
        issue_source_found_w = 1'b0;
        issue_source_index_w = '0;
        for (int source = 0; source < 16; source = source + 1) begin
            if (!issue_source_found_w && issue_work_mask_w[source]) begin
                issue_source_found_w = 1'b1;
                issue_source_index_w = source[3:0];
            end
        end
        issue_synthetic_parent_w = current_valid_w
            && !issue_source_found_w && current_parent_valid_w;
        issue_remaining_after_w = issue_work_mask_w;
        if (issue_source_found_w)
            issue_remaining_after_w[issue_source_index_w] = 1'b0;
        issue_last_w = current_valid_w
            && (issue_synthetic_parent_w || issue_remaining_after_w == 16'b0);

        issue_request_valid = exec_active_q && current_valid_w && !fault_q;
        issue_request_epoch = exec_epoch_q;
        issue_request_row_id = current_row_w;
        issue_request_first = !row_inflight_q;
        issue_request_last = issue_last_w;
        issue_request_source_valid = issue_source_found_w;
        issue_request_source_index = issue_source_index_w;
        issue_request_parent_valid = current_parent_valid_w;
        issue_request_parent_id = current_parent_id_w;
    end

    always_comb begin
        queue_count_w = {2'b0, slot0_valid_q} + {2'b0, slot1_valid_q};
        reserved_count_w = queue_count_w + {2'b0, read_pending_q};
        matching_parent_authoritative_w = !current_parent_valid_w
            || (slot0_valid_q
                && slot0_parent_id_q == current_parent_id_w
                && slot0_consumer_id_q == current_row_w);
        parent_ready_w = matching_parent_authoritative_w;
        parent_source_data_w = current_parent_valid_w
            && matching_parent_authoritative_w ? slot0_data_q : 1152'b0;
        issue_residual_effective_w = issue_synthetic_parent_w
            ? 1152'b0 : issue_residual_data;

        row_final_packed_w = '0;
        psum_write_data = '0;
        row_overflow_w = 1'b0;
        psum_overflow_w = 1'b0;
        residual_int8_format_ok_w = 1'b1;
        for (int lane = 0; lane < 96; lane = lane + 1) begin
            if (issue_residual_effective_w[lane*12 + 8 +: 4]
                    != {4{issue_residual_effective_w[lane*12 + 7]}})
                residual_int8_format_ok_w = 1'b0;
            row_partial_w[lane] =
                (issue_request_first ? 14'sd0 : $signed(row_acc_q[lane]))
                + $signed(issue_residual_effective_w[lane*12 +: 12]);
            row_final_w[lane] = row_partial_w[lane]
                + (current_parent_valid_w
                    ? $signed(parent_source_data_w[lane*12 +: 12])
                    : 14'sd0);
            psum_final_w[lane] =
                (issue_request_first
                    ? $signed(issue_psum_prior[lane*19 +: 19])
                    : $signed(psum_acc_q[lane]))
                + $signed(issue_residual_effective_w[lane*12 +: 12])
                + ((issue_last_w && current_parent_valid_w)
                    ? $signed(parent_source_data_w[lane*12 +: 12])
                    : 21'sd0);
            row_final_packed_w[lane*12 +: 12] = row_final_w[lane][11:0];
            psum_write_data[lane*19 +: 19] = psum_final_w[lane][18:0];
            if (row_final_w[lane] < -14'sd2048
                    || row_final_w[lane] > 14'sd2047)
                row_overflow_w = 1'b1;
            if (psum_final_w[lane] < -21'sd262144
                    || psum_final_w[lane] > 21'sd262143)
                psum_overflow_w = 1'b1;
        end

        // Fail closed before acceptance.  In particular a synthetic
        // parent-only beat is a protocol envelope with an exactly-zero
        // residual payload.  This combinational predicate gates every
        // architectural event; the sticky error is raised separately.
        preaccept_protocol_ok_w = residual_int8_format_ok_w
            && (!issue_synthetic_parent_w
                || issue_residual_data == 1152'b0);
        arithmetic_authoritative_w = issue_request_valid
            && (!issue_last_w || matching_parent_authoritative_w);

        final_sinks_ready_w = !issue_last_w
            || (psum_write_ready && row_complete_ready);
        base_issue_ready_w = issue_request_valid && parent_ready_w
            && final_sinks_ready_w
            && preaccept_protocol_ok_w
            && !(issue_last_w && (row_overflow_w || psum_overflow_w));

        // Hold a live final exactly when one read of the immediately next
        // consumer removes the otherwise unavoidable deadline miss.
        deadline_hold_w = issue_data_valid && base_issue_ready_w
            && issue_last_w && current_live_w
            && lookahead_immediate_next_w
            && reserved_count_w < 2
            && lookahead_written_w
            && lookahead_parent_w != current_row_w;
        issue_data_ready = base_issue_ready_w && !deadline_hold_w;
        issue_accept_w = issue_data_valid && issue_data_ready;
        consume_parent_w = issue_accept_w && issue_last_w
            && current_parent_valid_w;

        live_write_accept_w = issue_accept_w && issue_last_w && current_live_w;
        dead_elision_accept_w = issue_accept_w && issue_last_w && !current_live_w;
        stalled_same_address_w = issue_data_valid && issue_request_valid
            && preaccept_protocol_ok_w
            && issue_last_w && current_live_w && lookahead_valid_w
            && lookahead_parent_w == current_row_w && !base_issue_ready_w;
        forward_accept_w = live_write_accept_w && lookahead_valid_w
            && reserved_count_w < 2
            && lookahead_parent_w == current_row_w;
        macro_read_accept_w = lookahead_valid_w && reserved_count_w < 2
            && lookahead_written_w && !live_write_accept_w
            && !stalled_same_address_w
            && (!issue_data_valid || preaccept_protocol_ok_w);
        prefetch_accept_w = forward_accept_w || macro_read_accept_w;

        scratch_enable_w = live_write_accept_w || macro_read_accept_w;
        scratch_write_enable_w = live_write_accept_w;
        scratch_address_w = live_write_accept_w
            ? current_row_w : lookahead_parent_w;

        // These are atomic accepted-event pulses, not two independently
        // consumable channels: both sinks must be ready before either valid
        // rises.  The source request/payload remains held while either stalls.
        psum_write_valid = issue_data_valid && issue_request_valid
            && parent_ready_w && issue_last_w && final_sinks_ready_w
            && preaccept_protocol_ok_w
            && !row_overflow_w && !psum_overflow_w && !deadline_hold_w;
        psum_write_address = current_row_w;
        row_complete_valid = psum_write_valid;
        row_complete_id = current_row_w;
    end

    // Queue transition order is frozen: pop, prior macro response, then the
    // same-cycle forwarded new value.  Capacity never uses consume credit.
    always_comb begin
        slot0_valid_n = slot0_valid_q;
        slot0_parent_id_n = slot0_parent_id_q;
        slot0_consumer_id_n = slot0_consumer_id_q;
        slot0_data_n = slot0_data_q;
        slot1_valid_n = slot1_valid_q;
        slot1_parent_id_n = slot1_parent_id_q;
        slot1_consumer_id_n = slot1_consumer_id_q;
        slot1_data_n = slot1_data_q;
        queue_overflow_w = 1'b0;

        if (consume_parent_w) begin
            slot0_valid_n = slot1_valid_q;
            slot0_parent_id_n = slot1_parent_id_q;
            slot0_consumer_id_n = slot1_consumer_id_q;
            slot0_data_n = slot1_data_q;
            slot1_valid_n = 1'b0;
            slot1_parent_id_n = '0;
            slot1_consumer_id_n = '0;
            slot1_data_n = '0;
        end

        if (read_pending_q) begin
            if (!slot0_valid_n) begin
                slot0_valid_n = 1'b1;
                slot0_parent_id_n = read_pending_parent_q;
                slot0_consumer_id_n = read_pending_consumer_q;
                slot0_data_n = scratch_read_data_w;
            end else if (!slot1_valid_n) begin
                slot1_valid_n = 1'b1;
                slot1_parent_id_n = read_pending_parent_q;
                slot1_consumer_id_n = read_pending_consumer_q;
                slot1_data_n = scratch_read_data_w;
            end else begin
                queue_overflow_w = 1'b1;
            end
        end

        if (forward_accept_w) begin
            if (!slot0_valid_n) begin
                slot0_valid_n = 1'b1;
                slot0_parent_id_n = lookahead_parent_w;
                slot0_consumer_id_n = lookahead_consumer_w;
                slot0_data_n = row_final_packed_w;
            end else if (!slot1_valid_n) begin
                slot1_valid_n = 1'b1;
                slot1_parent_id_n = lookahead_parent_w;
                slot1_consumer_id_n = lookahead_consumer_w;
                slot1_data_n = row_final_packed_w;
            end else begin
                queue_overflow_w = 1'b1;
            end
        end
    end

    m528_dw1rw_parent_scratch_9x128_macro u_parent_scratch (
        .clk_core(clk_core),
        .enable(scratch_enable_w),
        .write_enable(scratch_write_enable_w),
        .address(scratch_address_w),
        .write_data(row_final_packed_w),
        .read_data(scratch_read_data_w)
    );

    always_comb begin
        task_drained_w = exec_active_q && !row_inflight_q
            && !selected_row_valid_w && !slot0_valid_q && !slot1_valid_q
            && !read_pending_q;
        fault_condition_w = 1'b0;
        if (prep_accept_w && !prep_semantic_ok_w)
            fault_condition_w = 1'b1;
        if (issue_data_valid && !issue_request_valid)
            fault_condition_w = 1'b1;
        if (issue_data_valid && arithmetic_authoritative_w && issue_last_w
                && (row_overflow_w || psum_overflow_w))
            fault_condition_w = 1'b1;
        if (issue_data_valid && issue_synthetic_parent_w
                && issue_residual_data != 1152'b0)
            fault_condition_w = 1'b1;
        if (issue_data_valid && issue_request_valid
                && !residual_int8_format_ok_w)
            fault_condition_w = 1'b1;
        if (slot1_valid_q && !slot0_valid_q)
            fault_condition_w = 1'b1;
        if (reserved_count_w > 2 || queue_overflow_w)
            fault_condition_w = 1'b1;
        if (macro_read_accept_w && !written_bitmap_q[lookahead_parent_w])
            fault_condition_w = 1'b1;
        if (scratch_write_enable_w && macro_read_accept_w)
            fault_condition_w = 1'b1;
        if (current_valid_w && current_directory_w[31:28] != 4'b0)
            fault_condition_w = 1'b1;
        if (current_valid_w && current_original_pop_w
                != popcount16(current_original_mask_w))
            fault_condition_w = 1'b1;
        if (current_valid_w
                && ((!current_parent_valid_w
                        && current_residual_mask_w != current_original_mask_w)
                    || (current_parent_valid_w
                        && current_residual_mask_w
                            != (current_original_mask_w
                                ^ mask_q[exec_bank_q][current_parent_id_w]))))
            fault_condition_w = 1'b1;
        if (current_valid_w && current_parent_valid_w
                && (!parent_live_q[exec_bank_q][current_parent_id_w]
                    || popcount16(mask_q[exec_bank_q][current_parent_id_w]) < 1
                    || ((mask_q[exec_bank_q][current_parent_id_w]
                            & current_original_mask_w)
                        != mask_q[exec_bank_q][current_parent_id_w])
                    || ((mask_q[exec_bank_q][current_parent_id_w]
                            == current_original_mask_w)
                        && current_parent_id_w >= current_row_w)))
            fault_condition_w = 1'b1;
        if (exec_active_q && bank_state_q[exec_bank_q] != BANK_EXEC)
            fault_condition_w = 1'b1;
    end

    assign protocol_error = fault_q;
    assign preprocess_busy = prep_active_q || match_active_q;
    assign execute_busy = exec_active_q;
    assign active_directory_bank = exec_bank_q;
    assign parent_queue_occupancy = queue_count_w[1:0];
    assign parent_reserved_occupancy = reserved_count_w;
    assign debug_parent_live_bitmap = exec_active_q
        ? parent_live_q[exec_bank_q] : 64'b0;
    assign debug_written_bitmap = written_bitmap_q;
    assign debug_scratch_read_event = macro_read_accept_w;
    assign debug_scratch_write_event = live_write_accept_w;
    assign debug_forward_event = forward_accept_w;
    assign debug_read_response_event = read_pending_q;
    assign debug_dual_enqueue_event = read_pending_q && forward_accept_w;
    assign debug_dead_write_elision_event = dead_elision_accept_w;
    assign debug_deadline_hold_event = deadline_hold_w;
    assign debug_overflow_block_event = issue_data_valid
        && arithmetic_authoritative_w && issue_last_w
        && (row_overflow_w || psum_overflow_w);
    assign debug_stalled_raw_event = stalled_same_address_w;

    integer reset_lane;
    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            bank_state_q[0] <= BANK_FREE;
            bank_state_q[1] <= BANK_FREE;
            bank_epoch_q[0] <= '0;
            bank_epoch_q[1] <= '0;
            parent_live_q[0] <= '0;
            parent_live_q[1] <= '0;
            prep_active_q <= 1'b0;
            prep_bank_q <= 1'b0;
            prep_epoch_q <= '0;
            prep_loaded_q <= '0;
            match_active_q <= 1'b0;
            match_bank_q <= 1'b0;
            match_row_q <= '0;
            epoch_seen_q <= 1'b0;
            newest_epoch_q <= '0;
            exec_active_q <= 1'b0;
            exec_bank_q <= 1'b0;
            exec_epoch_q <= '0;
            completed_bitmap_q <= '0;
            prefetched_edge_bitmap_q <= '0;
            written_bitmap_q <= '0;
            row_inflight_q <= 1'b0;
            current_row_q <= '0;
            residual_remaining_q <= '0;
            slot0_valid_q <= 1'b0;
            slot0_parent_id_q <= '0;
            slot0_consumer_id_q <= '0;
            slot0_data_q <= '0;
            slot1_valid_q <= 1'b0;
            slot1_parent_id_q <= '0;
            slot1_consumer_id_q <= '0;
            slot1_data_q <= '0;
            read_pending_q <= 1'b0;
            read_pending_parent_q <= '0;
            read_pending_consumer_q <= '0;
            task_done_valid <= 1'b0;
            task_done_epoch <= '0;
            fault_q <= 1'b0;
            count_issue_accepts <= '0;
            count_parent_edges <= '0;
            count_dead_write_elisions <= '0;
            count_macro_reads <= '0;
            count_macro_writes <= '0;
            count_forwards <= '0;
            count_deadline_holds <= '0;
            count_issue_stalls <= '0;
            count_psum_commits <= '0;
            count_row_completions <= '0;
            for (reset_lane = 0; reset_lane < 96; reset_lane = reset_lane + 1) begin
                row_acc_q[reset_lane] <= '0;
                psum_acc_q[reset_lane] <= '0;
            end
        end else begin
            task_done_valid <= 1'b0;
            if (fault_condition_w)
                fault_q <= 1'b1;

            // Load exactly one inactive bank and reject duplicate or stale
            // rows/epochs.  No partially loaded bank can become executable.
            if (prep_store_w) begin
                if (!prep_active_q) begin
                    prep_active_q <= 1'b1;
                    prep_bank_q <= free_bank_w;
                    prep_epoch_q <= prep_epoch;
                    prep_loaded_q <= row_bit(prep_row_id);
                    mask_q[free_bank_w][prep_row_id] <= prep_mask;
                    bank_epoch_q[free_bank_w] <= prep_epoch;
                    bank_state_q[free_bank_w] <= BANK_LOAD;
                    epoch_seen_q <= 1'b1;
                    newest_epoch_q <= prep_epoch;
                end else begin
                    prep_loaded_q <= prep_loaded_after_w;
                    mask_q[prep_bank_q][prep_row_id] <= prep_mask;
                    if (prep_task_last) begin
                        prep_active_q <= 1'b0;
                        match_active_q <= 1'b1;
                        match_bank_q <= prep_bank_q;
                        match_row_q <= '0;
                        parent_live_q[prep_bank_q] <= '0;
                        bank_state_q[prep_bank_q] <= BANK_MATCH;
                    end
                end
            end

            if (match_active_q) begin
                directory_q[match_bank_q][match_row_q] <= match_directory_w;
                if (match_best_valid_w)
                    parent_live_q[match_bank_q][match_best_id_w] <= 1'b1;
                if (match_row_q == 6'd63) begin
                    match_active_q <= 1'b0;
                    bank_state_q[match_bank_q] <= BANK_READY;
                end else begin
                    match_row_q <= match_row_q + 1'b1;
                end
            end

            // A ready bank becomes execution-owned only after all 64 matcher
            // rows and the complete live bitmap are committed.
            if (!exec_active_q && ready_bank_valid_w) begin
                exec_active_q <= 1'b1;
                exec_bank_q <= ready_bank_w;
                exec_epoch_q <= bank_epoch_q[ready_bank_w];
                bank_state_q[ready_bank_w] <= BANK_EXEC;
                completed_bitmap_q <= '0;
                prefetched_edge_bitmap_q <= '0;
                written_bitmap_q <= '0;
                row_inflight_q <= 1'b0;
                residual_remaining_q <= '0;
                slot0_valid_q <= 1'b0;
                slot1_valid_q <= 1'b0;
                read_pending_q <= 1'b0;
                count_issue_accepts <= '0;
                count_parent_edges <= '0;
                count_dead_write_elisions <= '0;
                count_macro_reads <= '0;
                count_macro_writes <= '0;
                count_forwards <= '0;
                count_deadline_holds <= '0;
                count_issue_stalls <= '0;
                count_psum_commits <= '0;
                count_row_completions <= '0;
            end

            if (exec_active_q) begin
                slot0_valid_q <= slot0_valid_n;
                slot0_parent_id_q <= slot0_parent_id_n;
                slot0_consumer_id_q <= slot0_consumer_id_n;
                slot0_data_q <= slot0_data_n;
                slot1_valid_q <= slot1_valid_n;
                slot1_parent_id_q <= slot1_parent_id_n;
                slot1_consumer_id_q <= slot1_consumer_id_n;
                slot1_data_q <= slot1_data_n;

                read_pending_q <= macro_read_accept_w;
                if (macro_read_accept_w) begin
                    read_pending_parent_q <= lookahead_parent_w;
                    read_pending_consumer_q <= lookahead_consumer_w;
                end
                if (prefetch_accept_w) begin
                    prefetched_edge_bitmap_q[lookahead_consumer_w] <= 1'b1;
                    count_parent_edges <= count_parent_edges + 1'b1;
                end
                if (live_write_accept_w) begin
                    written_bitmap_q[current_row_w] <= 1'b1;
                    count_macro_writes <= count_macro_writes + 1'b1;
                end
                if (macro_read_accept_w)
                    count_macro_reads <= count_macro_reads + 1'b1;
                if (forward_accept_w)
                    count_forwards <= count_forwards + 1'b1;
                if (dead_elision_accept_w)
                    count_dead_write_elisions <=
                        count_dead_write_elisions + 1'b1;
                if (deadline_hold_w)
                    count_deadline_holds <= count_deadline_holds + 1'b1;
                if (issue_data_valid && issue_request_valid
                        && preaccept_protocol_ok_w && !issue_data_ready)
                    count_issue_stalls <= count_issue_stalls + 1'b1;

                if (issue_accept_w) begin
                    count_issue_accepts <= count_issue_accepts + 1'b1;
                    if (issue_last_w) begin
                        completed_bitmap_q[current_row_w] <= 1'b1;
                        row_inflight_q <= 1'b0;
                        residual_remaining_q <= '0;
                        count_psum_commits <= count_psum_commits + 1'b1;
                        count_row_completions <= count_row_completions + 1'b1;
                    end else begin
                        row_inflight_q <= 1'b1;
                        current_row_q <= current_row_w;
                        residual_remaining_q <= issue_remaining_after_w;
                        for (int lane = 0; lane < 96; lane = lane + 1) begin
                            row_acc_q[lane] <= row_partial_w[lane][12:0];
                            psum_acc_q[lane] <= psum_final_w[lane][19:0];
                        end
                    end
                end

                if (task_drained_w) begin
                    exec_active_q <= 1'b0;
                    bank_state_q[exec_bank_q] <= BANK_FREE;
                    task_done_valid <= 1'b1;
                    task_done_epoch <= exec_epoch_q;
                    row_inflight_q <= 1'b0;
                end
            end
        end
    end
endmodule

`default_nettype wire
