`timescale 1ns/1ps
`default_nettype none

// Additive M935 timing-repair candidate for the M912/M528 C1 island.
//
// The product-capture algorithm, stable (popcount,row-id) order, one-port
// parent scratch, two-entry parent response queue, signed reconstruction and
// atomic architectural completion are unchanged.  The only architectural
// execution-side change remains the M912 metadata-only register boundary:
//   directory/mask arrays -> active/next row contexts -> 96-lane arithmetic
// and a separate registered prefetch token:
//   lookahead scan -> prefetch token -> one-port arbitration.
//
// No residual or psum payload is copied into a new pipeline register.  The
// existing two 1152-bit parent-response slots remain the only wide queue.
// Debug event pins are one-cycle registered observations; they never authorize
// a functional event and must not be used to hide a real functional path.
//
// M935 additionally replaces only the preprocess parent matcher with an exact
// three-stage, initiation-interval-one metadata pipeline:
//   F: register current {bank,row,mask,cached-popcount};
//   G: register eight balanced 8-candidate local winners;
//   R: reduce the eight winners, XOR the residual, and write the directory.
// Popcount is cached in the already allocated directory popcount field while
// prep loads each row; no second 64x5 array is introduced.  Candidate legality
// and the frozen strict-greater tie rule are unchanged: maximize parent
// popcount, and retain the lowest row id for an equal-popcount tie.  BANK_READY
// is asserted only on the edge that commits the row-63 R-stage result.
module m935_m912_three_stage_exact_parent_match_product_capture_island (
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
    input  logic          issue_data_valid,
    output logic          issue_data_ready,
    input  logic [1151:0] issue_residual_data,
    input  logic [1823:0] issue_psum_prior,

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

    // Candidate key is {invalid, original_popcount, row_id}.  A fixed six
    // level tournament preserves the frozen (popcount,row-id) order without
    // synthesizing the procedural 64-entry scan into a linear priority chain.
    function automatic logic [11:0] candidate_min(
        input logic [11:0] lhs,
        input logic [11:0] rhs
    );
        begin
            candidate_min = (lhs <= rhs) ? lhs : rhs;
        end
    endfunction

    // Parent tuple is {valid,popcount,row_id,mask}.  This comparator is the
    // exact tree form of the frozen ascending procedural scan: a larger
    // popcount wins, while equal-popcount ties retain the lower row id.
    function automatic logic [27:0] parent_winner(
        input logic [27:0] lhs,
        input logic [27:0] rhs
    );
        begin
            if (!lhs[27])
                parent_winner = rhs;
            else if (!rhs[27])
                parent_winner = lhs;
            else if (lhs[26:22] > rhs[26:22])
                parent_winner = lhs;
            else if (rhs[26:22] > lhs[26:22])
                parent_winner = rhs;
            else
                parent_winner = (lhs[21:16] <= rhs[21:16]) ? lhs : rhs;
        end
    endfunction

    // Frozen directory layout: residual[15:0], parent[21:16],
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
    logic match_issue_done_q;
    logic epoch_seen_q;
    logic [15:0] newest_epoch_q;

    logic free_bank_valid_w, free_bank_w;
    logic ready_bank_valid_w, ready_bank_w;
    logic prep_accept_w, prep_store_w, prep_semantic_ok_w;
    logic [63:0] prep_loaded_after_w;
    logic prep_last_expected_w;

    logic match_f_valid_q, match_f_bank_q;
    logic [5:0] match_f_row_q;
    logic [15:0] match_f_mask_q;
    logic [4:0] match_f_pop_q;
    logic [27:0] match_leaf_w [0:63];
    logic [27:0] match_pair_w [0:31];
    logic [27:0] match_quartet_w [0:15];
    logic [27:0] match_octet_w [0:7];

    logic match_g_valid_q, match_g_bank_q;
    logic [5:0] match_g_row_q;
    logic [15:0] match_g_mask_q;
    logic [4:0] match_g_pop_q;
    logic [27:0] match_g_winner_q [0:7];
    logic [27:0] match_r_pair_w [0:3];
    logic [27:0] match_r_quartet_w [0:1];
    logic [27:0] match_r_winner_w;
    logic [15:0] match_r_residual_w;
    logic [31:0] match_r_directory_w;

    logic exec_active_q, exec_bank_q;
    logic [15:0] exec_epoch_q;
    logic [63:0] completed_bitmap_q;
    logic [63:0] prefetched_edge_bitmap_q;
    logic [63:0] written_bitmap_q;

    // Metadata-only row dispatch boundary.  Neither context contains a wide
    // residual vector or psum vector.
    logic active_ctx_valid_q;
    logic [5:0] active_ctx_row_q;
    logic [15:0] active_ctx_original_mask_q;
    logic [15:0] active_ctx_residual_q;
    logic active_ctx_parent_valid_q;
    logic [5:0] active_ctx_parent_q;
    logic [4:0] active_ctx_original_pop_q;
    logic active_ctx_live_q;
    logic active_ctx_relation_ok_q;
    logic active_ctx_first_q;
    // A row is not exposed to the issue interface until one metadata-only
    // cycle has reserved its strict successor or proved no successor exists.
    // This keeps the 64-row selector out of functional ready/valid paths.
    logic active_ctx_primed_q;

    logic next_ctx_valid_q;
    logic [5:0] next_ctx_row_q;
    logic [15:0] next_ctx_original_mask_q;
    logic [15:0] next_ctx_residual_q;
    logic next_ctx_parent_valid_q;
    logic [5:0] next_ctx_parent_q;
    logic [4:0] next_ctx_original_pop_q;
    logic next_ctx_live_q;
    logic next_ctx_relation_ok_q;

    logic row_candidate_valid_w;
    logic [5:0] row_candidate_row_w;
    logic [4:0] row_candidate_pop_w;
    logic [31:0] row_candidate_directory_w;
    logic [15:0] row_candidate_original_mask_w;
    logic row_candidate_parent_valid_w;
    logic [5:0] row_candidate_parent_w;
    logic row_candidate_live_w;
    logic row_candidate_relation_ok_w;
    logic [11:0] row_key_s0_w [0:1][0:63];
    logic [11:0] row_key_s1_w [0:1][0:31];
    logic [11:0] row_key_s2_w [0:1][0:15];
    logic [11:0] row_key_s3_w [0:1][0:7];
    logic [11:0] row_key_s4_w [0:1][0:3];
    logic [11:0] row_key_s5_w [0:1][0:1];
    logic [11:0] row_key_s6_w [0:1];

    // Registered prefetch descriptor.  The wide response data continues to use
    // only the frozen two-entry response queue below.
    logic pf_token_valid_q;
    logic [5:0] pf_token_consumer_q;
    logic [5:0] pf_token_parent_q;
    logic pf_candidate_valid_w;
    logic [5:0] pf_candidate_consumer_w;
    logic [5:0] pf_candidate_parent_w;
    logic [4:0] pf_candidate_pop_w;
    logic pf_immediate_next_w, pf_parent_written_w;
    logic [11:0] pf_key_s0_w [0:1][0:63];
    logic [11:0] pf_key_s1_w [0:1][0:31];
    logic [11:0] pf_key_s2_w [0:1][0:15];
    logic [11:0] pf_key_s3_w [0:1][0:7];
    logic [11:0] pf_key_s4_w [0:1][0:3];
    logic [11:0] pf_key_s5_w [0:1][0:1];
    logic [11:0] pf_key_s6_w [0:1];

    logic [15:0] issue_work_mask_w, issue_remaining_after_w;
    logic issue_source_found_w;
    logic [3:0] issue_source_index_w;
    logic issue_synthetic_parent_w;
    logic issue_last_w;
    logic parent_ready_w;
    logic [1151:0] parent_source_data_w;

    logic signed [12:0] row_acc_q [0:95];
    logic signed [19:0] psum_acc_q [0:95];
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

    logic debug_scratch_read_q, debug_scratch_write_q;
    logic debug_forward_q, debug_read_response_q, debug_dual_enqueue_q;
    logic debug_dead_elision_q, debug_deadline_hold_q;
    logic debug_overflow_q, debug_stalled_raw_q;

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

    // G input logic.  Every candidate sees the F-stage row and the cached
    // prep-time popcount held in directory[27:23].  A candidate is legal iff
    // it is nonempty, is a subset of the current mask, and is not the current
    // row or a later equal-mask row.  The 64 leaves are reduced as eight
    // balanced 8-way trees; only their compact winners cross the G register.
    always_comb begin
        for (int candidate = 0; candidate < 64; candidate = candidate + 1) begin
            if (match_f_valid_q && match_f_pop_q >= 2
                    && directory_q[match_f_bank_q][candidate][27:23] >= 1
                    && ((mask_q[match_f_bank_q][candidate]
                            & match_f_mask_q)
                        == mask_q[match_f_bank_q][candidate])
                    && !((mask_q[match_f_bank_q][candidate]
                            == match_f_mask_q)
                        && candidate >= match_f_row_q)) begin
                match_leaf_w[candidate] = {
                    1'b1,
                    directory_q[match_f_bank_q][candidate][27:23],
                    candidate[5:0],
                    mask_q[match_f_bank_q][candidate]
                };
            end else begin
                match_leaf_w[candidate] = 28'b0;
            end
        end
        for (int node = 0; node < 32; node = node + 1)
            match_pair_w[node] = parent_winner(
                match_leaf_w[2*node], match_leaf_w[2*node+1]);
        for (int node = 0; node < 16; node = node + 1)
            match_quartet_w[node] = parent_winner(
                match_pair_w[2*node], match_pair_w[2*node+1]);
        for (int group = 0; group < 8; group = group + 1)
            match_octet_w[group] = parent_winner(
                match_quartet_w[2*group], match_quartet_w[2*group+1]);
    end

    // R combinational logic.  The registered local winners are reduced by a
    // balanced 8-to-1 tree.  This retains max-pop/lowest-id exactly and then
    // emits the frozen directory encoding.
    always_comb begin
        for (int node = 0; node < 4; node = node + 1)
            match_r_pair_w[node] = parent_winner(
                match_g_winner_q[2*node], match_g_winner_q[2*node+1]);
        for (int node = 0; node < 2; node = node + 1)
            match_r_quartet_w[node] = parent_winner(
                match_r_pair_w[2*node], match_r_pair_w[2*node+1]);
        match_r_winner_w = parent_winner(
            match_r_quartet_w[0], match_r_quartet_w[1]);
        match_r_residual_w = match_r_winner_w[27]
            ? (match_g_mask_q ^ match_r_winner_w[15:0])
            : match_g_mask_q;
        match_r_directory_w = {
            4'b0,
            match_g_pop_q,
            match_r_winner_w[27],
            match_r_winner_w[21:16],
            match_r_residual_w
        };
    end

    // Directory stage: build one balanced tournament per physical bank, then
    // select between the two bank-local winners.  The exec-bank bit therefore
    // crosses only the final mux instead of every leaf of a 64-row priority
    // chain.  Active and next contexts remain explicit reservations.
    always_comb begin
        for (int bank = 0; bank < 2; bank = bank + 1) begin
            for (int row = 0; row < 64; row = row + 1) begin
                if (mask_q[bank][row] != 16'b0
                        && !completed_bitmap_q[row]
                        && !(active_ctx_valid_q
                            && active_ctx_row_q == row[5:0])
                        && !(next_ctx_valid_q
                            && next_ctx_row_q == row[5:0])) begin
                    row_key_s0_w[bank][row] = {
                        1'b0, directory_q[bank][row][27:23], row[5:0]};
                end else begin
                    row_key_s0_w[bank][row] = 12'hfff;
                end
            end
            for (int node = 0; node < 32; node = node + 1)
                row_key_s1_w[bank][node] = candidate_min(
                    row_key_s0_w[bank][2*node],
                    row_key_s0_w[bank][2*node+1]);
            for (int node = 0; node < 16; node = node + 1)
                row_key_s2_w[bank][node] = candidate_min(
                    row_key_s1_w[bank][2*node],
                    row_key_s1_w[bank][2*node+1]);
            for (int node = 0; node < 8; node = node + 1)
                row_key_s3_w[bank][node] = candidate_min(
                    row_key_s2_w[bank][2*node],
                    row_key_s2_w[bank][2*node+1]);
            for (int node = 0; node < 4; node = node + 1)
                row_key_s4_w[bank][node] = candidate_min(
                    row_key_s3_w[bank][2*node],
                    row_key_s3_w[bank][2*node+1]);
            for (int node = 0; node < 2; node = node + 1)
                row_key_s5_w[bank][node] = candidate_min(
                    row_key_s4_w[bank][2*node],
                    row_key_s4_w[bank][2*node+1]);
            row_key_s6_w[bank] = candidate_min(
                row_key_s5_w[bank][0], row_key_s5_w[bank][1]);
        end

        row_candidate_valid_w = exec_active_q
            && !row_key_s6_w[exec_bank_q][11];
        row_candidate_row_w = row_candidate_valid_w
            ? row_key_s6_w[exec_bank_q][5:0] : 6'b0;
        row_candidate_pop_w = row_candidate_valid_w
            ? row_key_s6_w[exec_bank_q][10:6] : 5'd31;
        row_candidate_directory_w = row_candidate_valid_w
            ? directory_q[exec_bank_q][row_candidate_row_w] : 32'b0;
        row_candidate_original_mask_w = row_candidate_valid_w
            ? mask_q[exec_bank_q][row_candidate_row_w] : 16'b0;
        row_candidate_parent_valid_w = row_candidate_directory_w[22];
        row_candidate_parent_w = row_candidate_directory_w[21:16];
        row_candidate_live_w = row_candidate_valid_w
            && parent_live_q[exec_bank_q][row_candidate_row_w];
        row_candidate_relation_ok_w = row_candidate_valid_w
            && row_candidate_directory_w[31:28] == 4'b0
            && row_candidate_directory_w[27:23]
                == popcount16(row_candidate_original_mask_w)
            && ((!row_candidate_parent_valid_w
                    && row_candidate_directory_w[15:0]
                        == row_candidate_original_mask_w)
                || (row_candidate_parent_valid_w
                    && parent_live_q[exec_bank_q][row_candidate_parent_w]
                    && popcount16(mask_q[exec_bank_q]
                        [row_candidate_parent_w]) >= 1
                    && ((mask_q[exec_bank_q][row_candidate_parent_w]
                            & row_candidate_original_mask_w)
                        == mask_q[exec_bank_q][row_candidate_parent_w])
                    && !((mask_q[exec_bank_q][row_candidate_parent_w]
                            == row_candidate_original_mask_w)
                        && row_candidate_parent_w >= row_candidate_row_w)
                    && row_candidate_directory_w[15:0]
                        == (row_candidate_original_mask_w
                            ^ mask_q[exec_bank_q]
                                [row_candidate_parent_w])));
    end

    // Prefetch uses a separate balanced tournament because its reservation
    // set differs from row dispatch.  A held token may be replaced with the
    // already computed later winner on the same accepted edge.
    always_comb begin
        for (int bank = 0; bank < 2; bank = bank + 1) begin
            for (int row = 0; row < 64; row = row + 1) begin
                if (mask_q[bank][row] != 16'b0
                        && !completed_bitmap_q[row]
                        && directory_q[bank][row][22]
                        && !prefetched_edge_bitmap_q[row]
                        && !(pf_token_valid_q
                            && pf_token_consumer_q == row[5:0])) begin
                    pf_key_s0_w[bank][row] = {
                        1'b0, directory_q[bank][row][27:23], row[5:0]};
                end else begin
                    pf_key_s0_w[bank][row] = 12'hfff;
                end
            end
            for (int node = 0; node < 32; node = node + 1)
                pf_key_s1_w[bank][node] = candidate_min(
                    pf_key_s0_w[bank][2*node],
                    pf_key_s0_w[bank][2*node+1]);
            for (int node = 0; node < 16; node = node + 1)
                pf_key_s2_w[bank][node] = candidate_min(
                    pf_key_s1_w[bank][2*node],
                    pf_key_s1_w[bank][2*node+1]);
            for (int node = 0; node < 8; node = node + 1)
                pf_key_s3_w[bank][node] = candidate_min(
                    pf_key_s2_w[bank][2*node],
                    pf_key_s2_w[bank][2*node+1]);
            for (int node = 0; node < 4; node = node + 1)
                pf_key_s4_w[bank][node] = candidate_min(
                    pf_key_s3_w[bank][2*node],
                    pf_key_s3_w[bank][2*node+1]);
            for (int node = 0; node < 2; node = node + 1)
                pf_key_s5_w[bank][node] = candidate_min(
                    pf_key_s4_w[bank][2*node],
                    pf_key_s4_w[bank][2*node+1]);
            pf_key_s6_w[bank] = candidate_min(
                pf_key_s5_w[bank][0], pf_key_s5_w[bank][1]);
        end

        pf_candidate_valid_w = exec_active_q
            && !pf_key_s6_w[exec_bank_q][11];
        pf_candidate_consumer_w = pf_candidate_valid_w
            ? pf_key_s6_w[exec_bank_q][5:0] : 6'b0;
        pf_candidate_pop_w = pf_candidate_valid_w
            ? pf_key_s6_w[exec_bank_q][10:6] : 5'd31;
        pf_candidate_parent_w = pf_candidate_valid_w
            ? directory_q[exec_bank_q][pf_candidate_consumer_w][21:16]
            : 6'b0;
        pf_immediate_next_w = pf_token_valid_q && next_ctx_valid_q
            && pf_token_consumer_q == next_ctx_row_q;
        pf_parent_written_w = pf_token_valid_q
            && written_bitmap_q[pf_token_parent_q];
    end

    always_comb begin
        issue_work_mask_w = active_ctx_residual_q;
        issue_source_found_w = 1'b0;
        issue_source_index_w = '0;
        for (int source = 0; source < 16; source = source + 1) begin
            if (!issue_source_found_w && issue_work_mask_w[source]) begin
                issue_source_found_w = 1'b1;
                issue_source_index_w = source[3:0];
            end
        end
        issue_synthetic_parent_w = active_ctx_valid_q
            && !issue_source_found_w && active_ctx_parent_valid_q;
        issue_remaining_after_w = issue_work_mask_w;
        if (issue_source_found_w)
            issue_remaining_after_w[issue_source_index_w] = 1'b0;
        issue_last_w = active_ctx_valid_q
            && (issue_synthetic_parent_w || issue_remaining_after_w == 16'b0);

        issue_request_valid = exec_active_q && active_ctx_valid_q
            && active_ctx_primed_q && !fault_q;
        issue_request_epoch = exec_epoch_q;
        issue_request_row_id = active_ctx_row_q;
        issue_request_first = active_ctx_first_q;
        issue_request_last = issue_last_w;
        issue_request_source_valid = issue_source_found_w;
        issue_request_source_index = issue_source_index_w;
        issue_request_parent_valid = active_ctx_parent_valid_q;
        issue_request_parent_id = active_ctx_parent_q;
    end

    always_comb begin
        queue_count_w = {2'b0, slot0_valid_q} + {2'b0, slot1_valid_q};
        reserved_count_w = queue_count_w + {2'b0, read_pending_q};
        matching_parent_authoritative_w = !active_ctx_parent_valid_q
            || (slot0_valid_q
                && slot0_parent_id_q == active_ctx_parent_q
                && slot0_consumer_id_q == active_ctx_row_q);
        parent_ready_w = matching_parent_authoritative_w;
        parent_source_data_w = active_ctx_parent_valid_q
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
                (active_ctx_first_q ? 14'sd0 : $signed(row_acc_q[lane]))
                + $signed(issue_residual_effective_w[lane*12 +: 12]);
            row_final_w[lane] = row_partial_w[lane]
                + (active_ctx_parent_valid_q
                    ? $signed(parent_source_data_w[lane*12 +: 12])
                    : 14'sd0);
            psum_final_w[lane] =
                (active_ctx_first_q
                    ? $signed(issue_psum_prior[lane*19 +: 19])
                    : $signed(psum_acc_q[lane]))
                + $signed(issue_residual_effective_w[lane*12 +: 12])
                + ((issue_last_w && active_ctx_parent_valid_q)
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

        preaccept_protocol_ok_w = residual_int8_format_ok_w
            && (!issue_synthetic_parent_w
                || issue_residual_data == 1152'b0);
        arithmetic_authoritative_w = issue_request_valid
            && (!issue_last_w || matching_parent_authoritative_w);
        final_sinks_ready_w = !issue_last_w
            || (psum_write_ready && row_complete_ready);
        base_issue_ready_w = issue_request_valid && parent_ready_w
            && final_sinks_ready_w && preaccept_protocol_ok_w
            && !(issue_last_w && (row_overflow_w || psum_overflow_w));

        deadline_hold_w = issue_data_valid && base_issue_ready_w
            && issue_last_w && active_ctx_live_q
            && pf_immediate_next_w && reserved_count_w < 2
            && pf_parent_written_w
            && pf_token_parent_q != active_ctx_row_q;
        issue_data_ready = base_issue_ready_w && !deadline_hold_w;
        issue_accept_w = issue_data_valid && issue_data_ready;
        consume_parent_w = issue_accept_w && issue_last_w
            && active_ctx_parent_valid_q;

        live_write_accept_w = issue_accept_w && issue_last_w
            && active_ctx_live_q;
        dead_elision_accept_w = issue_accept_w && issue_last_w
            && !active_ctx_live_q;
        stalled_same_address_w = issue_data_valid && issue_request_valid
            && preaccept_protocol_ok_w && issue_last_w && active_ctx_live_q
            && pf_token_valid_q
            && pf_token_parent_q == active_ctx_row_q
            && !base_issue_ready_w;
        forward_accept_w = live_write_accept_w && pf_token_valid_q
            && reserved_count_w < 2
            && pf_token_parent_q == active_ctx_row_q;
        macro_read_accept_w = pf_token_valid_q && reserved_count_w < 2
            && pf_parent_written_w && !live_write_accept_w
            && !stalled_same_address_w
            && (!issue_data_valid || preaccept_protocol_ok_w);
        prefetch_accept_w = forward_accept_w || macro_read_accept_w;

        scratch_enable_w = live_write_accept_w || macro_read_accept_w;
        scratch_write_enable_w = live_write_accept_w;
        scratch_address_w = live_write_accept_w
            ? active_ctx_row_q : pf_token_parent_q;

        psum_write_valid = issue_data_valid && issue_request_valid
            && parent_ready_w && issue_last_w && final_sinks_ready_w
            && preaccept_protocol_ok_w
            && !row_overflow_w && !psum_overflow_w && !deadline_hold_w;
        psum_write_address = active_ctx_row_q;
        row_complete_valid = psum_write_valid;
        row_complete_id = active_ctx_row_q;
    end

    // Frozen queue update order: consume, prior synchronous response, then
    // same-cycle forward.  Capacity does not borrow consume credit.
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
                slot0_parent_id_n = pf_token_parent_q;
                slot0_consumer_id_n = pf_token_consumer_q;
                slot0_data_n = row_final_packed_w;
            end else if (!slot1_valid_n) begin
                slot1_valid_n = 1'b1;
                slot1_parent_id_n = pf_token_parent_q;
                slot1_consumer_id_n = pf_token_consumer_q;
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
        task_drained_w = exec_active_q && !active_ctx_valid_q
            && !next_ctx_valid_q && !row_candidate_valid_w
            && !pf_token_valid_q && !slot0_valid_q && !slot1_valid_q
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
        if (macro_read_accept_w && !written_bitmap_q[pf_token_parent_q])
            fault_condition_w = 1'b1;
        if (scratch_write_enable_w && macro_read_accept_w)
            fault_condition_w = 1'b1;
        if (active_ctx_valid_q && !active_ctx_relation_ok_q)
            fault_condition_w = 1'b1;
        if (next_ctx_valid_q && !next_ctx_relation_ok_q)
            fault_condition_w = 1'b1;
        if (active_ctx_valid_q && next_ctx_valid_q
                && active_ctx_row_q == next_ctx_row_q)
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
    assign debug_scratch_read_event = debug_scratch_read_q;
    assign debug_scratch_write_event = debug_scratch_write_q;
    assign debug_forward_event = debug_forward_q;
    assign debug_read_response_event = debug_read_response_q;
    assign debug_dual_enqueue_event = debug_dual_enqueue_q;
    assign debug_dead_write_elision_event = debug_dead_elision_q;
    assign debug_deadline_hold_event = debug_deadline_hold_q;
    assign debug_overflow_block_event = debug_overflow_q;
    assign debug_stalled_raw_event = debug_stalled_raw_q;

    integer reset_lane;
    integer reset_group;
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
            match_issue_done_q <= 1'b0;
            match_f_valid_q <= 1'b0;
            match_f_bank_q <= 1'b0;
            match_f_row_q <= '0;
            match_f_mask_q <= '0;
            match_f_pop_q <= '0;
            match_g_valid_q <= 1'b0;
            match_g_bank_q <= 1'b0;
            match_g_row_q <= '0;
            match_g_mask_q <= '0;
            match_g_pop_q <= '0;
            epoch_seen_q <= 1'b0;
            newest_epoch_q <= '0;
            exec_active_q <= 1'b0;
            exec_bank_q <= 1'b0;
            exec_epoch_q <= '0;
            completed_bitmap_q <= '0;
            prefetched_edge_bitmap_q <= '0;
            written_bitmap_q <= '0;
            active_ctx_valid_q <= 1'b0;
            active_ctx_row_q <= '0;
            active_ctx_original_mask_q <= '0;
            active_ctx_residual_q <= '0;
            active_ctx_parent_valid_q <= 1'b0;
            active_ctx_parent_q <= '0;
            active_ctx_original_pop_q <= '0;
            active_ctx_live_q <= 1'b0;
            active_ctx_relation_ok_q <= 1'b0;
            active_ctx_first_q <= 1'b0;
            active_ctx_primed_q <= 1'b0;
            next_ctx_valid_q <= 1'b0;
            next_ctx_row_q <= '0;
            next_ctx_original_mask_q <= '0;
            next_ctx_residual_q <= '0;
            next_ctx_parent_valid_q <= 1'b0;
            next_ctx_parent_q <= '0;
            next_ctx_original_pop_q <= '0;
            next_ctx_live_q <= 1'b0;
            next_ctx_relation_ok_q <= 1'b0;
            pf_token_valid_q <= 1'b0;
            pf_token_consumer_q <= '0;
            pf_token_parent_q <= '0;
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
            debug_scratch_read_q <= 1'b0;
            debug_scratch_write_q <= 1'b0;
            debug_forward_q <= 1'b0;
            debug_read_response_q <= 1'b0;
            debug_dual_enqueue_q <= 1'b0;
            debug_dead_elision_q <= 1'b0;
            debug_deadline_hold_q <= 1'b0;
            debug_overflow_q <= 1'b0;
            debug_stalled_raw_q <= 1'b0;
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
            for (reset_group = 0; reset_group < 8;
                    reset_group = reset_group + 1)
                match_g_winner_q[reset_group] <= '0;
        end else begin
            task_done_valid <= 1'b0;
            debug_scratch_read_q <= macro_read_accept_w;
            debug_scratch_write_q <= live_write_accept_w;
            debug_forward_q <= forward_accept_w;
            debug_read_response_q <= read_pending_q;
            debug_dual_enqueue_q <= read_pending_q && forward_accept_w;
            debug_dead_elision_q <= dead_elision_accept_w;
            debug_deadline_hold_q <= deadline_hold_w;
            debug_overflow_q <= issue_data_valid && arithmetic_authoritative_w
                && issue_last_w && (row_overflow_w || psum_overflow_w);
            debug_stalled_raw_q <= stalled_same_address_w;
            if (fault_condition_w)
                fault_q <= 1'b1;

            if (prep_store_w) begin
                if (!prep_active_q) begin
                    prep_active_q <= 1'b1;
                    prep_bank_q <= free_bank_w;
                    prep_epoch_q <= prep_epoch;
                    prep_loaded_q <= row_bit(prep_row_id);
                    mask_q[free_bank_w][prep_row_id] <= prep_mask;
                    directory_q[free_bank_w][prep_row_id] <= {
                        4'b0, popcount16(prep_mask), 23'b0};
                    bank_epoch_q[free_bank_w] <= prep_epoch;
                    bank_state_q[free_bank_w] <= BANK_LOAD;
                    epoch_seen_q <= 1'b1;
                    newest_epoch_q <= prep_epoch;
                end else begin
                    prep_loaded_q <= prep_loaded_after_w;
                    mask_q[prep_bank_q][prep_row_id] <= prep_mask;
                    directory_q[prep_bank_q][prep_row_id] <= {
                        4'b0, popcount16(prep_mask), 23'b0};
                    if (prep_task_last) begin
                        prep_active_q <= 1'b0;
                        match_active_q <= 1'b1;
                        match_bank_q <= prep_bank_q;
                        match_row_q <= '0;
                        match_issue_done_q <= 1'b0;
                        match_f_valid_q <= 1'b0;
                        match_g_valid_q <= 1'b0;
                        parent_live_q[prep_bank_q] <= '0;
                        bank_state_q[prep_bank_q] <= BANK_MATCH;
                    end
                end
            end

            // F stage accepts one row on every cycle until row 63.  The
            // cached popcount is read from the directory field seeded during
            // prep, so no 16-bit popcount tree appears in every G leaf.
            match_f_valid_q <= match_active_q && !match_issue_done_q;
            if (match_active_q && !match_issue_done_q) begin
                match_f_bank_q <= match_bank_q;
                match_f_row_q <= match_row_q;
                match_f_mask_q <= mask_q[match_bank_q][match_row_q];
                match_f_pop_q <=
                    directory_q[match_bank_q][match_row_q][27:23];
                if (match_row_q == 6'd63)
                    match_issue_done_q <= 1'b1;
                else
                    match_row_q <= match_row_q + 1'b1;
            end

            // G stage holds only eight compact local winners plus the current
            // row metadata.  No 1152-bit residual/psum payload is duplicated.
            match_g_valid_q <= match_f_valid_q;
            if (match_f_valid_q) begin
                match_g_bank_q <= match_f_bank_q;
                match_g_row_q <= match_f_row_q;
                match_g_mask_q <= match_f_mask_q;
                match_g_pop_q <= match_f_pop_q;
                for (int group = 0; group < 8; group = group + 1)
                    match_g_winner_q[group] <= match_octet_w[group];
            end

            // R stage is the sole directory writer during BANK_MATCH.
            // BANK_READY is delayed through the full drain and is asserted on
            // the same edge that writes row 63's exact result.
            if (match_g_valid_q) begin
                directory_q[match_g_bank_q][match_g_row_q]
                    <= match_r_directory_w;
                if (match_r_winner_w[27])
                    parent_live_q[match_g_bank_q]
                        [match_r_winner_w[21:16]] <= 1'b1;
                if (match_g_row_q == 6'd63) begin
                    match_active_q <= 1'b0;
                    match_issue_done_q <= 1'b0;
                    bank_state_q[match_g_bank_q] <= BANK_READY;
                end
            end

            if (!exec_active_q && ready_bank_valid_w) begin
                exec_active_q <= 1'b1;
                exec_bank_q <= ready_bank_w;
                exec_epoch_q <= bank_epoch_q[ready_bank_w];
                bank_state_q[ready_bank_w] <= BANK_EXEC;
                completed_bitmap_q <= '0;
                prefetched_edge_bitmap_q <= '0;
                written_bitmap_q <= '0;
                active_ctx_valid_q <= 1'b0;
                active_ctx_primed_q <= 1'b0;
                next_ctx_valid_q <= 1'b0;
                pf_token_valid_q <= 1'b0;
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
                    read_pending_parent_q <= pf_token_parent_q;
                    read_pending_consumer_q <= pf_token_consumer_q;
                end

                // Consume the current prefetch token and install the already
                // computed later token in the same edge.  When empty, fill it.
                if (prefetch_accept_w) begin
                    if (pf_candidate_valid_w) begin
                        pf_token_valid_q <= 1'b1;
                        pf_token_consumer_q <= pf_candidate_consumer_w;
                        pf_token_parent_q <= pf_candidate_parent_w;
                    end else begin
                        pf_token_valid_q <= 1'b0;
                    end
                end else if (!pf_token_valid_q && pf_candidate_valid_w) begin
                    pf_token_valid_q <= 1'b1;
                    pf_token_consumer_q <= pf_candidate_consumer_w;
                    pf_token_parent_q <= pf_candidate_parent_w;
                end
                if (prefetch_accept_w) begin
                    prefetched_edge_bitmap_q[pf_token_consumer_q] <= 1'b1;
                    count_parent_edges <= count_parent_edges + 1'b1;
                end

                if (live_write_accept_w) begin
                    written_bitmap_q[active_ctx_row_q] <= 1'b1;
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
                        completed_bitmap_q[active_ctx_row_q] <= 1'b1;
                        count_psum_commits <= count_psum_commits + 1'b1;
                        count_row_completions <= count_row_completions + 1'b1;

                        // Promote without an inter-row bubble.  The directory
                        // candidate excludes both current reservations, so it
                        // is the exact next free stable-order row.
                        if (next_ctx_valid_q) begin
                            active_ctx_valid_q <= 1'b1;
                            active_ctx_row_q <= next_ctx_row_q;
                            active_ctx_original_mask_q <=
                                next_ctx_original_mask_q;
                            active_ctx_residual_q <= next_ctx_residual_q;
                            active_ctx_parent_valid_q <=
                                next_ctx_parent_valid_q;
                            active_ctx_parent_q <= next_ctx_parent_q;
                            active_ctx_original_pop_q <=
                                next_ctx_original_pop_q;
                            active_ctx_live_q <= next_ctx_live_q;
                            active_ctx_relation_ok_q <=
                                next_ctx_relation_ok_q;
                            active_ctx_first_q <= 1'b1;
                            // Successor metadata is installed in this same
                            // edge, so the promoted row is fully primed.
                            active_ctx_primed_q <= 1'b1;
                            if (row_candidate_valid_w) begin
                                next_ctx_valid_q <= 1'b1;
                                next_ctx_row_q <= row_candidate_row_w;
                                next_ctx_original_mask_q <=
                                    row_candidate_original_mask_w;
                                next_ctx_residual_q <=
                                    row_candidate_directory_w[15:0];
                                next_ctx_parent_valid_q <=
                                    row_candidate_parent_valid_w;
                                next_ctx_parent_q <= row_candidate_parent_w;
                                next_ctx_original_pop_q <=
                                    row_candidate_directory_w[27:23];
                                next_ctx_live_q <= row_candidate_live_w;
                                next_ctx_relation_ok_q <=
                                    row_candidate_relation_ok_w;
                            end else begin
                                next_ctx_valid_q <= 1'b0;
                            end
                        end else if (row_candidate_valid_w) begin
                            active_ctx_valid_q <= 1'b1;
                            active_ctx_row_q <= row_candidate_row_w;
                            active_ctx_original_mask_q <=
                                row_candidate_original_mask_w;
                            active_ctx_residual_q <=
                                row_candidate_directory_w[15:0];
                            active_ctx_parent_valid_q <=
                                row_candidate_parent_valid_w;
                            active_ctx_parent_q <= row_candidate_parent_w;
                            active_ctx_original_pop_q <=
                                row_candidate_directory_w[27:23];
                            active_ctx_live_q <= row_candidate_live_w;
                            active_ctx_relation_ok_q <=
                                row_candidate_relation_ok_w;
                            active_ctx_first_q <= 1'b1;
                            // Defensive direct promotion cannot issue until a
                            // following metadata-only priming cycle.
                            active_ctx_primed_q <= 1'b0;
                            next_ctx_valid_q <= 1'b0;
                        end else begin
                            active_ctx_valid_q <= 1'b0;
                            active_ctx_primed_q <= 1'b0;
                            next_ctx_valid_q <= 1'b0;
                        end
                    end else begin
                        active_ctx_first_q <= 1'b0;
                        active_ctx_residual_q <= issue_remaining_after_w;
                        for (int lane = 0; lane < 96; lane = lane + 1) begin
                            row_acc_q[lane] <= row_partial_w[lane][12:0];
                            psum_acc_q[lane] <= psum_final_w[lane][19:0];
                        end
                        if (!next_ctx_valid_q && row_candidate_valid_w) begin
                            next_ctx_valid_q <= 1'b1;
                            next_ctx_row_q <= row_candidate_row_w;
                            next_ctx_original_mask_q <=
                                row_candidate_original_mask_w;
                            next_ctx_residual_q <=
                                row_candidate_directory_w[15:0];
                            next_ctx_parent_valid_q <=
                                row_candidate_parent_valid_w;
                            next_ctx_parent_q <= row_candidate_parent_w;
                            next_ctx_original_pop_q <=
                                row_candidate_directory_w[27:23];
                            next_ctx_live_q <= row_candidate_live_w;
                            next_ctx_relation_ok_q <=
                                row_candidate_relation_ok_w;
                        end
                    end
                end else if (!active_ctx_valid_q && row_candidate_valid_w) begin
                    active_ctx_valid_q <= 1'b1;
                    active_ctx_row_q <= row_candidate_row_w;
                    active_ctx_original_mask_q <=
                        row_candidate_original_mask_w;
                    active_ctx_residual_q <= row_candidate_directory_w[15:0];
                    active_ctx_parent_valid_q <=
                        row_candidate_parent_valid_w;
                    active_ctx_parent_q <= row_candidate_parent_w;
                    active_ctx_original_pop_q <=
                        row_candidate_directory_w[27:23];
                    active_ctx_live_q <= row_candidate_live_w;
                    active_ctx_relation_ok_q <= row_candidate_relation_ok_w;
                    active_ctx_first_q <= 1'b1;
                    active_ctx_primed_q <= 1'b0;
                end else if (active_ctx_valid_q
                        && !active_ctx_primed_q) begin
                    // Register the selector result before exposing the row.
                    // A single-row task also pays this cycle so that the
                    // absence of a candidate is never a combinational input
                    // to issue_request_valid.
                    active_ctx_primed_q <= 1'b1;
                    if (!next_ctx_valid_q && row_candidate_valid_w) begin
                        next_ctx_valid_q <= 1'b1;
                        next_ctx_row_q <= row_candidate_row_w;
                        next_ctx_original_mask_q <=
                            row_candidate_original_mask_w;
                        next_ctx_residual_q <=
                            row_candidate_directory_w[15:0];
                        next_ctx_parent_valid_q <=
                            row_candidate_parent_valid_w;
                        next_ctx_parent_q <= row_candidate_parent_w;
                        next_ctx_original_pop_q <=
                            row_candidate_directory_w[27:23];
                        next_ctx_live_q <= row_candidate_live_w;
                        next_ctx_relation_ok_q <=
                            row_candidate_relation_ok_w;
                    end
                end else if (active_ctx_valid_q && !next_ctx_valid_q
                        && row_candidate_valid_w) begin
                    // Defensive refill for a future implementation that may
                    // invalidate next_ctx independently.  It is not on the
                    // current steady-state row-to-row path.
                    next_ctx_valid_q <= 1'b1;
                    next_ctx_row_q <= row_candidate_row_w;
                    next_ctx_original_mask_q <=
                        row_candidate_original_mask_w;
                    next_ctx_residual_q <= row_candidate_directory_w[15:0];
                    next_ctx_parent_valid_q <=
                        row_candidate_parent_valid_w;
                    next_ctx_parent_q <= row_candidate_parent_w;
                    next_ctx_original_pop_q <=
                        row_candidate_directory_w[27:23];
                    next_ctx_live_q <= row_candidate_live_w;
                    next_ctx_relation_ok_q <= row_candidate_relation_ok_w;
                end

                if (task_drained_w) begin
                    exec_active_q <= 1'b0;
                    bank_state_q[exec_bank_q] <= BANK_FREE;
                    task_done_valid <= 1'b1;
                    task_done_epoch <= exec_epoch_q;
                    active_ctx_valid_q <= 1'b0;
                    active_ctx_primed_q <= 1'b0;
                    next_ctx_valid_q <= 1'b0;
                    pf_token_valid_q <= 1'b0;
                end
            end
        end
    end
endmodule

`default_nettype wire
