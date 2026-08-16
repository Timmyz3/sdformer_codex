`timescale 1ns/1ps
`default_nettype none

// Fixed-capacity head slots shared by TERM-CSR and RAW replay formats.
// The behavioral memory has one write stream and one read stream; physical
// SRAM macro selection is intentionally left to synthesis handoff.
module gatestack_head_slot_sram_adapter #(
    parameter int CONTEXTS       = 2,
    parameter int HEADS          = 24,
    parameter int HEAD_BITS      = 6642,
    parameter int WORD_W         = 64,
    parameter int SLOT_CAPACITY_BITS =
        ((HEAD_BITS + WORD_W - 1) / WORD_W) * WORD_W,
    parameter int TAG_W          = 32,
    parameter int SIZE_W         = 16,
    parameter int FORMAT_W       = 2,
    parameter int COUNTER_W      = 32,
    parameter int CONTEXT_ID_W   = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W      = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int WORD_INDEX_W   =
        (((SLOT_CAPACITY_BITS + WORD_W - 1) / WORD_W) <= 1) ? 1 :
        $clog2((SLOT_CAPACITY_BITS + WORD_W - 1) / WORD_W)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         commit_begin_valid,
    output logic                         commit_begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      commit_context_id,
    input  logic [HEAD_ID_W-1:0]         commit_head_id,
    input  logic [TAG_W-1:0]             commit_tag,
    input  logic                         commit_mode_is_csr,
    input  logic [SIZE_W-1:0]            commit_payload_bits,

    input  logic                         commit_word_valid,
    output logic                         commit_word_ready,
    input  logic [WORD_W-1:0]            commit_word_data,
    input  logic                         commit_word_last,

    input  logic                         inspect_valid,
    output logic                         inspect_ready,
    input  logic [CONTEXT_ID_W-1:0]      inspect_context_id,
    input  logic [HEAD_ID_W-1:0]         inspect_head_id,
    output logic                         inspect_meta_valid,
    input  logic                         inspect_meta_ready,
    output logic                         inspect_exists,
    output logic [TAG_W-1:0]             inspect_tag,
    output logic                         inspect_mode_is_csr,
    output logic [FORMAT_W-1:0]          inspect_format,
    output logic [SIZE_W-1:0]            inspect_payload_bits,
    output logic [SIZE_W-1:0]            inspect_word_count,

    input  logic                         replay_begin_valid,
    output logic                         replay_begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      replay_context_id,
    input  logic [HEAD_ID_W-1:0]         replay_head_id,
    input  logic [WORD_INDEX_W-1:0]      replay_start_word,

    output logic                         replay_word_valid,
    input  logic                         replay_word_ready,
    output logic [WORD_W-1:0]            replay_word_data,
    output logic [WORD_INDEX_W-1:0]      replay_word_index,
    output logic                         replay_word_last,
    output logic [TAG_W-1:0]             replay_tag,
    output logic                         replay_mode_is_csr,
    output logic [FORMAT_W-1:0]          replay_format,
    output logic [SIZE_W-1:0]            replay_payload_bits,

    input  logic                         release_valid,
    output logic                         release_ready,
    input  logic [CONTEXT_ID_W-1:0]      release_context_id,
    input  logic [HEAD_ID_W-1:0]         release_head_id,

    output logic                         commit_session_active,
    output logic                         replay_session_active,
    output logic [(CONTEXTS*HEADS)-1:0]  slot_valid_flat,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_commit_heads,
    output logic [COUNTER_W-1:0]         count_replay_heads,
    output logic [COUNTER_W-1:0]         count_release_heads,
    output logic [COUNTER_W-1:0]         count_invalid_headers,
    output logic [COUNTER_W-1:0]         count_commit_stall_cycles,
    output logic [COUNTER_W-1:0]         count_replay_stall_cycles
);

    localparam int WORDS_PER_HEAD =
        (SLOT_CAPACITY_BITS + WORD_W - 1) / WORD_W;
    localparam int TOTAL_SLOTS = CONTEXTS * HEADS;
    localparam int TOTAL_WORDS = TOTAL_SLOTS * WORDS_PER_HEAD;
    localparam int SLOT_INDEX_W = (TOTAL_SLOTS <= 1) ? 1 : $clog2(TOTAL_SLOTS);
    localparam int MEM_ADDR_W = (TOTAL_WORDS <= 1) ? 1 : $clog2(TOTAL_WORDS);
    localparam int WORD_COUNT_W = (WORDS_PER_HEAD + 1 <= 2) ?
                                  1 : $clog2(WORDS_PER_HEAD + 1);
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);
    localparam logic [FORMAT_W-1:0] FORMAT_INVALID = FORMAT_W'(3);
    localparam logic [15:0] IPD32W_MAGIC = 16'h4753;
    localparam logic [15:0] FADC24_MAGIC = 16'h4641;

    logic [WORD_W-1:0] slot_mem [0:TOTAL_WORDS-1];
    logic [TOTAL_SLOTS-1:0] slot_valid_q;
    logic [TAG_W-1:0] slot_tag_q [0:TOTAL_SLOTS-1];
    logic slot_mode_is_csr_q [0:TOTAL_SLOTS-1];
    logic [FORMAT_W-1:0] slot_format_q [0:TOTAL_SLOTS-1];
    logic [SIZE_W-1:0] slot_payload_bits_q [0:TOTAL_SLOTS-1];
    logic [WORD_COUNT_W-1:0] slot_words_q [0:TOTAL_SLOTS-1];

    logic commit_active_q;
    logic [SLOT_INDEX_W-1:0] commit_slot_q;
    logic [MEM_ADDR_W-1:0] commit_base_q;
    logic [WORD_INDEX_W-1:0] commit_word_index_q;
    logic [WORD_COUNT_W-1:0] commit_expected_words_q;
    logic [TAG_W-1:0] commit_tag_q;
    logic commit_mode_is_csr_q;
    logic [FORMAT_W-1:0] commit_format_q;
    logic commit_header_error_q;
    logic [SIZE_W-1:0] commit_payload_bits_q;

    logic replay_active_q;
    logic [SLOT_INDEX_W-1:0] replay_slot_q;
    logic [MEM_ADDR_W-1:0] replay_base_q;
    logic [WORD_INDEX_W-1:0] replay_next_index_q;
    logic [WORD_COUNT_W-1:0] replay_expected_words_q;
    logic [TAG_W-1:0] replay_tag_q;
    logic replay_mode_is_csr_q;
    logic [FORMAT_W-1:0] replay_format_q;
    logic [SIZE_W-1:0] replay_payload_bits_q;

    logic replay_word_valid_q;
    logic [WORD_W-1:0] replay_word_data_q;
    logic [WORD_INDEX_W-1:0] replay_word_index_q;
    logic replay_word_last_q;

    logic commit_context_in_range;
    logic commit_head_in_range;
    logic commit_payload_in_range;
    logic replay_context_in_range;
    logic replay_head_in_range;
    logic replay_start_in_range;
    logic release_context_in_range;
    logic release_head_in_range;
    logic inspect_context_in_range;
    logic inspect_head_in_range;
    logic [SLOT_INDEX_W-1:0] commit_slot_comb;
    logic [SLOT_INDEX_W-1:0] replay_slot_comb;
    logic [SLOT_INDEX_W-1:0] release_slot_comb;
    logic [SLOT_INDEX_W-1:0] inspect_slot_comb;
    logic [WORD_COUNT_W-1:0] commit_words_comb;
    logic commit_begin_fire;
    logic commit_word_fire;
    logic replay_begin_fire;
    logic replay_word_fire;
    logic release_fire;
    logic inspect_fire;
    logic commit_expected_last;
    logic commit_last_matches;
    logic replay_target_conflict;
    logic release_target_conflict;
    logic inspect_meta_valid_q;
    logic inspect_exists_q;
    logic [TAG_W-1:0] inspect_tag_q;
    logic inspect_mode_is_csr_q;
    logic [FORMAT_W-1:0] inspect_format_q;
    logic [SIZE_W-1:0] inspect_payload_bits_q;
    logic [SIZE_W-1:0] inspect_word_count_q;
    logic [FORMAT_W-1:0] commit_word_format_comb;
    logic commit_word_header_legal_comb;

    assign commit_context_in_range = 32'(commit_context_id) < CONTEXTS;
    assign commit_head_in_range = 32'(commit_head_id) < HEADS;
    assign commit_payload_in_range = (commit_payload_bits != '0) &&
        (32'(commit_payload_bits) <= SLOT_CAPACITY_BITS);
    assign replay_context_in_range = 32'(replay_context_id) < CONTEXTS;
    assign replay_head_in_range = 32'(replay_head_id) < HEADS;
    assign replay_start_in_range =
        32'(replay_start_word) < 32'(slot_words_q[replay_slot_comb]);
    assign release_context_in_range = 32'(release_context_id) < CONTEXTS;
    assign release_head_in_range = 32'(release_head_id) < HEADS;
    assign inspect_context_in_range = 32'(inspect_context_id) < CONTEXTS;
    assign inspect_head_in_range = 32'(inspect_head_id) < HEADS;

    always_comb begin
        commit_slot_comb = '0;
        replay_slot_comb = '0;
        release_slot_comb = '0;
        inspect_slot_comb = '0;
        if (commit_context_in_range && commit_head_in_range) begin
            commit_slot_comb = SLOT_INDEX_W'(
                (32'(commit_context_id) * 32'(HEADS)) +
                32'(commit_head_id));
        end
        if (replay_context_in_range && replay_head_in_range) begin
            replay_slot_comb = SLOT_INDEX_W'(
                (32'(replay_context_id) * 32'(HEADS)) +
                32'(replay_head_id));
        end
        if (release_context_in_range && release_head_in_range) begin
            release_slot_comb = SLOT_INDEX_W'(
                (32'(release_context_id) * 32'(HEADS)) +
                32'(release_head_id));
        end
        if (inspect_context_in_range && inspect_head_in_range) begin
            inspect_slot_comb = SLOT_INDEX_W'(
                (32'(inspect_context_id) * 32'(HEADS)) +
                32'(inspect_head_id));
        end
        commit_words_comb = WORD_COUNT_W'(
            (32'(commit_payload_bits) + WORD_W - 1) / WORD_W);
        commit_word_format_comb = commit_format_q;
        commit_word_header_legal_comb = !commit_header_error_q;
        if (commit_word_index_q == '0) begin
            if (!commit_mode_is_csr_q) begin
                commit_word_format_comb = FORMAT_RAW;
                commit_word_header_legal_comb = 1'b1;
            end else if (commit_word_data[15:0] == IPD32W_MAGIC) begin
                commit_word_format_comb = FORMAT_IPD32W;
                commit_word_header_legal_comb =
                    commit_word_data[19:16] == 4'd1 &&
                    commit_word_data[20] && commit_word_data[31:21] == '0 &&
                    commit_word_data[63:32] == 32'(commit_tag_q);
            end else if (commit_word_data[15:0] == FADC24_MAGIC) begin
                commit_word_format_comb = FORMAT_FADC24;
                commit_word_header_legal_comb =
                    commit_word_data[23:16] == 8'd1 &&
                    commit_word_data[31:24] == '0 &&
                    commit_word_data[63:32] == 32'(commit_tag_q);
            end else begin
                commit_word_format_comb = FORMAT_INVALID;
                commit_word_header_legal_comb = 1'b0;
            end
        end
    end

    assign replay_target_conflict = commit_active_q &&
                                    (commit_slot_q == replay_slot_comb);
    assign release_target_conflict = (commit_active_q &&
                                      (commit_slot_q == release_slot_comb)) ||
                                     (replay_active_q &&
                                      (replay_slot_q == release_slot_comb)) ||
                                     (replay_word_valid_q &&
                                      (replay_slot_q == release_slot_comb));

    assign commit_begin_ready = !commit_active_q &&
                                commit_context_in_range &&
                                commit_head_in_range &&
                                commit_payload_in_range &&
                                !slot_valid_q[commit_slot_comb] &&
                                !(replay_active_q &&
                                  (replay_slot_q == commit_slot_comb)) &&
                                !(replay_word_valid_q &&
                                  (replay_slot_q == commit_slot_comb)) &&
                                !(release_valid && release_context_in_range &&
                                  release_head_in_range &&
                                  (release_slot_comb == commit_slot_comb));
    assign commit_word_ready = commit_active_q;
    assign replay_begin_ready = !replay_active_q && !replay_word_valid_q &&
                                replay_context_in_range &&
                                replay_head_in_range &&
                                slot_valid_q[replay_slot_comb] &&
                                replay_start_in_range &&
                                !replay_target_conflict &&
                                !(release_valid && release_context_in_range &&
                                  release_head_in_range &&
                                  (release_slot_comb == replay_slot_comb));
    assign release_ready = release_context_in_range && release_head_in_range &&
                           slot_valid_q[release_slot_comb] &&
                           !release_target_conflict;
    assign inspect_ready = !inspect_meta_valid_q &&
        inspect_context_in_range && inspect_head_in_range &&
        !(commit_active_q && commit_slot_q == inspect_slot_comb) &&
        !(commit_begin_valid && commit_begin_ready &&
          commit_slot_comb == inspect_slot_comb) &&
        !(release_valid && release_ready &&
          release_slot_comb == inspect_slot_comb);

    assign commit_begin_fire = commit_begin_valid && commit_begin_ready;
    assign commit_word_fire = commit_word_valid && commit_word_ready;
    assign replay_begin_fire = replay_begin_valid && replay_begin_ready;
    assign replay_word_fire = replay_word_valid_q && replay_word_ready;
    assign release_fire = release_valid && release_ready;
    assign inspect_fire = inspect_valid && inspect_ready;
    assign commit_expected_last =
        WORD_COUNT_W'(commit_word_index_q) ==
        (commit_expected_words_q - 1'b1);
    assign commit_last_matches = commit_word_last == commit_expected_last;

    assign replay_word_valid = replay_word_valid_q;
    assign replay_word_data = replay_word_data_q;
    assign replay_word_index = replay_word_index_q;
    assign replay_word_last = replay_word_last_q;
    assign replay_tag = replay_tag_q;
    assign replay_mode_is_csr = replay_mode_is_csr_q;
    assign replay_format = replay_format_q;
    assign replay_payload_bits = replay_payload_bits_q;
    assign commit_session_active = commit_active_q;
    assign replay_session_active = replay_active_q || replay_word_valid_q;
    assign slot_valid_flat = slot_valid_q;
    assign inspect_meta_valid = inspect_meta_valid_q;
    assign inspect_exists = inspect_exists_q;
    assign inspect_tag = inspect_tag_q;
    assign inspect_mode_is_csr = inspect_mode_is_csr_q;
    assign inspect_format = inspect_format_q;
    assign inspect_payload_bits = inspect_payload_bits_q;
    assign inspect_word_count = inspect_word_count_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            slot_valid_q <= '0;
            commit_active_q <= 1'b0;
            commit_slot_q <= '0;
            commit_base_q <= '0;
            commit_word_index_q <= '0;
            commit_expected_words_q <= '0;
            commit_tag_q <= '0;
            commit_mode_is_csr_q <= 1'b0;
            commit_format_q <= FORMAT_RAW;
            commit_header_error_q <= 1'b0;
            commit_payload_bits_q <= '0;
            replay_active_q <= 1'b0;
            replay_slot_q <= '0;
            replay_base_q <= '0;
            replay_next_index_q <= '0;
            replay_expected_words_q <= '0;
            replay_tag_q <= '0;
            replay_mode_is_csr_q <= 1'b0;
            replay_format_q <= FORMAT_RAW;
            replay_payload_bits_q <= '0;
            replay_word_valid_q <= 1'b0;
            replay_word_data_q <= '0;
            replay_word_index_q <= '0;
            replay_word_last_q <= 1'b0;
            inspect_meta_valid_q <= 1'b0;
            inspect_exists_q <= 1'b0;
            inspect_tag_q <= '0;
            inspect_mode_is_csr_q <= 1'b0;
            inspect_format_q <= FORMAT_RAW;
            inspect_payload_bits_q <= '0;
            inspect_word_count_q <= '0;
            protocol_error <= 1'b0;
            count_commit_heads <= '0;
            count_replay_heads <= '0;
            count_release_heads <= '0;
            count_invalid_headers <= '0;
            count_commit_stall_cycles <= '0;
            count_replay_stall_cycles <= '0;
        end else begin
            if (commit_begin_fire) begin
                commit_active_q <= 1'b1;
                commit_slot_q <= commit_slot_comb;
                commit_base_q <= MEM_ADDR_W'(
                    32'(commit_slot_comb) * 32'(WORDS_PER_HEAD));
                commit_word_index_q <= '0;
                commit_expected_words_q <= commit_words_comb;
                commit_tag_q <= commit_tag;
                commit_mode_is_csr_q <= commit_mode_is_csr;
                commit_format_q <= commit_mode_is_csr ?
                    FORMAT_IPD32W : FORMAT_RAW;
                commit_header_error_q <= 1'b0;
                commit_payload_bits_q <= commit_payload_bits;
            end

            if (commit_word_fire) begin
                commit_format_q <= commit_word_format_comb;
                commit_header_error_q <= !commit_word_header_legal_comb;
                slot_mem[commit_base_q + MEM_ADDR_W'(commit_word_index_q)] <=
                    commit_word_data;
                if (!commit_last_matches) begin
                    commit_active_q <= 1'b0;
                    protocol_error <= 1'b1;
                end else if (commit_expected_last &&
                             !commit_word_header_legal_comb) begin
                    commit_active_q <= 1'b0;
                    protocol_error <= 1'b1;
                    count_invalid_headers <= count_invalid_headers + 1'b1;
                end else if (commit_expected_last) begin
                    commit_active_q <= 1'b0;
                    slot_valid_q[commit_slot_q] <= 1'b1;
                    slot_tag_q[commit_slot_q] <= commit_tag_q;
                    slot_mode_is_csr_q[commit_slot_q] <= commit_mode_is_csr_q;
                    slot_format_q[commit_slot_q] <= commit_word_format_comb;
                    slot_payload_bits_q[commit_slot_q] <=
                        commit_payload_bits_q;
                    slot_words_q[commit_slot_q] <= commit_expected_words_q;
                    count_commit_heads <= count_commit_heads + 1'b1;
                end else begin
                    commit_word_index_q <= commit_word_index_q + 1'b1;
                end
            end

            if (replay_begin_fire) begin
                replay_active_q <= 1'b1;
                replay_slot_q <= replay_slot_comb;
                replay_base_q <= MEM_ADDR_W'(
                    32'(replay_slot_comb) * 32'(WORDS_PER_HEAD)) +
                    MEM_ADDR_W'(replay_start_word);
                replay_next_index_q <= '0;
                replay_expected_words_q <=
                    slot_words_q[replay_slot_comb] -
                    WORD_COUNT_W'(replay_start_word);
                replay_tag_q <= slot_tag_q[replay_slot_comb];
                replay_mode_is_csr_q <=
                    slot_mode_is_csr_q[replay_slot_comb];
                replay_format_q <= slot_format_q[replay_slot_comb];
                replay_payload_bits_q <=
                    slot_payload_bits_q[replay_slot_comb];
                count_replay_heads <= count_replay_heads + 1'b1;
            end

            if (inspect_fire) begin
                inspect_meta_valid_q <= 1'b1;
                inspect_exists_q <= slot_valid_q[inspect_slot_comb];
                inspect_tag_q <= slot_valid_q[inspect_slot_comb] ?
                    slot_tag_q[inspect_slot_comb] : '0;
                inspect_mode_is_csr_q <= slot_valid_q[inspect_slot_comb] ?
                    slot_mode_is_csr_q[inspect_slot_comb] : 1'b0;
                inspect_format_q <= slot_valid_q[inspect_slot_comb] ?
                    slot_format_q[inspect_slot_comb] : FORMAT_RAW;
                inspect_payload_bits_q <= slot_valid_q[inspect_slot_comb] ?
                    slot_payload_bits_q[inspect_slot_comb] : '0;
                inspect_word_count_q <= slot_valid_q[inspect_slot_comb] ?
                    SIZE_W'(slot_words_q[inspect_slot_comb]) : '0;
            end else if (inspect_meta_valid_q && inspect_meta_ready) begin
                inspect_meta_valid_q <= 1'b0;
            end

            if (replay_word_fire) begin
                replay_word_valid_q <= 1'b0;
            end
            if (replay_active_q &&
                (!replay_word_valid_q || replay_word_ready)) begin
                replay_word_valid_q <= 1'b1;
                replay_word_data_q <= slot_mem[
                    replay_base_q + MEM_ADDR_W'(replay_next_index_q)];
                replay_word_index_q <= replay_next_index_q;
                replay_word_last_q <=
                    WORD_COUNT_W'(replay_next_index_q) ==
                    (replay_expected_words_q - 1'b1);
                if (WORD_COUNT_W'(replay_next_index_q) ==
                    (replay_expected_words_q - 1'b1)) begin
                    replay_active_q <= 1'b0;
                end else begin
                    replay_next_index_q <= replay_next_index_q + 1'b1;
                end
            end

            if (release_fire) begin
                slot_valid_q[release_slot_comb] <= 1'b0;
                count_release_heads <= count_release_heads + 1'b1;
            end

            if (commit_begin_valid && !commit_begin_ready) begin
                count_commit_stall_cycles <= count_commit_stall_cycles + 1'b1;
            end
            if (replay_begin_valid && !replay_begin_ready) begin
                count_replay_stall_cycles <= count_replay_stall_cycles + 1'b1;
            end
            if ((commit_begin_valid && !commit_context_in_range) ||
                (commit_begin_valid && !commit_head_in_range) ||
                (commit_begin_valid && !commit_payload_in_range) ||
                (replay_begin_valid && !replay_context_in_range) ||
                (replay_begin_valid && !replay_head_in_range) ||
                (replay_begin_valid && replay_context_in_range &&
                 replay_head_in_range &&
                 slot_valid_q[replay_slot_comb] &&
                 !replay_start_in_range) ||
                (release_valid && !release_context_in_range) ||
                (release_valid && !release_head_in_range) ||
                (inspect_valid && !inspect_context_in_range) ||
                (inspect_valid && !inspect_head_in_range)) begin
                protocol_error <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
