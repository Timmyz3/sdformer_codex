`timescale 1ns/1ps
`default_nettype none

// Capacity-bounded C0 serializer. The complete payload is built and checked in
// a private slot buffer before commit_begin is exposed, so malformed input can
// never make a partially encoded head visible to the shared slot SRAM.
module gatestack_typed_payload_serializer #(
    parameter int TOKENS             = 162,
    parameter int LANES              = 32,
    parameter int GATE_W             = 9,
    parameter int MAX_TERMS          = 128,
    parameter int SLOT_WORDS         = 104,
    parameter int WORD_W             = 64,
    parameter int TAG_W              = 32,
    parameter int FORMAT_W           = 2,
    parameter int SIZE_W             = 16,
    parameter int CONTEXTS           = 2,
    parameter int HEADS              = 24,
    parameter int COUNTER_W          = 32,
    parameter int BITMAP_BYPASS_ENABLE = 0,
    parameter int CONTEXT_ID_W       = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W          = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int TOKEN_ID_W         = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int TERM_INDEX_W       = (MAX_TERMS <= 1) ? 1 : $clog2(MAX_TERMS),
    parameter int WORD_INDEX_W       = (SLOT_WORDS <= 1) ? 1 : $clog2(SLOT_WORDS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         begin_valid,
    output logic                         begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      begin_context_id,
    input  logic [HEAD_ID_W-1:0]         begin_head_id,
    input  logic [TAG_W-1:0]             begin_tag,
    input  logic [FORMAT_W-1:0]          begin_format,
    input  logic [SIZE_W-1:0]            begin_expected_payload_bits,
    input  logic [3:0]                   begin_active_classes,
    input  logic [7:0]                   begin_active_tokens,
    input  logic [7:0]                   begin_term_count,
    input  logic [12:0]                  begin_event_count,
    input  logic [7:0]                   begin_bitmap_term_count,
    input  logic [12:0]                  begin_fadc_destination_bytes,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [8:0]                   descriptor_gate_code,
    input  logic [4:0]                   descriptor_lane_id,
    input  logic [7:0]                   descriptor_destination_count,
    input  logic                         descriptor_last,

    input  logic                         destination_valid,
    output logic                         destination_ready,
    input  logic [7:0]                   destination_token_id,
    input  logic                         destination_last_for_term,
    input  logic                         destination_bitmap_valid,
    output logic                         destination_bitmap_ready,
    input  logic [TOKENS-1:0]            destination_bitmap,

    input  logic                         raw_token_valid,
    output logic                         raw_token_ready,
    input  logic [7:0]                   raw_token_id,
    input  logic [GATE_W-1:0]            raw_gate_code,
    input  logic [LANES-1:0]             raw_k_bits,

    output logic                         commit_begin_valid,
    input  logic                         commit_begin_ready,
    output logic [CONTEXT_ID_W-1:0]      commit_context_id,
    output logic [HEAD_ID_W-1:0]         commit_head_id,
    output logic [TAG_W-1:0]             commit_tag,
    output logic                         commit_mode_is_csr,
    output logic [SIZE_W-1:0]            commit_payload_bits,

    output logic                         commit_word_valid,
    input  logic                         commit_word_ready,
    output logic [WORD_W-1:0]            commit_word_data,
    output logic                         commit_word_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_W-1:0]             done_tag,
    output logic [FORMAT_W-1:0]          done_format,
    output logic                         done_error,
    output logic [7:0]                   done_word_count,

    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_heads,
    output logic [COUNTER_W-1:0]         count_committed_heads,
    output logic [COUNTER_W-1:0]         count_aborted_heads,
    output logic [COUNTER_W-1:0]         count_committed_words,
    output logic [COUNTER_W-1:0]         count_input_stall_cycles,
    output logic [COUNTER_W-1:0]         count_output_stall_cycles
);

    localparam int RAW_PAYLOAD_BITS = TOKENS * (LANES + GATE_W);
    localparam int SLOT_CAPACITY_BITS = SLOT_WORDS * WORD_W;
    localparam int BITMAP_BYTES = (TOKENS + 7) / 8;
    localparam int BUFFER_WORDS = 1 << WORD_INDEX_W;
    localparam int BITMAP_PADDED_BITS = BITMAP_BYTES * 8;
    localparam LOOP_BITMAP_BYTE_BITS = 32'd8;
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_HEADER0,
        ST_HEADER1,
        ST_DESCRIPTORS,
        ST_IPD_PAD,
        ST_DESTINATIONS,
        ST_BITMAP_EMIT,
        ST_RAW_TOKENS,
        ST_FINALIZE,
        ST_CHECK,
        ST_COMMIT_BEGIN,
        ST_COMMIT_WORDS,
        ST_DONE
    } state_t;

    state_t state_q;
    // The physical C0 buffer is rounded to the next power-of-two depth. Only
    // SLOT_WORDS entries are legal payload storage and visible to commit.
    logic [WORD_W-1:0] payload_mem [0:BUFFER_WORDS-1];
    logic [23:0] term_mem [0:MAX_TERMS-1];

    logic [CONTEXT_ID_W-1:0] context_q;
    logic [HEAD_ID_W-1:0] head_q;
    logic [TAG_W-1:0] tag_q;
    logic [FORMAT_W-1:0] format_q;
    logic [SIZE_W-1:0] expected_payload_bits_q;
    logic [2:0] active_classes_q;
    logic [7:0] active_tokens_q;
    logic [7:0] term_total_q;
    logic [12:0] event_total_q;
    logic [7:0] bitmap_term_total_q;
    logic [12:0] fadc_destination_bytes_q;

    logic [63:0] pack_buffer_q;
    logic [6:0] pack_count_q;
    logic [7:0] write_word_count_q;
    logic [15:0] total_bits_appended_q;
    logic [TERM_INDEX_W-1:0] descriptor_index_q;
    logic [12:0] descriptor_event_sum_q;
    logic [7:0] descriptor_bitmap_sum_q;
    logic [12:0] descriptor_fadc_destination_sum_q;
    logic [TERM_INDEX_W-1:0] destination_term_index_q;
    logic [7:0] destination_seen_q;
    logic [12:0] destination_event_sum_q;
    logic [7:0] previous_token_q;
    logic [TOKENS-1:0] bitmap_q;
    logic [4:0] bitmap_byte_index_q;
    logic [7:0] bitmap_popcount_q;
    logic [7:0] raw_token_count_q;
    logic [WORD_INDEX_W-1:0] commit_word_index_q;
    logic session_error_q;

    logic begin_fire;
    logic descriptor_fire;
    logic destination_fire;
    logic destination_bitmap_fire;
    logic raw_token_fire;
    logic commit_begin_fire;
    logic commit_word_fire;
    logic done_fire;
    logic append_valid;
    logic append_ready;
    logic append_fire;
    logic [63:0] append_data;
    logic [6:0] append_width;
    logic [7:0] append_total_bits;
    logic append_writes_word;
    logic [127:0] append_combined;
    logic flush_valid;
    logic flush_ready;
    logic flush_fire;
    logic [63:0] header_word0;
    logic [63:0] header_word1;
    logic [31:0] descriptor_ipd;
    logic [23:0] descriptor_fadc;
    logic descriptor_bitmap_mode;
    logic descriptor_expected_last;
    logic [13:0] descriptor_event_sum_ext;
    logic [13:0] descriptor_fadc_destination_sum_ext;
    logic [8:0] descriptor_bitmap_sum_ext;
    logic [7:0] selected_destination_count;
    logic selected_bitmap_mode;
    logic destination_expected_last;
    logic destination_token_in_range;
    logic destination_token_ordered;
    logic [BITMAP_PADDED_BITS-1:0] bitmap_padded;
    logic [7:0] current_bitmap_byte;
    logic [3:0] current_bitmap_byte_popcount;
    logic [8:0] bitmap_popcount_sum_ext;
    logic [31:0] ipd_payload_bits_comb;
    logic [31:0] fadc_payload_bits_comb;
    logic [31:0] expected_word_count_comb;
    logic begin_contract_ok;
    logic check_contract_ok;
    logic input_waiting;

    assign begin_ready = state_q == ST_IDLE;
    assign begin_fire = begin_valid && begin_ready;
    assign descriptor_fire = descriptor_valid && descriptor_ready;
    assign destination_fire = destination_valid && destination_ready;
    assign destination_bitmap_fire =
        destination_bitmap_valid && destination_bitmap_ready;
    assign raw_token_fire = raw_token_valid && raw_token_ready;
    assign commit_begin_fire = commit_begin_valid && commit_begin_ready;
    assign commit_word_fire = commit_word_valid && commit_word_ready;
    assign done_fire = done_valid && done_ready;

    assign ipd_payload_bits_comb =
        (32'd16 + ((((32'(begin_term_count) + 1) >> 1)) << 3) +
         32'(begin_event_count)) << 3;
    assign fadc_payload_bits_comb =
        (32'd16 + 32'(begin_term_count) + (32'(begin_term_count) << 1) +
         32'(begin_fadc_destination_bytes)) << 3;

    always_comb begin
        begin_contract_ok =
            (begin_format == FORMAT_RAW || begin_format == FORMAT_IPD32W ||
             begin_format == FORMAT_FADC24) &&
            32'(begin_context_id) < CONTEXTS &&
            32'(begin_head_id) < HEADS;
        if (begin_format == FORMAT_RAW) begin
            begin_contract_ok = begin_contract_ok &&
                32'(begin_expected_payload_bits) == RAW_PAYLOAD_BITS;
        end else begin
            begin_contract_ok = begin_contract_ok &&
                32'(begin_term_count) <= MAX_TERMS &&
                ((begin_term_count == 0) == (begin_event_count == 0)) &&
                32'(begin_active_tokens) <= TOKENS &&
                32'(begin_expected_payload_bits) <= SLOT_CAPACITY_BITS &&
                begin_expected_payload_bits[2:0] == 3'd0;
            if (begin_format == FORMAT_IPD32W) begin
                begin_contract_ok = begin_contract_ok &&
                    32'(begin_active_classes) <= 7 &&
                    32'(begin_expected_payload_bits) == ipd_payload_bits_comb;
            end else begin
                begin_contract_ok = begin_contract_ok &&
                    32'(begin_bitmap_term_count) <=
                        32'(begin_term_count) &&
                    32'(begin_expected_payload_bits) ==
                        fadc_payload_bits_comb;
            end
        end
    end

    always_comb begin
        header_word0 = '0;
        header_word1 = '0;
        if (format_q == FORMAT_IPD32W) begin
            header_word0 = 64'(16'h4753) | (64'(1) << 16) |
                           (64'(1) << 20) | (64'(tag_q) << 32);
            header_word1 = 64'(expected_payload_bits_q) |
                           (64'(term_total_q) << 13) |
                           (64'(event_total_q) << 21) |
                           (64'(active_classes_q) << 34) |
                           (64'(active_tokens_q) << 37) |
                           ((64'd16 +
                             (((64'(term_total_q) + 1) >> 1) << 3)) << 45);
        end else begin
            header_word0 = 64'(16'h4641) | (64'(1) << 16) |
                           (64'(tag_q) << 32);
            header_word1 = (64'(expected_payload_bits_q) >> 3) |
                           (64'(term_total_q) << 16) |
                           (64'(event_total_q) << 24) |
                           (64'(bitmap_term_total_q) << 37) |
                           ((64'd16 + 64'(term_total_q) +
                             (64'(term_total_q) << 1)) << 45);
        end
    end

    assign descriptor_bitmap_mode =
        32'(descriptor_destination_count) > BITMAP_BYTES;
    assign descriptor_ipd = 32'(descriptor_gate_code) |
                            (32'(descriptor_lane_id) << 9) |
                            (32'(descriptor_destination_count) << 14);
    assign descriptor_fadc = 24'(descriptor_gate_code) |
                             (24'(descriptor_lane_id) << 9) |
                             (24'(descriptor_destination_count) << 14) |
                             (24'(descriptor_bitmap_mode) << 22);
    assign descriptor_expected_last =
        32'(descriptor_index_q) + 1 == 32'(term_total_q);
    assign descriptor_event_sum_ext =
        14'(descriptor_event_sum_q) +
        14'(descriptor_destination_count);
    assign descriptor_fadc_destination_sum_ext =
        14'(descriptor_fadc_destination_sum_q) +
        (descriptor_bitmap_mode ? 14'(BITMAP_BYTES) :
                                  14'(descriptor_destination_count));
    assign descriptor_bitmap_sum_ext =
        9'(descriptor_bitmap_sum_q) + 9'(descriptor_bitmap_mode);

    assign selected_destination_count =
        term_mem[destination_term_index_q][21:14];
    assign selected_bitmap_mode =
        term_mem[destination_term_index_q][22];
    assign destination_expected_last =
        32'(destination_seen_q) + 1 == 32'(selected_destination_count);
    assign destination_token_in_range =
        32'(destination_token_id) < TOKENS;
    assign destination_token_ordered = destination_seen_q == 0 ||
        32'(destination_token_id) > 32'(previous_token_q);

    always_comb begin
        bitmap_padded = '0;
        bitmap_padded[TOKENS-1:0] = bitmap_q;
        current_bitmap_byte = bitmap_padded[
            32'(bitmap_byte_index_q) * 8 +: 8];
        current_bitmap_byte_popcount = '0;
        for (int bit_index = 32'd0;
             bit_index < LOOP_BITMAP_BYTE_BITS;
             bit_index = bit_index + 32'd1)
            current_bitmap_byte_popcount = current_bitmap_byte_popcount +
                4'(current_bitmap_byte[bit_index]);
    end
    assign bitmap_popcount_sum_ext = 9'(bitmap_popcount_q) +
        9'(current_bitmap_byte_popcount);

    always_comb begin
        append_valid = 1'b0;
        append_data = '0;
        append_width = '0;
        if (state_q == ST_HEADER0) begin
            append_valid = 1'b1;
            append_data = header_word0;
            append_width = 7'd64;
        end else if (state_q == ST_HEADER1) begin
            append_valid = 1'b1;
            append_data = header_word1;
            append_width = 7'd64;
        end else if (state_q == ST_DESCRIPTORS && descriptor_valid) begin
            append_valid = 1'b1;
            append_data = (format_q == FORMAT_IPD32W) ?
                          64'(descriptor_ipd) : 64'(descriptor_fadc);
            append_width = (format_q == FORMAT_IPD32W) ? 7'd32 : 7'd24;
        end else if (state_q == ST_IPD_PAD) begin
            append_valid = 1'b1;
            append_data = '0;
            append_width = 7'd32;
        end else if (state_q == ST_DESTINATIONS && destination_valid &&
                     !selected_bitmap_mode) begin
            append_valid = 1'b1;
            append_data = 64'(destination_token_id);
            append_width = 7'd8;
        end else if (state_q == ST_BITMAP_EMIT) begin
            append_valid = 1'b1;
            append_data = 64'(current_bitmap_byte);
            append_width = 7'd8;
        end else if (state_q == ST_RAW_TOKENS && raw_token_valid) begin
            append_valid = 1'b1;
            append_data = 64'({raw_gate_code, raw_k_bits});
            append_width = 7'(LANES + GATE_W);
        end
    end

    assign append_total_bits = 8'(pack_count_q) + 8'(append_width);
    assign append_writes_word = 32'(append_total_bits) >= WORD_W;
    assign append_ready = !append_writes_word ||
                          32'(write_word_count_q) < SLOT_WORDS;
    assign append_fire = append_valid && append_ready;
    assign append_combined =
        ({64'd0, append_data} << 32'(pack_count_q)) |
        128'(pack_buffer_q);
    assign flush_valid = state_q == ST_FINALIZE && pack_count_q != 7'd0;
    assign flush_ready = 32'(write_word_count_q) < SLOT_WORDS;
    assign flush_fire = flush_valid && flush_ready;

    assign descriptor_ready = state_q == ST_DESCRIPTORS && append_ready;
    assign destination_ready = state_q == ST_DESTINATIONS &&
        (!selected_bitmap_mode || BITMAP_BYPASS_ENABLE == 32'd0) &&
        (selected_bitmap_mode || append_ready);
    assign destination_bitmap_ready = state_q == ST_DESTINATIONS &&
        selected_bitmap_mode && BITMAP_BYPASS_ENABLE != 32'd0;
    assign raw_token_ready = state_q == ST_RAW_TOKENS && append_ready;

    assign commit_begin_valid = state_q == ST_COMMIT_BEGIN;
    assign commit_context_id = context_q;
    assign commit_head_id = head_q;
    assign commit_tag = tag_q;
    assign commit_mode_is_csr = format_q != FORMAT_RAW;
    assign commit_payload_bits = expected_payload_bits_q;
    assign commit_word_valid = state_q == ST_COMMIT_WORDS;
    assign commit_word_data = payload_mem[commit_word_index_q];
    assign commit_word_last = commit_word_valid &&
        32'(commit_word_index_q) + 1 == 32'(write_word_count_q);

    assign done_valid = state_q == ST_DONE;
    assign done_tag = tag_q;
    assign done_format = format_q;
    assign done_error = session_error_q;
    assign done_word_count = write_word_count_q;

    assign expected_word_count_comb =
        (32'(expected_payload_bits_q) + WORD_W - 1) / WORD_W;
    always_comb begin
        check_contract_ok =
            !session_error_q &&
            32'(total_bits_appended_q) ==
                32'(expected_payload_bits_q) &&
            32'(write_word_count_q) == expected_word_count_comb;
        if (format_q == FORMAT_RAW) begin
            check_contract_ok = check_contract_ok &&
                32'(raw_token_count_q) == TOKENS;
        end else begin
            check_contract_ok = check_contract_ok &&
                descriptor_event_sum_q == event_total_q &&
                destination_event_sum_q == event_total_q;
            if (format_q == FORMAT_FADC24) begin
                check_contract_ok = check_contract_ok &&
                    descriptor_bitmap_sum_q == bitmap_term_total_q &&
                    descriptor_fadc_destination_sum_q ==
                        fadc_destination_bytes_q;
            end
        end
    end

    assign input_waiting =
        (state_q == ST_DESCRIPTORS && descriptor_valid && !descriptor_ready) ||
        (state_q == ST_DESTINATIONS && destination_valid &&
         !destination_ready) ||
        (state_q == ST_DESTINATIONS && destination_bitmap_valid &&
         !destination_bitmap_ready) ||
        (state_q == ST_RAW_TOKENS && raw_token_valid && !raw_token_ready);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pack_buffer_q <= '0;
            pack_count_q <= '0;
            write_word_count_q <= '0;
            total_bits_appended_q <= '0;
        end else if (begin_fire) begin
            pack_buffer_q <= '0;
            pack_count_q <= '0;
            write_word_count_q <= '0;
            total_bits_appended_q <= '0;
        end else if (append_fire) begin
            total_bits_appended_q <= total_bits_appended_q +
                                     16'(append_width);
            if (append_writes_word) begin
                payload_mem[WORD_INDEX_W'(write_word_count_q)] <=
                    append_combined[63:0];
                write_word_count_q <= write_word_count_q + 1'b1;
                pack_buffer_q <= append_combined[127:64];
                pack_count_q <= 7'(32'(append_total_bits) - WORD_W);
            end else begin
                pack_buffer_q <= append_combined[63:0];
                pack_count_q <= 7'(append_total_bits);
            end
        end else if (flush_fire) begin
            payload_mem[WORD_INDEX_W'(write_word_count_q)] <= pack_buffer_q;
            write_word_count_q <= write_word_count_q + 1'b1;
            pack_buffer_q <= '0;
            pack_count_q <= '0;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            context_q <= '0;
            head_q <= '0;
            tag_q <= '0;
            format_q <= FORMAT_RAW;
            expected_payload_bits_q <= '0;
            active_classes_q <= '0;
            active_tokens_q <= '0;
            term_total_q <= '0;
            event_total_q <= '0;
            bitmap_term_total_q <= '0;
            fadc_destination_bytes_q <= '0;
            descriptor_index_q <= '0;
            descriptor_event_sum_q <= '0;
            descriptor_bitmap_sum_q <= '0;
            descriptor_fadc_destination_sum_q <= '0;
            destination_term_index_q <= '0;
            destination_seen_q <= '0;
            destination_event_sum_q <= '0;
            previous_token_q <= '0;
            bitmap_q <= '0;
            bitmap_byte_index_q <= '0;
            bitmap_popcount_q <= '0;
            raw_token_count_q <= '0;
            commit_word_index_q <= '0;
            session_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_heads <= '0;
            count_committed_heads <= '0;
            count_aborted_heads <= '0;
            count_committed_words <= '0;
            count_input_stall_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (input_waiting)
                count_input_stall_cycles <= count_input_stall_cycles + 1'b1;
            if ((commit_begin_valid && !commit_begin_ready) ||
                (commit_word_valid && !commit_word_ready))
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;

            case (state_q)
                ST_IDLE: begin
                    if (begin_fire) begin
                        context_q <= begin_context_id;
                        head_q <= begin_head_id;
                        tag_q <= begin_tag;
                        format_q <= begin_format;
                        expected_payload_bits_q <= begin_expected_payload_bits;
                        active_classes_q <= begin_active_classes[2:0];
                        active_tokens_q <= begin_active_tokens;
                        term_total_q <= begin_term_count;
                        event_total_q <= begin_event_count;
                        bitmap_term_total_q <= begin_bitmap_term_count;
                        fadc_destination_bytes_q <=
                            begin_fadc_destination_bytes;
                        descriptor_index_q <= '0;
                        descriptor_event_sum_q <= '0;
                        descriptor_bitmap_sum_q <= '0;
                        descriptor_fadc_destination_sum_q <= '0;
                        destination_term_index_q <= '0;
                        destination_seen_q <= '0;
                        destination_event_sum_q <= '0;
                        previous_token_q <= '0;
                        bitmap_q <= '0;
                        bitmap_byte_index_q <= '0;
                        bitmap_popcount_q <= '0;
                        raw_token_count_q <= '0;
                        commit_word_index_q <= '0;
                        session_error_q <= !begin_contract_ok;
                        count_heads <= count_heads + 1'b1;
                        if (!begin_contract_ok) begin
                            state_q <= ST_DONE;
                            protocol_error <= 1'b1;
                            count_aborted_heads <= count_aborted_heads + 1'b1;
                        end else if (begin_format == FORMAT_RAW) begin
                            state_q <= ST_RAW_TOKENS;
                        end else begin
                            state_q <= ST_HEADER0;
                        end
                    end
                end

                ST_HEADER0: begin
                    if (append_fire)
                        state_q <= ST_HEADER1;
                end

                ST_HEADER1: begin
                    if (append_fire) begin
                        if (term_total_q == 0)
                            state_q <= ST_FINALIZE;
                        else
                            state_q <= ST_DESCRIPTORS;
                    end
                end

                ST_DESCRIPTORS: begin
                    if (descriptor_fire) begin
                        if (descriptor_destination_count == 0 ||
                            32'(descriptor_lane_id) >= LANES ||
                            descriptor_last != descriptor_expected_last ||
                            descriptor_event_sum_ext[13] ||
                            descriptor_fadc_destination_sum_ext[13] ||
                            descriptor_bitmap_sum_ext[8]) begin
                            state_q <= ST_DONE;
                            session_error_q <= 1'b1;
                            protocol_error <= 1'b1;
                            count_aborted_heads <= count_aborted_heads + 1'b1;
                        end else begin
                            term_mem[descriptor_index_q] <=
                                (format_q == FORMAT_FADC24) ?
                                descriptor_fadc : 24'(descriptor_ipd);
                            descriptor_event_sum_q <=
                                descriptor_event_sum_ext[12:0];
                            descriptor_fadc_destination_sum_q <=
                                descriptor_fadc_destination_sum_ext[12:0];
                            descriptor_bitmap_sum_q <=
                                descriptor_bitmap_sum_ext[7:0];
                            if (descriptor_expected_last) begin
                                destination_term_index_q <= '0;
                                destination_seen_q <= '0;
                                destination_event_sum_q <= '0;
                                previous_token_q <= '0;
                                if (format_q == FORMAT_IPD32W &&
                                    term_total_q[0])
                                    state_q <= ST_IPD_PAD;
                                else
                                    state_q <= ST_DESTINATIONS;
                            end else begin
                                descriptor_index_q <= descriptor_index_q + 1'b1;
                            end
                        end
                    end
                end

                ST_IPD_PAD: begin
                    if (append_fire)
                        state_q <= ST_DESTINATIONS;
                end

                ST_DESTINATIONS: begin
                    if (destination_bitmap_fire) begin
                        bitmap_q <= destination_bitmap;
                        bitmap_byte_index_q <= '0;
                        bitmap_popcount_q <= '0;
                        destination_event_sum_q <=
                            destination_event_sum_q +
                            13'(selected_destination_count);
                        destination_seen_q <= '0;
                        previous_token_q <= '0;
                        state_q <= ST_BITMAP_EMIT;
                    end else if (destination_fire) begin
                        if (!destination_token_in_range ||
                            !destination_token_ordered ||
                            destination_last_for_term !=
                                destination_expected_last) begin
                            state_q <= ST_DONE;
                            session_error_q <= 1'b1;
                            protocol_error <= 1'b1;
                            count_aborted_heads <= count_aborted_heads + 1'b1;
                        end else begin
                            destination_event_sum_q <=
                                destination_event_sum_q + 1'b1;
                            previous_token_q <= destination_token_id;
                            if (selected_bitmap_mode)
                                bitmap_q[TOKEN_ID_W'(
                                    destination_token_id)] <= 1'b1;
                            if (destination_expected_last) begin
                                destination_seen_q <= '0;
                                previous_token_q <= '0;
                                if (selected_bitmap_mode) begin
                                    bitmap_byte_index_q <= '0;
                                    state_q <= ST_BITMAP_EMIT;
                                end else if (32'(destination_term_index_q) + 1 ==
                                             32'(term_total_q)) begin
                                    state_q <= ST_FINALIZE;
                                end else begin
                                    destination_term_index_q <=
                                        destination_term_index_q + 1'b1;
                                end
                            end else begin
                                destination_seen_q <= destination_seen_q + 1'b1;
                            end
                        end
                    end
                end

                ST_BITMAP_EMIT: begin
                    if (append_fire) begin
                        if (BITMAP_BYPASS_ENABLE != 32'd0)
                            bitmap_popcount_q <= bitmap_popcount_sum_ext[7:0];
                        if (32'(bitmap_byte_index_q) + 1 == BITMAP_BYTES) begin
                            bitmap_q <= '0;
                            bitmap_byte_index_q <= '0;
                            bitmap_popcount_q <= '0;
                            if (BITMAP_BYPASS_ENABLE != 32'd0 &&
                                bitmap_popcount_sum_ext !=
                                9'(selected_destination_count)) begin
                                state_q <= ST_DONE;
                                session_error_q <= 1'b1;
                                protocol_error <= 1'b1;
                                count_aborted_heads <=
                                    count_aborted_heads + 1'b1;
                            end else if (32'(destination_term_index_q) + 1 ==
                                32'(term_total_q)) begin
                                state_q <= ST_FINALIZE;
                            end else begin
                                destination_term_index_q <=
                                    destination_term_index_q + 1'b1;
                                state_q <= ST_DESTINATIONS;
                            end
                        end else begin
                            bitmap_byte_index_q <= bitmap_byte_index_q + 1'b1;
                        end
                    end
                end

                ST_RAW_TOKENS: begin
                    if (raw_token_fire) begin
                        if (32'(raw_token_id) != 32'(raw_token_count_q)) begin
                            state_q <= ST_DONE;
                            session_error_q <= 1'b1;
                            protocol_error <= 1'b1;
                            count_aborted_heads <= count_aborted_heads + 1'b1;
                        end else if (32'(raw_token_count_q) + 1 == TOKENS) begin
                            raw_token_count_q <= raw_token_count_q + 1'b1;
                            state_q <= ST_FINALIZE;
                        end else begin
                            raw_token_count_q <= raw_token_count_q + 1'b1;
                        end
                    end
                end

                ST_FINALIZE: begin
                    if (pack_count_q == 0 || flush_fire)
                        state_q <= ST_CHECK;
                    else if (!flush_ready) begin
                        state_q <= ST_DONE;
                        session_error_q <= 1'b1;
                        protocol_error <= 1'b1;
                        count_aborted_heads <= count_aborted_heads + 1'b1;
                    end
                end

                ST_CHECK: begin
                    if (check_contract_ok) begin
                        state_q <= ST_COMMIT_BEGIN;
                    end else begin
                        state_q <= ST_DONE;
                        session_error_q <= 1'b1;
                        protocol_error <= 1'b1;
                        count_aborted_heads <= count_aborted_heads + 1'b1;
                    end
                end

                ST_COMMIT_BEGIN: begin
                    if (commit_begin_fire) begin
                        commit_word_index_q <= '0;
                        state_q <= ST_COMMIT_WORDS;
                    end
                end

                ST_COMMIT_WORDS: begin
                    if (commit_word_fire) begin
                        count_committed_words <= count_committed_words + 1'b1;
                        if (commit_word_last) begin
                            state_q <= ST_DONE;
                            count_committed_heads <= count_committed_heads + 1'b1;
                        end else begin
                            commit_word_index_q <= commit_word_index_q + 1'b1;
                        end
                    end
                end

                ST_DONE: begin
                    if (done_fire)
                        state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_IDLE;
                    session_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
