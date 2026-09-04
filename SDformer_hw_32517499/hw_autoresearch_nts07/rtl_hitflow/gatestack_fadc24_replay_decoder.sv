`timescale 1ns/1ps
`default_nettype none

// FADC24 leaf decoder. A head payload is buffered first, then each 24-bit
// descriptor selects either an 8-bit token list or a 162-bit token bitmap.
// This first implementation freezes exact format/protocol behavior; streaming
// overlap is intentionally left for a later microarchitecture iteration.
module gatestack_fadc24_replay_decoder #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int MAX_TERMS       = 128,
    parameter int EVENT_WAYS      = 4,
    parameter int SLOT_WORDS      = 104,
    parameter int TAG_W           = 32,
    parameter int WORD_INDEX_W    = 7,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = 8,
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int TERM_INDEX_W    = (MAX_TERMS <= 1) ? 1 : $clog2(MAX_TERMS),
    parameter int EVENT_COUNT_W   = 13,
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1)
) (
    input  logic                               clk_core,
    input  logic                               rst_core,

    input  logic                               start_valid,
    output logic                               start_ready,

    input  logic                               word_valid,
    output logic                               word_ready,
    input  logic [63:0]                        word_data,
    input  logic [WORD_INDEX_W-1:0]            word_index,
    input  logic                               word_last,

    output logic                               descriptor_begin_valid,
    input  logic                               descriptor_begin_ready,
    output logic [TAG_W-1:0]                   descriptor_begin_tag,
    output logic [7:0]                         descriptor_begin_term_count,

    output logic                               term_valid,
    input  logic                               term_ready,
    output logic [8:0]                         term_gate_code,
    output logic [LANE_ID_W-1:0]               term_lane_id,
    output logic [7:0]                         term_destination_count,
    output logic [TERM_INDEX_W-1:0]            term_index,
    output logic                               term_head_last,

    output logic                               event_valid,
    input  logic                               event_ready,
    output logic [8:0]                         event_gate_code,
    output logic [LANE_ID_W-1:0]               event_lane_id,
    output logic [EVENT_WAYS-1:0]              event_token_valid,
    output logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    output logic [WAY_COUNT_W-1:0]             event_count,
    output logic                               event_term_first,
    output logic                               event_term_last,
    output logic                               event_head_last,

    output logic                               done_valid,
    input  logic                               done_ready,
    output logic [TAG_W-1:0]                   done_tag,
    output logic                               done_error,

    output logic                               protocol_error,
    output logic [COUNTER_W-1:0]               count_heads,
    output logic [COUNTER_W-1:0]               count_terms,
    output logic [COUNTER_W-1:0]               count_events,
    output logic [COUNTER_W-1:0]               count_bitmap_terms,
    output logic [COUNTER_W-1:0]               count_input_stall_cycles,
    output logic [COUNTER_W-1:0]               count_term_stall_cycles,
    output logic [COUNTER_W-1:0]               count_output_stall_cycles
);
    localparam logic [15:0] MAGIC = 16'h4641;
    localparam logic [7:0] VERSION = 8'd1;
    localparam int HEADER_BYTES = 32'd16;
    localparam int DESCRIPTOR_BYTES = 32'd3;
    localparam int BITMAP_BYTES = 32'd21;
    localparam int SLOT_BYTES = SLOT_WORDS * 8;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD,
        ST_DESCRIPTOR_BEGIN,
        ST_LOAD_TERM,
        ST_EVENTS,
        ST_ERROR_DRAIN,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [63:0] payload_mem [0:SLOT_WORDS-1];
    logic [TAG_W-1:0] tag_q;
    logic [15:0] payload_bytes_q;
    logic [7:0] term_total_q;
    logic [EVENT_COUNT_W-1:0] event_total_q;
    logic [7:0] bitmap_term_total_q;
    logic [10:0] destination_offset_q;
    logic [WORD_INDEX_W-1:0] expected_word_index_q;
    logic [TERM_INDEX_W-1:0] term_index_q;
    logic [10:0] destination_cursor_q;
    logic [10:0] current_data_offset_q;
    logic [8:0] current_gate_q;
    logic [LANE_ID_W-1:0] current_lane_q;
    logic [7:0] current_remaining_q;
    logic [7:0] current_emitted_q;
    logic current_bitmap_mode_q;
    logic [167:0] current_bitmap_q;
    logic [EVENT_COUNT_W-1:0] events_emitted_q;
    logic [7:0] bitmap_terms_seen_q;
    logic session_error_q;

    logic word_fire, descriptor_begin_fire, term_fire, event_fire, done_fire;
    logic [23:0] selected_descriptor;
    logic [10:0] selected_descriptor_base;
    logic [10:0] selected_descriptor_byte0_index;
    logic [10:0] selected_descriptor_byte1_index;
    logic [10:0] selected_descriptor_byte2_index;
    logic [7:0] selected_descriptor_byte0;
    logic [7:0] selected_descriptor_byte1;
    logic [7:0] selected_descriptor_byte2;
    logic [8:0] selected_gate;
    logic [4:0] selected_lane;
    logic [7:0] selected_count;
    logic selected_bitmap_mode;
    logic selected_descriptor_valid;
    logic [10:0] selected_encoded_bytes;
    logic [11:0] selected_data_end;
    logic [167:0] selected_bitmap;
    logic [8:0] selected_bitmap_popcount;
    logic selected_bitmap_valid;
    logic [EVENT_WAYS-1:0] event_token_valid_comb;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids_comb;
    logic [WAY_COUNT_W-1:0] event_count_comb;
    logic event_tokens_in_range;
    logic [167:0] bitmap_after_emit;
    integer bitmap_selected_comb;
    logic [15:0] header_payload_bytes;
    logic [7:0] header_term_count;
    logic [EVENT_COUNT_W-1:0] header_event_count;
    logic [7:0] header_bitmap_terms;
    logic [10:0] header_destination_offset;
    logic [15:0] header_expected_words;
    logic header_contract_ok;

    assign start_ready = state_q == ST_IDLE;
    assign word_ready = state_q == ST_LOAD || state_q == ST_ERROR_DRAIN;
    assign done_valid = state_q == ST_DONE;
    assign done_tag = tag_q;
    assign done_error = session_error_q;
    assign word_fire = word_valid && word_ready;
    assign descriptor_begin_fire = descriptor_begin_valid &&
                                   descriptor_begin_ready;
    assign term_fire = term_valid && term_ready;
    assign event_fire = event_valid && event_ready;
    assign done_fire = done_valid && done_ready;

    assign descriptor_begin_valid = state_q == ST_DESCRIPTOR_BEGIN;
    assign descriptor_begin_tag = tag_q;
    assign descriptor_begin_term_count = term_total_q;

    assign header_payload_bytes = word_data[15:0];
    assign header_term_count = word_data[23:16];
    assign header_event_count = word_data[36:24];
    assign header_bitmap_terms = word_data[44:37];
    assign header_destination_offset = word_data[55:45];
    assign header_expected_words = (header_payload_bytes + 7) >> 3;
    assign header_contract_ok = word_data[63:56] == 0 &&
        32'(header_payload_bytes) >= HEADER_BYTES &&
        32'(header_payload_bytes) <= SLOT_BYTES &&
        32'(header_term_count) <= MAX_TERMS &&
        32'(header_bitmap_terms) <= 32'(header_term_count) &&
        32'(header_destination_offset) ==
            HEADER_BYTES + DESCRIPTOR_BYTES * 32'(header_term_count) &&
        ((header_term_count == 0) == (header_event_count == 0)) &&
        ((header_term_count == 0) == (header_bitmap_terms == 0));

    assign selected_descriptor_base = 11'(
        32'(HEADER_BYTES) + 32'(term_index_q) * 32'(DESCRIPTOR_BYTES));
    assign selected_descriptor_byte0_index = selected_descriptor_base;
    assign selected_descriptor_byte1_index = selected_descriptor_base + 1'b1;
    assign selected_descriptor_byte2_index = selected_descriptor_base + 11'd2;
    assign selected_descriptor_byte0 =
        8'(payload_mem[WORD_INDEX_W'(selected_descriptor_byte0_index >> 3)] >>
           (32'(selected_descriptor_byte0_index[2:0]) * 32'd8));
    assign selected_descriptor_byte1 =
        8'(payload_mem[WORD_INDEX_W'(selected_descriptor_byte1_index >> 3)] >>
           (32'(selected_descriptor_byte1_index[2:0]) * 32'd8));
    assign selected_descriptor_byte2 =
        8'(payload_mem[WORD_INDEX_W'(selected_descriptor_byte2_index >> 3)] >>
           (32'(selected_descriptor_byte2_index[2:0]) * 32'd8));
    assign selected_descriptor = {selected_descriptor_byte2,
                                  selected_descriptor_byte1,
                                  selected_descriptor_byte0};
    assign selected_gate = selected_descriptor[8:0];
    assign selected_lane = selected_descriptor[13:9];
    assign selected_count = selected_descriptor[21:14];
    assign selected_bitmap_mode = selected_descriptor[22];
    assign selected_encoded_bytes = selected_bitmap_mode ?
                                    11'(BITMAP_BYTES) : 11'(selected_count);
    assign selected_data_end = 12'(destination_cursor_q) +
                               12'(selected_encoded_bytes);
    assign selected_descriptor_valid = !selected_descriptor[23] &&
        selected_count != 0 && 32'(selected_lane) < LANES &&
        16'(selected_data_end) <= payload_bytes_q;
    always_comb begin
        selected_bitmap = '0;
        for (int byte_lane = 0; byte_lane < 32'd21; byte_lane = byte_lane + 1) begin
            selected_bitmap[(byte_lane*8) +: 8] =
                8'(payload_mem[WORD_INDEX_W'(
                    (32'(destination_cursor_q) + 32'(byte_lane)) >> 3)] >>
                    (((32'(destination_cursor_q) + 32'(byte_lane)) & 32'd7) *
                     32'd8));
        end
    end

    always_comb begin
        selected_bitmap_popcount = '0;
        for (int bit_index = 0; bit_index < 32'd162; bit_index = bit_index + 1)
            selected_bitmap_popcount = selected_bitmap_popcount +
                                       9'(selected_bitmap[bit_index]);
    end
    assign selected_bitmap_valid = selected_bitmap[167:162] == '0 &&
        selected_bitmap_popcount == 9'(selected_count);

    always_comb begin
        term_valid = 1'b0;
        term_gate_code = selected_gate;
        term_lane_id = LANE_ID_W'(selected_lane);
        term_destination_count = selected_count;
        term_index = term_index_q;
        term_head_last = 32'(term_index_q) + 1 == 32'(term_total_q);
        if (state_q == ST_LOAD_TERM && selected_descriptor_valid &&
            (!selected_bitmap_mode || selected_bitmap_valid)) begin
            term_valid = 1'b1;
        end
    end

    always_comb begin
        event_token_valid_comb = '0;
        event_token_ids_comb = '0;
        event_count_comb = '0;
        bitmap_after_emit = current_bitmap_q;
        bitmap_selected_comb = 32'd0;
        if (current_bitmap_mode_q) begin
            for (int token = 0; token < 32'd162; token = token + 1) begin
                if (current_bitmap_q[token] &&
                    bitmap_selected_comb < 32'(EVENT_WAYS)) begin
                    event_token_valid_comb[bitmap_selected_comb] = 1'b1;
                    event_token_ids_comb[(bitmap_selected_comb*TOKEN_ID_W) +:
                                         TOKEN_ID_W] =
                        TOKEN_ID_W'(token);
                    bitmap_after_emit[token] = 1'b0;
                    bitmap_selected_comb = bitmap_selected_comb + 1;
                end
            end
            event_count_comb = WAY_COUNT_W'(bitmap_selected_comb);
        end else begin
            for (int way = 0; way < 32'd4; way = way + 1) begin
                if (32'(way) < current_remaining_q) begin
                    event_token_valid_comb[way] = 1'b1;
                    event_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W] =
                        8'(payload_mem[WORD_INDEX_W'(
                            (32'(current_data_offset_q) +
                             32'(current_emitted_q) + 32'(way)) >> 3)] >>
                            (((32'(current_data_offset_q) +
                               32'(current_emitted_q) + 32'(way)) & 32'd7) *
                             32'd8));
                    event_count_comb = event_count_comb + 1'b1;
                end
            end
        end
    end

    always_comb begin
        event_tokens_in_range = 1'b1;
        for (int way = 0; way < 32'd4; way = way + 1) begin
            if (event_token_valid_comb[way] &&
                32'(event_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W]) >=
                TOKENS) begin
                event_tokens_in_range = 1'b0;
            end
        end
    end

    assign event_valid = state_q == ST_EVENTS &&
                         event_count_comb != 0 && event_tokens_in_range;
    assign event_gate_code = current_gate_q;
    assign event_lane_id = current_lane_q;
    assign event_token_valid = event_token_valid_comb;
    assign event_token_ids = event_token_ids_comb;
    assign event_count = event_count_comb;
    assign event_term_first = current_emitted_q == 8'd0;
    assign event_term_last = 8'(event_count_comb) == current_remaining_q;
    assign event_head_last = event_term_last &&
        (32'(term_index_q) + 1 == 32'(term_total_q));

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            payload_bytes_q <= '0;
            term_total_q <= '0;
            event_total_q <= '0;
            bitmap_term_total_q <= '0;
            destination_offset_q <= '0;
            expected_word_index_q <= '0;
            term_index_q <= '0;
            destination_cursor_q <= '0;
            current_data_offset_q <= '0;
            current_gate_q <= '0;
            current_lane_q <= '0;
            current_remaining_q <= '0;
            current_emitted_q <= '0;
            current_bitmap_mode_q <= 1'b0;
            current_bitmap_q <= '0;
            events_emitted_q <= '0;
            bitmap_terms_seen_q <= '0;
            session_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_heads <= '0;
            count_terms <= '0;
            count_events <= '0;
            count_bitmap_terms <= '0;
            count_input_stall_cycles <= '0;
            count_term_stall_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (start_valid && start_ready) begin
                state_q <= ST_LOAD;
                tag_q <= '0;
                payload_bytes_q <= '0;
                term_total_q <= '0;
                event_total_q <= '0;
                bitmap_term_total_q <= '0;
                destination_offset_q <= '0;
                expected_word_index_q <= '0;
                term_index_q <= '0;
                destination_cursor_q <= '0;
                events_emitted_q <= '0;
                bitmap_terms_seen_q <= '0;
                session_error_q <= 1'b0;
            end

            unique case (state_q)
                ST_IDLE: begin
                    // The start handshake owns the transition to ST_LOAD.
                end

                ST_LOAD: begin
                    if (word_fire) begin
                        if (32'(word_index) >= SLOT_WORDS ||
                            word_index != expected_word_index_q) begin
                            protocol_error <= 1'b1;
                            session_error_q <= 1'b1;
                            state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            payload_mem[word_index] <= word_data;
                            expected_word_index_q <= expected_word_index_q + 1'b1;
                            if (word_index == 0) begin
                                tag_q <= TAG_W'(word_data[63:32]);
                                if (word_data[15:0] != MAGIC ||
                                    word_data[23:16] != VERSION ||
                                    word_data[31:24] != 0 || word_last) begin
                                    protocol_error <= 1'b1;
                                    session_error_q <= 1'b1;
                                    state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                                end
                            end else if (word_index == 1) begin
                                payload_bytes_q <= header_payload_bytes;
                                term_total_q <= header_term_count;
                                event_total_q <= header_event_count;
                                bitmap_term_total_q <= header_bitmap_terms;
                                destination_offset_q <= header_destination_offset;
                                if (!header_contract_ok ||
                                    word_last != (header_expected_words == 2)) begin
                                    protocol_error <= 1'b1;
                                    session_error_q <= 1'b1;
                                    state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                                end else if (word_last) begin
                                    state_q <= ST_DESCRIPTOR_BEGIN;
                                end
                            end else if (word_last) begin
                                if (32'(word_index) + 1 !=
                                    ((32'(payload_bytes_q) + 7) >> 3)) begin
                                    protocol_error <= 1'b1;
                                    session_error_q <= 1'b1;
                                    state_q <= ST_DONE;
                                end else begin
                                    state_q <= ST_DESCRIPTOR_BEGIN;
                                end
                            end
                        end
                    end
                end

                ST_DESCRIPTOR_BEGIN: begin
                    if (descriptor_begin_fire) begin
                        destination_cursor_q <= destination_offset_q;
                        if (term_total_q == 0) begin
                            if (payload_bytes_q != 16'(HEADER_BYTES)) begin
                                protocol_error <= 1'b1;
                                session_error_q <= 1'b1;
                            end
                            count_heads <= count_heads + 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            state_q <= ST_LOAD_TERM;
                        end
                    end
                end

                ST_LOAD_TERM: begin
                    if (!selected_descriptor_valid ||
                        (selected_bitmap_mode && !selected_bitmap_valid)) begin
                        protocol_error <= 1'b1;
                        session_error_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else if (term_fire) begin
                        current_data_offset_q <= destination_cursor_q;
                        destination_cursor_q <= 11'(selected_data_end);
                        current_gate_q <= selected_gate;
                        current_lane_q <= LANE_ID_W'(selected_lane);
                        current_remaining_q <= selected_count;
                        current_emitted_q <= '0;
                        current_bitmap_mode_q <= selected_bitmap_mode;
                        current_bitmap_q <= selected_bitmap_mode ?
                                            selected_bitmap : '0;
                        bitmap_terms_seen_q <= bitmap_terms_seen_q +
                                               8'(selected_bitmap_mode);
                        if (selected_bitmap_mode)
                            count_bitmap_terms <= count_bitmap_terms + 1'b1;
                        state_q <= ST_EVENTS;
                    end
                end

                ST_EVENTS: begin
                    if (!event_tokens_in_range && event_count_comb != 0) begin
                        protocol_error <= 1'b1;
                        session_error_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else if (event_fire) begin
                        current_remaining_q <= current_remaining_q -
                                               8'(event_count_comb);
                        current_emitted_q <= current_emitted_q +
                                             8'(event_count_comb);
                        current_bitmap_q <= bitmap_after_emit;
                        events_emitted_q <= events_emitted_q +
                                            EVENT_COUNT_W'(event_count_comb);
                        count_events <= count_events + COUNTER_W'(event_count_comb);
                        if (event_term_last) begin
                            count_terms <= count_terms + 1'b1;
                            if (event_head_last) begin
                                if (events_emitted_q +
                                    EVENT_COUNT_W'(event_count_comb) != event_total_q ||
                                    16'(destination_cursor_q) != payload_bytes_q ||
                                    bitmap_terms_seen_q != bitmap_term_total_q) begin
                                    protocol_error <= 1'b1;
                                    session_error_q <= 1'b1;
                                end
                                count_heads <= count_heads + 1'b1;
                                state_q <= ST_DONE;
                            end else begin
                                term_index_q <= term_index_q + 1'b1;
                                state_q <= ST_LOAD_TERM;
                            end
                        end
                    end
                end

                ST_ERROR_DRAIN: begin
                    if (word_fire && word_last)
                        state_q <= ST_DONE;
                end

                ST_DONE: begin
                    if (done_fire)
                        state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase

            if (word_valid && !word_ready)
                count_input_stall_cycles <= count_input_stall_cycles + 1'b1;
            if (term_valid && !term_ready)
                count_term_stall_cycles <= count_term_stall_cycles + 1'b1;
            if (event_valid && !event_ready)
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;
        end
    end
endmodule

`default_nettype wire
