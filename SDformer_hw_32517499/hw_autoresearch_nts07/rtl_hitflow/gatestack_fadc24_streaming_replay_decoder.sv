`timescale 1ns/1ps
`default_nettype none

// Streaming FADC24 decoder. A 256-bit byte reservoir repacks 24-bit
// descriptors and token lists; bitmap payloads are loaded into one 168-bit
// scan register. No random byte access to the full head slot is required.
module gatestack_fadc24_streaming_replay_decoder #(
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

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_HEADER0,
        ST_HEADER1,
        ST_DESCRIPTOR_BEGIN,
        ST_DESCRIPTORS,
        ST_LOAD_TERM,
        ST_LIST_EVENTS,
        ST_BITMAP_LOAD,
        ST_BITMAP_EVENTS,
        ST_ERROR_DRAIN,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [23:0] descriptor_mem [0:MAX_TERMS-1];
    logic [TAG_W-1:0] tag_q;
    logic [15:0] payload_bytes_q;
    logic [7:0] term_total_q;
    logic [EVENT_COUNT_W-1:0] event_total_q;
    logic [7:0] bitmap_term_total_q;
    logic [10:0] destination_offset_q;
    logic [15:0] payload_bytes_received_q;
    logic [WORD_INDEX_W-1:0] expected_word_index_q;
    logic input_last_seen_q;
    logic [255:0] reservoir_q;
    logic [5:0] reservoir_bytes_q;
    logic [TERM_INDEX_W-1:0] descriptor_index_q;
    logic [EVENT_COUNT_W-1:0] descriptor_event_sum_q;
    logic [10:0] descriptor_encoded_sum_q;
    logic [7:0] descriptor_bitmap_sum_q;
    logic [TERM_INDEX_W-1:0] term_index_q;
    logic [8:0] current_gate_q;
    logic [LANE_ID_W-1:0] current_lane_q;
    logic [7:0] current_remaining_q;
    logic [7:0] current_emitted_q;
    logic [4:0] bitmap_bytes_loaded_q;
    logic [167:0] bitmap_q;
    logic [3:0] bitmap_segment_q;
    logic [EVENT_COUNT_W-1:0] events_emitted_q;
    logic session_error_q;

    logic word_fire, descriptor_begin_fire, term_fire, event_fire, done_fire;
    logic descriptor_consume, bitmap_byte_consume;
    logic [5:0] consume_bytes_comb;
    logic [5:0] reservoir_bytes_after_consume;
    logic [255:0] reservoir_after_consume;
    logic [3:0] input_valid_bytes;
    logic [15:0] payload_bytes_after_input;
    logic input_word_is_last;
    logic [23:0] incoming_descriptor;
    logic [4:0] incoming_lane;
    logic [7:0] incoming_count;
    logic incoming_bitmap_mode;
    logic incoming_descriptor_valid;
    logic [EVENT_COUNT_W-1:0] descriptor_event_sum_with;
    logic [10:0] descriptor_encoded_sum_with;
    logic [7:0] descriptor_bitmap_sum_with;
    logic [23:0] selected_descriptor;
    logic [8:0] selected_gate;
    logic [4:0] selected_lane;
    logic [7:0] selected_count;
    logic selected_bitmap_mode;
    logic selected_descriptor_valid;
    logic [EVENT_WAYS-1:0] list_token_valid_comb;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] list_token_ids_comb;
    logic [WAY_COUNT_W-1:0] list_event_count_comb;
    logic list_tokens_in_range;
    logic [167:0] bitmap_with_input_byte;
    logic bitmap_loaded_valid;
    logic [17:0] bitmap_segment_bits;
    logic [17:0] bitmap_segment_after_emit;
    logic [EVENT_WAYS-1:0] bitmap_token_valid_comb;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] bitmap_token_ids_comb;
    logic [WAY_COUNT_W-1:0] bitmap_event_count_comb;
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
    assign header_expected_words = (header_payload_bytes + 16'd7) >> 3;
    assign header_contract_ok = word_data[63:56] == 0 &&
        32'(header_payload_bytes) >= HEADER_BYTES &&
        32'(header_payload_bytes) <= SLOT_BYTES &&
        32'(header_term_count) <= MAX_TERMS &&
        32'(header_bitmap_terms) <= 32'(header_term_count) &&
        32'(header_destination_offset) ==
            HEADER_BYTES + DESCRIPTOR_BYTES * 32'(header_term_count) &&
        ((header_term_count == 0) == (header_event_count == 0)) &&
        (header_term_count != 0 || header_bitmap_terms == 0);

    assign incoming_descriptor = reservoir_q[23:0];
    assign incoming_lane = incoming_descriptor[13:9];
    assign incoming_count = incoming_descriptor[21:14];
    assign incoming_bitmap_mode = incoming_descriptor[22];
    assign incoming_descriptor_valid = !incoming_descriptor[23] &&
        incoming_count != 0 && 32'(incoming_lane) < LANES;
    assign descriptor_consume = state_q == ST_DESCRIPTORS &&
                                reservoir_bytes_q >= 6'd3;
    assign descriptor_event_sum_with = descriptor_event_sum_q +
                                       EVENT_COUNT_W'(incoming_count);
    assign descriptor_encoded_sum_with = descriptor_encoded_sum_q +
        (incoming_bitmap_mode ? 11'(BITMAP_BYTES) : 11'(incoming_count));
    assign descriptor_bitmap_sum_with = descriptor_bitmap_sum_q +
                                        8'(incoming_bitmap_mode);

    assign selected_descriptor = descriptor_mem[term_index_q];
    assign selected_gate = selected_descriptor[8:0];
    assign selected_lane = selected_descriptor[13:9];
    assign selected_count = selected_descriptor[21:14];
    assign selected_bitmap_mode = selected_descriptor[22];
    assign selected_descriptor_valid = !selected_descriptor[23] &&
        selected_count != 0 && 32'(selected_lane) < LANES;

    always_comb begin
        list_token_valid_comb = '0;
        list_token_ids_comb = '0;
        list_event_count_comb = '0;
        for (int way = 0; way < 32'd4; way = way + 1) begin
            if (32'(way) < reservoir_bytes_q &&
                32'(way) < current_remaining_q) begin
                list_token_valid_comb[way] = 1'b1;
                list_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W] =
                    reservoir_q[(way*8) +: 8];
                list_event_count_comb = list_event_count_comb + 1'b1;
            end
        end
    end
    always_comb begin
        list_tokens_in_range = 1'b1;
        for (int way = 0; way < 32'd4; way = way + 1) begin
            if (list_token_valid_comb[way] &&
                32'(list_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W]) >=
                TOKENS)
                list_tokens_in_range = 1'b0;
        end
    end

    assign bitmap_byte_consume = state_q == ST_BITMAP_LOAD &&
                                 reservoir_bytes_q != 6'd0;
    assign bitmap_with_input_byte = {reservoir_q[7:0], bitmap_q[167:8]};
    assign bitmap_loaded_valid = bitmap_with_input_byte[167:162] == 6'd0;
    assign bitmap_segment_bits = 18'(
        bitmap_q >> (32'(bitmap_segment_q) * 32'd18));

    always_comb begin
        bitmap_token_valid_comb = '0;
        bitmap_token_ids_comb = '0;
        bitmap_event_count_comb = '0;
        bitmap_after_emit = bitmap_q;
        bitmap_segment_after_emit = bitmap_segment_bits;
        bitmap_selected_comb = 32'd0;
        for (int lane = 0; lane < 32'd18; lane = lane + 1) begin
            if (bitmap_segment_bits[lane] &&
                bitmap_selected_comb < 32'(EVENT_WAYS)) begin
                bitmap_token_valid_comb[bitmap_selected_comb] = 1'b1;
                bitmap_token_ids_comb[(bitmap_selected_comb*TOKEN_ID_W) +:
                                      TOKEN_ID_W] = TOKEN_ID_W'(
                    32'(bitmap_segment_q) * 32'd18 + 32'(lane));
                bitmap_after_emit[
                    32'(bitmap_segment_q) * 32'd18 + 32'(lane)] = 1'b0;
                bitmap_segment_after_emit[lane] = 1'b0;
                bitmap_selected_comb = bitmap_selected_comb + 1;
            end
        end
        bitmap_event_count_comb = WAY_COUNT_W'(bitmap_selected_comb);
    end

    always_comb begin
        consume_bytes_comb = '0;
        if (descriptor_consume)
            consume_bytes_comb = 6'd3;
        else if (state_q == ST_LIST_EVENTS && event_fire)
            consume_bytes_comb = 6'(list_event_count_comb);
        else if (bitmap_byte_consume)
            consume_bytes_comb = 6'd1;
        reservoir_bytes_after_consume = reservoir_bytes_q - consume_bytes_comb;
        reservoir_after_consume = reservoir_q >> (32'(consume_bytes_comb) * 8);
    end

    assign input_valid_bytes =
        (32'(payload_bytes_q) - 32'(payload_bytes_received_q) >= 8) ?
        4'd8 : 4'(payload_bytes_q - payload_bytes_received_q);
    assign payload_bytes_after_input = payload_bytes_received_q +
                                       16'(input_valid_bytes);
    assign input_word_is_last = payload_bytes_after_input == payload_bytes_q;
    always_comb begin
        word_ready = 1'b0;
        if (state_q == ST_HEADER0 || state_q == ST_HEADER1 ||
            state_q == ST_ERROR_DRAIN) begin
            word_ready = 1'b1;
        end else if (state_q == ST_DESCRIPTORS ||
                     state_q == ST_LIST_EVENTS ||
                     state_q == ST_BITMAP_LOAD) begin
            word_ready = !input_last_seen_q &&
                         payload_bytes_received_q < payload_bytes_q &&
                         reservoir_bytes_after_consume <= 24 &&
                         !(event_valid && !event_ready);
        end
    end

    always_comb begin
        term_valid = state_q == ST_LOAD_TERM && selected_descriptor_valid;
        term_gate_code = selected_gate;
        term_lane_id = LANE_ID_W'(selected_lane);
        term_destination_count = selected_count;
        term_index = term_index_q;
        term_head_last = 32'(term_index_q) + 1 == 32'(term_total_q);
    end

    always_comb begin
        event_valid = 1'b0;
        event_gate_code = current_gate_q;
        event_lane_id = current_lane_q;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        if (state_q == ST_LIST_EVENTS && list_event_count_comb != 0 &&
            list_tokens_in_range) begin
            event_valid = 1'b1;
            event_token_valid = list_token_valid_comb;
            event_token_ids = list_token_ids_comb;
            event_count = list_event_count_comb;
        end else if (state_q == ST_BITMAP_EVENTS &&
                     bitmap_event_count_comb != 0) begin
            event_valid = 1'b1;
            event_token_valid = bitmap_token_valid_comb;
            event_token_ids = bitmap_token_ids_comb;
            event_count = bitmap_event_count_comb;
        end
        event_term_first = current_emitted_q == 8'd0;
        event_term_last = 8'(event_count) == current_remaining_q;
        event_head_last = event_term_last &&
            (32'(term_index_q) + 1 == 32'(term_total_q));
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            payload_bytes_q <= '0;
            term_total_q <= '0;
            event_total_q <= '0;
            bitmap_term_total_q <= '0;
            destination_offset_q <= '0;
            payload_bytes_received_q <= '0;
            expected_word_index_q <= '0;
            input_last_seen_q <= 1'b0;
            reservoir_q <= '0;
            reservoir_bytes_q <= '0;
            descriptor_index_q <= '0;
            descriptor_event_sum_q <= '0;
            descriptor_encoded_sum_q <= '0;
            descriptor_bitmap_sum_q <= '0;
            term_index_q <= '0;
            current_gate_q <= '0;
            current_lane_q <= '0;
            current_remaining_q <= '0;
            current_emitted_q <= '0;
            bitmap_bytes_loaded_q <= '0;
            bitmap_q <= '0;
            bitmap_segment_q <= '0;
            events_emitted_q <= '0;
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
                state_q <= ST_HEADER0;
                tag_q <= '0;
                payload_bytes_q <= '0;
                term_total_q <= '0;
                event_total_q <= '0;
                bitmap_term_total_q <= '0;
                destination_offset_q <= '0;
                payload_bytes_received_q <= '0;
                expected_word_index_q <= '0;
                input_last_seen_q <= 1'b0;
                reservoir_q <= '0;
                reservoir_bytes_q <= '0;
                descriptor_index_q <= '0;
                descriptor_event_sum_q <= '0;
                descriptor_encoded_sum_q <= '0;
                descriptor_bitmap_sum_q <= '0;
                term_index_q <= '0;
                events_emitted_q <= '0;
                session_error_q <= 1'b0;
            end

            unique case (state_q)
                ST_IDLE: begin end
                ST_HEADER0: begin
                    if (word_fire) begin
                        tag_q <= TAG_W'(word_data[63:32]);
                        expected_word_index_q <= expected_word_index_q + 1'b1;
                        payload_bytes_received_q <= 16'd8;
                        if (word_index != 0 || word_last ||
                            word_data[15:0] != MAGIC ||
                            word_data[23:16] != VERSION ||
                            word_data[31:24] != 0) begin
                            protocol_error <= 1'b1;
                            session_error_q <= 1'b1;
                            state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            state_q <= ST_HEADER1;
                        end
                    end
                end
                ST_HEADER1: begin
                    if (word_fire) begin
                        payload_bytes_q <= header_payload_bytes;
                        term_total_q <= header_term_count;
                        event_total_q <= header_event_count;
                        bitmap_term_total_q <= header_bitmap_terms;
                        destination_offset_q <= header_destination_offset;
                        expected_word_index_q <= expected_word_index_q + 1'b1;
                        payload_bytes_received_q <= 16'd16;
                        input_last_seen_q <= word_last;
                        if (word_index != 1 || !header_contract_ok ||
                            word_last != (header_expected_words == 2)) begin
                            protocol_error <= 1'b1;
                            session_error_q <= 1'b1;
                            state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            state_q <= ST_DESCRIPTOR_BEGIN;
                        end
                    end
                end
                ST_DESCRIPTOR_BEGIN: begin
                    if (descriptor_begin_fire) begin
                        if (term_total_q == 0) begin
                            count_heads <= count_heads + 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            state_q <= ST_DESCRIPTORS;
                        end
                    end
                end
                ST_DESCRIPTORS: begin
                    reservoir_q <= reservoir_after_consume;
                    reservoir_bytes_q <= reservoir_bytes_after_consume;
                    if (descriptor_consume) begin
                        if (!incoming_descriptor_valid) begin
                            protocol_error <= 1'b1;
                            session_error_q <= 1'b1;
                            state_q <= input_last_seen_q ? ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            descriptor_mem[descriptor_index_q] <= incoming_descriptor;
                            descriptor_event_sum_q <= descriptor_event_sum_with;
                            descriptor_encoded_sum_q <= descriptor_encoded_sum_with;
                            descriptor_bitmap_sum_q <= descriptor_bitmap_sum_with;
                            if (32'(descriptor_index_q) + 1 ==
                                32'(term_total_q)) begin
                                if (descriptor_event_sum_with != event_total_q ||
                                    descriptor_bitmap_sum_with != bitmap_term_total_q ||
                                    16'(destination_offset_q) +
                                        16'(descriptor_encoded_sum_with) !=
                                        payload_bytes_q) begin
                                    protocol_error <= 1'b1;
                                    session_error_q <= 1'b1;
                                    state_q <= input_last_seen_q ?
                                               ST_DONE : ST_ERROR_DRAIN;
                                end else begin
                                    term_index_q <= '0;
                                    state_q <= ST_LOAD_TERM;
                                end
                            end else begin
                                descriptor_index_q <= descriptor_index_q + 1'b1;
                            end
                        end
                    end
                end
                ST_LOAD_TERM: begin
                    if (!selected_descriptor_valid) begin
                        protocol_error <= 1'b1;
                        session_error_q <= 1'b1;
                        state_q <= input_last_seen_q ? ST_DONE : ST_ERROR_DRAIN;
                    end else if (term_fire) begin
                        current_gate_q <= selected_gate;
                        current_lane_q <= LANE_ID_W'(selected_lane);
                        current_remaining_q <= selected_count;
                        current_emitted_q <= '0;
                        if (selected_bitmap_mode) begin
                            bitmap_bytes_loaded_q <= '0;
                            bitmap_q <= '0;
                            bitmap_segment_q <= '0;
                            count_bitmap_terms <= count_bitmap_terms + 1'b1;
                            state_q <= ST_BITMAP_LOAD;
                        end else begin
                            state_q <= ST_LIST_EVENTS;
                        end
                    end
                end
                ST_LIST_EVENTS: begin
                    reservoir_q <= reservoir_after_consume;
                    reservoir_bytes_q <= reservoir_bytes_after_consume;
                    if (!list_tokens_in_range && list_event_count_comb != 0) begin
                        protocol_error <= 1'b1;
                        session_error_q <= 1'b1;
                        state_q <= input_last_seen_q ? ST_DONE : ST_ERROR_DRAIN;
                    end else if (event_fire) begin
                        current_remaining_q <= current_remaining_q - 8'(event_count);
                        current_emitted_q <= current_emitted_q + 8'(event_count);
                        events_emitted_q <= events_emitted_q +
                                            EVENT_COUNT_W'(event_count);
                        count_events <= count_events + COUNTER_W'(event_count);
                        if (event_term_last) begin
                            count_terms <= count_terms + 1'b1;
                            if (event_head_last) begin
                                if (events_emitted_q + EVENT_COUNT_W'(event_count) !=
                                        event_total_q ||
                                    !input_last_seen_q ||
                                    reservoir_bytes_after_consume != 0) begin
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
                ST_BITMAP_LOAD: begin
                    reservoir_q <= reservoir_after_consume;
                    reservoir_bytes_q <= reservoir_bytes_after_consume;
                    if (bitmap_byte_consume) begin
                        bitmap_q <= bitmap_with_input_byte;
                        bitmap_bytes_loaded_q <= bitmap_bytes_loaded_q + 1'b1;
                        if (bitmap_bytes_loaded_q == 5'd20) begin
                            if (!bitmap_loaded_valid) begin
                                protocol_error <= 1'b1;
                                session_error_q <= 1'b1;
                                state_q <= input_last_seen_q ?
                                           ST_DONE : ST_ERROR_DRAIN;
                            end else begin
                                bitmap_segment_q <= '0;
                                state_q <= ST_BITMAP_EVENTS;
                            end
                        end
                    end
                end
                ST_BITMAP_EVENTS: begin
                    if (bitmap_segment_bits == 0 && current_remaining_q != 0) begin
                        if (bitmap_segment_q == 4'd8) begin
                            protocol_error <= 1'b1;
                            session_error_q <= 1'b1;
                            state_q <= input_last_seen_q ?
                                       ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            bitmap_segment_q <= bitmap_segment_q + 1'b1;
                        end
                    end else if (event_fire) begin
                        bitmap_q <= bitmap_after_emit;
                        current_remaining_q <= current_remaining_q - 8'(event_count);
                        current_emitted_q <= current_emitted_q + 8'(event_count);
                        events_emitted_q <= events_emitted_q +
                                            EVENT_COUNT_W'(event_count);
                        count_events <= count_events + COUNTER_W'(event_count);
                        if (event_term_last) begin
                            count_terms <= count_terms + 1'b1;
                            if (event_head_last) begin
                                if (events_emitted_q + EVENT_COUNT_W'(event_count) !=
                                        event_total_q ||
                                    !input_last_seen_q ||
                                    reservoir_bytes_q != 0 ||
                                    bitmap_after_emit != 0) begin
                                    protocol_error <= 1'b1;
                                    session_error_q <= 1'b1;
                                end
                                count_heads <= count_heads + 1'b1;
                                state_q <= ST_DONE;
                            end else begin
                                term_index_q <= term_index_q + 1'b1;
                                state_q <= ST_LOAD_TERM;
                            end
                        end else if (bitmap_segment_after_emit == 0) begin
                            if (bitmap_segment_q == 4'd8) begin
                                protocol_error <= 1'b1;
                                session_error_q <= 1'b1;
                                state_q <= input_last_seen_q ?
                                           ST_DONE : ST_ERROR_DRAIN;
                            end else begin
                                bitmap_segment_q <= bitmap_segment_q + 1'b1;
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

            if (word_fire && state_q != ST_HEADER0 && state_q != ST_HEADER1 &&
                state_q != ST_ERROR_DRAIN) begin
                reservoir_q <= reservoir_after_consume |
                    (256'(word_data) << (32'(reservoir_bytes_after_consume) * 8));
                reservoir_bytes_q <= reservoir_bytes_after_consume +
                                     6'(input_valid_bytes);
                payload_bytes_received_q <= payload_bytes_after_input;
                expected_word_index_q <= expected_word_index_q + 1'b1;
                input_last_seen_q <= word_last;
                if (word_index != expected_word_index_q ||
                    word_last != input_word_is_last) begin
                    protocol_error <= 1'b1;
                    session_error_q <= 1'b1;
                    state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                end
            end

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
