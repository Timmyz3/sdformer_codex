`timescale 1ns/1ps
`default_nettype none

// Decodes a sequential IPD32W head-slot stream into up to four destinations
// per cycle. Descriptors are buffered as 64-bit pairs; token IDs use a
// 128-bit reservoir so input refill can overlap four-way output consumption.
module gatestack_ipd32w_replay_decoder #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int MAX_TERMS       = 128,
    parameter int EVENT_WAYS      = 4,
    parameter int TAG_W           = 32,
    parameter int WORD_INDEX_W    = 7,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = 8,
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int TERM_INDEX_W    = (MAX_TERMS <= 1) ? 1 : $clog2(MAX_TERMS),
    parameter int EVENT_COUNT_W   = 13,
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1)
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              start_valid,
    output logic                              start_ready,

    input  logic                              word_valid,
    output logic                              word_ready,
    input  logic [63:0]                       word_data,
    input  logic [WORD_INDEX_W-1:0]           word_index,
    input  logic                              word_last,

    output logic                              descriptor_begin_valid,
    input  logic                              descriptor_begin_ready,
    output logic [TAG_W-1:0]                  descriptor_begin_tag,
    output logic [7:0]                        descriptor_begin_term_count,

    output logic                              term_valid,
    input  logic                              term_ready,
    output logic [8:0]                        term_gate_code,
    output logic [LANE_ID_W-1:0]              term_lane_id,
    output logic [7:0]                        term_destination_count,
    output logic [TERM_INDEX_W-1:0]           term_index,
    output logic                              term_head_last,

    output logic                              event_valid,
    input  logic                              event_ready,
    output logic [8:0]                        event_gate_code,
    output logic [LANE_ID_W-1:0]              event_lane_id,
    output logic [EVENT_WAYS-1:0]             event_token_valid,
    output logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    output logic [WAY_COUNT_W-1:0]            event_count,
    output logic                              event_term_first,
    output logic                              event_term_last,
    output logic                              event_head_last,

    output logic                              done_valid,
    input  logic                              done_ready,
    output logic [TAG_W-1:0]                  done_tag,
    output logic                              done_error,

    output logic                              protocol_error,
    output logic [COUNTER_W-1:0]              count_heads,
    output logic [COUNTER_W-1:0]              count_terms,
    output logic [COUNTER_W-1:0]              count_events,
    output logic [COUNTER_W-1:0]              count_input_stall_cycles,
    output logic [COUNTER_W-1:0]              count_term_stall_cycles,
    output logic [COUNTER_W-1:0]              count_output_stall_cycles
);

    localparam logic [15:0] MAGIC = 16'h4753;
    localparam logic [3:0] VERSION = 4'd1;
    localparam int DESC_PAIRS = (MAX_TERMS + 1) / 2;
    localparam int DESC_PAIR_INDEX_W = (DESC_PAIRS <= 1) ?
                                         1 : $clog2(DESC_PAIRS);

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_HEADER0,
        ST_HEADER1,
        ST_DESCRIPTOR_BEGIN,
        ST_DESC,
        ST_LOAD_TERM,
        ST_TOKENS,
        ST_ERROR_DRAIN,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [63:0] descriptor_pair_mem [0:DESC_PAIRS-1];
    logic [TAG_W-1:0] tag_q;
    logic [7:0] term_total_q;
    logic [EVENT_COUNT_W-1:0] event_total_q;
    logic [WORD_INDEX_W-1:0] expected_word_index_q;
    logic [DESC_PAIR_INDEX_W-1:0] desc_pair_index_q;
    logic [EVENT_COUNT_W-1:0] desc_event_sum_q;

    logic [TERM_INDEX_W-1:0] term_index_q;
    logic [8:0] current_gate_q;
    logic [LANE_ID_W-1:0] current_lane_q;
    logic [7:0] current_term_remaining_q;
    logic current_term_started_q;
    logic next_term_prefetched_q;
    logic [8:0] next_gate_q;
    logic [LANE_ID_W-1:0] next_lane_q;
    logic [7:0] next_term_count_q;

    logic [127:0] token_reservoir_q;
    logic [4:0] token_bytes_q;
    logic [EVENT_COUNT_W-1:0] tokens_received_q;
    logic [EVENT_COUNT_W-1:0] tokens_emitted_q;
    logic input_last_seen_q;
    logic session_error_q;

    logic word_fire;
    logic term_fire;
    logic event_fire;
    logic done_fire;
    logic descriptor_begin_fire;
    logic [WAY_COUNT_W-1:0] event_count_comb;
    logic [EVENT_WAYS-1:0] event_token_valid_comb;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids_comb;
    logic event_tokens_in_range;
    logic [4:0] bytes_after_consume;
    logic [127:0] reservoir_after_consume;
    logic [3:0] token_word_valid_bytes;
    logic [EVENT_COUNT_W-1:0] events_after_input;
    logic token_input_is_last;
    logic [63:0] selected_descriptor_pair;
    logic [31:0] selected_descriptor;
    logic [7:0] selected_term_count;
    logic [8:0] selected_gate;
    logic [4:0] selected_lane;
    logic selected_descriptor_valid;
    logic [TERM_INDEX_W-1:0] next_term_index_comb;
    logic [63:0] next_descriptor_pair;
    logic [31:0] next_descriptor;
    logic [8:0] next_selected_gate;
    logic [4:0] next_selected_lane;
    logic [7:0] next_selected_count;
    logic next_descriptor_valid;
    logic has_next_term;
    logic desc_pair_is_last;
    logic [7:0] low_desc_count;
    logic [7:0] high_desc_count;
    logic low_desc_valid;
    logic high_desc_valid;
    logic [EVENT_COUNT_W-1:0] desc_sum_with_word;
    logic [7:0] header_term_count;
    logic [EVENT_COUNT_W-1:0] header_event_count;
    logic [2:0] header_active_classes;
    logic [7:0] header_active_tokens;
    logic [9:0] header_token_offset;
    logic [12:0] header_payload_bits;
    logic [10:0] expected_token_offset;
    logic [13:0] expected_payload_bits;

    assign start_ready = state_q == ST_IDLE;
    assign done_valid = state_q == ST_DONE;
    assign done_tag = tag_q;
    assign done_error = session_error_q;
    assign word_fire = word_valid && word_ready;
    assign term_fire = term_valid && term_ready;
    assign event_fire = event_valid && event_ready;
    assign done_fire = done_valid && done_ready;
    assign descriptor_begin_valid = state_q == ST_DESCRIPTOR_BEGIN;
    assign descriptor_begin_tag = tag_q;
    assign descriptor_begin_term_count = term_total_q;
    assign descriptor_begin_fire = descriptor_begin_valid &&
                                   descriptor_begin_ready;

    assign header_payload_bits = word_data[12:0];
    assign header_term_count = word_data[20:13];
    assign header_event_count = word_data[33:21];
    assign header_active_classes = word_data[36:34];
    assign header_active_tokens = word_data[44:37];
    assign header_token_offset = word_data[54:45];
    assign expected_token_offset = 11'(16) +
        (11'((32'(header_term_count) + 1) >> 1) << 3);
    assign expected_payload_bits =
        (14'(header_token_offset) + 14'(header_event_count)) << 3;

    assign desc_pair_is_last =
        32'(desc_pair_index_q) == ((32'(term_total_q) + 1) / 2 - 1);
    assign low_desc_valid = 32'(desc_pair_index_q) * 2 < term_total_q;
    assign high_desc_valid = 32'(desc_pair_index_q) * 2 + 1 < term_total_q;
    assign low_desc_count = word_data[21:14];
    assign high_desc_count = word_data[53:46];
    assign desc_sum_with_word = desc_event_sum_q +
                                EVENT_COUNT_W'(low_desc_count) +
                                (high_desc_valid ?
                                 EVENT_COUNT_W'(high_desc_count) :
                                 EVENT_COUNT_W'(0));

    assign selected_descriptor_pair = descriptor_pair_mem[
        DESC_PAIR_INDEX_W'(term_index_q >> 1)];
    assign selected_descriptor = term_index_q[0] ?
                                 selected_descriptor_pair[63:32] :
                                 selected_descriptor_pair[31:0];
    assign selected_gate = selected_descriptor[8:0];
    assign selected_lane = selected_descriptor[13:9];
    assign selected_term_count = selected_descriptor[21:14];
    assign selected_descriptor_valid = selected_descriptor[31:22] == '0 &&
                                       selected_term_count != 0 &&
                                       32'(selected_lane) < LANES;

    assign next_term_index_comb = term_index_q + 1'b1;
    assign has_next_term = 32'(term_index_q) + 1 < 32'(term_total_q);
    assign next_descriptor_pair = descriptor_pair_mem[
        DESC_PAIR_INDEX_W'(next_term_index_comb >> 1)];
    assign next_descriptor = next_term_index_comb[0] ?
                             next_descriptor_pair[63:32] :
                             next_descriptor_pair[31:0];
    assign next_selected_gate = next_descriptor[8:0];
    assign next_selected_lane = next_descriptor[13:9];
    assign next_selected_count = next_descriptor[21:14];
    assign next_descriptor_valid = next_descriptor[31:22] == '0 &&
                                   next_selected_count != 0 &&
                                   32'(next_selected_lane) < LANES;

    always_comb begin
        term_valid = 1'b0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_index = '0;
        term_head_last = 1'b0;
        if (state_q == ST_LOAD_TERM) begin
            term_valid = selected_descriptor_valid;
            term_gate_code = selected_gate;
            term_lane_id = LANE_ID_W'(selected_lane);
            term_destination_count = selected_term_count;
            term_index = term_index_q;
            term_head_last = !has_next_term;
        end else if (state_q == ST_TOKENS && has_next_term &&
                     !next_term_prefetched_q) begin
            term_valid = next_descriptor_valid;
            term_gate_code = next_selected_gate;
            term_lane_id = LANE_ID_W'(next_selected_lane);
            term_destination_count = next_selected_count;
            term_index = next_term_index_comb;
            term_head_last =
                32'(next_term_index_comb) + 1 == 32'(term_total_q);
        end
    end

    always_comb begin
        event_count_comb = '0;
        event_token_valid_comb = '0;
        event_token_ids_comb = '0;
        for (int way = 0; way < EVENT_WAYS; way = way + 1) begin
            if (32'(way) < token_bytes_q &&
                32'(way) < current_term_remaining_q) begin
                event_token_valid_comb[way] = 1'b1;
                event_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W] =
                    token_reservoir_q[(way*8) +: 8];
                event_count_comb = event_count_comb + 1'b1;
            end
        end
    end

    always_comb begin
        event_tokens_in_range = 1'b1;
        for (int way = 0; way < EVENT_WAYS; way = way + 1) begin
            if (event_token_valid_comb[way] &&
                32'(event_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W]) >=
                TOKENS) begin
                event_tokens_in_range = 1'b0;
            end
        end
    end

    assign event_valid = state_q == ST_TOKENS &&
                         event_count_comb != 0 && event_tokens_in_range;
    assign event_gate_code = current_gate_q;
    assign event_lane_id = current_lane_q;
    assign event_token_valid = event_token_valid_comb;
    assign event_token_ids = event_token_ids_comb;
    assign event_count = event_count_comb;
    assign event_term_first = !current_term_started_q;
    assign event_term_last = 8'(event_count_comb) ==
                             current_term_remaining_q;
    assign event_head_last = event_term_last &&
                             (32'(term_index_q) + 1 == 32'(term_total_q));

    assign bytes_after_consume = token_bytes_q -
        (event_fire ? 5'(event_count_comb) : 5'(0));
    assign reservoir_after_consume = token_reservoir_q >>
        ((event_fire ? 32'(event_count_comb) : 32'(0)) * 8);
    assign token_word_valid_bytes =
        ((32'(event_total_q) - 32'(tokens_received_q)) >= 8) ?
        4'd8 : 4'(event_total_q - tokens_received_q);
    assign events_after_input = tokens_received_q +
        EVENT_COUNT_W'(token_word_valid_bytes);
    assign token_input_is_last = events_after_input == event_total_q;

    always_comb begin
        word_ready = 1'b0;
        unique case (state_q)
            ST_HEADER0, ST_HEADER1, ST_DESC: word_ready = 1'b1;
            ST_TOKENS: begin
                word_ready = !input_last_seen_q &&
                             tokens_received_q < event_total_q &&
                             bytes_after_consume <= 8 &&
                             !(event_valid && !event_ready);
            end
            ST_ERROR_DRAIN: word_ready = 1'b1;
            default: word_ready = 1'b0;
        endcase
    end

    task automatic flag_session_error;
        begin
            protocol_error <= 1'b1;
            session_error_q <= 1'b1;
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            term_total_q <= '0;
            event_total_q <= '0;
            expected_word_index_q <= '0;
            desc_pair_index_q <= '0;
            desc_event_sum_q <= '0;
            term_index_q <= '0;
            current_gate_q <= '0;
            current_lane_q <= '0;
            current_term_remaining_q <= '0;
            current_term_started_q <= 1'b0;
            next_term_prefetched_q <= 1'b0;
            next_gate_q <= '0;
            next_lane_q <= '0;
            next_term_count_q <= '0;
            token_reservoir_q <= '0;
            token_bytes_q <= '0;
            tokens_received_q <= '0;
            tokens_emitted_q <= '0;
            input_last_seen_q <= 1'b0;
            session_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_heads <= '0;
            count_terms <= '0;
            count_events <= '0;
            count_input_stall_cycles <= '0;
            count_term_stall_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (start_valid && start_ready) begin
                state_q <= ST_HEADER0;
                tag_q <= '0;
                term_total_q <= '0;
                event_total_q <= '0;
                expected_word_index_q <= '0;
                desc_pair_index_q <= '0;
                desc_event_sum_q <= '0;
                term_index_q <= '0;
                token_reservoir_q <= '0;
                token_bytes_q <= '0;
                tokens_received_q <= '0;
                tokens_emitted_q <= '0;
                input_last_seen_q <= 1'b0;
                next_term_prefetched_q <= 1'b0;
                session_error_q <= 1'b0;
            end

            unique case (state_q)
                ST_IDLE: begin
                    // start handshake above owns the IDLE-to-HEADER0 transition.
                end

                ST_HEADER0: begin
                    if (word_fire) begin
                        expected_word_index_q <= expected_word_index_q + 1'b1;
                        tag_q <= TAG_W'(word_data[63:32]);
                        if (word_index != 0 || word_last ||
                            word_data[15:0] != MAGIC ||
                            word_data[19:16] != VERSION ||
                            !word_data[20] || word_data[31:21] != 0) begin
                            flag_session_error();
                            state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            state_q <= ST_HEADER1;
                        end
                    end
                end

                ST_HEADER1: begin
                    if (word_fire) begin
                        expected_word_index_q <= expected_word_index_q + 1'b1;
                        term_total_q <= header_term_count;
                        event_total_q <= header_event_count;
                        if (word_index != expected_word_index_q ||
                            word_data[63:55] != 0 ||
                            header_active_classes > 4 ||
                            32'(header_active_tokens) > TOKENS ||
                            32'(header_term_count) > MAX_TERMS ||
                            11'(header_token_offset) != expected_token_offset ||
                            14'(header_payload_bits) != expected_payload_bits ||
                            header_payload_bits > 6642 ||
                            ((header_term_count == 0) !=
                             (header_event_count == 0)) ||
                            (header_term_count == 0 && !word_last) ||
                            (header_term_count != 0 && word_last)) begin
                            flag_session_error();
                            state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                        end else begin
                            desc_pair_index_q <= '0;
                            desc_event_sum_q <= '0;
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
                            state_q <= ST_DESC;
                        end
                    end
                end

                ST_DESC: begin
                    if (word_fire) begin
                        descriptor_pair_mem[desc_pair_index_q] <= word_data;
                        expected_word_index_q <= expected_word_index_q + 1'b1;
                        desc_event_sum_q <= desc_sum_with_word;
                        if (word_index != expected_word_index_q || word_last ||
                            (low_desc_valid &&
                             (word_data[31:22] != 0 || low_desc_count == 0 ||
                              32'(word_data[13:9]) >= LANES)) ||
                            (high_desc_valid &&
                             (word_data[63:54] != 0 || high_desc_count == 0 ||
                              32'(word_data[45:41]) >= LANES)) ||
                            (!high_desc_valid && word_data[63:32] != 0)) begin
                            flag_session_error();
                            state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                        end else if (desc_pair_is_last) begin
                            if (desc_sum_with_word != event_total_q) begin
                                flag_session_error();
                                state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                            end else begin
                                term_index_q <= '0;
                                state_q <= ST_LOAD_TERM;
                            end
                        end else begin
                            desc_pair_index_q <= desc_pair_index_q + 1'b1;
                        end
                    end
                end

                ST_LOAD_TERM: begin
                    if (!selected_descriptor_valid) begin
                        flag_session_error();
                        state_q <= ST_ERROR_DRAIN;
                    end else if (term_fire) begin
                        current_gate_q <= selected_gate;
                        current_lane_q <= LANE_ID_W'(selected_lane);
                        current_term_remaining_q <= selected_term_count;
                        current_term_started_q <= 1'b0;
                        next_term_prefetched_q <= 1'b0;
                        state_q <= ST_TOKENS;
                    end
                end

                ST_TOKENS: begin
                    token_reservoir_q <= reservoir_after_consume;
                    token_bytes_q <= bytes_after_consume;
                    if (!event_tokens_in_range && event_count_comb != 0) begin
                        flag_session_error();
                        token_reservoir_q <= '0;
                        token_bytes_q <= '0;
                        state_q <= input_last_seen_q ? ST_DONE : ST_ERROR_DRAIN;
                    end else begin
                        if (term_fire) begin
                            next_term_prefetched_q <= 1'b1;
                            next_gate_q <= next_selected_gate;
                            next_lane_q <= LANE_ID_W'(next_selected_lane);
                            next_term_count_q <= next_selected_count;
                        end
                        if (event_fire) begin
                            tokens_emitted_q <= tokens_emitted_q +
                                EVENT_COUNT_W'(event_count_comb);
                            count_events <= count_events +
                                COUNTER_W'(event_count_comb);
                            current_term_started_q <= 1'b1;
                            if (event_term_last) begin
                                count_terms <= count_terms + 1'b1;
                                if (event_head_last) begin
                                    if ((tokens_emitted_q +
                                         EVENT_COUNT_W'(event_count_comb) !=
                                         event_total_q) ||
                                        !input_last_seen_q ||
                                        bytes_after_consume != 0) begin
                                        flag_session_error();
                                    end
                                    count_heads <= count_heads + 1'b1;
                                    state_q <= ST_DONE;
                                end else begin
                                    term_index_q <= term_index_q + 1'b1;
                                    current_term_started_q <= 1'b0;
                                    if (next_term_prefetched_q || term_fire) begin
                                        current_gate_q <= term_fire ?
                                            next_selected_gate : next_gate_q;
                                        current_lane_q <= term_fire ?
                                            LANE_ID_W'(next_selected_lane) :
                                            next_lane_q;
                                        current_term_remaining_q <= term_fire ?
                                            next_selected_count :
                                            next_term_count_q;
                                        next_term_prefetched_q <= 1'b0;
                                        state_q <= ST_TOKENS;
                                    end else begin
                                        state_q <= ST_LOAD_TERM;
                                    end
                                end
                            end else begin
                                current_term_remaining_q <=
                                    current_term_remaining_q -
                                    8'(event_count_comb);
                            end
                        end

                        if (word_fire) begin
                            expected_word_index_q <=
                                expected_word_index_q + 1'b1;
                            token_reservoir_q <= reservoir_after_consume |
                                (128'(word_data) << (32'(bytes_after_consume) * 8));
                            token_bytes_q <= bytes_after_consume +
                                5'(token_word_valid_bytes);
                            tokens_received_q <= events_after_input;
                            input_last_seen_q <= word_last;
                            if (word_index != expected_word_index_q ||
                                word_last != token_input_is_last) begin
                                flag_session_error();
                                state_q <= word_last ? ST_DONE : ST_ERROR_DRAIN;
                            end
                        end
                    end
                end

                ST_ERROR_DRAIN: begin
                    if (word_fire && word_last) begin
                        state_q <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    if (done_fire) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase

            if (word_valid && !word_ready) begin
                count_input_stall_cycles <= count_input_stall_cycles + 1'b1;
            end
            if (term_valid && !term_ready) begin
                count_term_stall_cycles <= count_term_stall_cycles + 1'b1;
            end
            if (event_valid && !event_ready) begin
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
