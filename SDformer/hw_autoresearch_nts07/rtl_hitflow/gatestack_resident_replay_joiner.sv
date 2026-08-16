`timescale 1ns/1ps
`default_nettype none

// Joins resident IPD32W descriptors with the token-only head-slot substream.
// One next descriptor and its product command can be prefetched while the
// current term multicasts destinations.
module gatestack_resident_replay_joiner #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int MAX_TERMS       = 80,
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
    input  logic                               clk_core,
    input  logic                               rst_core,

    input  logic                               start_valid,
    output logic                               start_ready,
    input  logic [TAG_W-1:0]                   start_tag,
    input  logic [7:0]                         start_term_count,
    input  logic [EVENT_COUNT_W-1:0]           start_event_count,

    input  logic                               descriptor_valid,
    output logic                               descriptor_ready,
    input  logic [8:0]                         descriptor_gate_code,
    input  logic [LANE_ID_W-1:0]               descriptor_lane_id,
    input  logic [7:0]                         descriptor_destination_count,
    input  logic [TERM_INDEX_W-1:0]            descriptor_term_index,
    input  logic                               descriptor_last,

    input  logic                               word_valid,
    output logic                               word_ready,
    input  logic [63:0]                        word_data,
    input  logic [WORD_INDEX_W-1:0]            word_index,
    input  logic                               word_last,

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
    output logic [COUNTER_W-1:0]               count_descriptor_stall_cycles,
    output logic [COUNTER_W-1:0]               count_input_stall_cycles,
    output logic [COUNTER_W-1:0]               count_term_stall_cycles,
    output logic [COUNTER_W-1:0]               count_output_stall_cycles
);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_WAIT_CURRENT_DESC,
        ST_WAIT_CURRENT_TERM,
        ST_TOKENS,
        ST_ADVANCE,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [7:0] term_total_q;
    logic [EVENT_COUNT_W-1:0] event_total_q;
    logic session_error_q;

    logic [8:0] current_gate_q;
    logic [LANE_ID_W-1:0] current_lane_q;
    logic [7:0] current_remaining_q;
    logic [TERM_INDEX_W-1:0] current_index_q;
    logic current_last_q;
    logic current_started_q;

    logic next_valid_q;
    logic [8:0] next_gate_q;
    logic [LANE_ID_W-1:0] next_lane_q;
    logic [7:0] next_count_q;
    logic [TERM_INDEX_W-1:0] next_index_q;
    logic next_last_q;
    logic next_term_issued_q;

    logic [7:0] descriptors_received_q;
    logic [EVENT_COUNT_W-1:0] descriptor_event_sum_q;
    logic descriptor_last_seen_q;
    logic [WORD_INDEX_W-1:0] expected_word_index_q;

    logic [127:0] token_reservoir_q;
    logic [4:0] token_bytes_q;
    logic [EVENT_COUNT_W-1:0] tokens_received_q;
    logic [EVENT_COUNT_W-1:0] tokens_emitted_q;
    logic input_last_seen_q;

    logic descriptor_fire;
    logic term_fire;
    logic event_fire;
    logic word_fire;
    logic done_fire;
    logic descriptor_is_valid;
    logic descriptor_expected_last;
    logic [EVENT_COUNT_W-1:0] descriptor_sum_next;
    logic [WAY_COUNT_W-1:0] event_count_comb;
    logic [EVENT_WAYS-1:0] event_token_valid_comb;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids_comb;
    logic event_tokens_in_range;
    logic [4:0] bytes_after_consume;
    logic [127:0] reservoir_after_consume;
    logic [3:0] token_word_valid_bytes;
    logic [EVENT_COUNT_W-1:0] events_after_input;
    logic token_input_is_last;
    logic final_input_seen_comb;
    logic current_term_finishes;
    logic next_term_available_comb;

    assign start_ready = state_q == ST_IDLE;
    assign done_valid = state_q == ST_DONE;
    assign done_tag = tag_q;
    assign done_error = session_error_q;

    assign descriptor_ready =
        (state_q == ST_WAIT_CURRENT_DESC) ||
        ((state_q == ST_TOKENS || state_q == ST_ADVANCE) &&
         !current_last_q && !next_valid_q);
    assign descriptor_fire = descriptor_valid && descriptor_ready;
    assign descriptor_expected_last =
        32'(descriptors_received_q) + 1 == 32'(term_total_q);
    assign descriptor_is_valid = descriptor_destination_count != 0 &&
        32'(descriptor_lane_id) < LANES &&
        32'(descriptor_term_index) == 32'(descriptors_received_q) &&
        descriptor_last == descriptor_expected_last;
    assign descriptor_sum_next = descriptor_event_sum_q +
        EVENT_COUNT_W'(descriptor_destination_count);

    always_comb begin
        term_valid = 1'b0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_index = '0;
        term_head_last = 1'b0;
        if (state_q == ST_WAIT_CURRENT_TERM) begin
            term_valid = 1'b1;
            term_gate_code = current_gate_q;
            term_lane_id = current_lane_q;
            term_destination_count = current_remaining_q;
            term_index = current_index_q;
            term_head_last = current_last_q;
        end else if ((state_q == ST_TOKENS || state_q == ST_ADVANCE) &&
                     next_valid_q && !next_term_issued_q) begin
            term_valid = 1'b1;
            term_gate_code = next_gate_q;
            term_lane_id = next_lane_q;
            term_destination_count = next_count_q;
            term_index = next_index_q;
            term_head_last = next_last_q;
        end
    end
    assign term_fire = term_valid && term_ready;

    always_comb begin
        event_count_comb = '0;
        event_token_valid_comb = '0;
        event_token_ids_comb = '0;
        for (int way = 0; way < 4; way = way + 1) begin
            if (32'(way) < token_bytes_q &&
                32'(way) < current_remaining_q) begin
                event_token_valid_comb[way] = 1'b1;
                event_token_ids_comb[(way*TOKEN_ID_W) +: TOKEN_ID_W] =
                    token_reservoir_q[(way*8) +: 8];
                event_count_comb = event_count_comb + 1'b1;
            end
        end
    end

    always_comb begin
        event_tokens_in_range = 1'b1;
        for (int way = 0; way < 4; way = way + 1) begin
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
    assign event_term_first = !current_started_q;
    assign event_term_last = 8'(event_count_comb) == current_remaining_q;
    assign event_head_last = event_term_last && current_last_q;
    assign event_fire = event_valid && event_ready;
    assign current_term_finishes = event_fire && event_term_last;

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
    assign word_ready = state_q == ST_TOKENS && !input_last_seen_q &&
        tokens_received_q < event_total_q && bytes_after_consume <= 8 &&
        !(event_valid && !event_ready);
    assign word_fire = word_valid && word_ready;
    assign final_input_seen_comb = input_last_seen_q ||
        (word_fire && token_input_is_last && word_last);
    assign done_fire = done_valid && done_ready;
    assign next_term_available_comb = next_valid_q &&
        (next_term_issued_q || term_fire);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            term_total_q <= '0;
            event_total_q <= '0;
            session_error_q <= 1'b0;
            current_gate_q <= '0;
            current_lane_q <= '0;
            current_remaining_q <= '0;
            current_index_q <= '0;
            current_last_q <= 1'b0;
            current_started_q <= 1'b0;
            next_valid_q <= 1'b0;
            next_gate_q <= '0;
            next_lane_q <= '0;
            next_count_q <= '0;
            next_index_q <= '0;
            next_last_q <= 1'b0;
            next_term_issued_q <= 1'b0;
            descriptors_received_q <= '0;
            descriptor_event_sum_q <= '0;
            descriptor_last_seen_q <= 1'b0;
            expected_word_index_q <= '0;
            token_reservoir_q <= '0;
            token_bytes_q <= '0;
            tokens_received_q <= '0;
            tokens_emitted_q <= '0;
            input_last_seen_q <= 1'b0;
            protocol_error <= 1'b0;
            count_heads <= '0;
            count_terms <= '0;
            count_events <= '0;
            count_descriptor_stall_cycles <= '0;
            count_input_stall_cycles <= '0;
            count_term_stall_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (start_valid && start_ready) begin
                tag_q <= start_tag;
                term_total_q <= start_term_count;
                event_total_q <= start_event_count;
                session_error_q <= 1'b0;
                descriptors_received_q <= '0;
                descriptor_event_sum_q <= '0;
                descriptor_last_seen_q <= 1'b0;
                expected_word_index_q <= '0;
                token_reservoir_q <= '0;
                token_bytes_q <= '0;
                tokens_received_q <= '0;
                tokens_emitted_q <= '0;
                input_last_seen_q <= 1'b0;
                next_valid_q <= 1'b0;
                next_term_issued_q <= 1'b0;
                if (32'(start_term_count) > MAX_TERMS ||
                    ((start_term_count == 0) !=
                     (start_event_count == 0))) begin
                    session_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    state_q <= ST_DONE;
                end else if (start_term_count == 0) begin
                    count_heads <= count_heads + 1'b1;
                    state_q <= ST_DONE;
                end else begin
                    state_q <= ST_WAIT_CURRENT_DESC;
                end
            end

            if (descriptor_valid && !descriptor_ready &&
                state_q != ST_IDLE && state_q != ST_DONE) begin
                count_descriptor_stall_cycles <=
                    count_descriptor_stall_cycles + 1'b1;
            end
            if (word_valid && !word_ready) begin
                count_input_stall_cycles <= count_input_stall_cycles + 1'b1;
            end
            if (term_valid && !term_ready) begin
                count_term_stall_cycles <= count_term_stall_cycles + 1'b1;
            end
            if (event_valid && !event_ready) begin
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;
            end

            if (descriptor_fire) begin
                descriptors_received_q <= descriptors_received_q + 1'b1;
                descriptor_event_sum_q <= descriptor_sum_next;
                if (!descriptor_is_valid ||
                    (descriptor_expected_last &&
                     descriptor_sum_next != event_total_q)) begin
                    session_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    state_q <= ST_DONE;
                end else begin
                    if (descriptor_last) begin
                        descriptor_last_seen_q <= 1'b1;
                    end
                    if (state_q == ST_WAIT_CURRENT_DESC) begin
                        current_gate_q <= descriptor_gate_code;
                        current_lane_q <= descriptor_lane_id;
                        current_remaining_q <= descriptor_destination_count;
                        current_index_q <= descriptor_term_index;
                        current_last_q <= descriptor_last;
                        current_started_q <= 1'b0;
                        state_q <= ST_WAIT_CURRENT_TERM;
                    end else begin
                        next_valid_q <= 1'b1;
                        next_gate_q <= descriptor_gate_code;
                        next_lane_q <= descriptor_lane_id;
                        next_count_q <= descriptor_destination_count;
                        next_index_q <= descriptor_term_index;
                        next_last_q <= descriptor_last;
                        next_term_issued_q <= 1'b0;
                    end
                end
            end

            if (term_fire) begin
                count_terms <= count_terms + 1'b1;
                if (state_q == ST_WAIT_CURRENT_TERM) begin
                    state_q <= ST_TOKENS;
                end else begin
                    next_term_issued_q <= 1'b1;
                end
            end

            if (word_fire) begin
                token_reservoir_q <= reservoir_after_consume |
                    (128'(word_data) << (32'(bytes_after_consume) * 8));
                token_bytes_q <= bytes_after_consume +
                                 5'(token_word_valid_bytes);
                tokens_received_q <= events_after_input;
                expected_word_index_q <= expected_word_index_q + 1'b1;
                if (word_index != expected_word_index_q ||
                    word_last != token_input_is_last) begin
                    session_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    state_q <= ST_DONE;
                end else if (token_input_is_last) begin
                    input_last_seen_q <= 1'b1;
                end
            end else if (event_fire) begin
                token_reservoir_q <= reservoir_after_consume;
                token_bytes_q <= bytes_after_consume;
            end

            if (state_q == ST_TOKENS && event_count_comb != 0 &&
                !event_tokens_in_range) begin
                session_error_q <= 1'b1;
                protocol_error <= 1'b1;
                state_q <= ST_DONE;
            end else if (event_fire) begin
                tokens_emitted_q <= tokens_emitted_q +
                                    EVENT_COUNT_W'(event_count_comb);
                count_events <= count_events + COUNTER_W'(event_count_comb);
                current_remaining_q <= current_remaining_q -
                                       8'(event_count_comb);
                current_started_q <= 1'b1;
                if (current_term_finishes) begin
                    if (current_last_q) begin
                        if (!descriptor_last_seen_q ||
                            32'(descriptors_received_q) != 32'(term_total_q) ||
                            descriptor_event_sum_q != event_total_q ||
                            tokens_emitted_q +
                                EVENT_COUNT_W'(event_count_comb) !=
                                event_total_q ||
                            !final_input_seen_comb) begin
                            session_error_q <= 1'b1;
                            protocol_error <= 1'b1;
                        end
                        count_heads <= count_heads + 1'b1;
                        state_q <= ST_DONE;
                    end else if (next_term_available_comb) begin
                        current_gate_q <= next_gate_q;
                        current_lane_q <= next_lane_q;
                        current_remaining_q <= next_count_q;
                        current_index_q <= next_index_q;
                        current_last_q <= next_last_q;
                        current_started_q <= 1'b0;
                        next_valid_q <= 1'b0;
                        next_term_issued_q <= 1'b0;
                    end else begin
                        state_q <= ST_ADVANCE;
                    end
                end
            end

            if (state_q == ST_ADVANCE && next_term_available_comb) begin
                current_gate_q <= next_gate_q;
                current_lane_q <= next_lane_q;
                current_remaining_q <= next_count_q;
                current_index_q <= next_index_q;
                current_last_q <= next_last_q;
                current_started_q <= 1'b0;
                next_valid_q <= 1'b0;
                next_term_issued_q <= 1'b0;
                state_q <= ST_TOKENS;
            end

            if (done_fire) begin
                state_q <= ST_IDLE;
            end
        end
    end

endmodule

`default_nettype wire
