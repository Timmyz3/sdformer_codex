`timescale 1ns/1ps
`default_nettype none

// Typed exact CSR frontend. Commit-time metadata directly selects IPD32W or
// FADC24; a legacy path can still buffer word0 and select the child by magic.
module gatestack_adaptive_csr_replay_decoder #(
    parameter int TOKENS = 162,
    parameter int LANES = 32,
    parameter int MAX_TERMS = 128,
    parameter int EVENT_WAYS = 4,
    parameter int TAG_W = 32,
    parameter int WORD_INDEX_W = 7,
    parameter int COUNTER_W = 32,
    parameter int TOKEN_ID_W = 8,
    parameter int LANE_ID_W = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int TERM_INDEX_W = (MAX_TERMS <= 1) ? 1 : $clog2(MAX_TERMS),
    parameter int EVENT_COUNT_W = 13,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input  logic                               clk_core,
    input  logic                               rst_core,
    input  logic                               start_valid,
    output logic                               start_ready,
    input  logic                               start_format_valid,
    input  logic                               start_select_fadc,
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
    output logic [COUNTER_W-1:0]               count_input_stall_cycles,
    output logic [COUNTER_W-1:0]               count_term_stall_cycles,
    output logic [COUNTER_W-1:0]               count_output_stall_cycles
);
    localparam logic [15:0] FADC24_MAGIC = 16'h4641;

    typedef enum logic [1:0] {ST_IDLE, ST_PEEK, ST_START, ST_RUN} state_t;
    state_t state_q;
    logic select_fadc_q, first_word_pending_q;
    logic [63:0] first_word_data_q;
    logic [WORD_INDEX_W-1:0] first_word_index_q;
    logic first_word_last_q;

    logic [1:0] child_start_valid, child_start_ready;
    logic [1:0] child_word_valid, child_word_ready;
    logic [1:0] child_descriptor_valid, child_descriptor_ready;
    logic [(2*TAG_W)-1:0] child_descriptor_tag;
    logic [15:0] child_descriptor_terms;
    logic [1:0] child_term_valid, child_term_ready;
    logic [17:0] child_term_gate;
    logic [(2*LANE_ID_W)-1:0] child_term_lane;
    logic [15:0] child_term_destinations;
    logic [(2*TERM_INDEX_W)-1:0] child_term_index;
    logic [1:0] child_term_head_last;
    logic [1:0] child_event_valid, child_event_ready;
    logic [17:0] child_event_gate;
    logic [(2*LANE_ID_W)-1:0] child_event_lane;
    logic [(2*EVENT_WAYS)-1:0] child_event_token_valid;
    logic [(2*EVENT_WAYS*TOKEN_ID_W)-1:0] child_event_token_ids;
    logic [(2*WAY_COUNT_W)-1:0] child_event_count;
    logic [1:0] child_event_first, child_event_last, child_event_head_last;
    logic [1:0] child_done_valid, child_done_ready, child_done_error;
    logic [(2*TAG_W)-1:0] child_done_tag;
    logic [1:0] child_protocol_error;
    logic [(2*COUNTER_W)-1:0] child_count_heads, child_count_terms;
    logic [(2*COUNTER_W)-1:0] child_count_events;
    logic [(2*COUNTER_W)-1:0] child_count_input_stall;
    logic [(2*COUNTER_W)-1:0] child_count_term_stall;
    logic [(2*COUNTER_W)-1:0] child_count_output_stall;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] fadc_count_bitmap_terms;
    /* verilator lint_on UNUSEDSIGNAL */

    logic selected_word_ready;
    logic [63:0] forwarded_word_data;
    logic [WORD_INDEX_W-1:0] forwarded_word_index;
    logic forwarded_word_last;
    logic peek_fire, selected_done_fire;

    assign start_ready = state_q == ST_IDLE;
    assign peek_fire = state_q == ST_PEEK && word_valid && word_ready;
    assign selected_word_ready = child_word_ready[select_fadc_q];
    assign forwarded_word_data = first_word_pending_q ? first_word_data_q : word_data;
    assign forwarded_word_index = first_word_pending_q ? first_word_index_q : word_index;
    assign forwarded_word_last = first_word_pending_q ? first_word_last_q : word_last;
    assign word_ready = state_q == ST_PEEK ||
        (state_q == ST_RUN && !first_word_pending_q && selected_word_ready);

    always_comb begin
        child_start_valid = '0;
        child_word_valid = '0;
        child_descriptor_ready = '0;
        child_term_ready = '0;
        child_event_ready = '0;
        child_done_ready = '0;
        if (state_q == ST_START)
            child_start_valid[select_fadc_q] = 1'b1;
        if (state_q == ST_RUN) begin
            child_word_valid[select_fadc_q] = first_word_pending_q || word_valid;
            child_descriptor_ready[select_fadc_q] = descriptor_begin_ready;
            child_term_ready[select_fadc_q] = term_ready;
            child_event_ready[select_fadc_q] = event_ready;
            child_done_ready[select_fadc_q] = done_ready;
        end
    end

    assign descriptor_begin_valid = child_descriptor_valid[select_fadc_q];
    assign descriptor_begin_tag = child_descriptor_tag[select_fadc_q*TAG_W +: TAG_W];
    assign descriptor_begin_term_count = child_descriptor_terms[select_fadc_q*8 +: 8];
    assign term_valid = child_term_valid[select_fadc_q];
    assign term_gate_code = child_term_gate[select_fadc_q*9 +: 9];
    assign term_lane_id = child_term_lane[select_fadc_q*LANE_ID_W +: LANE_ID_W];
    assign term_destination_count = child_term_destinations[select_fadc_q*8 +: 8];
    assign term_index = child_term_index[select_fadc_q*TERM_INDEX_W +: TERM_INDEX_W];
    assign term_head_last = child_term_head_last[select_fadc_q];
    assign event_valid = child_event_valid[select_fadc_q];
    assign event_gate_code = child_event_gate[select_fadc_q*9 +: 9];
    assign event_lane_id = child_event_lane[select_fadc_q*LANE_ID_W +: LANE_ID_W];
    assign event_token_valid = child_event_token_valid[select_fadc_q*EVENT_WAYS +: EVENT_WAYS];
    assign event_token_ids = child_event_token_ids[
        select_fadc_q*EVENT_WAYS*TOKEN_ID_W +: EVENT_WAYS*TOKEN_ID_W];
    assign event_count = child_event_count[select_fadc_q*WAY_COUNT_W +: WAY_COUNT_W];
    assign event_term_first = child_event_first[select_fadc_q];
    assign event_term_last = child_event_last[select_fadc_q];
    assign event_head_last = child_event_head_last[select_fadc_q];
    assign done_valid = child_done_valid[select_fadc_q];
    assign done_tag = child_done_tag[select_fadc_q*TAG_W +: TAG_W];
    assign done_error = child_done_error[select_fadc_q];
    assign selected_done_fire = done_valid && done_ready;
    assign protocol_error = |child_protocol_error;
    assign count_heads = child_count_heads[COUNTER_W-1:0] +
                         child_count_heads[COUNTER_W +: COUNTER_W];
    assign count_terms = child_count_terms[COUNTER_W-1:0] +
                         child_count_terms[COUNTER_W +: COUNTER_W];
    assign count_events = child_count_events[COUNTER_W-1:0] +
                          child_count_events[COUNTER_W +: COUNTER_W];
    assign count_input_stall_cycles = child_count_input_stall[COUNTER_W-1:0] +
        child_count_input_stall[COUNTER_W +: COUNTER_W];
    assign count_term_stall_cycles = child_count_term_stall[COUNTER_W-1:0] +
        child_count_term_stall[COUNTER_W +: COUNTER_W];
    assign count_output_stall_cycles = child_count_output_stall[COUNTER_W-1:0] +
        child_count_output_stall[COUNTER_W +: COUNTER_W];

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            select_fadc_q <= 1'b0;
            first_word_pending_q <= 1'b0;
            first_word_data_q <= '0;
            first_word_index_q <= '0;
            first_word_last_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: if (start_valid && start_ready) begin
                    first_word_pending_q <= 1'b0;
                    if (start_format_valid) begin
                        select_fadc_q <= start_select_fadc;
                        state_q <= ST_START;
                    end else begin
                        state_q <= ST_PEEK;
                    end
                end
                ST_PEEK: if (peek_fire) begin
                    select_fadc_q <= word_data[15:0] == FADC24_MAGIC;
                    first_word_pending_q <= 1'b1;
                    first_word_data_q <= word_data;
                    first_word_index_q <= word_index;
                    first_word_last_q <= word_last;
                    state_q <= ST_START;
                end
                ST_START: if (child_start_valid[select_fadc_q] &&
                              child_start_ready[select_fadc_q]) state_q <= ST_RUN;
                ST_RUN: begin
                    if (first_word_pending_q && child_word_valid[select_fadc_q] &&
                        selected_word_ready) first_word_pending_q <= 1'b0;
                    if (selected_done_fire) state_q <= ST_IDLE;
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end

    gatestack_ipd32w_replay_decoder #(
        .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(MAX_TERMS),
        .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W), .WORD_INDEX_W(WORD_INDEX_W),
        .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W), .TERM_INDEX_W(TERM_INDEX_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .WAY_COUNT_W(WAY_COUNT_W)
    ) u_ipd32w (.*,
        .start_valid(child_start_valid[0]), .start_ready(child_start_ready[0]),
        .word_valid(child_word_valid[0]), .word_ready(child_word_ready[0]),
        .word_data(forwarded_word_data), .word_index(forwarded_word_index),
        .word_last(forwarded_word_last),
        .descriptor_begin_valid(child_descriptor_valid[0]),
        .descriptor_begin_ready(child_descriptor_ready[0]),
        .descriptor_begin_tag(child_descriptor_tag[0 +: TAG_W]),
        .descriptor_begin_term_count(child_descriptor_terms[0 +: 8]),
        .term_valid(child_term_valid[0]), .term_ready(child_term_ready[0]),
        .term_gate_code(child_term_gate[0 +: 9]),
        .term_lane_id(child_term_lane[0 +: LANE_ID_W]),
        .term_destination_count(child_term_destinations[0 +: 8]),
        .term_index(child_term_index[0 +: TERM_INDEX_W]),
        .term_head_last(child_term_head_last[0]),
        .event_valid(child_event_valid[0]), .event_ready(child_event_ready[0]),
        .event_gate_code(child_event_gate[0 +: 9]),
        .event_lane_id(child_event_lane[0 +: LANE_ID_W]),
        .event_token_valid(child_event_token_valid[0 +: EVENT_WAYS]),
        .event_token_ids(child_event_token_ids[0 +: EVENT_WAYS*TOKEN_ID_W]),
        .event_count(child_event_count[0 +: WAY_COUNT_W]),
        .event_term_first(child_event_first[0]),
        .event_term_last(child_event_last[0]),
        .event_head_last(child_event_head_last[0]),
        .done_valid(child_done_valid[0]), .done_ready(child_done_ready[0]),
        .done_tag(child_done_tag[0 +: TAG_W]), .done_error(child_done_error[0]),
        .protocol_error(child_protocol_error[0]),
        .count_heads(child_count_heads[0 +: COUNTER_W]),
        .count_terms(child_count_terms[0 +: COUNTER_W]),
        .count_events(child_count_events[0 +: COUNTER_W]),
        .count_input_stall_cycles(child_count_input_stall[0 +: COUNTER_W]),
        .count_term_stall_cycles(child_count_term_stall[0 +: COUNTER_W]),
        .count_output_stall_cycles(child_count_output_stall[0 +: COUNTER_W]));

    gatestack_fadc24_streaming_replay_decoder #(
        .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(MAX_TERMS),
        .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W), .WORD_INDEX_W(WORD_INDEX_W),
        .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W), .TERM_INDEX_W(TERM_INDEX_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .WAY_COUNT_W(WAY_COUNT_W)
    ) u_fadc24 (.*,
        .start_valid(child_start_valid[1]), .start_ready(child_start_ready[1]),
        .word_valid(child_word_valid[1]), .word_ready(child_word_ready[1]),
        .word_data(forwarded_word_data), .word_index(forwarded_word_index),
        .word_last(forwarded_word_last),
        .descriptor_begin_valid(child_descriptor_valid[1]),
        .descriptor_begin_ready(child_descriptor_ready[1]),
        .descriptor_begin_tag(child_descriptor_tag[TAG_W +: TAG_W]),
        .descriptor_begin_term_count(child_descriptor_terms[8 +: 8]),
        .term_valid(child_term_valid[1]), .term_ready(child_term_ready[1]),
        .term_gate_code(child_term_gate[9 +: 9]),
        .term_lane_id(child_term_lane[LANE_ID_W +: LANE_ID_W]),
        .term_destination_count(child_term_destinations[8 +: 8]),
        .term_index(child_term_index[TERM_INDEX_W +: TERM_INDEX_W]),
        .term_head_last(child_term_head_last[1]),
        .event_valid(child_event_valid[1]), .event_ready(child_event_ready[1]),
        .event_gate_code(child_event_gate[9 +: 9]),
        .event_lane_id(child_event_lane[LANE_ID_W +: LANE_ID_W]),
        .event_token_valid(child_event_token_valid[EVENT_WAYS +: EVENT_WAYS]),
        .event_token_ids(child_event_token_ids[EVENT_WAYS*TOKEN_ID_W +:
                                               EVENT_WAYS*TOKEN_ID_W]),
        .event_count(child_event_count[WAY_COUNT_W +: WAY_COUNT_W]),
        .event_term_first(child_event_first[1]),
        .event_term_last(child_event_last[1]),
        .event_head_last(child_event_head_last[1]),
        .done_valid(child_done_valid[1]), .done_ready(child_done_ready[1]),
        .done_tag(child_done_tag[TAG_W +: TAG_W]),
        .done_error(child_done_error[1]),
        .protocol_error(child_protocol_error[1]),
        .count_heads(child_count_heads[COUNTER_W +: COUNTER_W]),
        .count_terms(child_count_terms[COUNTER_W +: COUNTER_W]),
        .count_events(child_count_events[COUNTER_W +: COUNTER_W]),
        .count_bitmap_terms(fadc_count_bitmap_terms),
        .count_input_stall_cycles(child_count_input_stall[COUNTER_W +: COUNTER_W]),
        .count_term_stall_cycles(child_count_term_stall[COUNTER_W +: COUNTER_W]),
        .count_output_stall_cycles(child_count_output_stall[COUNTER_W +: COUNTER_W]));
endmodule

`default_nettype wire
