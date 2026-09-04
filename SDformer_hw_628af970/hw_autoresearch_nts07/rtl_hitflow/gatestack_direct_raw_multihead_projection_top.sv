`timescale 1ns/1ps
`default_nettype none

// Physically stripped RAW41 baseline. It keeps the same multihead projection
// backend but contains no resident/CSR decoder, replay mux, or cache-fill path.
module gatestack_direct_raw_multihead_projection_top #(
    parameter int TOKENS = 162,
    parameter int LANES = 32,
    parameter int EVENT_WAYS = 4,
    parameter int OUT_TILE = 8,
    parameter int BANKS = 2,
    parameter int SEGMENT_TOKENS = 18,
    parameter int GATE_W = 9,
    parameter int WEIGHT_W = 8,
    parameter int PRODUCT_W = GATE_W + WEIGHT_W,
    parameter int ACC_W = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int ISSUE_SEQ_W = 13,
    parameter int HEAD_COUNT_W = 6,
    parameter int WORD_INDEX_W = 7,
    parameter int COUNTER_W = 32,
    parameter int TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      tile_start_valid,
    output logic                                      tile_start_ready,
    input  logic [TAG_W-1:0]                          tile_start_tag,
    input  logic [OUTPUT_TILE_W-1:0]                  tile_start_output_tile,
    input  logic [HEAD_COUNT_W-1:0]                   tile_start_head_count,
    input  logic                                      head_start_valid,
    output logic                                      head_start_ready,
    input  logic [TAG_W-1:0]                          head_start_tag,
    input  logic [TAG_W-1:0]                          head_start_payload_tag,
    input  logic [HEAD_COUNT_W-1:0]                   head_start_index,
    input  logic [INPUT_CH_W-1:0]                     head_start_input_channel_base,
    input  logic                                      head_start_last,
    input  logic                                      raw_word_valid,
    output logic                                      raw_word_ready,
    input  logic [63:0]                               raw_word_data,
    input  logic [WORD_INDEX_W-1:0]                   raw_word_index,
    input  logic                                      raw_word_last,
    output logic                                      decoder_done_valid,
    input  logic                                      decoder_done_ready,
    output logic [TAG_W-1:0]                          decoder_done_payload_tag,
    output logic                                      decoder_done_error,
    output logic                                      head_done_valid,
    input  logic                                      head_done_ready,
    output logic [TAG_W-1:0]                          head_done_tag,
    output logic [HEAD_COUNT_W-1:0]                   head_done_index,
    output logic                                      head_done_last,
    output logic                                      head_done_error,
    output logic                                      weight_req_valid,
    input  logic                                      weight_req_ready,
    output logic [TAG_W-1:0]                          weight_req_tag,
    output logic [INPUT_CH_W-1:0]                     weight_req_input_channel,
    output logic [OUTPUT_TILE_W-1:0]                  weight_req_output_tile,
    input  logic                                      weight_rsp_valid,
    output logic                                      weight_rsp_ready,
    input  logic [TAG_W-1:0]                          weight_rsp_tag,
    input  logic [INPUT_CH_W-1:0]                     weight_rsp_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]                  weight_rsp_output_tile,
    input  logic [(OUT_TILE*WEIGHT_W)-1:0]            weight_rsp_weights,
    output logic                                      bias_req_valid,
    input  logic                                      bias_req_ready,
    output logic [TAG_W-1:0]                          bias_req_tag,
    output logic [OUTPUT_TILE_W-1:0]                  bias_req_output_tile,
    output logic [TOKEN_ID_W-1:0]                     bias_req_token_id,
    input  logic                                      bias_rsp_valid,
    output logic                                      bias_rsp_ready,
    input  logic [TAG_W-1:0]                          bias_rsp_tag,
    input  logic [TOKEN_ID_W-1:0]                     bias_rsp_token_id,
    input  logic [(OUT_TILE*ACC_W)-1:0]               bias_rsp_values,
    output logic [BANKS-1:0]                          final_valid,
    input  logic [BANKS-1:0]                          final_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]             final_token_ids,
    output logic [TAG_W-1:0]                          final_tag,
    output logic [(BANKS*OUT_TILE*ACC_W)-1:0]         final_values,
    output logic                                      tile_done_valid,
    input  logic                                      tile_done_ready,
    output logic [TAG_W-1:0]                          tile_done_tag,
    output logic                                      protocol_error,
    output logic                                      accumulator_overflow,
    output logic [COUNTER_W-1:0]                      count_heads,
    output logic [COUNTER_W-1:0]                      count_terms,
    output logic [COUNTER_W-1:0]                      count_completed_terms,
    output logic [COUNTER_W-1:0]                      count_bias_commits,
    output logic [COUNTER_W-1:0]                      count_raw_records,
    output logic [COUNTER_W-1:0]                      count_raw_events
);
    logic projection_head_start_valid, projection_head_start_ready;
    logic raw_start_valid, raw_start_ready;
    logic raw_direct_valid, raw_direct_ready;
    logic [GATE_W-1:0] raw_direct_gate;
    logic [LANE_ID_W-1:0] raw_direct_lane;
    logic [TOKEN_ID_W-1:0] raw_direct_token;
    /* verilator lint_off UNUSEDSIGNAL */
    logic raw_direct_head_last;
    /* verilator lint_on UNUSEDSIGNAL */
    logic raw_done_valid, raw_done_ready;
    logic [TAG_W-1:0] raw_done_tag;
    logic raw_done_error, raw_protocol_error;
    logic raw_tail_valid, raw_tail_ready;
    logic [GATE_W-1:0] raw_tail_gate;
    logic [LANE_ID_W-1:0] raw_tail_lane;
    logic [TOKEN_ID_W-1:0] raw_tail_token;
    logic raw_tail_head_last;
    logic raw_tail_done_valid, raw_tail_done_ready;
    logic [TAG_W-1:0] raw_tail_done_tag;
    logic raw_tail_done_error, raw_tail_protocol_error;
    logic adapter_term_valid, adapter_term_ready;
    logic [GATE_W-1:0] adapter_term_gate;
    logic [LANE_ID_W-1:0] adapter_term_lane;
    logic [7:0] adapter_term_destinations;
    logic [ISSUE_SEQ_W-1:0] adapter_term_issue_seq;
    logic adapter_term_head_last;
    logic adapter_event_valid, adapter_event_ready;
    logic [GATE_W-1:0] adapter_event_gate;
    logic [LANE_ID_W-1:0] adapter_event_lane;
    logic [EVENT_WAYS-1:0] adapter_event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] adapter_event_token_ids;
    logic [WAY_COUNT_W-1:0] adapter_event_count;
    logic [ISSUE_SEQ_W-1:0] adapter_event_issue_seq;
    logic adapter_event_first, adapter_event_last, adapter_event_head_last;
    logic source_done_valid, source_done_ready, source_done_error;
    logic [TAG_W-1:0] source_done_tag;
    logic raw_adapter_busy_q, raw_completion_blocked;
    logic head_start_fire, source_done_fire;
    logic [TAG_W-1:0] expected_payload_tag_q, active_execution_tag_q;
    logic decoder_done_valid_q, decoder_done_error_q;
    logic [TAG_W-1:0] decoder_done_payload_tag_q;
    logic projection_protocol_error;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] count_raw_heads, count_raw_kzero;
    logic [COUNTER_W-1:0] count_raw_input_stall, count_raw_output_stall;
    logic [COUNTER_W-1:0] count_tail_inputs, count_tail_outputs;
    logic [COUNTER_W-1:0] count_tail_empty;
    logic [COUNTER_W-1:0] count_adapter_inputs;
    logic [COUNTER_W-1:0] count_adapter_term_stall;
    logic [COUNTER_W-1:0] count_adapter_event_stall;
    /* verilator lint_on UNUSEDSIGNAL */

    assign head_start_ready = projection_head_start_ready && raw_start_ready &&
                              !decoder_done_valid_q;
    assign head_start_fire = head_start_valid && head_start_ready;
    assign projection_head_start_valid = head_start_valid && raw_start_ready &&
                                         !decoder_done_valid_q;
    assign raw_start_valid = head_start_valid && projection_head_start_ready &&
                             !decoder_done_valid_q;
    assign decoder_done_valid = decoder_done_valid_q;
    assign decoder_done_payload_tag = decoder_done_payload_tag_q;
    assign decoder_done_error = decoder_done_error_q;

    assign raw_completion_blocked = raw_adapter_busy_q || adapter_event_valid;
    assign raw_tail_done_ready = source_done_ready && !raw_completion_blocked;
    assign source_done_valid = raw_tail_done_valid && !raw_completion_blocked;
    assign source_done_tag = active_execution_tag_q;
    assign source_done_error = raw_tail_done_error ||
                               raw_tail_done_tag != expected_payload_tag_q;
    assign source_done_fire = source_done_valid && source_done_ready;
    assign protocol_error = raw_protocol_error || raw_tail_protocol_error ||
                            projection_protocol_error ||
                            (raw_tail_done_valid && source_done_error);

    gatestack_raw41_replay_decoder #(
        .TOKENS(TOKENS), .LANES(LANES), .TAG_W(TAG_W),
        .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W)
    ) u_raw_decoder (
        .clk_core(clk_core), .rst_core(rst_core),
        .start_valid(raw_start_valid), .start_ready(raw_start_ready),
        .start_tag(head_start_payload_tag),
        .word_valid(raw_word_valid), .word_ready(raw_word_ready),
        .word_data(raw_word_data), .word_index(raw_word_index),
        .word_last(raw_word_last),
        .direct_valid(raw_direct_valid), .direct_ready(raw_direct_ready),
        .direct_gate_code(raw_direct_gate), .direct_lane_id(raw_direct_lane),
        .direct_token_id(raw_direct_token),
        .direct_head_last(raw_direct_head_last),
        .done_valid(raw_done_valid), .done_ready(raw_done_ready),
        .done_tag(raw_done_tag), .done_error(raw_done_error),
        .protocol_error(raw_protocol_error), .count_heads(count_raw_heads),
        .count_records(count_raw_records),
        .count_kzero_records(count_raw_kzero),
        .count_direct_events(count_raw_events),
        .count_input_stall_cycles(count_raw_input_stall),
        .count_output_stall_cycles(count_raw_output_stall)
    );

    gatestack_raw_tail_retimer #(
        .TAG_W(TAG_W), .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W), .COUNTER_W(COUNTER_W)
    ) u_raw_tail_retimer (
        .clk_core(clk_core), .rst_core(rst_core),
        .input_valid(raw_direct_valid), .input_ready(raw_direct_ready),
        .input_gate_code(raw_direct_gate), .input_lane_id(raw_direct_lane),
        .input_token_id(raw_direct_token),
        .input_done_valid(raw_done_valid), .input_done_ready(raw_done_ready),
        .input_done_tag(raw_done_tag), .input_done_error(raw_done_error),
        .output_valid(raw_tail_valid), .output_ready(raw_tail_ready),
        .output_gate_code(raw_tail_gate), .output_lane_id(raw_tail_lane),
        .output_token_id(raw_tail_token),
        .output_head_last(raw_tail_head_last),
        .output_done_valid(raw_tail_done_valid),
        .output_done_ready(raw_tail_done_ready),
        .output_done_tag(raw_tail_done_tag),
        .output_done_error(raw_tail_done_error),
        .protocol_error(raw_tail_protocol_error),
        .count_inputs(count_tail_inputs), .count_outputs(count_tail_outputs),
        .count_empty_sessions(count_tail_empty)
    );

    gatestack_raw_issue_adapter #(
        .EVENT_WAYS(EVENT_WAYS), .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .WAY_COUNT_W(WAY_COUNT_W), .COUNTER_W(COUNTER_W)
    ) u_raw_adapter (
        .clk_core(clk_core), .rst_core(rst_core),
        .direct_valid(raw_tail_valid), .direct_ready(raw_tail_ready),
        .direct_gate_code(raw_tail_gate), .direct_lane_id(raw_tail_lane),
        .direct_token_id(raw_tail_token),
        .direct_head_last(raw_tail_head_last),
        .term_valid(adapter_term_valid), .term_ready(adapter_term_ready),
        .term_gate_code(adapter_term_gate), .term_lane_id(adapter_term_lane),
        .term_destination_count(adapter_term_destinations),
        .term_issue_seq(adapter_term_issue_seq),
        .term_head_last(adapter_term_head_last),
        .event_valid(adapter_event_valid), .event_ready(adapter_event_ready),
        .event_gate_code(adapter_event_gate), .event_lane_id(adapter_event_lane),
        .event_token_valid(adapter_event_token_valid),
        .event_token_ids(adapter_event_token_ids),
        .event_count(adapter_event_count),
        .event_issue_seq(adapter_event_issue_seq),
        .event_term_first(adapter_event_first),
        .event_term_last(adapter_event_last),
        .event_head_last(adapter_event_head_last),
        .count_direct_inputs(count_adapter_inputs),
        .count_term_stall_cycles(count_adapter_term_stall),
        .count_event_stall_cycles(count_adapter_event_stall)
    );

    gatestack_multihead_tile_projection_top #(
        .TOKENS(TOKENS), .LANES(LANES), .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(SEGMENT_TOKENS), .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W), .ACC_W(ACC_W),
        .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .TAG_W(TAG_W), .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) u_projection (
        .clk_core(clk_core), .rst_core(rst_core),
        .tile_start_valid(tile_start_valid), .tile_start_ready(tile_start_ready),
        .tile_start_tag(tile_start_tag),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_start_valid(projection_head_start_valid),
        .head_start_ready(projection_head_start_ready),
        .head_start_tag(head_start_tag), .head_start_index(head_start_index),
        .head_start_input_channel_base(head_start_input_channel_base),
        .head_start_last(head_start_last),
        .term_valid(adapter_term_valid), .term_ready(adapter_term_ready),
        .term_gate_code(adapter_term_gate), .term_lane_id(adapter_term_lane),
        .term_destination_count(adapter_term_destinations),
        .term_issue_seq(adapter_term_issue_seq),
        .term_head_last(adapter_term_head_last),
        .event_valid(adapter_event_valid), .event_ready(adapter_event_ready),
        .event_gate_code(adapter_event_gate),
        .event_lane_id(adapter_event_lane),
        .event_token_valid(adapter_event_token_valid),
        .event_token_ids(adapter_event_token_ids),
        .event_count(adapter_event_count),
        .event_issue_seq(adapter_event_issue_seq),
        .event_term_first(adapter_event_first),
        .event_term_last(adapter_event_last),
        .event_head_last(adapter_event_head_last),
        .source_done_valid(source_done_valid),
        .source_done_ready(source_done_ready),
        .source_done_tag(source_done_tag), .source_done_error(source_done_error),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag), .head_done_index(head_done_index),
        .head_done_last(head_done_last), .head_done_error(head_done_error),
        .weight_req_valid(weight_req_valid), .weight_req_ready(weight_req_ready),
        .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid), .weight_rsp_ready(weight_rsp_ready),
        .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .bias_req_valid(bias_req_valid), .bias_req_ready(bias_req_ready),
        .bias_req_tag(bias_req_tag),
        .bias_req_output_tile(bias_req_output_tile),
        .bias_req_token_id(bias_req_token_id),
        .bias_rsp_valid(bias_rsp_valid), .bias_rsp_ready(bias_rsp_ready),
        .bias_rsp_tag(bias_rsp_tag),
        .bias_rsp_token_id(bias_rsp_token_id),
        .bias_rsp_values(bias_rsp_values),
        .final_valid(final_valid), .final_ready(final_ready),
        .final_token_ids(final_token_ids), .final_tag(final_tag),
        .final_values(final_values),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag),
        .protocol_error(projection_protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_heads(count_heads), .count_terms(count_terms),
        .count_completed_terms(count_completed_terms),
        .count_bias_commits(count_bias_commits)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            raw_adapter_busy_q <= 1'b0;
            expected_payload_tag_q <= '0;
            active_execution_tag_q <= '0;
            decoder_done_valid_q <= 1'b0;
            decoder_done_payload_tag_q <= '0;
            decoder_done_error_q <= 1'b0;
        end else begin
            if (decoder_done_valid_q && decoder_done_ready)
                decoder_done_valid_q <= 1'b0;
            if (head_start_fire) begin
                expected_payload_tag_q <= head_start_payload_tag;
                active_execution_tag_q <= head_start_tag;
                raw_adapter_busy_q <= 1'b0;
            end
            if (raw_tail_valid && raw_tail_ready)
                raw_adapter_busy_q <= 1'b1;
            if (adapter_event_valid && adapter_event_ready)
                raw_adapter_busy_q <= 1'b0;
            if (source_done_fire) begin
                decoder_done_valid_q <= 1'b1;
                decoder_done_payload_tag_q <= raw_tail_done_tag;
                decoder_done_error_q <= source_done_error;
            end
        end
    end
endmodule

`default_nettype wire
