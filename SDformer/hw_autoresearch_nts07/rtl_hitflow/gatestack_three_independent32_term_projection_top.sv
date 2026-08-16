`timescale 1ns/1ps
`default_nettype none

// Projection-only comparison wrapper. The three engines share only clock and
// reset; all control, term/event, memory, completion, and counter ports remain
// independent and are exposed as packed engine-major vectors.
module gatestack_three_independent32_term_projection_top #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int EVENT_WAYS      = 4,
    parameter int BANKS           = 2,
    parameter int SEGMENT_TOKENS  = 18,
    parameter int GATE_W          = 9,
    parameter int WEIGHT_W        = 8,
    parameter int PRODUCT_W       = GATE_W + WEIGHT_W,
    parameter int ACC_W           = 32,
    parameter int TAG_W           = 32,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int HEAD_COUNT_W    = 6,
    parameter int COUNTER_W       = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter int TOKEN_ID_W      = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic [2:0]                                tile_start_valid,
    output logic [2:0]                                tile_start_ready,
    input  logic [(3*TAG_W)-1:0]                      tile_start_tags,
    input  logic [(3*OUTPUT_TILE_W)-1:0]              tile_start_output_tiles,
    input  logic [(3*HEAD_COUNT_W)-1:0]               tile_start_head_counts,

    input  logic [2:0]                                head_start_valid,
    output logic [2:0]                                head_start_ready,
    input  logic [(3*TAG_W)-1:0]                      head_start_tags,
    input  logic [(3*HEAD_COUNT_W)-1:0]               head_start_indices,
    input  logic [(3*INPUT_CH_W)-1:0]                 head_start_input_channel_bases,
    input  logic [2:0]                                head_start_last,

    input  logic [2:0]                                term_valid,
    output logic [2:0]                                term_ready,
    input  logic [(3*GATE_W)-1:0]                     term_gate_codes,
    input  logic [(3*LANE_ID_W)-1:0]                  term_lane_ids,
    input  logic [(3*8)-1:0]                          term_destination_counts,
    input  logic [(3*ISSUE_SEQ_W)-1:0]                term_issue_seqs,
    input  logic [2:0]                                term_head_last,

    input  logic [2:0]                                event_valid,
    output logic [2:0]                                event_ready,
    input  logic [(3*GATE_W)-1:0]                     event_gate_codes,
    input  logic [(3*LANE_ID_W)-1:0]                  event_lane_ids,
    input  logic [(3*EVENT_WAYS)-1:0]                 event_token_valids,
    input  logic [(3*EVENT_WAYS*TOKEN_ID_W)-1:0]      event_token_ids,
    input  logic [(3*WAY_COUNT_W)-1:0]                event_counts,
    input  logic [(3*ISSUE_SEQ_W)-1:0]                event_issue_seqs,
    input  logic [2:0]                                event_term_first,
    input  logic [2:0]                                event_term_last,
    input  logic [2:0]                                event_head_last,

    input  logic [2:0]                                source_done_valid,
    output logic [2:0]                                source_done_ready,
    input  logic [(3*TAG_W)-1:0]                      source_done_tags,
    input  logic [2:0]                                source_done_error,

    output logic [2:0]                                head_done_valid,
    input  logic [2:0]                                head_done_ready,
    output logic [(3*TAG_W)-1:0]                      head_done_tags,
    output logic [(3*HEAD_COUNT_W)-1:0]               head_done_indices,
    output logic [2:0]                                head_done_last,
    output logic [2:0]                                head_done_error,

    output logic [2:0]                                weight_req_valid,
    input  logic [2:0]                                weight_req_ready,
    output logic [(3*TAG_W)-1:0]                      weight_req_tags,
    output logic [(3*INPUT_CH_W)-1:0]                 weight_req_input_channels,
    output logic [(3*OUTPUT_TILE_W)-1:0]              weight_req_output_tiles,
    input  logic [2:0]                                weight_rsp_valid,
    output logic [2:0]                                weight_rsp_ready,
    input  logic [(3*TAG_W)-1:0]                      weight_rsp_tags,
    input  logic [(3*INPUT_CH_W)-1:0]                 weight_rsp_input_channels,
    input  logic [(3*OUTPUT_TILE_W)-1:0]              weight_rsp_output_tiles,
    input  logic [(3*32*WEIGHT_W)-1:0]                weight_rsp_weights,

    output logic [2:0]                                bias_req_valid,
    input  logic [2:0]                                bias_req_ready,
    output logic [(3*TAG_W)-1:0]                      bias_req_tags,
    output logic [(3*OUTPUT_TILE_W)-1:0]              bias_req_output_tiles,
    output logic [(3*TOKEN_ID_W)-1:0]                 bias_req_token_ids,
    input  logic [2:0]                                bias_rsp_valid,
    output logic [2:0]                                bias_rsp_ready,
    input  logic [(3*TAG_W)-1:0]                      bias_rsp_tags,
    input  logic [(3*TOKEN_ID_W)-1:0]                 bias_rsp_token_ids,
    input  logic [(3*32*ACC_W)-1:0]                   bias_rsp_values,

    output logic [(3*BANKS)-1:0]                      final_valid,
    input  logic [(3*BANKS)-1:0]                      final_ready,
    output logic [(3*BANKS*TOKEN_ID_W)-1:0]           final_token_ids,
    output logic [(3*TAG_W)-1:0]                      final_tags,
    output logic [(3*BANKS*32*ACC_W)-1:0]             final_values,

    output logic [2:0]                                tile_done_valid,
    input  logic [2:0]                                tile_done_ready,
    output logic [(3*TAG_W)-1:0]                      tile_done_tags,
    output logic [2:0]                                protocol_error,
    output logic [2:0]                                accumulator_overflow,
    output logic [(3*COUNTER_W)-1:0]                  count_heads,
    output logic [(3*COUNTER_W)-1:0]                  count_terms,
    output logic [(3*COUNTER_W)-1:0]                  count_completed_terms,
    output logic [(3*COUNTER_W)-1:0]                  count_bias_commits
);
    for (genvar engine = 0; engine < 3; engine = engine + 1) begin : g_engine
        gatestack_multihead_tile_projection_top #(
            .TOKENS(TOKENS),
            .LANES(LANES),
            .EVENT_WAYS(EVENT_WAYS),
            .OUT_TILE(32),
            .BANKS(BANKS),
            .SEGMENT_TOKENS(SEGMENT_TOKENS),
            .GATE_W(GATE_W),
            .WEIGHT_W(WEIGHT_W),
            .PRODUCT_W(PRODUCT_W),
            .ACC_W(ACC_W),
            .TAG_W(TAG_W),
            .INPUT_CH_W(INPUT_CH_W),
            .OUTPUT_TILE_W(OUTPUT_TILE_W),
            .ISSUE_SEQ_W(ISSUE_SEQ_W),
            .HEAD_COUNT_W(HEAD_COUNT_W),
            .COUNTER_W(COUNTER_W),
            .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
            .TOKEN_ID_W(TOKEN_ID_W),
            .LANE_ID_W(LANE_ID_W),
            .WAY_COUNT_W(WAY_COUNT_W)
        ) u_projection (
            .clk_core(clk_core),
            .rst_core(rst_core),
            .tile_start_valid(tile_start_valid[engine]),
            .tile_start_ready(tile_start_ready[engine]),
            .tile_start_tag(tile_start_tags[(engine*TAG_W) +: TAG_W]),
            .tile_start_output_tile(tile_start_output_tiles[
                (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
            .tile_start_head_count(tile_start_head_counts[
                (engine*HEAD_COUNT_W) +: HEAD_COUNT_W]),
            .head_start_valid(head_start_valid[engine]),
            .head_start_ready(head_start_ready[engine]),
            .head_start_tag(head_start_tags[(engine*TAG_W) +: TAG_W]),
            .head_start_index(head_start_indices[
                (engine*HEAD_COUNT_W) +: HEAD_COUNT_W]),
            .head_start_input_channel_base(head_start_input_channel_bases[
                (engine*INPUT_CH_W) +: INPUT_CH_W]),
            .head_start_last(head_start_last[engine]),
            .term_valid(term_valid[engine]),
            .term_ready(term_ready[engine]),
            .term_gate_code(term_gate_codes[(engine*GATE_W) +: GATE_W]),
            .term_lane_id(term_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W]),
            .term_destination_count(term_destination_counts[(engine*8) +: 8]),
            .term_issue_seq(term_issue_seqs[
                (engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W]),
            .term_head_last(term_head_last[engine]),
            .event_valid(event_valid[engine]),
            .event_ready(event_ready[engine]),
            .event_gate_code(event_gate_codes[(engine*GATE_W) +: GATE_W]),
            .event_lane_id(event_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W]),
            .event_token_valid(event_token_valids[
                (engine*EVENT_WAYS) +: EVENT_WAYS]),
            .event_token_ids(event_token_ids[
                (engine*EVENT_WAYS*TOKEN_ID_W) +: (EVENT_WAYS*TOKEN_ID_W)]),
            .event_count(event_counts[(engine*WAY_COUNT_W) +: WAY_COUNT_W]),
            .event_issue_seq(event_issue_seqs[
                (engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W]),
            .event_term_first(event_term_first[engine]),
            .event_term_last(event_term_last[engine]),
            .event_head_last(event_head_last[engine]),
            .source_done_valid(source_done_valid[engine]),
            .source_done_ready(source_done_ready[engine]),
            .source_done_tag(source_done_tags[(engine*TAG_W) +: TAG_W]),
            .source_done_error(source_done_error[engine]),
            .head_done_valid(head_done_valid[engine]),
            .head_done_ready(head_done_ready[engine]),
            .head_done_tag(head_done_tags[(engine*TAG_W) +: TAG_W]),
            .head_done_index(head_done_indices[
                (engine*HEAD_COUNT_W) +: HEAD_COUNT_W]),
            .head_done_last(head_done_last[engine]),
            .head_done_error(head_done_error[engine]),
            .weight_req_valid(weight_req_valid[engine]),
            .weight_req_ready(weight_req_ready[engine]),
            .weight_req_tag(weight_req_tags[(engine*TAG_W) +: TAG_W]),
            .weight_req_input_channel(weight_req_input_channels[
                (engine*INPUT_CH_W) +: INPUT_CH_W]),
            .weight_req_output_tile(weight_req_output_tiles[
                (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
            .weight_rsp_valid(weight_rsp_valid[engine]),
            .weight_rsp_ready(weight_rsp_ready[engine]),
            .weight_rsp_tag(weight_rsp_tags[(engine*TAG_W) +: TAG_W]),
            .weight_rsp_input_channel(weight_rsp_input_channels[
                (engine*INPUT_CH_W) +: INPUT_CH_W]),
            .weight_rsp_output_tile(weight_rsp_output_tiles[
                (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
            .weight_rsp_weights(weight_rsp_weights[
                (engine*32*WEIGHT_W) +: (32*WEIGHT_W)]),
            .bias_req_valid(bias_req_valid[engine]),
            .bias_req_ready(bias_req_ready[engine]),
            .bias_req_tag(bias_req_tags[(engine*TAG_W) +: TAG_W]),
            .bias_req_output_tile(bias_req_output_tiles[
                (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
            .bias_req_token_id(bias_req_token_ids[
                (engine*TOKEN_ID_W) +: TOKEN_ID_W]),
            .bias_rsp_valid(bias_rsp_valid[engine]),
            .bias_rsp_ready(bias_rsp_ready[engine]),
            .bias_rsp_tag(bias_rsp_tags[(engine*TAG_W) +: TAG_W]),
            .bias_rsp_token_id(bias_rsp_token_ids[
                (engine*TOKEN_ID_W) +: TOKEN_ID_W]),
            .bias_rsp_values(bias_rsp_values[
                (engine*32*ACC_W) +: (32*ACC_W)]),
            .final_valid(final_valid[(engine*BANKS) +: BANKS]),
            .final_ready(final_ready[(engine*BANKS) +: BANKS]),
            .final_token_ids(final_token_ids[
                (engine*BANKS*TOKEN_ID_W) +: (BANKS*TOKEN_ID_W)]),
            .final_tag(final_tags[(engine*TAG_W) +: TAG_W]),
            .final_values(final_values[
                (engine*BANKS*32*ACC_W) +: (BANKS*32*ACC_W)]),
            .tile_done_valid(tile_done_valid[engine]),
            .tile_done_ready(tile_done_ready[engine]),
            .tile_done_tag(tile_done_tags[(engine*TAG_W) +: TAG_W]),
            .protocol_error(protocol_error[engine]),
            .accumulator_overflow(accumulator_overflow[engine]),
            .count_heads(count_heads[(engine*COUNTER_W) +: COUNTER_W]),
            .count_terms(count_terms[(engine*COUNTER_W) +: COUNTER_W]),
            .count_completed_terms(count_completed_terms[
                (engine*COUNTER_W) +: COUNTER_W]),
            .count_bias_commits(count_bias_commits[
                (engine*COUNTER_W) +: COUNTER_W])
        );
    end
endmodule

`default_nettype wire
