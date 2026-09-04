`timescale 1ns/1ps
`default_nettype none

// Three replay representations share one tile-lifetime multihead projection
// backend. The replay route is acquired atomically with each head session.
module gatestack_routed_multihead_tile_projection_top #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int SOURCES         = 3,
    parameter int EVENT_WAYS      = 4,
    parameter int OUT_TILE        = 8,
    parameter int BANKS           = 2,
    parameter int SEGMENT_TOKENS  = 18,
    parameter int GATE_W          = 9,
    parameter int WEIGHT_W        = 8,
    parameter int PRODUCT_W       = GATE_W + WEIGHT_W,
    parameter int ACC_W           = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter int TAG_W           = 32,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int HEAD_COUNT_W    = 6,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1),
    parameter int ROUTE_W         = (SOURCES <= 1) ? 1 : $clog2(SOURCES)
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
    input  logic [HEAD_COUNT_W-1:0]                   head_start_index,
    input  logic [ROUTE_W-1:0]                        head_start_route_select,
    input  logic [INPUT_CH_W-1:0]                     head_start_input_channel_base,
    input  logic                                      head_start_last,

    input  logic [SOURCES-1:0]                        source_term_valid,
    output logic [SOURCES-1:0]                        source_term_ready,
    input  logic [(SOURCES*GATE_W)-1:0]               source_term_gate_code,
    input  logic [(SOURCES*LANE_ID_W)-1:0]            source_term_lane_id,
    input  logic [(SOURCES*8)-1:0]                    source_term_destination_count,
    input  logic [SOURCES-1:0]                        source_term_head_last,
    input  logic [SOURCES-1:0]                        source_event_valid,
    output logic [SOURCES-1:0]                        source_event_ready,
    input  logic [(SOURCES*GATE_W)-1:0]               source_event_gate_code,
    input  logic [(SOURCES*LANE_ID_W)-1:0]            source_event_lane_id,
    input  logic [(SOURCES*EVENT_WAYS)-1:0]           source_event_token_valid,
    input  logic [(SOURCES*EVENT_WAYS*TOKEN_ID_W)-1:0] source_event_token_ids,
    input  logic [(SOURCES*WAY_COUNT_W)-1:0]          source_event_count,
    input  logic [SOURCES-1:0]                        source_event_term_first,
    input  logic [SOURCES-1:0]                        source_event_term_last,
    input  logic [SOURCES-1:0]                        source_event_head_last,
    input  logic [SOURCES-1:0]                        source_done_valid,
    output logic [SOURCES-1:0]                        source_done_ready,
    input  logic [(SOURCES*TAG_W)-1:0]                source_done_tag,
    input  logic [SOURCES-1:0]                        source_done_error,

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
    output logic                                      route_active,
    output logic [ROUTE_W-1:0]                        route_active_select,
    output logic                                      protocol_error,
    output logic                                      accumulator_overflow,
    output logic [COUNTER_W-1:0]                      count_heads,
    output logic [COUNTER_W-1:0]                      count_terms,
    output logic [COUNTER_W-1:0]                      count_completed_terms,
    output logic [COUNTER_W-1:0]                      count_bias_commits
);
    logic projection_head_start_valid;
    logic projection_head_start_ready;
    logic mux_route_start_valid;
    logic mux_route_start_ready;
    logic route_legal;
    logic mux_term_valid, mux_term_ready;
    logic [GATE_W-1:0] mux_term_gate;
    logic [LANE_ID_W-1:0] mux_term_lane;
    logic [7:0] mux_term_destination_count;
    logic [ISSUE_SEQ_W-1:0] mux_term_issue_seq;
    logic mux_term_head_last;
    logic mux_event_valid, mux_event_ready;
    logic [GATE_W-1:0] mux_event_gate;
    logic [LANE_ID_W-1:0] mux_event_lane;
    logic [EVENT_WAYS-1:0] mux_event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] mux_event_token_ids;
    logic [WAY_COUNT_W-1:0] mux_event_count;
    logic [ISSUE_SEQ_W-1:0] mux_event_issue_seq;
    logic mux_event_term_first, mux_event_term_last, mux_event_head_last;
    logic mux_done_valid, mux_done_ready;
    logic [TAG_W-1:0] mux_done_tag;
    logic mux_done_error;
    logic mux_protocol_error, projection_protocol_error;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] mux_count_completed;
    logic [(SOURCES*COUNTER_W)-1:0] mux_count_routes;
    /* verilator lint_on UNUSEDSIGNAL */

    assign route_legal = 32'(head_start_route_select) < SOURCES;
    assign head_start_ready = route_legal && projection_head_start_ready &&
                              mux_route_start_ready;
    assign projection_head_start_valid = head_start_valid && route_legal &&
                                         mux_route_start_ready;
    assign mux_route_start_valid = head_start_valid && route_legal &&
                                   projection_head_start_ready;
    assign protocol_error = mux_protocol_error || projection_protocol_error ||
                            (head_start_valid && !route_legal);

    gatestack_replay_mux #(
        .SOURCES(SOURCES), .EVENT_WAYS(EVENT_WAYS),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W), .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W), .ROUTE_W(ROUTE_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) u_replay_mux (
        .clk_core(clk_core), .rst_core(rst_core),
        .route_start_valid(mux_route_start_valid),
        .route_start_ready(mux_route_start_ready),
        .route_start_select(head_start_route_select),
        .route_active(route_active), .route_active_select(route_active_select),
        .source_term_valid(source_term_valid),
        .source_term_ready(source_term_ready),
        .source_term_gate_code(source_term_gate_code),
        .source_term_lane_id(source_term_lane_id),
        .source_term_destination_count(source_term_destination_count),
        .source_term_head_last(source_term_head_last),
        .source_event_valid(source_event_valid),
        .source_event_ready(source_event_ready),
        .source_event_gate_code(source_event_gate_code),
        .source_event_lane_id(source_event_lane_id),
        .source_event_token_valid(source_event_token_valid),
        .source_event_token_ids(source_event_token_ids),
        .source_event_count(source_event_count),
        .source_event_term_first(source_event_term_first),
        .source_event_term_last(source_event_term_last),
        .source_event_head_last(source_event_head_last),
        .source_done_valid(source_done_valid),
        .source_done_ready(source_done_ready),
        .source_done_tag(source_done_tag),
        .source_done_error(source_done_error),
        .term_valid(mux_term_valid), .term_ready(mux_term_ready),
        .term_gate_code(mux_term_gate), .term_lane_id(mux_term_lane),
        .term_destination_count(mux_term_destination_count),
        .term_issue_seq(mux_term_issue_seq),
        .term_head_last(mux_term_head_last),
        .event_valid(mux_event_valid), .event_ready(mux_event_ready),
        .event_gate_code(mux_event_gate), .event_lane_id(mux_event_lane),
        .event_token_valid(mux_event_token_valid),
        .event_token_ids(mux_event_token_ids),
        .event_count(mux_event_count), .event_issue_seq(mux_event_issue_seq),
        .event_term_first(mux_event_term_first),
        .event_term_last(mux_event_term_last),
        .event_head_last(mux_event_head_last), .done_valid(mux_done_valid),
        .done_ready(mux_done_ready), .done_tag(mux_done_tag),
        .done_error(mux_done_error), .protocol_error(mux_protocol_error),
        .count_completed_heads(mux_count_completed),
        .count_route_heads(mux_count_routes)
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
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready), .tile_start_tag(tile_start_tag),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_start_valid(projection_head_start_valid),
        .head_start_ready(projection_head_start_ready),
        .head_start_tag(head_start_tag), .head_start_index(head_start_index),
        .head_start_input_channel_base(head_start_input_channel_base),
        .head_start_last(head_start_last),
        .term_valid(mux_term_valid), .term_ready(mux_term_ready),
        .term_gate_code(mux_term_gate), .term_lane_id(mux_term_lane),
        .term_destination_count(mux_term_destination_count),
        .term_issue_seq(mux_term_issue_seq),
        .term_head_last(mux_term_head_last),
        .event_valid(mux_event_valid), .event_ready(mux_event_ready),
        .event_gate_code(mux_event_gate), .event_lane_id(mux_event_lane),
        .event_token_valid(mux_event_token_valid),
        .event_token_ids(mux_event_token_ids),
        .event_count(mux_event_count), .event_issue_seq(mux_event_issue_seq),
        .event_term_first(mux_event_term_first),
        .event_term_last(mux_event_term_last),
        .event_head_last(mux_event_head_last),
        .source_done_valid(mux_done_valid), .source_done_ready(mux_done_ready),
        .source_done_tag(mux_done_tag), .source_done_error(mux_done_error),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag), .head_done_index(head_done_index),
        .head_done_last(head_done_last), .head_done_error(head_done_error),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready), .weight_rsp_tag(weight_rsp_tag),
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
        .bias_rsp_values(bias_rsp_values), .final_valid(final_valid),
        .final_ready(final_ready), .final_token_ids(final_token_ids),
        .final_tag(final_tag), .final_values(final_values),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag),
        .protocol_error(projection_protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_heads(count_heads), .count_terms(count_terms),
        .count_completed_terms(count_completed_terms),
        .count_bias_commits(count_bias_commits)
    );
endmodule

`default_nettype wire
