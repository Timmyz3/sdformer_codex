`timescale 1ns/1ps
`default_nettype none

// Structural 12-block Local5 numerical shell. The frame scheduler and one
// time-multiplexed T450/OUT32 tile executor share the exact production
// tile/head completion protocol. Relation memoization and post-projection
// bias/BN/requant/residual are deliberately outside this shell.
module qfit_local5_encoder_t450_numeric_shell #(
    parameter int TAG_W = 24,
    parameter int WINDOW_W = 9,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter bit USE_SCORE_ACTIVE_FRONT = 1'b0
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         start_frame,
    output logic                         frame_busy,
    output logic                         frame_done,

    output logic                         token_req_valid,
    input  logic                         token_req_ready,
    output logic [TAG_W-1:0]             token_req_tag,
    output logic [HEAD_W-1:0]            token_req_input_head,
    output logic [8:0]                   token_req_token_id,
    output logic                         token_req_plane,
    output logic [3:0]                   token_req_y,
    output logic [3:0]                   token_req_x,
    input  logic                         token_rsp_valid,
    output logic                         token_rsp_ready,
    input  logic [TAG_W-1:0]             token_rsp_tag,
    input  logic [HEAD_W-1:0]            token_rsp_input_head,
    input  logic [8:0]                   token_rsp_token_id,
    input  logic [31:0]                  token_rsp_q,
    input  logic [159:0]                 token_rsp_k,
    input  logic [4:0]                   token_rsp_valid_mask,
    input  logic                         token_rsp_error,

    output logic                         weight_req_valid,
    input  logic                         weight_req_ready,
    output logic [TAG_W-1:0]             weight_req_tag,
    output logic [HEAD_W-1:0]            weight_req_input_head,
    output logic [OUTPUT_TILE_W-1:0]     weight_req_output_tile,
    output logic [4:0]                   weight_req_lane,
    output logic [4:0]                   weight_req_out,
    input  logic                         weight_rsp_valid,
    output logic                         weight_rsp_ready,
    input  logic [TAG_W-1:0]             weight_rsp_tag,
    input  logic [HEAD_W-1:0]            weight_rsp_input_head,
    input  logic [OUTPUT_TILE_W-1:0]     weight_rsp_output_tile,
    input  logic [4:0]                   weight_rsp_lane,
    input  logic [4:0]                   weight_rsp_out,
    input  logic signed [7:0]            weight_rsp_data,
    input  logic                         weight_rsp_error,

    output logic                         tile_result_valid,
    input  logic                         tile_result_ready,
    output logic [TAG_W-1:0]             tile_result_tag,
    output logic [OUTPUT_TILE_W-1:0]     tile_result_output_tile,
    output logic                         tile_result_plane,
    output logic [3:0]                   tile_result_y,
    output logic [3:0]                   tile_result_x,
    output logic [4:0]                   tile_result_out,
    output logic signed [31:0]           tile_result_data,
    output logic                         tile_result_last,

    output logic                         protocol_error,
    output logic [31:0]                  perf_window_groups,
    output logic [31:0]                  perf_output_tiles,
    output logic [31:0]                  perf_head_replays,
    output logic [31:0]                  perf_decode_intent_jobs,
    output logic [31:0]                  perf_release_intent_jobs,
    output logic [31:0]                  perf_numeric_tiles,
    output logic [31:0]                  perf_numeric_heads,
    output logic [31:0]                  perf_partial_results,
    output logic [31:0]                  perf_accumulator_writes,
    output logic [31:0]                  perf_final_results
);
    logic tile_start_valid;
    logic tile_start_ready;
    logic [TAG_W-1:0] tile_start_tag;
    logic [1:0] tile_start_stage;
    logic [2:0] tile_start_block;
    logic [WINDOW_W-1:0] tile_start_window;
    logic [OUTPUT_TILE_W-1:0] tile_start_output_tile;
    logic [5:0] tile_start_head_count;
    logic head_job_valid;
    logic head_job_ready;
    logic [TAG_W-1:0] head_job_tag;
    logic [1:0] head_job_stage;
    logic [2:0] head_job_block;
    logic [WINDOW_W-1:0] head_job_window;
    logic [HEAD_W-1:0] head_job_input_head;
    logic [INPUT_CH_W-1:0] head_job_input_channel_base;
    logic [OUTPUT_TILE_W-1:0] head_job_output_tile;
    logic head_job_decode_required;
    logic head_job_cache_release;
    logic head_job_last_input_head;
    logic head_job_last_output_tile;
    logic head_done_valid;
    logic head_done_ready;
    logic [TAG_W-1:0] head_done_tag;
    logic [HEAD_W-1:0] head_done_input_head;
    logic head_done_error;
    logic tile_done_valid;
    logic tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic tile_done_error;
    logic scheduler_protocol_error;
    logic executor_protocol_error;

    qfit_local5_encoder_job_scheduler #(
        .TAG_W(TAG_W), .WINDOW_W(WINDOW_W), .HEAD_W(HEAD_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .INPUT_CH_W(INPUT_CH_W)
    ) u_scheduler (
        .clk_core(clk_core), .rst_core(rst_core),
        .start_frame(start_frame), .frame_busy(frame_busy),
        .frame_done(frame_done),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready), .tile_start_tag(tile_start_tag),
        .tile_start_stage(tile_start_stage),
        .tile_start_block(tile_start_block),
        .tile_start_window(tile_start_window),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_job_valid(head_job_valid), .head_job_ready(head_job_ready),
        .head_job_tag(head_job_tag), .head_job_stage(head_job_stage),
        .head_job_block(head_job_block), .head_job_window(head_job_window),
        .head_job_input_head(head_job_input_head),
        .head_job_input_channel_base(head_job_input_channel_base),
        .head_job_output_tile(head_job_output_tile),
        .head_job_decode_required(head_job_decode_required),
        .head_job_cache_release(head_job_cache_release),
        .head_job_last_input_head(head_job_last_input_head),
        .head_job_last_output_tile(head_job_last_output_tile),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_input_head(head_done_input_head),
        .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag), .tile_done_error(tile_done_error),
        .protocol_error(scheduler_protocol_error),
        .perf_window_groups(perf_window_groups),
        .perf_output_tiles(perf_output_tiles),
        .perf_head_replays(perf_head_replays),
        .perf_decode_intent_jobs(perf_decode_intent_jobs),
        .perf_release_intent_jobs(perf_release_intent_jobs)
    );

    qfit_local5_cross_head_tile_executor #(
        .TAG_W(TAG_W), .WINDOW_W(WINDOW_W), .HEAD_W(HEAD_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .INPUT_CH_W(INPUT_CH_W),
        .USE_SCORE_ACTIVE_FRONT(USE_SCORE_ACTIVE_FRONT)
    ) u_tile_executor (
        .clk_core(clk_core), .rst_core(rst_core),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready), .tile_start_tag(tile_start_tag),
        .tile_start_stage(tile_start_stage),
        .tile_start_block(tile_start_block),
        .tile_start_window(tile_start_window),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_job_valid(head_job_valid), .head_job_ready(head_job_ready),
        .head_job_tag(head_job_tag), .head_job_stage(head_job_stage),
        .head_job_block(head_job_block), .head_job_window(head_job_window),
        .head_job_input_head(head_job_input_head),
        .head_job_input_channel_base(head_job_input_channel_base),
        .head_job_output_tile(head_job_output_tile),
        .head_job_decode_required(head_job_decode_required),
        .head_job_cache_release(head_job_cache_release),
        .head_job_last_input_head(head_job_last_input_head),
        .head_job_last_output_tile(head_job_last_output_tile),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_input_head(head_done_input_head),
        .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag), .tile_done_error(tile_done_error),
        .token_req_valid(token_req_valid), .token_req_ready(token_req_ready),
        .token_req_tag(token_req_tag),
        .token_req_input_head(token_req_input_head),
        .token_req_token_id(token_req_token_id),
        .token_req_plane(token_req_plane), .token_req_y(token_req_y),
        .token_req_x(token_req_x),
        .token_rsp_valid(token_rsp_valid), .token_rsp_ready(token_rsp_ready),
        .token_rsp_tag(token_rsp_tag),
        .token_rsp_input_head(token_rsp_input_head),
        .token_rsp_token_id(token_rsp_token_id), .token_rsp_q(token_rsp_q),
        .token_rsp_k(token_rsp_k),
        .token_rsp_valid_mask(token_rsp_valid_mask),
        .token_rsp_error(token_rsp_error),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_head(weight_req_input_head),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_req_lane(weight_req_lane), .weight_req_out(weight_req_out),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready),
        .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_head(weight_rsp_input_head),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_lane(weight_rsp_lane), .weight_rsp_out(weight_rsp_out),
        .weight_rsp_data(weight_rsp_data),
        .weight_rsp_error(weight_rsp_error),
        .tile_result_valid(tile_result_valid),
        .tile_result_ready(tile_result_ready),
        .tile_result_tag(tile_result_tag),
        .tile_result_output_tile(tile_result_output_tile),
        .tile_result_plane(tile_result_plane), .tile_result_y(tile_result_y),
        .tile_result_x(tile_result_x), .tile_result_out(tile_result_out),
        .tile_result_data(tile_result_data),
        .tile_result_last(tile_result_last),
        .protocol_error(executor_protocol_error),
        .perf_tiles(perf_numeric_tiles),
        .perf_heads(perf_numeric_heads),
        .perf_partial_results(perf_partial_results),
        .perf_accumulator_writes(perf_accumulator_writes),
        .perf_final_results(perf_final_results)
    );

    assign protocol_error = scheduler_protocol_error
                          || executor_protocol_error;
endmodule

`default_nettype wire
