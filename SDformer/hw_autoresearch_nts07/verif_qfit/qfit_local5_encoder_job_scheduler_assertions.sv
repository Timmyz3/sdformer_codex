`timescale 1ns/1ps
`default_nettype none

module qfit_local5_encoder_job_scheduler_assertions #(
    parameter int TAG_W = 24,
    parameter int WINDOW_W = 9,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter int COUNTER_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic frame_done,
    input logic tile_start_valid,
    input logic tile_start_ready,
    input logic [TAG_W-1:0] tile_start_tag,
    input logic [1:0] tile_start_stage,
    input logic [2:0] tile_start_block,
    input logic [WINDOW_W-1:0] tile_start_window,
    input logic [OUTPUT_TILE_W-1:0] tile_start_output_tile,
    input logic [5:0] tile_start_head_count,
    input logic head_job_valid,
    input logic head_job_ready,
    input logic [TAG_W-1:0] head_job_tag,
    input logic [1:0] head_job_stage,
    input logic [2:0] head_job_block,
    input logic [WINDOW_W-1:0] head_job_window,
    input logic [HEAD_W-1:0] head_job_input_head,
    input logic [INPUT_CH_W-1:0] head_job_input_channel_base,
    input logic [OUTPUT_TILE_W-1:0] head_job_output_tile,
    input logic head_job_decode_required,
    input logic head_job_cache_release,
    input logic head_job_last_input_head,
    input logic head_job_last_output_tile,
    input logic protocol_error,
    input logic [COUNTER_W-1:0] perf_window_groups,
    input logic [COUNTER_W-1:0] perf_output_tiles,
    input logic [COUNTER_W-1:0] perf_head_replays,
    input logic [COUNTER_W-1:0] perf_decode_intent_jobs,
    input logic [COUNTER_W-1:0] perf_release_intent_jobs
);
    function automatic logic legal_tile_geometry(
        input logic [1:0] stage,
        input logic [2:0] block_id,
        input logic [WINDOW_W-1:0] window_id,
        input logic [OUTPUT_TILE_W-1:0] output_tile,
        input logic [5:0] head_count
    );
        case (stage)
            2'd0: legal_tile_geometry = 32'(block_id) < 2
                && 32'(window_id) < 440 && 32'(output_tile) < 3
                && head_count == 6'd3;
            2'd1: legal_tile_geometry = 32'(block_id) < 2
                && 32'(window_id) < 120 && 32'(output_tile) < 6
                && head_count == 6'd6;
            2'd2: legal_tile_geometry = 32'(block_id) < 6
                && 32'(window_id) < 30 && 32'(output_tile) < 12
                && head_count == 6'd12;
            default: legal_tile_geometry = 32'(block_id) < 2
                && 32'(window_id) < 10 && 32'(output_tile) < 24
                && head_count == 6'd24;
        endcase
    endfunction

    function automatic logic legal_head_geometry(
        input logic [1:0] stage,
        input logic [2:0] block_id,
        input logic [WINDOW_W-1:0] window_id,
        input logic [OUTPUT_TILE_W-1:0] output_tile,
        input logic [HEAD_W-1:0] input_head
    );
        case (stage)
            2'd0: legal_head_geometry = 32'(block_id) < 2
                && 32'(window_id) < 440 && 32'(output_tile) < 3
                && 32'(input_head) < 3;
            2'd1: legal_head_geometry = 32'(block_id) < 2
                && 32'(window_id) < 120 && 32'(output_tile) < 6
                && 32'(input_head) < 6;
            2'd2: legal_head_geometry = 32'(block_id) < 6
                && 32'(window_id) < 30 && 32'(output_tile) < 12
                && 32'(input_head) < 12;
            default: legal_head_geometry = 32'(block_id) < 2
                && 32'(window_id) < 10 && 32'(output_tile) < 24
                && 32'(input_head) < 24;
        endcase
    endfunction

    property p_tile_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            tile_start_valid && !tile_start_ready
            |=> tile_start_valid
                && $stable(tile_start_tag)
                && $stable(tile_start_stage)
                && $stable(tile_start_block)
                && $stable(tile_start_window)
                && $stable(tile_start_output_tile)
                && $stable(tile_start_head_count);
    endproperty

    property p_head_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            head_job_valid && !head_job_ready
            |=> head_job_valid
                && $stable(head_job_tag)
                && $stable(head_job_stage)
                && $stable(head_job_block)
                && $stable(head_job_window)
                && $stable(head_job_input_head)
                && $stable(head_job_input_channel_base)
                && $stable(head_job_output_tile)
                && $stable(head_job_decode_required)
                && $stable(head_job_cache_release)
                && $stable(head_job_last_input_head)
                && $stable(head_job_last_output_tile);
    endproperty

    property p_tile_geometry_is_legal;
        @(posedge clk_core) disable iff (rst_core)
            tile_start_valid |-> legal_tile_geometry(
                tile_start_stage, tile_start_block, tile_start_window,
                tile_start_output_tile, tile_start_head_count);
    endproperty

    property p_head_geometry_is_legal;
        @(posedge clk_core) disable iff (rst_core)
            head_job_valid |-> legal_head_geometry(
                head_job_stage, head_job_block, head_job_window,
                head_job_output_tile, head_job_input_head);
    endproperty

    property p_tile_tag_encodes_output_tile;
        @(posedge clk_core) disable iff (rst_core)
            tile_start_valid
            |-> tile_start_tag[4:0] == 5'(tile_start_output_tile);
    endproperty

    property p_head_tag_encodes_output_tile;
        @(posedge clk_core) disable iff (rst_core)
            head_job_valid
            |-> head_job_tag[4:0] == 5'(head_job_output_tile);
    endproperty

    property p_decode_only_on_first_output_tile;
        @(posedge clk_core) disable iff (rst_core)
            head_job_valid
            |-> head_job_decode_required == (head_job_output_tile == '0);
    endproperty

    property p_release_only_on_last_output_tile;
        @(posedge clk_core) disable iff (rst_core)
            head_job_valid
            |-> head_job_cache_release == head_job_last_output_tile;
    endproperty

    property p_input_channel_matches_head;
        @(posedge clk_core) disable iff (rst_core)
            head_job_valid
            |-> 32'(head_job_input_channel_base)
                == 32'(head_job_input_head) * 32;
    endproperty

    property p_frame_done_is_single_cycle;
        @(posedge clk_core) disable iff (rst_core)
            frame_done |=> !frame_done;
    endproperty

    property p_protocol_error_is_sticky;
        @(posedge clk_core) disable iff (rst_core)
            $past(protocol_error) |-> protocol_error;
    endproperty

    property p_counters_are_bounded;
        @(posedge clk_core) disable iff (rst_core)
            32'(perf_window_groups) <= 1320
            && 32'(perf_output_tiles) <= 6720
            && 32'(perf_head_replays) <= 54000
            && 32'(perf_decode_intent_jobs) <= 6720
            && 32'(perf_release_intent_jobs) <= 6720;
    endproperty

    assert property (p_tile_stable_under_backpressure);
    assert property (p_head_stable_under_backpressure);
    assert property (p_tile_geometry_is_legal);
    assert property (p_head_geometry_is_legal);
    assert property (p_tile_tag_encodes_output_tile);
    assert property (p_head_tag_encodes_output_tile);
    assert property (p_decode_only_on_first_output_tile);
    assert property (p_release_only_on_last_output_tile);
    assert property (p_input_channel_matches_head);
    assert property (p_frame_done_is_single_cycle);
    assert property (p_protocol_error_is_sticky);
    assert property (p_counters_are_bounded);

    property p_error_stops_dispatch_and_completion;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error
            |-> !tile_start_valid && !head_job_valid && !frame_done;
    endproperty
    assert property (p_error_stops_dispatch_and_completion);

    property p_done_requires_clean_protocol;
        @(posedge clk_core) disable iff (rst_core)
            frame_done |-> !protocol_error;
    endproperty
    assert property (p_done_requires_clean_protocol);
endmodule

bind qfit_local5_encoder_job_scheduler
    qfit_local5_encoder_job_scheduler_assertions #(
        .TAG_W(TAG_W),
        .WINDOW_W(WINDOW_W),
        .HEAD_W(HEAD_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .INPUT_CH_W(INPUT_CH_W),
        .COUNTER_W(COUNTER_W)
    ) u_qfit_local5_encoder_job_scheduler_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .frame_done(frame_done),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready),
        .tile_start_tag(tile_start_tag),
        .tile_start_stage(tile_start_stage),
        .tile_start_block(tile_start_block),
        .tile_start_window(tile_start_window),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_job_valid(head_job_valid),
        .head_job_ready(head_job_ready),
        .head_job_tag(head_job_tag),
        .head_job_stage(head_job_stage),
        .head_job_block(head_job_block),
        .head_job_window(head_job_window),
        .head_job_input_head(head_job_input_head),
        .head_job_input_channel_base(head_job_input_channel_base),
        .head_job_output_tile(head_job_output_tile),
        .head_job_decode_required(head_job_decode_required),
        .head_job_cache_release(head_job_cache_release),
        .head_job_last_input_head(head_job_last_input_head),
        .head_job_last_output_tile(head_job_last_output_tile),
        .protocol_error(protocol_error),
        .perf_window_groups(perf_window_groups),
        .perf_output_tiles(perf_output_tiles),
        .perf_head_replays(perf_head_replays),
        .perf_decode_intent_jobs(perf_decode_intent_jobs),
        .perf_release_intent_jobs(perf_release_intent_jobs)
    );

`default_nettype wire
