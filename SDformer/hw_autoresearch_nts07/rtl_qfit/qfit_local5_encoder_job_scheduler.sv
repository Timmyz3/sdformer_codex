`timescale 1ns/1ps
`default_nettype none

// 12-block Local5 frame schedule. Each window owns one semantic group. The
// first output-tile visit decodes/caches every input head; later visits replay
// the exact cached term stream and the final visit releases the head slot.
module qfit_local5_encoder_job_scheduler #(
    parameter int TAG_W = 24,
    parameter int WINDOW_W = 9,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter int COUNTER_W = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         start_frame,
    output logic                         frame_busy,
    output logic                         frame_done,

    output logic                         tile_start_valid,
    input  logic                         tile_start_ready,
    output logic [TAG_W-1:0]             tile_start_tag,
    output logic [1:0]                   tile_start_stage,
    output logic [2:0]                   tile_start_block,
    output logic [WINDOW_W-1:0]          tile_start_window,
    output logic [OUTPUT_TILE_W-1:0]     tile_start_output_tile,
    output logic [5:0]                   tile_start_head_count,

    output logic                         head_job_valid,
    input  logic                         head_job_ready,
    output logic [TAG_W-1:0]             head_job_tag,
    output logic [1:0]                   head_job_stage,
    output logic [2:0]                   head_job_block,
    output logic [WINDOW_W-1:0]          head_job_window,
    output logic [HEAD_W-1:0]            head_job_input_head,
    output logic [INPUT_CH_W-1:0]        head_job_input_channel_base,
    output logic [OUTPUT_TILE_W-1:0]     head_job_output_tile,
    output logic                         head_job_decode_required,
    output logic                         head_job_cache_release,
    output logic                         head_job_last_input_head,
    output logic                         head_job_last_output_tile,

    input  logic                         head_done_valid,
    output logic                         head_done_ready,
    input  logic [TAG_W-1:0]             head_done_tag,
    input  logic [HEAD_W-1:0]            head_done_input_head,
    input  logic                         head_done_error,

    input  logic                         tile_done_valid,
    output logic                         tile_done_ready,
    input  logic [TAG_W-1:0]             tile_done_tag,
    input  logic                         tile_done_error,

    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         perf_window_groups,
    output logic [COUNTER_W-1:0]         perf_output_tiles,
    output logic [COUNTER_W-1:0]         perf_head_replays,
    output logic [COUNTER_W-1:0]         perf_decode_intent_jobs,
    output logic [COUNTER_W-1:0]         perf_release_intent_jobs
);
    localparam int EXPECTED_GROUPS = 1320;
    localparam int EXPECTED_OUTPUT_TILES = 6720;
    localparam int EXPECTED_HEAD_REPLAYS = 54000;
    localparam int DESCRIPTOR_W = 4;
    localparam int GROUP_ID_W = $clog2(EXPECTED_GROUPS);

    typedef enum logic [2:0] {
        F_IDLE,
        F_GROUP_ISSUE,
        F_GROUP_WAIT,
        F_DONE,
        F_ERROR
    } frame_state_t;

    frame_state_t frame_state_q;
    logic [DESCRIPTOR_W-1:0] descriptor_q;
    logic [WINDOW_W-1:0] window_q;
    logic [GROUP_ID_W-1:0] group_id_q;
    logic [1:0] descriptor_stage;
    logic [2:0] descriptor_block;
    logic [5:0] descriptor_heads;
    logic [WINDOW_W-1:0] descriptor_windows;
    logic [TAG_W-1:0] group_tag;

    logic group_valid;
    logic group_ready;
    logic group_done_valid;
    logic group_done_ready;
    logic [TAG_W-1:0] group_done_tag;
    logic group_done_error;
    logic scheduler_protocol_error;
    logic group_fire;
    logic group_done_fire;

    logic unused_tile_start_fire;
    logic unused_head_context;
    logic head_job_last_head;
    logic head_job_last_tile;
    logic [COUNTER_W-1:0] unused_count_groups;
    logic [COUNTER_W-1:0] unused_count_tiles;
    logic [COUNTER_W-1:0] unused_count_heads;
    logic [COUNTER_W-1:0] unused_count_errors;
    logic protocol_error_q;
    logic frame_done_q;
    logic final_group;
    logic counters_complete;
    logic start_conflict;
    logic scheduler_rst_core;

    always_comb begin
        descriptor_stage = 2'd0;
        descriptor_block = 3'd0;
        descriptor_heads = 6'd3;
        descriptor_windows = WINDOW_W'(440);
        unique case (descriptor_q)
            4'd0: begin descriptor_stage = 0; descriptor_block = 0; descriptor_heads = 3;  descriptor_windows = WINDOW_W'(440); end
            4'd1: begin descriptor_stage = 0; descriptor_block = 1; descriptor_heads = 3;  descriptor_windows = WINDOW_W'(440); end
            4'd2: begin descriptor_stage = 1; descriptor_block = 0; descriptor_heads = 6;  descriptor_windows = WINDOW_W'(120); end
            4'd3: begin descriptor_stage = 1; descriptor_block = 1; descriptor_heads = 6;  descriptor_windows = WINDOW_W'(120); end
            4'd4: begin descriptor_stage = 2; descriptor_block = 0; descriptor_heads = 12; descriptor_windows = WINDOW_W'(30);  end
            4'd5: begin descriptor_stage = 2; descriptor_block = 1; descriptor_heads = 12; descriptor_windows = WINDOW_W'(30);  end
            4'd6: begin descriptor_stage = 2; descriptor_block = 2; descriptor_heads = 12; descriptor_windows = WINDOW_W'(30);  end
            4'd7: begin descriptor_stage = 2; descriptor_block = 3; descriptor_heads = 12; descriptor_windows = WINDOW_W'(30);  end
            4'd8: begin descriptor_stage = 2; descriptor_block = 4; descriptor_heads = 12; descriptor_windows = WINDOW_W'(30);  end
            4'd9: begin descriptor_stage = 2; descriptor_block = 5; descriptor_heads = 12; descriptor_windows = WINDOW_W'(30);  end
            4'd10: begin descriptor_stage = 3; descriptor_block = 0; descriptor_heads = 24; descriptor_windows = WINDOW_W'(10); end
            default: begin descriptor_stage = 3; descriptor_block = 1; descriptor_heads = 24; descriptor_windows = WINDOW_W'(10); end
        endcase
    end

    assign group_tag = TAG_W'(32'(group_id_q) * 32);
    assign group_valid = frame_state_q == F_GROUP_ISSUE;
    assign group_fire = group_valid && group_ready;
    assign group_done_ready = frame_state_q == F_GROUP_WAIT;
    assign group_done_fire = group_done_valid && group_done_ready;
    assign final_group = descriptor_q == 4'd11
                       && 32'(window_q) + 1 == 32'(descriptor_windows);
    assign counters_complete =
        32'(perf_window_groups) == EXPECTED_GROUPS
        && 32'(perf_output_tiles) == EXPECTED_OUTPUT_TILES
        && 32'(perf_head_replays) == EXPECTED_HEAD_REPLAYS
        && 32'(perf_decode_intent_jobs) == EXPECTED_OUTPUT_TILES
        && 32'(perf_release_intent_jobs) == EXPECTED_OUTPUT_TILES;

    assign start_conflict = start_frame && frame_state_q != F_IDLE;
    assign scheduler_rst_core = rst_core || start_conflict;

    assign frame_busy = frame_state_q != F_IDLE && frame_state_q != F_DONE;
    assign frame_done = frame_done_q;
    assign protocol_error = protocol_error_q || scheduler_protocol_error;

    assign tile_start_stage = descriptor_stage;
    assign tile_start_block = descriptor_block;
    assign tile_start_window = window_q;
    assign head_job_stage = descriptor_stage;
    assign head_job_block = descriptor_block;
    assign head_job_window = window_q;
    assign head_job_decode_required = head_job_output_tile == '0;
    assign head_job_cache_release = head_job_last_tile;
    assign head_job_last_input_head = head_job_last_head;
    assign head_job_last_output_tile = head_job_last_tile;

    gatestack_output_tile_scheduler #(
        .CONTEXTS(1),
        .HEADS(24),
        .LANES(32),
        .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .OUTPUT_TILE_COUNT_W(6),
        .HEAD_COUNT_W(6),
        .COUNTER_W(COUNTER_W),
        .CONTEXT_ID_W(1),
        .HEAD_ID_W(HEAD_W)
    ) u_output_tile_scheduler (
        .clk_core(clk_core),
        .rst_core(scheduler_rst_core),
        .group_valid(group_valid),
        .group_ready(group_ready),
        .group_context_id(1'b0),
        .group_tag(group_tag),
        .group_head_count(descriptor_heads),
        .group_first_output_tile('0),
        .group_output_tile_count(descriptor_heads),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready),
        .tile_start_tag(tile_start_tag),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_issue_valid(head_job_valid),
        .head_issue_ready(head_job_ready),
        .head_issue_context_id(unused_head_context),
        .head_issue_tag(head_job_tag),
        .head_issue_head_id(head_job_input_head),
        .head_issue_head_index(),
        .head_issue_input_channel_base(head_job_input_channel_base),
        .head_issue_output_tile(head_job_output_tile),
        .head_issue_last_head(head_job_last_head),
        .head_issue_last_output_tile(head_job_last_tile),
        .head_done_valid(head_done_valid),
        .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_head_id(head_done_input_head),
        .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid),
        .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag),
        .tile_done_error(tile_done_error),
        .group_done_valid(group_done_valid),
        .group_done_ready(group_done_ready),
        .group_done_tag(group_done_tag),
        .group_done_error(group_done_error),
        .protocol_error(scheduler_protocol_error),
        .count_groups(unused_count_groups),
        .count_tile_starts(unused_count_tiles),
        .count_head_issues(unused_count_heads),
        .count_group_errors(unused_count_errors)
    );

    assign unused_tile_start_fire = tile_start_valid && tile_start_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            frame_state_q <= F_IDLE;
            descriptor_q <= '0;
            window_q <= '0;
            group_id_q <= '0;
            protocol_error_q <= 1'b0;
            frame_done_q <= 1'b0;
            perf_window_groups <= '0;
            perf_output_tiles <= '0;
            perf_head_replays <= '0;
            perf_decode_intent_jobs <= '0;
            perf_release_intent_jobs <= '0;
        end else begin
            frame_done_q <= 1'b0;
            if (start_conflict || scheduler_protocol_error) begin
                protocol_error_q <= 1'b1;
                frame_state_q <= F_ERROR;
            end else begin
                if (unused_tile_start_fire)
                    perf_output_tiles <= perf_output_tiles + 1'b1;
                if (head_job_valid && head_job_ready) begin
                    perf_head_replays <= perf_head_replays + 1'b1;
                    if (head_job_decode_required)
                        perf_decode_intent_jobs <=
                            perf_decode_intent_jobs + 1'b1;
                    if (head_job_cache_release)
                        perf_release_intent_jobs <=
                            perf_release_intent_jobs + 1'b1;
                end

                unique case (frame_state_q)
                F_IDLE: begin
                    if (start_frame) begin
                        descriptor_q <= '0;
                        window_q <= '0;
                        group_id_q <= '0;
                        protocol_error_q <= 1'b0;
                        perf_window_groups <= '0;
                        perf_output_tiles <= '0;
                        perf_head_replays <= '0;
                        perf_decode_intent_jobs <= '0;
                        perf_release_intent_jobs <= '0;
                        frame_state_q <= F_GROUP_ISSUE;
                    end
                end

                F_GROUP_ISSUE: begin
                    if (group_fire) begin
                        perf_window_groups <= perf_window_groups + 1'b1;
                        frame_state_q <= F_GROUP_WAIT;
                    end
                end

                F_GROUP_WAIT: begin
                    if (group_done_fire) begin
                        if (group_done_error || group_done_tag != group_tag) begin
                            protocol_error_q <= 1'b1;
                            frame_state_q <= F_ERROR;
                        end else if (final_group) begin
                            if (!counters_complete) begin
                                protocol_error_q <= 1'b1;
                                frame_state_q <= F_ERROR;
                            end else begin
                                frame_done_q <= 1'b1;
                                frame_state_q <= F_DONE;
                            end
                        end else begin
                            group_id_q <= group_id_q + 1'b1;
                            if (32'(window_q) + 1 < 32'(descriptor_windows)) begin
                                window_q <= window_q + 1'b1;
                            end else begin
                                descriptor_q <= descriptor_q + 1'b1;
                                window_q <= '0;
                            end
                            frame_state_q <= F_GROUP_ISSUE;
                        end
                    end
                end

                F_DONE: frame_state_q <= F_IDLE;
                    default: frame_state_q <= F_ERROR;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
