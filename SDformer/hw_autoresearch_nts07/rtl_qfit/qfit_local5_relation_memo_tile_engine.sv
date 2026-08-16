`timescale 1ns/1ps
`default_nettype none

module qfit_local5_relation_memo_tile_engine #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int MAX_HEADS = 24,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int HEAD_W = (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS),
    parameter int PTR_W = $clog2(513),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,

    input  logic                       tile_start,
    output logic                       tile_ready,
    input  logic                       tile_prefer_replay,
    input  logic [HEAD_W-1:0]          tile_head_index,
    output logic                       tile_done,
    output logic                       fallback_taken,
    output logic                       recompute_request,
    input  logic                       recompute_grant,

    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [4:0]                 in_candidate_valid,
    input  logic [HEAD_DIM-1:0]        in_k_self,
    input  logic [5*GATE_W-1:0]        in_direction_gates,
    output logic                       plane_idle,

    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,

    input  logic                       read_valid,
    output logic                       read_ready,
    input  logic [PLANE_W-1:0]         read_plane,
    input  logic [Y_W-1:0]             read_y,
    input  logic [X_W-1:0]             read_x,
    input  logic [OUT_W-1:0]           read_out,
    output logic                       read_data_valid,
    output logic signed [ACC_W-1:0]    read_data,

    output logic                       descriptor_valid,
    output logic                       descriptor_ready,
    output logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    output logic [Y_W-1:0]             descriptor_y,
    output logic [X_W-1:0]             descriptor_x,
    output logic [HEAD_DIM-1:0]        descriptor_k,
    output logic [5*GATE_W-1:0]        descriptor_gates,
    output logic [4:0]                 descriptor_valid_mask,
    output logic                       descriptor_last,
    output logic                       descriptor_stream_idle,

    output logic                       head_done,
    output logic                       head_resident,
    output logic                       head_critical,
    output logic                       head_overflow,
    output logic [31:0]                head_service_cycles,
    output logic [PTR_W-1:0]           head_record_count,
    output logic                       protocol_error,
    output logic [31:0]                perf_speculative_writes,
    output logic [31:0]                perf_discarded_writes,
    output logic [31:0]                perf_committed_records,
    output logic [31:0]                perf_replay_reads,
    output logic [31:0]                perf_capacity_misses,
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates
);
    logic use_replay;
    logic replay_start;
    logic replay_cmd_ready;
    logic [HEAD_W-1:0] replay_head_index;
    logic replay_done;
    logic replay_miss;
    logic head_start;
    logic head_ready;
    logic [HEAD_W-1:0] head_index;
    logic projection_start;
    logic projection_close;
    logic projection_close_ready;
    logic projection_busy;
    logic projection_done;
    logic controller_protocol_error;
    logic datapath_protocol_error;

    qfit_relation_memo_tile_controller #(
        .MAX_HEADS(MAX_HEADS)
    ) u_controller (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .tile_start(tile_start),
        .tile_ready(tile_ready),
        .tile_prefer_replay(tile_prefer_replay),
        .tile_head_index(tile_head_index),
        .tile_done(tile_done),
        .fallback_taken(fallback_taken),
        .use_replay(use_replay),
        .replay_start(replay_start),
        .replay_cmd_ready(replay_cmd_ready),
        .replay_head_index(replay_head_index),
        .replay_done(replay_done),
        .replay_miss(replay_miss),
        .recompute_request(recompute_request),
        .recompute_grant(recompute_grant),
        .head_start(head_start),
        .head_ready(head_ready),
        .head_index(head_index),
        .head_done(head_done),
        .descriptor_stream_idle(descriptor_stream_idle),
        .projection_start(projection_start),
        .projection_accumulate(1'b0),
        .projection_close(projection_close),
        .projection_close_ready(projection_close_ready),
        .projection_done(projection_done),
        .protocol_error(controller_protocol_error)
    );

    qfit_fcsr_relation_memo_projection_top #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .MAX_HEADS(MAX_HEADS)
    ) u_datapath (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .head_start(head_start),
        .head_ready(head_ready),
        .head_index(head_index),
        .head_done(head_done),
        .head_resident(head_resident),
        .head_critical(head_critical),
        .head_overflow(head_overflow),
        .head_service_cycles(head_service_cycles),
        .head_record_count(head_record_count),
        .plane_start(plane_start),
        .plane_id(plane_id),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_y(in_y),
        .in_x(in_x),
        .in_candidate_valid(in_candidate_valid),
        .in_k_self(in_k_self),
        .in_direction_gates(in_direction_gates),
        .plane_idle(plane_idle),
        .use_replay(use_replay),
        .replay_start(replay_start),
        .replay_cmd_ready(replay_cmd_ready),
        .replay_head_index(replay_head_index),
        .replay_done(replay_done),
        .replay_miss(replay_miss),
        .weight_valid(weight_valid),
        .weight_ready(weight_ready),
        .weight_lane(weight_lane),
        .weight_out(weight_out),
        .weight_data(weight_data),
        .weight_last(weight_last),
        .weight_context_release(1'b0),
        .weight_context_release_ready(),
        .projection_start(projection_start),
        .projection_close(projection_close),
        .projection_close_ready(projection_close_ready),
        .projection_busy(projection_busy),
        .projection_done(projection_done),
        .read_valid(read_valid),
        .read_ready(read_ready),
        .read_plane(read_plane),
        .read_y(read_y),
        .read_x(read_x),
        .read_out(read_out),
        .read_data_valid(read_data_valid),
        .read_data(read_data),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_gates(descriptor_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .descriptor_last(descriptor_last),
        .descriptor_stream_idle(descriptor_stream_idle),
        .protocol_error(datapath_protocol_error),
        .perf_speculative_writes(perf_speculative_writes),
        .perf_discarded_writes(perf_discarded_writes),
        .perf_committed_records(perf_committed_records),
        .perf_replay_reads(perf_replay_reads),
        .perf_capacity_misses(perf_capacity_misses),
        .perf_descriptors(perf_descriptors),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates)
    );

    assign protocol_error = controller_protocol_error
                         || datapath_protocol_error;

    logic unused_projection_busy;
    assign unused_projection_busy = projection_busy;
endmodule

`default_nettype wire
