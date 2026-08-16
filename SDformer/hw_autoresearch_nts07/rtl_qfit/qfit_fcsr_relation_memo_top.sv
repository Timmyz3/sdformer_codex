`timescale 1ns/1ps
`default_nettype none

module qfit_fcsr_relation_memo_top #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int MAX_HEADS = 24,
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int HEAD_W = (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS),
    parameter int PTR_W = $clog2(513)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       head_start,
    output logic                       head_ready,
    input  logic [HEAD_W-1:0]          head_index,

    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [4:0]                 in_candidate_valid,
    input  logic [HEAD_DIM-1:0]        in_k_self,
    input  logic [5*GATE_W-1:0]        in_direction_gates,

    output logic                       live_valid,
    input  logic                       live_ready,
    output logic [SOURCE_ID_W-1:0]     live_source_id,
    output logic [Y_W-1:0]             live_y,
    output logic [X_W-1:0]             live_x,
    output logic [HEAD_DIM-1:0]        live_k,
    output logic [5*GATE_W-1:0]        live_gates,
    output logic [4:0]                 live_valid_mask,
    output logic                       live_last,

    output logic                       head_done,
    output logic                       head_resident,
    output logic                       head_critical,
    output logic                       head_overflow,
    output logic [31:0]                head_service_cycles,
    output logic [PTR_W-1:0]           head_record_count,

    input  logic                       replay_start,
    output logic                       replay_cmd_ready,
    input  logic [HEAD_W-1:0]          replay_head_index,
    output logic                       replay_valid,
    input  logic                       replay_ready,
    output logic [SOURCE_ID_W-1:0]     replay_source_id,
    output logic [Y_W-1:0]             replay_y,
    output logic [X_W-1:0]             replay_x,
    output logic [HEAD_DIM-1:0]        replay_k,
    output logic [5*GATE_W-1:0]        replay_gates,
    output logic [4:0]                 replay_valid_mask,
    output logic                       replay_last,
    output logic                       replay_done,
    output logic                       replay_miss,

    output logic                       plane_idle,
    output logic                       protocol_error,
    output logic [31:0]                perf_speculative_writes,
    output logic [31:0]                perf_discarded_writes,
    output logic [31:0]                perf_committed_records,
    output logic [31:0]                perf_replay_reads,
    output logic [31:0]                perf_capacity_misses
);
    localparam int TOTAL_SOURCES = HEIGHT * WIDTH * TIME_PLANES;

    logic fcsr_descriptor_valid;
    logic fcsr_descriptor_ready;
    logic [SOURCE_ID_W-1:0] fcsr_descriptor_source_id;
    logic [Y_W-1:0] fcsr_descriptor_y;
    logic [X_W-1:0] fcsr_descriptor_x;
    logic [HEAD_DIM-1:0] fcsr_descriptor_k;
    logic [5*GATE_W-1:0] fcsr_descriptor_gates;
    logic [4:0] fcsr_descriptor_valid_mask;
    logic [31:0] unused_producer_stalls;
    logic [2:0] unused_max_pending;
    logic vault_protocol_error;

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(0),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(HEAD_DIM),
        .GATE_W(GATE_W)
    ) u_fcsr (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .plane_start(plane_start),
        .plane_id(plane_id),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_y(in_y),
        .in_x(in_x),
        .in_candidate_valid(in_candidate_valid),
        .in_k_self(in_k_self),
        .in_direction_gates(in_direction_gates),
        .descriptor_valid(fcsr_descriptor_valid),
        .descriptor_ready(fcsr_descriptor_ready),
        .descriptor_source_id(fcsr_descriptor_source_id),
        .descriptor_y(fcsr_descriptor_y),
        .descriptor_x(fcsr_descriptor_x),
        .descriptor_k(fcsr_descriptor_k),
        .descriptor_incoming_gates(fcsr_descriptor_gates),
        .descriptor_valid_mask(fcsr_descriptor_valid_mask),
        .plane_idle(plane_idle),
        .perf_producer_stalls(unused_producer_stalls),
        .perf_max_pending(unused_max_pending)
    );

    qfit_exposure_relation_vault #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .MAX_HEADS(MAX_HEADS)
    ) u_vault (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .head_start(head_start),
        .head_ready(head_ready),
        .head_index(head_index),
        .in_valid(fcsr_descriptor_valid),
        .in_ready(fcsr_descriptor_ready),
        .in_source_id(fcsr_descriptor_source_id),
        .in_y(fcsr_descriptor_y),
        .in_x(fcsr_descriptor_x),
        .in_k(fcsr_descriptor_k),
        .in_gates(fcsr_descriptor_gates),
        .in_valid_mask(fcsr_descriptor_valid_mask),
        .in_last(
            fcsr_descriptor_source_id == SOURCE_ID_W'(TOTAL_SOURCES - 1)
        ),
        .live_valid(live_valid),
        .live_ready(live_ready),
        .live_source_id(live_source_id),
        .live_y(live_y),
        .live_x(live_x),
        .live_k(live_k),
        .live_gates(live_gates),
        .live_valid_mask(live_valid_mask),
        .live_last(live_last),
        .head_done(head_done),
        .head_resident(head_resident),
        .head_critical(head_critical),
        .head_overflow(head_overflow),
        .head_service_cycles(head_service_cycles),
        .head_record_count(head_record_count),
        .replay_start(replay_start),
        .replay_cmd_ready(replay_cmd_ready),
        .replay_head_index(replay_head_index),
        .replay_valid(replay_valid),
        .replay_ready(replay_ready),
        .replay_source_id(replay_source_id),
        .replay_y(replay_y),
        .replay_x(replay_x),
        .replay_k(replay_k),
        .replay_gates(replay_gates),
        .replay_valid_mask(replay_valid_mask),
        .replay_last(replay_last),
        .replay_done(replay_done),
        .replay_miss(replay_miss),
        .protocol_error(vault_protocol_error),
        .perf_speculative_writes(perf_speculative_writes),
        .perf_discarded_writes(perf_discarded_writes),
        .perf_committed_records(perf_committed_records),
        .perf_replay_reads(perf_replay_reads),
        .perf_capacity_misses(perf_capacity_misses)
    );

    assign protocol_error = vault_protocol_error;
endmodule

`default_nettype wire
