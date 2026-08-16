`timescale 1ns/1ps
`default_nettype none

// Scheduler-facing Local5 output-tile transaction. One tagged T450 engine is
// reused across all input heads. The baseline materializes per-head Acc32
// results in a shared 1RW memory; the in-place mode keeps the five TCFM5
// vector accumulators resident and exposes results only for the final head.
module qfit_local5_cross_head_tile_executor #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 32,
    parameter int TAG_W = 24,
    parameter int WINDOW_W = 9,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter int ACC_W = 32,
    parameter int W_W = 8,
    parameter bit USE_RELATION_MEMO = 1'b0,
    parameter bit USE_SCORE_ACTIVE_FRONT = 1'b0,
    parameter bit USE_INPLACE_CROSS_HEAD_ACC = 1'b0,
    parameter bit VECTOR_RESULT_MODE = 1'b0,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter int CROSS_HEAD_MEMORY_IMPL = 0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int TOKEN_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int RESULT_ADDR_W =
        (HEIGHT * WIDTH * TIME_PLANES * OUT_DIM <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES * OUT_DIM)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         tile_start_valid,
    output logic                         tile_start_ready,
    input  logic [TAG_W-1:0]             tile_start_tag,
    input  logic [1:0]                   tile_start_stage,
    input  logic [2:0]                   tile_start_block,
    input  logic [WINDOW_W-1:0]          tile_start_window,
    input  logic [OUTPUT_TILE_W-1:0]     tile_start_output_tile,
    input  logic [5:0]                   tile_start_head_count,

    input  logic                         head_job_valid,
    output logic                         head_job_ready,
    input  logic [TAG_W-1:0]             head_job_tag,
    input  logic [1:0]                   head_job_stage,
    input  logic [2:0]                   head_job_block,
    input  logic [WINDOW_W-1:0]          head_job_window,
    input  logic [HEAD_W-1:0]            head_job_input_head,
    input  logic [INPUT_CH_W-1:0]        head_job_input_channel_base,
    input  logic [OUTPUT_TILE_W-1:0]     head_job_output_tile,
    input  logic                         head_job_decode_required,
    input  logic                         head_job_cache_release,
    input  logic                         head_job_last_input_head,
    input  logic                         head_job_last_output_tile,

    output logic                         head_done_valid,
    input  logic                         head_done_ready,
    output logic [TAG_W-1:0]             head_done_tag,
    output logic [HEAD_W-1:0]            head_done_input_head,
    output logic                         head_done_error,

    output logic                         tile_done_valid,
    input  logic                         tile_done_ready,
    output logic [TAG_W-1:0]             tile_done_tag,
    output logic                         tile_done_error,

    output logic                         token_req_valid,
    input  logic                         token_req_ready,
    output logic [TAG_W-1:0]             token_req_tag,
    output logic [HEAD_W-1:0]            token_req_input_head,
    output logic [TOKEN_ID_W-1:0]        token_req_token_id,
    output logic [PLANE_W-1:0]           token_req_plane,
    output logic [Y_W-1:0]               token_req_y,
    output logic [X_W-1:0]               token_req_x,

    input  logic                         token_rsp_valid,
    output logic                         token_rsp_ready,
    input  logic [TAG_W-1:0]             token_rsp_tag,
    input  logic [HEAD_W-1:0]            token_rsp_input_head,
    input  logic [TOKEN_ID_W-1:0]        token_rsp_token_id,
    input  logic [31:0]                  token_rsp_q,
    input  logic [5*32-1:0]              token_rsp_k,
    input  logic [4:0]                   token_rsp_valid_mask,
    input  logic                         token_rsp_error,

    output logic                         weight_req_valid,
    input  logic                         weight_req_ready,
    output logic [TAG_W-1:0]             weight_req_tag,
    output logic [HEAD_W-1:0]            weight_req_input_head,
    output logic [OUTPUT_TILE_W-1:0]     weight_req_output_tile,
    output logic [LANE_W-1:0]            weight_req_lane,
    output logic [OUT_W-1:0]             weight_req_out,

    input  logic                         weight_rsp_valid,
    output logic                         weight_rsp_ready,
    input  logic [TAG_W-1:0]             weight_rsp_tag,
    input  logic [HEAD_W-1:0]            weight_rsp_input_head,
    input  logic [OUTPUT_TILE_W-1:0]     weight_rsp_output_tile,
    input  logic [LANE_W-1:0]            weight_rsp_lane,
    input  logic [OUT_W-1:0]             weight_rsp_out,
    input  logic signed [W_W-1:0]        weight_rsp_data,
    input  logic                         weight_rsp_error,

    output logic                         tile_result_valid,
    input  logic                         tile_result_ready,
    output logic [TAG_W-1:0]             tile_result_tag,
    output logic [OUTPUT_TILE_W-1:0]     tile_result_output_tile,
    output logic [PLANE_W-1:0]           tile_result_plane,
    output logic [Y_W-1:0]               tile_result_y,
    output logic [X_W-1:0]               tile_result_x,
    output logic [OUT_W-1:0]             tile_result_out,
    output logic signed [ACC_W-1:0]      tile_result_data,
    output logic                         tile_result_last,

    output logic                         protocol_error,
    output logic [31:0]                  perf_tiles,
    output logic [31:0]                  perf_heads,
    output logic [31:0]                  perf_partial_results,
    output logic [31:0]                  perf_accumulator_writes,
    output logic [31:0]                  perf_final_results
);
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;

    typedef enum logic [3:0] {
        TX_IDLE,
        TX_WAIT_HEAD,
        TX_RUN_HEAD,
        TX_HEAD_DONE,
        TX_DRAIN_REQ,
        TX_DRAIN_WAIT,
        TX_DRAIN_OUT,
        TX_TILE_DONE,
        TX_ERROR,
        TX_VECTOR_FLUSH,
        TX_VECTOR_DRAIN_REQ,
        TX_VECTOR_DRAIN_WAIT,
        TX_VECTOR_DRAIN_DONE
    } tx_state_t;
    typedef enum logic { ACC_IDLE, ACC_RMW_WAIT } acc_state_t;

    tx_state_t tx_state_q;
    acc_state_t acc_state_q;
    logic [TAG_W-1:0] tile_tag_q;
    logic [1:0] tile_stage_q;
    logic [2:0] tile_block_q;
    logic [WINDOW_W-1:0] tile_window_q;
    logic [OUTPUT_TILE_W-1:0] tile_output_tile_q;
    logic [5:0] tile_head_count_q;
    logic [HEAD_W-1:0] expected_head_q;
    logic [HEAD_W-1:0] active_head_q;
    logic active_last_head_q;
    logic [31:0] partial_index_q;
    logic [RESULT_ADDR_W-1:0] pending_addr_q;
    logic signed [ACC_W-1:0] pending_delta_q;
    logic [RESULT_ADDR_W-1:0] drain_addr_q;
    logic [PLANE_W-1:0] drain_plane_q;
    logic [Y_W-1:0] drain_y_q;
    logic [X_W-1:0] drain_x_q;
    logic [OUT_W-1:0] drain_out_q;
    logic signed [ACC_W-1:0] drain_data_q;
    logic protocol_error_q;

    logic child_job_valid;
    wire child_job_ready;
    wire child_job_done_valid;
    logic child_job_done_ready;
    wire [TAG_W-1:0] child_job_done_tag;
    wire [HEAD_W-1:0] child_job_done_input_head;
    wire child_job_done_error;
    wire child_result_valid;
    logic child_result_ready;
    wire [TAG_W-1:0] child_result_tag;
    wire [HEAD_W-1:0] child_result_input_head;
    wire [OUTPUT_TILE_W-1:0] child_result_output_tile;
    wire [PLANE_W-1:0] child_result_plane;
    wire [Y_W-1:0] child_result_y;
    wire [X_W-1:0] child_result_x;
    wire [OUT_W-1:0] child_result_out;
    wire signed [ACC_W-1:0] child_result_data;
    wire child_result_last;
    wire child_vector_result_valid;
    logic child_vector_result_ready;
    wire [OUT_DIM*ACC_W-1:0] child_vector_result_data;
    wire child_protocol_error;
    wire [31:0] unused_child_jobs;
    wire [31:0] child_token_requests;
    logic [31:0] perf_memo_hits;
    logic [31:0] perf_memo_fallbacks;
    logic [31:0] perf_memo_resident_builds;
    logic [31:0] perf_cache_release_intents;
    logic [31:0] perf_replay_records;
    wire [31:0] unused_child_token_responses;
    wire [31:0] unused_child_weight_requests;
    wire [31:0] unused_child_weight_responses;
    wire [31:0] unused_child_results;
    wire [31:0] child_result_jobs;

    wire child_token_req_valid;
    wire child_token_req_ready;
    wire [TAG_W-1:0] child_token_req_tag;
    wire [HEAD_W-1:0] child_token_req_input_head;
    wire [TOKEN_ID_W-1:0] child_token_req_token_id;
    wire [PLANE_W-1:0] child_token_req_plane;
    wire [Y_W-1:0] child_token_req_y;
    wire [X_W-1:0] child_token_req_x;
    wire child_token_rsp_valid;
    wire child_token_rsp_ready;
    wire child_weight_req_valid;
    wire child_weight_req_ready;
    wire [TAG_W-1:0] child_weight_req_tag;
    wire [HEAD_W-1:0] child_weight_req_input_head;
    wire [OUTPUT_TILE_W-1:0] child_weight_req_output_tile;
    wire [LANE_W-1:0] child_weight_req_lane;
    wire [OUT_W-1:0] child_weight_req_out;
    wire child_weight_rsp_valid;
    wire child_weight_rsp_ready;
    logic child_service_enable;

    logic memory_command_valid;
    logic memory_command_write;
    logic [RESULT_ADDR_W-1:0] memory_command_addr;
    logic [ACC_W-1:0] memory_command_write_data;
    logic memory_read_data_valid;
    logic [ACC_W-1:0] memory_read_data;
    logic [RESULT_ADDR_W-1:0] child_result_addr;
    logic child_result_matches;
    logic child_vector_result_matches;
    logic child_result_expected;
    logic child_result_fire;
    logic child_vector_result_fire;
    logic child_job_done_fire;
    logic tile_start_fire;
    logic head_job_fire;
    logic head_job_legal;
    logic expected_last_head;
    logic final_drain_result;
    logic tile_result_fire;
    logic head_done_fire;
    logic tile_done_fire;

    logic vector_acc_run_start;
    logic vector_acc_update_valid;
    logic vector_acc_update_ready;
    logic vector_acc_flush_valid;
    logic vector_acc_flush_ready;
    logic vector_acc_flush_done;
    logic vector_acc_read_valid;
    logic vector_acc_read_ready;
    logic vector_acc_read_data_valid;
    logic [OUT_DIM*ACC_W-1:0] vector_acc_read_data;
    logic vector_acc_protocol_error;
    logic serializer_in_valid;
    logic serializer_in_ready;
    logic [PLANE_W-1:0] serializer_in_plane;
    logic [Y_W-1:0] serializer_in_y;
    logic [X_W-1:0] serializer_in_x;
    logic [OUT_DIM*ACC_W-1:0] serializer_in_data;
    logic serializer_in_last;
    logic serializer_out_valid;
    logic serializer_out_ready;
    logic [PLANE_W-1:0] serializer_out_plane;
    logic [Y_W-1:0] serializer_out_y;
    logic [X_W-1:0] serializer_out_x;
    logic [OUT_W-1:0] serializer_out_index;
    logic signed [ACC_W-1:0] serializer_out_data;
    logic serializer_out_last;
    logic vector_acc_read_fire;
    logic vector_acc_data_fire;

    initial begin
        if (HEIGHT != 15 || WIDTH != 15 || TIME_PLANES != 2)
            $error("cross-head Local5 executor currently requires T450");
        if (HEAD_DIM != 32 || OUT_DIM != 32)
            $error("cross-head Local5 executor requires HEAD_DIM=OUT_DIM=32");
        if (USE_RELATION_MEMO && USE_SCORE_ACTIVE_FRONT)
            $error("memo and production score-active child are exclusive");
        if (USE_SCORE_ACTIVE_FRONT && VECTOR_RESULT_MODE)
            $error("production score-active child currently uses scalar readout");
    end

    assign tile_start_ready = tx_state_q == TX_IDLE;
    assign tile_start_fire = tile_start_valid && tile_start_ready;
    assign expected_last_head =
        32'(expected_head_q) + 1 == 32'(tile_head_count_q);
    assign head_job_legal = head_job_tag == tile_tag_q
        && head_job_stage == tile_stage_q
        && head_job_block == tile_block_q
        && head_job_window == tile_window_q
        && head_job_input_head == expected_head_q
        && 32'(head_job_input_channel_base) == 32'(expected_head_q) * HEAD_DIM
        && head_job_output_tile == tile_output_tile_q
        && head_job_decode_required == (tile_output_tile_q == '0)
        && head_job_cache_release == head_job_last_output_tile
        && head_job_last_input_head == expected_last_head;
    assign head_job_ready = tx_state_q == TX_WAIT_HEAD
        && (head_job_legal ? child_job_ready : 1'b1);
    assign head_job_fire = head_job_valid && head_job_ready;
    assign child_job_valid = tx_state_q == TX_WAIT_HEAD
        && head_job_valid && head_job_legal;

    assign head_done_valid = tx_state_q == TX_HEAD_DONE;
    assign head_done_tag = tile_tag_q;
    assign head_done_input_head = active_head_q;
    assign head_done_error = protocol_error_q || child_job_done_error;
    assign head_done_fire = head_done_valid && head_done_ready;
    assign tile_done_valid = tx_state_q == TX_TILE_DONE;
    assign tile_done_tag = tile_tag_q;
    assign tile_done_error = protocol_error_q;
    assign tile_done_fire = tile_done_valid && tile_done_ready;

    assign child_result_addr = RESULT_ADDR_W'(
        (((32'(child_result_plane) * HEIGHT + 32'(child_result_y)) * WIDTH
          + 32'(child_result_x)) * OUT_DIM) + 32'(child_result_out)
    );
    assign child_result_expected = !USE_INPLACE_CROSS_HEAD_ACC
                                || active_last_head_q;
    assign child_result_matches = child_result_expected
        && child_result_tag == tile_tag_q
        && child_result_input_head == active_head_q
        && child_result_output_tile == tile_output_tile_q
        && 32'(child_result_plane) < TIME_PLANES
        && 32'(child_result_y) < HEIGHT
        && 32'(child_result_x) < WIDTH
        && 32'(child_result_out) < OUT_DIM
        && 32'(child_result_addr) == partial_index_q
        && child_result_last == (partial_index_q + 1 == TOTAL_RESULTS);
    assign child_vector_result_matches = child_result_expected
        && child_result_tag == tile_tag_q
        && child_result_input_head == active_head_q
        && child_result_output_tile == tile_output_tile_q
        && 32'(child_result_plane) < TIME_PLANES
        && 32'(child_result_y) < HEIGHT
        && 32'(child_result_x) < WIDTH
        && 32'(
            (32'(child_result_plane) * HEIGHT + 32'(child_result_y)) * WIDTH
            + 32'(child_result_x)
        ) == partial_index_q
        && child_result_last == (partial_index_q + 1 == TOTAL_TOKENS);
    assign child_result_ready = tx_state_q == TX_RUN_HEAD
        && !VECTOR_RESULT_MODE
        && (USE_INPLACE_CROSS_HEAD_ACC
            ? (child_result_matches ? tile_result_ready : 1'b1)
            : acc_state_q == ACC_IDLE);
    assign child_result_fire = child_result_valid && child_result_ready;
    assign child_vector_result_ready = tx_state_q == TX_RUN_HEAD
        && VECTOR_RESULT_MODE
        && (child_vector_result_matches
            ? (USE_INPLACE_CROSS_HEAD_ACC
                ? serializer_in_ready : vector_acc_update_ready)
            : 1'b1);
    assign child_vector_result_fire = child_vector_result_valid
                                   && child_vector_result_ready;
    assign child_job_done_ready = tx_state_q == TX_RUN_HEAD
        && (VECTOR_RESULT_MODE
            || USE_INPLACE_CROSS_HEAD_ACC || acc_state_q == ACC_IDLE);
    assign child_job_done_fire = child_job_done_valid && child_job_done_ready;

    assign final_drain_result = 32'(drain_addr_q) + 1 == TOTAL_RESULTS;
    assign tile_result_valid = VECTOR_RESULT_MODE
        ? serializer_out_valid
        : USE_INPLACE_CROSS_HEAD_ACC
        ? (tx_state_q == TX_RUN_HEAD
           && active_last_head_q
           && child_result_valid
           && child_result_matches
           && !protocol_error)
        : tx_state_q == TX_DRAIN_OUT;
    assign tile_result_tag = tile_tag_q;
    assign tile_result_output_tile = tile_output_tile_q;
    assign tile_result_plane = VECTOR_RESULT_MODE
        ? serializer_out_plane
        : USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_plane : drain_plane_q;
    assign tile_result_y = VECTOR_RESULT_MODE
        ? serializer_out_y
        : USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_y : drain_y_q;
    assign tile_result_x = VECTOR_RESULT_MODE
        ? serializer_out_x
        : USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_x : drain_x_q;
    assign tile_result_out = VECTOR_RESULT_MODE
        ? serializer_out_index
        : USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_out : drain_out_q;
    assign tile_result_data = VECTOR_RESULT_MODE
        ? serializer_out_data
        : USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_data : drain_data_q;
    assign tile_result_last = VECTOR_RESULT_MODE
        ? serializer_out_last
        : USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_last : final_drain_result;
    assign tile_result_fire = tile_result_valid && tile_result_ready;
    assign protocol_error = protocol_error_q || child_protocol_error
                         || vector_acc_protocol_error;
    assign child_service_enable = tx_state_q == TX_RUN_HEAD
                               && !protocol_error;
    assign serializer_out_ready = VECTOR_RESULT_MODE && tile_result_ready;
    assign token_req_valid = child_service_enable && child_token_req_valid;
    assign token_req_tag = child_token_req_tag;
    assign token_req_input_head = child_token_req_input_head;
    assign token_req_token_id = child_token_req_token_id;
    assign token_req_plane = child_token_req_plane;
    assign token_req_y = child_token_req_y;
    assign token_req_x = child_token_req_x;
    assign child_token_req_ready = child_service_enable && token_req_ready;
    assign child_token_rsp_valid = child_service_enable && token_rsp_valid;
    assign token_rsp_ready = child_service_enable && child_token_rsp_ready;

    assign weight_req_valid = child_service_enable && child_weight_req_valid;
    assign weight_req_tag = child_weight_req_tag;
    assign weight_req_input_head = child_weight_req_input_head;
    assign weight_req_output_tile = child_weight_req_output_tile;
    assign weight_req_lane = child_weight_req_lane;
    assign weight_req_out = child_weight_req_out;
    assign child_weight_req_ready = child_service_enable && weight_req_ready;
    assign child_weight_rsp_valid = child_service_enable && weight_rsp_valid;
    assign weight_rsp_ready = child_service_enable && child_weight_rsp_ready;

    assign vector_acc_run_start = VECTOR_RESULT_MODE
                               && !USE_INPLACE_CROSS_HEAD_ACC
                               && tile_start_fire;
    assign vector_acc_update_valid = VECTOR_RESULT_MODE
                                  && !USE_INPLACE_CROSS_HEAD_ACC
                                  && tx_state_q == TX_RUN_HEAD
                                  && child_vector_result_valid
                                  && child_vector_result_matches;
    assign vector_acc_flush_valid = VECTOR_RESULT_MODE
                                 && !USE_INPLACE_CROSS_HEAD_ACC
                                 && tx_state_q == TX_VECTOR_FLUSH;
    assign vector_acc_read_valid = VECTOR_RESULT_MODE
                                && !USE_INPLACE_CROSS_HEAD_ACC
                                && tx_state_q == TX_VECTOR_DRAIN_REQ
                                && serializer_in_ready;
    assign vector_acc_read_fire = vector_acc_read_valid
                               && vector_acc_read_ready;
    assign vector_acc_data_fire = vector_acc_read_data_valid
                               && serializer_in_ready;

    assign serializer_in_valid = VECTOR_RESULT_MODE && (
        USE_INPLACE_CROSS_HEAD_ACC
        ? (tx_state_q == TX_RUN_HEAD
           && active_last_head_q
           && child_vector_result_valid
           && child_vector_result_matches)
        : (tx_state_q == TX_VECTOR_DRAIN_WAIT
           && vector_acc_read_data_valid)
    );
    assign serializer_in_plane = USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_plane : drain_plane_q;
    assign serializer_in_y = USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_y : drain_y_q;
    assign serializer_in_x = USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_x : drain_x_q;
    assign serializer_in_data = USE_INPLACE_CROSS_HEAD_ACC
        ? child_vector_result_data : vector_acc_read_data;
    assign serializer_in_last = USE_INPLACE_CROSS_HEAD_ACC
        ? child_result_last
        : (32'(drain_plane_q) + 1 == TIME_PLANES
           && 32'(drain_y_q) + 1 == HEIGHT
           && 32'(drain_x_q) + 1 == WIDTH);

    generate
        if (VECTOR_RESULT_MODE && !USE_INPLACE_CROSS_HEAD_ACC) begin : g_vector_acc
            qfit_local5_vector_cross_head_acc #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES), .OUT_DIM(OUT_DIM),
                .ACC_W(ACC_W), .MEMORY_IMPL(CROSS_HEAD_MEMORY_IMPL)
            ) u_vector_cross_head_acc (
                .clk_core(clk_core), .rst_core(rst_core),
                .run_start(vector_acc_run_start),
                .update_valid(vector_acc_update_valid),
                .update_ready(vector_acc_update_ready),
                .update_plane(child_result_plane),
                .update_y(child_result_y), .update_x(child_result_x),
                .update_delta(child_vector_result_data),
                .flush_valid(vector_acc_flush_valid),
                .flush_ready(vector_acc_flush_ready),
                .flush_done(vector_acc_flush_done),
                .read_valid(vector_acc_read_valid),
                .read_ready(vector_acc_read_ready),
                .read_plane(drain_plane_q), .read_y(drain_y_q),
                .read_x(drain_x_q),
                .read_data_valid(vector_acc_read_data_valid),
                .read_data(vector_acc_read_data),
                .protocol_error(vector_acc_protocol_error)
            );
        end else begin : g_no_vector_acc
            assign vector_acc_update_ready = 1'b0;
            assign vector_acc_flush_ready = 1'b0;
            assign vector_acc_flush_done = 1'b0;
            assign vector_acc_read_ready = 1'b0;
            assign vector_acc_read_data_valid = 1'b0;
            assign vector_acc_read_data = '0;
            assign vector_acc_protocol_error = 1'b0;
        end

        if (VECTOR_RESULT_MODE) begin : g_vector_serializer
            qfit_acc32_vector_serializer #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W)
            ) u_vector_serializer (
                .clk_core(clk_core), .rst_core(rst_core),
                .in_valid(serializer_in_valid), .in_ready(serializer_in_ready),
                .in_plane(serializer_in_plane), .in_y(serializer_in_y),
                .in_x(serializer_in_x), .in_data(serializer_in_data),
                .in_last(serializer_in_last),
                .out_valid(serializer_out_valid),
                .out_ready(serializer_out_ready),
                .out_plane(serializer_out_plane), .out_y(serializer_out_y),
                .out_x(serializer_out_x), .out_index(serializer_out_index),
                .out_data(serializer_out_data), .out_last(serializer_out_last)
            );
        end else begin : g_no_vector_serializer
            assign serializer_in_ready = 1'b0;
            assign serializer_out_valid = 1'b0;
            assign serializer_out_plane = '0;
            assign serializer_out_y = '0;
            assign serializer_out_x = '0;
            assign serializer_out_index = '0;
            assign serializer_out_data = '0;
            assign serializer_out_last = 1'b0;
        end
    endgenerate

    generate
        if (!USE_INPLACE_CROSS_HEAD_ACC && !VECTOR_RESULT_MODE) begin : g_scalar_cross_head_acc
            always_comb begin
                memory_command_valid = 1'b0;
                memory_command_write = 1'b0;
                memory_command_addr = '0;
                memory_command_write_data = '0;
                if (tx_state_q == TX_RUN_HEAD && child_result_fire
                    && child_result_matches) begin
                    memory_command_valid = 1'b1;
                    memory_command_addr = child_result_addr;
                    if (expected_head_q == '0) begin
                        memory_command_write = 1'b1;
                        memory_command_write_data = child_result_data;
                    end
                end else if (tx_state_q == TX_RUN_HEAD
                             && acc_state_q == ACC_RMW_WAIT
                             && memory_read_data_valid) begin
                    memory_command_valid = 1'b1;
                    memory_command_write = 1'b1;
                    memory_command_addr = pending_addr_q;
                    memory_command_write_data = ACC_W'(
                        signed'(memory_read_data)
                        + signed'(pending_delta_q)
                    );
                end else if (tx_state_q == TX_DRAIN_REQ) begin
                    memory_command_valid = 1'b1;
                    memory_command_addr = drain_addr_q;
                end
            end
            qfit_single_port_acc_memory #(
                .DEPTH(TOTAL_RESULTS),
                .VEC_W(ACC_W),
                .ADDR_W(RESULT_ADDR_W),
                .MEMORY_IMPL(0)
            ) u_cross_head_accumulator (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .command_valid(memory_command_valid),
                .command_write(memory_command_write),
                .command_addr(memory_command_addr),
                .command_write_data(memory_command_write_data),
                .read_data_valid(memory_read_data_valid),
                .read_data(memory_read_data)
            );
        end else begin : g_inplace_cross_head_acc
            assign memory_command_valid = 1'b0;
            assign memory_command_write = 1'b0;
            assign memory_command_addr = '0;
            assign memory_command_write_data = '0;
            assign memory_read_data_valid = 1'b0;
            assign memory_read_data = '0;
        end
    endgenerate

    generate
        if (USE_RELATION_MEMO) begin : g_memo_head_engine
            qfit_local5_memo_tagged_t450_job_engine #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .TAG_W(TAG_W),
                .HEAD_W(HEAD_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
                .W_W(W_W), .ACC_W(ACC_W),
                .VECTOR_RESULT_MODE(VECTOR_RESULT_MODE),
                .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
            ) u_head_engine (
                .clk_core(clk_core), .rst_core(rst_core),
                .job_valid(child_job_valid), .job_ready(child_job_ready),
                .job_tag(head_job_tag),
                .job_input_head(head_job_input_head),
                .job_output_tile(head_job_output_tile),
                .job_decode_required(head_job_decode_required),
                .job_cache_release(head_job_cache_release),
                .job_accumulate(
                    USE_INPLACE_CROSS_HEAD_ACC
                    && head_job_input_head != '0
                ),
                .job_emit_results(
                    !USE_INPLACE_CROSS_HEAD_ACC
                    || head_job_last_input_head
                ),
                .job_done_valid(child_job_done_valid),
                .job_done_ready(child_job_done_ready),
                .job_done_tag(child_job_done_tag),
                .job_done_input_head(child_job_done_input_head),
                .job_done_error(child_job_done_error),
                .token_req_valid(child_token_req_valid),
                .token_req_ready(child_token_req_ready),
                .token_req_tag(child_token_req_tag),
                .token_req_input_head(child_token_req_input_head),
                .token_req_token_id(child_token_req_token_id),
                .token_req_plane(child_token_req_plane),
                .token_req_y(child_token_req_y),
                .token_req_x(child_token_req_x),
                .token_rsp_valid(child_token_rsp_valid),
                .token_rsp_ready(child_token_rsp_ready),
                .token_rsp_tag(token_rsp_tag),
                .token_rsp_input_head(token_rsp_input_head),
                .token_rsp_token_id(token_rsp_token_id),
                .token_rsp_q(token_rsp_q), .token_rsp_k(token_rsp_k),
                .token_rsp_valid_mask(token_rsp_valid_mask),
                .token_rsp_error(token_rsp_error),
                .weight_req_valid(child_weight_req_valid),
                .weight_req_ready(child_weight_req_ready),
                .weight_req_tag(child_weight_req_tag),
                .weight_req_input_head(child_weight_req_input_head),
                .weight_req_output_tile(child_weight_req_output_tile),
                .weight_req_lane(child_weight_req_lane),
                .weight_req_out(child_weight_req_out),
                .weight_rsp_valid(child_weight_rsp_valid),
                .weight_rsp_ready(child_weight_rsp_ready),
                .weight_rsp_tag(weight_rsp_tag),
                .weight_rsp_input_head(weight_rsp_input_head),
                .weight_rsp_output_tile(weight_rsp_output_tile),
                .weight_rsp_lane(weight_rsp_lane),
                .weight_rsp_out(weight_rsp_out),
                .weight_rsp_data(weight_rsp_data),
                .weight_rsp_error(weight_rsp_error),
                .result_valid(child_result_valid),
                .result_ready(child_result_ready),
                .result_tag(child_result_tag),
                .result_input_head(child_result_input_head),
                .result_output_tile(child_result_output_tile),
                .result_plane(child_result_plane), .result_y(child_result_y),
                .result_x(child_result_x), .result_out(child_result_out),
                .result_data(child_result_data),
                .result_last(child_result_last),
                .result_vector_valid(child_vector_result_valid),
                .result_vector_ready(child_vector_result_ready),
                .result_vector_data(child_vector_result_data),
                .protocol_error(child_protocol_error),
                .perf_jobs(unused_child_jobs),
                .perf_token_requests(child_token_requests),
                .perf_token_responses(unused_child_token_responses),
                .perf_weight_requests(unused_child_weight_requests),
                .perf_weight_responses(unused_child_weight_responses),
                .perf_results(unused_child_results),
                .perf_result_jobs(child_result_jobs),
                .perf_memo_hits(perf_memo_hits),
                .perf_memo_fallbacks(perf_memo_fallbacks),
                .perf_memo_resident_builds(perf_memo_resident_builds),
                .perf_cache_release_intents(perf_cache_release_intents),
                .perf_replay_records(perf_replay_records)
            );
        end else begin : g_baseline_head_engine
            assign perf_memo_hits = '0;
            assign perf_memo_fallbacks = '0;
            assign perf_memo_resident_builds = '0;
            assign perf_cache_release_intents = '0;
            assign perf_replay_records = '0;
            qfit_local5_tagged_t450_job_engine #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .TAG_W(TAG_W),
                .HEAD_W(HEAD_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
                .W_W(W_W), .ACC_W(ACC_W),
                .VECTOR_RESULT_MODE(VECTOR_RESULT_MODE),
                .USE_SCORE_ACTIVE_FRONT(USE_SCORE_ACTIVE_FRONT),
                .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
            ) u_head_engine (
                .clk_core(clk_core), .rst_core(rst_core),
                .job_valid(child_job_valid), .job_ready(child_job_ready),
                .job_tag(head_job_tag),
                .job_input_head(head_job_input_head),
                .job_output_tile(head_job_output_tile),
                .job_accumulate(
                    USE_INPLACE_CROSS_HEAD_ACC
                    && head_job_input_head != '0
                ),
                .job_emit_results(
                    !USE_INPLACE_CROSS_HEAD_ACC
                    || head_job_last_input_head
                ),
                .job_done_valid(child_job_done_valid),
                .job_done_ready(child_job_done_ready),
                .job_done_tag(child_job_done_tag),
                .job_done_input_head(child_job_done_input_head),
                .job_done_error(child_job_done_error),
                .token_req_valid(child_token_req_valid),
                .token_req_ready(child_token_req_ready),
                .token_req_tag(child_token_req_tag),
                .token_req_input_head(child_token_req_input_head),
                .token_req_token_id(child_token_req_token_id),
                .token_req_plane(child_token_req_plane),
                .token_req_y(child_token_req_y),
                .token_req_x(child_token_req_x),
                .token_rsp_valid(child_token_rsp_valid),
                .token_rsp_ready(child_token_rsp_ready),
                .token_rsp_tag(token_rsp_tag),
                .token_rsp_input_head(token_rsp_input_head),
                .token_rsp_token_id(token_rsp_token_id),
                .token_rsp_q(token_rsp_q), .token_rsp_k(token_rsp_k),
                .token_rsp_valid_mask(token_rsp_valid_mask),
                .token_rsp_error(token_rsp_error),
                .weight_req_valid(child_weight_req_valid),
                .weight_req_ready(child_weight_req_ready),
                .weight_req_tag(child_weight_req_tag),
                .weight_req_input_head(child_weight_req_input_head),
                .weight_req_output_tile(child_weight_req_output_tile),
                .weight_req_lane(child_weight_req_lane),
                .weight_req_out(child_weight_req_out),
                .weight_rsp_valid(child_weight_rsp_valid),
                .weight_rsp_ready(child_weight_rsp_ready),
                .weight_rsp_tag(weight_rsp_tag),
                .weight_rsp_input_head(weight_rsp_input_head),
                .weight_rsp_output_tile(weight_rsp_output_tile),
                .weight_rsp_lane(weight_rsp_lane),
                .weight_rsp_out(weight_rsp_out),
                .weight_rsp_data(weight_rsp_data),
                .weight_rsp_error(weight_rsp_error),
                .result_valid(child_result_valid),
                .result_ready(child_result_ready),
                .result_tag(child_result_tag),
                .result_input_head(child_result_input_head),
                .result_output_tile(child_result_output_tile),
                .result_plane(child_result_plane), .result_y(child_result_y),
                .result_x(child_result_x), .result_out(child_result_out),
                .result_data(child_result_data),
                .result_last(child_result_last),
                .result_vector_valid(child_vector_result_valid),
                .result_vector_ready(child_vector_result_ready),
                .result_vector_data(child_vector_result_data),
                .protocol_error(child_protocol_error),
                .perf_jobs(unused_child_jobs),
                .perf_token_requests(child_token_requests),
                .perf_token_responses(unused_child_token_responses),
                .perf_weight_requests(unused_child_weight_requests),
                .perf_weight_responses(unused_child_weight_responses),
                .perf_results(unused_child_results),
                .perf_result_jobs(child_result_jobs)
            );
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            tx_state_q <= TX_IDLE;
            acc_state_q <= ACC_IDLE;
            tile_tag_q <= '0;
            tile_stage_q <= '0;
            tile_block_q <= '0;
            tile_window_q <= '0;
            tile_output_tile_q <= '0;
            tile_head_count_q <= '0;
            expected_head_q <= '0;
            active_head_q <= '0;
            active_last_head_q <= 1'b0;
            partial_index_q <= '0;
            pending_addr_q <= '0;
            pending_delta_q <= '0;
            drain_addr_q <= '0;
            drain_plane_q <= '0;
            drain_y_q <= '0;
            drain_x_q <= '0;
            drain_out_q <= '0;
            drain_data_q <= '0;
            protocol_error_q <= 1'b0;
            perf_tiles <= '0;
            perf_heads <= '0;
            perf_partial_results <= '0;
            perf_accumulator_writes <= '0;
            perf_final_results <= '0;
        end else begin
            if (VECTOR_RESULT_MODE && tile_result_fire)
                perf_final_results <= perf_final_results + 1'b1;
            if (tx_state_q == TX_RUN_HEAD
                && acc_state_q == ACC_RMW_WAIT
                && memory_read_data_valid) begin
                acc_state_q <= ACC_IDLE;
                perf_accumulator_writes <= perf_accumulator_writes + 1'b1;
            end

            case (tx_state_q)
                TX_IDLE: begin
                    if (tile_start_fire) begin
                        if (tile_start_head_count == 0
                            || 32'(tile_start_head_count) > (32'(1) << HEAD_W)) begin
                            protocol_error_q <= 1'b1;
                            tx_state_q <= TX_ERROR;
                        end else begin
                            tile_tag_q <= tile_start_tag;
                            tile_stage_q <= tile_start_stage;
                            tile_block_q <= tile_start_block;
                            tile_window_q <= tile_start_window;
                            tile_output_tile_q <= tile_start_output_tile;
                            tile_head_count_q <= tile_start_head_count;
                            expected_head_q <= '0;
                            perf_tiles <= perf_tiles + 1'b1;
                            tx_state_q <= TX_WAIT_HEAD;
                        end
                    end
                end

                TX_WAIT_HEAD: begin
                    if (head_job_fire) begin
                        active_head_q <= head_job_input_head;
                        active_last_head_q <= head_job_last_input_head;
                        partial_index_q <= '0;
                        if (!head_job_legal) begin
                            protocol_error_q <= 1'b1;
                            tx_state_q <= TX_HEAD_DONE;
                        end else begin
                            perf_heads <= perf_heads + 1'b1;
                            tx_state_q <= TX_RUN_HEAD;
                        end
                    end
                end

                TX_RUN_HEAD: begin
                    if (VECTOR_RESULT_MODE && child_vector_result_fire) begin
                        if (!child_vector_result_matches) begin
                            protocol_error_q <= 1'b1;
                            tx_state_q <= TX_HEAD_DONE;
                        end else begin
                            partial_index_q <= partial_index_q + 1'b1;
                            if (!USE_INPLACE_CROSS_HEAD_ACC) begin
                                perf_partial_results <=
                                    perf_partial_results + 1'b1;
                                perf_accumulator_writes <=
                                    perf_accumulator_writes + 1'b1;
                            end
                        end
                    end else if (child_result_fire) begin
                        if (!child_result_matches) begin
                            protocol_error_q <= 1'b1;
                            tx_state_q <= TX_HEAD_DONE;
                        end else if (USE_INPLACE_CROSS_HEAD_ACC) begin
                            partial_index_q <= partial_index_q + 1'b1;
                            perf_final_results <=
                                perf_final_results + 1'b1;
                        end else begin
                            partial_index_q <= partial_index_q + 1'b1;
                            perf_partial_results <=
                                perf_partial_results + 1'b1;
                            if (expected_head_q == '0) begin
                                perf_accumulator_writes <=
                                    perf_accumulator_writes + 1'b1;
                            end else begin
                                pending_addr_q <= child_result_addr;
                                pending_delta_q <= child_result_data;
                                acc_state_q <= ACC_RMW_WAIT;
                            end
                        end
                    end
                    if (child_job_done_fire) begin
                        if (child_job_done_error
                            || child_job_done_tag != tile_tag_q
                            || child_job_done_input_head != active_head_q
                            || partial_index_q
                               != (VECTOR_RESULT_MODE
                                   ? (USE_INPLACE_CROSS_HEAD_ACC
                                      && !active_last_head_q
                                      ? 0 : TOTAL_TOKENS)
                                   : (USE_INPLACE_CROSS_HEAD_ACC
                                      && !active_last_head_q
                                      ? 0 : TOTAL_RESULTS))) begin
                            protocol_error_q <= 1'b1;
                        end
                        tx_state_q <= TX_HEAD_DONE;
                    end
                end

                TX_HEAD_DONE: begin
                    if (head_done_fire) begin
                        if (head_done_error) begin
                            tx_state_q <= TX_ERROR;
                        end else if (active_last_head_q) begin
                            if (VECTOR_RESULT_MODE
                                && USE_INPLACE_CROSS_HEAD_ACC) begin
                                tx_state_q <= TX_VECTOR_DRAIN_DONE;
                            end else if (VECTOR_RESULT_MODE) begin
                                tx_state_q <= TX_VECTOR_FLUSH;
                            end else if (USE_INPLACE_CROSS_HEAD_ACC) begin
                                tx_state_q <= TX_TILE_DONE;
                            end else begin
                                drain_addr_q <= '0;
                                drain_plane_q <= '0;
                                drain_y_q <= '0;
                                drain_x_q <= '0;
                                drain_out_q <= '0;
                                tx_state_q <= TX_DRAIN_REQ;
                            end
                        end else begin
                            expected_head_q <= expected_head_q + 1'b1;
                            tx_state_q <= TX_WAIT_HEAD;
                        end
                    end
                end

                TX_DRAIN_REQ: tx_state_q <= TX_DRAIN_WAIT;

                TX_DRAIN_WAIT: begin
                    if (memory_read_data_valid) begin
                        drain_data_q <= memory_read_data;
                        tx_state_q <= TX_DRAIN_OUT;
                    end
                end

                TX_DRAIN_OUT: begin
                    if (tile_result_fire) begin
                        perf_final_results <= perf_final_results + 1'b1;
                        if (final_drain_result) begin
                            tx_state_q <= TX_TILE_DONE;
                        end else begin
                            drain_addr_q <= drain_addr_q + 1'b1;
                            if (32'(drain_out_q) + 1 < OUT_DIM) begin
                                drain_out_q <= drain_out_q + 1'b1;
                            end else if (32'(drain_x_q) + 1 < WIDTH) begin
                                drain_out_q <= '0;
                                drain_x_q <= drain_x_q + 1'b1;
                            end else if (32'(drain_y_q) + 1 < HEIGHT) begin
                                drain_out_q <= '0;
                                drain_x_q <= '0;
                                drain_y_q <= drain_y_q + 1'b1;
                            end else begin
                                drain_out_q <= '0;
                                drain_x_q <= '0;
                                drain_y_q <= '0;
                                drain_plane_q <= drain_plane_q + 1'b1;
                            end
                            tx_state_q <= TX_DRAIN_REQ;
                        end
                    end
                end

                TX_VECTOR_FLUSH: begin
                    if (vector_acc_flush_done) begin
                        drain_plane_q <= '0;
                        drain_y_q <= '0;
                        drain_x_q <= '0;
                        drain_out_q <= '0;
                        tx_state_q <= TX_VECTOR_DRAIN_REQ;
                    end
                end

                TX_VECTOR_DRAIN_REQ: begin
                    if (vector_acc_read_fire)
                        tx_state_q <= TX_VECTOR_DRAIN_WAIT;
                end

                TX_VECTOR_DRAIN_WAIT: begin
                    if (vector_acc_data_fire) begin
                        if (32'(drain_plane_q) + 1 == TIME_PLANES
                            && 32'(drain_y_q) + 1 == HEIGHT
                            && 32'(drain_x_q) + 1 == WIDTH) begin
                            tx_state_q <= TX_VECTOR_DRAIN_DONE;
                        end else if (32'(drain_x_q) + 1 < WIDTH) begin
                            drain_x_q <= drain_x_q + 1'b1;
                            tx_state_q <= TX_VECTOR_DRAIN_REQ;
                        end else if (32'(drain_y_q) + 1 < HEIGHT) begin
                            drain_x_q <= '0;
                            drain_y_q <= drain_y_q + 1'b1;
                            tx_state_q <= TX_VECTOR_DRAIN_REQ;
                        end else begin
                            drain_x_q <= '0;
                            drain_y_q <= '0;
                            drain_plane_q <= drain_plane_q + 1'b1;
                            tx_state_q <= TX_VECTOR_DRAIN_REQ;
                        end
                    end
                end

                TX_VECTOR_DRAIN_DONE: begin
                    if (tile_result_fire && serializer_out_last)
                        tx_state_q <= TX_TILE_DONE;
                end

                TX_TILE_DONE: begin
                    if (tile_done_fire)
                        tx_state_q <= TX_IDLE;
                end

                default: tx_state_q <= TX_ERROR;
            endcase

            // A failed child cannot progress after the service firewall closes.
            if ((child_protocol_error || vector_acc_protocol_error)
                && (tx_state_q == TX_RUN_HEAD
                    || tx_state_q == TX_VECTOR_FLUSH
                    || tx_state_q == TX_VECTOR_DRAIN_REQ
                    || tx_state_q == TX_VECTOR_DRAIN_WAIT)) begin
                protocol_error_q <= 1'b1;
                tx_state_q <= TX_ERROR;
            end
        end
    end
endmodule

`default_nettype wire
