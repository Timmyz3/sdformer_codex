`timescale 1ns/1ps
`default_nettype none

// One tagged Local5 T450 head/output-tile job. The engine requests all weights
// and Q/K tokens from external services, executes the exact numerical tile,
// drains every Acc32 result under backpressure, then releases the weight
// context before acknowledging the job.
module qfit_local5_tagged_t450_job_engine #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int TAG_W = 24,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter bit VECTOR_RESULT_MODE = 1'b0,
    // Default preserves the sealed legacy child. Set for the current
    // QS/FCSR/source-owned/TCFM5 production score-to-Acc32 path.
    parameter bit USE_SCORE_ACTIVE_FRONT = 1'b0,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W =
        (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int TOKEN_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         job_valid,
    output logic                         job_ready,
    input  logic [TAG_W-1:0]             job_tag,
    input  logic [HEAD_W-1:0]            job_input_head,
    input  logic [OUTPUT_TILE_W-1:0]     job_output_tile,
    input  logic                         job_accumulate,
    input  logic                         job_emit_results,

    output logic                         job_done_valid,
    input  logic                         job_done_ready,
    output logic [TAG_W-1:0]             job_done_tag,
    output logic [HEAD_W-1:0]            job_done_input_head,
    output logic                         job_done_error,

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

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_W-1:0]             result_tag,
    output logic [HEAD_W-1:0]            result_input_head,
    output logic [OUTPUT_TILE_W-1:0]     result_output_tile,
    output logic [PLANE_W-1:0]           result_plane,
    output logic [Y_W-1:0]               result_y,
    output logic [X_W-1:0]               result_x,
    output logic [OUT_W-1:0]             result_out,
    output logic signed [ACC_W-1:0]      result_data,
    output logic                         result_last,
    output logic                         result_vector_valid,
    input  logic                         result_vector_ready,
    output logic [OUT_DIM*ACC_W-1:0]     result_vector_data,

    output logic                         protocol_error,
    output logic [31:0]                  perf_jobs,
    output logic [31:0]                  perf_token_requests,
    output logic [31:0]                  perf_token_responses,
    output logic [31:0]                  perf_weight_requests,
    output logic [31:0]                  perf_weight_responses,
    output logic [31:0]                  perf_results,
    output logic [31:0]                  perf_result_jobs
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;
    localparam int TOTAL_TOKENS = TIME_PLANES * TOKENS_PER_PLANE;
    localparam int TOTAL_WEIGHTS = HEAD_DIM * OUT_DIM;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;
    localparam int ACC_VEC_W = OUT_DIM * ACC_W;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_WEIGHT_REQ,
        ST_WEIGHT_WAIT,
        ST_START,
        ST_PLANE_START,
        ST_TOKEN_REQ,
        ST_TOKEN_WAIT,
        ST_STREAM_DRAIN,
        ST_CLOSE,
        ST_RUN_DONE,
        ST_READ_REQ,
        ST_READ_WAIT,
        ST_RESULT_OUT,
        ST_WEIGHT_RELEASE,
        ST_JOB_DONE,
        ST_ERROR,
        ST_RELATION_SEAL
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] job_tag_q;
    logic [HEAD_W-1:0] job_input_head_q;
    logic [OUTPUT_TILE_W-1:0] job_output_tile_q;
    logic job_accumulate_q;
    logic job_emit_results_q;
    logic [LANE_W-1:0] weight_lane_q;
    logic [OUT_W-1:0] weight_out_q;
    logic [PLANE_W-1:0] token_plane_q;
    logic [Y_W-1:0] token_y_q;
    logic [X_W-1:0] token_x_q;
    logic [PLANE_W-1:0] result_plane_q;
    logic [Y_W-1:0] result_y_q;
    logic [X_W-1:0] result_x_q;
    logic [OUT_W-1:0] result_out_q;
    logic signed [ACC_W-1:0] result_data_q;
    logic [ACC_VEC_W-1:0] result_vector_data_q;
    logic protocol_error_q;

    logic tile_weight_valid;
    logic tile_weight_ready;
    logic tile_weight_last;
    logic tile_weight_context_release;
    logic tile_weight_context_release_ready;
    logic tile_projection_start;
    logic tile_projection_start_ready;
    logic tile_projection_close;
    logic tile_projection_close_ready;
    logic tile_projection_busy;
    logic tile_projection_done;
    logic tile_stream_idle;
    logic tile_plane_start;
    logic tile_plane_start_ready;
    logic tile_relation_seal;
    logic tile_relation_seal_ready;
    logic tile_in_valid;
    logic tile_in_ready;
    logic tile_read_valid;
    logic tile_read_ready;
    logic tile_read_data_valid;
    logic signed [ACC_W-1:0] tile_read_data;
    logic tile_vector_read_valid;
    logic tile_vector_read_ready;
    logic tile_vector_read_data_valid;
    logic [ACC_VEC_W-1:0] tile_vector_read_data;
    logic tile_protocol_error;
    logic [31:0] unused_descriptors;
    logic [31:0] unused_terms;
    logic [31:0] unused_updates;
    logic [31:0] unused_relation_stalls;

    logic job_fire;
    logic token_req_fire;
    logic token_rsp_fire;
    logic weight_req_fire;
    logic weight_rsp_fire;
    logic result_fire;
    logic job_done_fire;
    logic token_rsp_matches;
    logic weight_rsp_matches;
    logic final_weight;
    logic final_token;
    logic final_result;

    initial begin
        if (HEIGHT != 15 || WIDTH != 15 || TIME_PLANES != 2)
            $error("tagged Local5 job engine currently requires T450");
        if (HEAD_DIM != 32)
            $error("tagged Local5 job engine currently requires HEAD_DIM=32");
        if (USE_SCORE_ACTIVE_FRONT && VECTOR_RESULT_MODE)
            $error("production score-active child currently uses scalar readout");
    end

    assign job_ready = state_q == ST_IDLE && !tile_protocol_error;
    assign job_fire = job_valid && job_ready;
    assign job_done_valid = state_q == ST_JOB_DONE || state_q == ST_ERROR;
    assign job_done_tag = job_tag_q;
    assign job_done_input_head = job_input_head_q;
    assign job_done_error = state_q == ST_ERROR || protocol_error_q;
    assign job_done_fire = job_done_valid && job_done_ready;

    assign weight_req_valid = state_q == ST_WEIGHT_REQ;
    assign weight_req_tag = job_tag_q;
    assign weight_req_input_head = job_input_head_q;
    assign weight_req_output_tile = job_output_tile_q;
    assign weight_req_lane = weight_lane_q;
    assign weight_req_out = weight_out_q;
    assign weight_req_fire = weight_req_valid && weight_req_ready;
    assign weight_rsp_matches = weight_rsp_tag == job_tag_q
        && weight_rsp_input_head == job_input_head_q
        && weight_rsp_output_tile == job_output_tile_q
        && weight_rsp_lane == weight_lane_q
        && weight_rsp_out == weight_out_q
        && !weight_rsp_error;
    assign weight_rsp_ready = state_q == ST_WEIGHT_WAIT
        && (!weight_rsp_matches || tile_weight_ready);
    assign weight_rsp_fire = weight_rsp_valid && weight_rsp_ready;
    assign tile_weight_valid = state_q == ST_WEIGHT_WAIT
        && weight_rsp_valid && weight_rsp_matches;
    assign final_weight = 32'(weight_lane_q) + 1 == HEAD_DIM
                       && 32'(weight_out_q) + 1 == OUT_DIM;
    assign tile_weight_last = final_weight;

    assign tile_projection_start = state_q == ST_START
                                && tile_projection_start_ready;
    assign tile_plane_start = state_q == ST_PLANE_START
                           && tile_plane_start_ready;

    assign token_req_valid = state_q == ST_TOKEN_REQ;
    assign token_req_tag = job_tag_q;
    assign token_req_input_head = job_input_head_q;
    assign token_req_plane = token_plane_q;
    assign token_req_y = token_y_q;
    assign token_req_x = token_x_q;
    assign token_req_token_id = TOKEN_ID_W'(
        32'(token_plane_q) * TOKENS_PER_PLANE
        + 32'(token_y_q) * WIDTH
        + 32'(token_x_q)
    );
    assign token_req_fire = token_req_valid && token_req_ready;
    assign token_rsp_matches = token_rsp_tag == job_tag_q
        && token_rsp_input_head == job_input_head_q
        && token_rsp_token_id == token_req_token_id
        && !token_rsp_error;
    assign token_rsp_ready = state_q == ST_TOKEN_WAIT
        && (!token_rsp_matches || tile_in_ready);
    assign token_rsp_fire = token_rsp_valid && token_rsp_ready;
    assign tile_in_valid = state_q == ST_TOKEN_WAIT
        && token_rsp_valid && token_rsp_matches;
    assign final_token = 32'(token_plane_q) + 1 == TIME_PLANES
                      && 32'(token_y_q) + 1 == HEIGHT
                      && 32'(token_x_q) + 1 == WIDTH;

    assign tile_projection_close = state_q == ST_CLOSE
                                && tile_projection_close_ready;
    assign tile_relation_seal = state_q == ST_RELATION_SEAL
                              && tile_relation_seal_ready;
    assign tile_read_valid = !VECTOR_RESULT_MODE
                          && state_q == ST_READ_REQ && tile_read_ready;
    assign tile_vector_read_valid = VECTOR_RESULT_MODE
                                 && state_q == ST_READ_REQ
                                 && tile_vector_read_ready;
    assign final_result = 32'(result_plane_q) + 1 == TIME_PLANES
                       && 32'(result_y_q) + 1 == HEIGHT
                       && 32'(result_x_q) + 1 == WIDTH
                       && (VECTOR_RESULT_MODE
                           || 32'(result_out_q) + 1 == OUT_DIM);

    assign result_valid = !VECTOR_RESULT_MODE && state_q == ST_RESULT_OUT;
    assign result_vector_valid = VECTOR_RESULT_MODE
                              && state_q == ST_RESULT_OUT;
    assign result_tag = job_tag_q;
    assign result_input_head = job_input_head_q;
    assign result_output_tile = job_output_tile_q;
    assign result_plane = result_plane_q;
    assign result_y = result_y_q;
    assign result_x = result_x_q;
    assign result_out = result_out_q;
    assign result_data = result_data_q;
    assign result_vector_data = result_vector_data_q;
    assign result_last = final_result;
    assign result_fire = VECTOR_RESULT_MODE
        ? result_vector_valid && result_vector_ready
        : result_valid && result_ready;

    assign tile_weight_context_release = state_q == ST_WEIGHT_RELEASE
                                      && tile_weight_context_release_ready;
    assign protocol_error = protocol_error_q || tile_protocol_error;

    generate
        if (USE_SCORE_ACTIVE_FRONT) begin : g_score_active_tile
            logic [31:0] unused_score_rows;
            logic [31:0] unused_score_service_cycles;
            logic [31:0] unused_score_direct_rows;
            logic [31:0] unused_relation_writes;
            logic [31:0] unused_active_source_reads;
            logic [31:0] unused_dense_reads_avoided;
            logic [31:0] unused_memory_wait_cycles;
            logic [31:0] unused_qsilent_rows;
            logic [31:0] unused_identk_rows;
            logic [31:0] unused_overlap_accepts;
            logic [31:0] unused_cache_hits;
            logic [31:0] unused_cache_misses;
            logic [31:0] unused_tag_compares;
            logic [31:0] unused_lru_writes;
            logic [31:0] unused_product_reads;
            logic [31:0] unused_product_writes;
            logic [31:0] unused_product_starts;
            logic [31:0] unused_weight_reads;

            assign tile_projection_start_ready = 1'b1;
            assign tile_plane_start_ready = 1'b1;
            assign tile_stream_idle = tile_relation_seal_ready;
            assign tile_vector_read_ready = 1'b0;
            assign tile_vector_read_data_valid = 1'b0;
            assign tile_vector_read_data = '0;

            qfit_local5_score_active_projection_tile #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES), .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM), .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL), .BACKEND_KIND(0),
                .RELATION_SCHED_MODE(0), .ARCH_QSILENT(1'b1),
                .ARCH_IDENTK(1'b1), .ARCH_QSILENT_OVERLAP(1'b1)
            ) u_tile (
                .clk_core(clk_core), .rst_core(rst_core),
                .weight_valid(tile_weight_valid),
                .weight_ready(tile_weight_ready),
                .weight_lane(weight_rsp_lane), .weight_out(weight_rsp_out),
                .weight_data(weight_rsp_data), .weight_last(tile_weight_last),
                .weight_context_release(tile_weight_context_release),
                .weight_context_release_ready(
                    tile_weight_context_release_ready
                ),
                .projection_start(tile_projection_start),
                .projection_accumulate(job_accumulate_q),
                .projection_close(tile_projection_close),
                .projection_close_ready(tile_projection_close_ready),
                .projection_busy(tile_projection_busy),
                .projection_done(tile_projection_done),
                .relation_start(tile_projection_start),
                .relation_seal(tile_relation_seal),
                .relation_seal_ready(tile_relation_seal_ready),
                .relation_active(), .relation_done(),
                .row_valid(tile_in_valid), .row_ready(tile_in_ready),
                .row_plane(token_plane_q),
                .row_destination_y(token_y_q),
                .row_destination_x(token_x_q), .row_q(token_rsp_q),
                .row_candidate_k(token_rsp_k),
                .row_candidate_valid(token_rsp_valid_mask),
                .read_valid(tile_read_valid), .read_ready(tile_read_ready),
                .read_plane(result_plane_q), .read_y(result_y_q),
                .read_x(result_x_q), .read_out(result_out_q),
                .read_data_valid(tile_read_data_valid),
                .read_data(tile_read_data), .protocol_error(tile_protocol_error),
                .perf_score_rows(unused_score_rows),
                .perf_score_service_cycles(unused_score_service_cycles),
                .perf_score_direct_rows(unused_score_direct_rows),
                .perf_relation_writes(unused_relation_writes),
                .perf_active_source_reads(unused_active_source_reads),
                .perf_dense_reads_avoided(unused_dense_reads_avoided),
                .perf_memory_wait_cycles(unused_memory_wait_cycles),
                .perf_descriptors(unused_descriptors),
                .perf_product_terms(unused_terms),
                .perf_destination_updates(unused_updates),
                .perf_qsilent_rows(unused_qsilent_rows),
                .perf_identk_rows(unused_identk_rows),
                .perf_overlap_accepts(unused_overlap_accepts),
                .perf_cache_hits(unused_cache_hits),
                .perf_cache_misses(unused_cache_misses),
                .perf_tag_compares(unused_tag_compares),
                .perf_lru_writes(unused_lru_writes),
                .perf_product_reads(unused_product_reads),
                .perf_product_writes(unused_product_writes),
                .perf_product_starts(unused_product_starts),
                .perf_weight_reads(unused_weight_reads)
            );
        end else begin : g_legacy_tile
            assign tile_relation_seal_ready = 1'b0;
            qfit_local5_projection_tile #(
                .HEIGHT(HEIGHT), .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES), .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM), .TAG_W(16), .BACKEND_KIND(0),
                .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
                .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL),
                .ENABLE_VECTOR_READ(VECTOR_RESULT_MODE)
            ) u_tile (
                .clk_core(clk_core), .rst_core(rst_core),
                .weight_valid(tile_weight_valid),
                .weight_ready(tile_weight_ready),
                .weight_lane(weight_rsp_lane), .weight_out(weight_rsp_out),
                .weight_data(weight_rsp_data), .weight_last(tile_weight_last),
                .weight_context_release(tile_weight_context_release),
                .weight_context_release_ready(
                    tile_weight_context_release_ready
                ),
                .projection_start(tile_projection_start),
                .projection_accumulate(job_accumulate_q),
                .projection_start_ready(tile_projection_start_ready),
                .projection_close(tile_projection_close),
                .term_issue_enable(1'b1),
                .projection_close_ready(tile_projection_close_ready),
                .projection_busy(tile_projection_busy),
                .projection_done(tile_projection_done),
                .stream_idle(tile_stream_idle),
                .plane_start(tile_plane_start), .plane_id(token_plane_q),
                .plane_start_ready(tile_plane_start_ready),
                .in_valid(tile_in_valid), .in_ready(tile_in_ready),
                .in_y(token_y_q), .in_x(token_x_q), .in_q(token_rsp_q),
                .in_k(token_rsp_k), .in_valid_mask(token_rsp_valid_mask),
                .read_valid(tile_read_valid), .read_ready(tile_read_ready),
                .read_plane(result_plane_q), .read_y(result_y_q),
                .read_x(result_x_q), .read_out(result_out_q),
                .read_data_valid(tile_read_data_valid),
                .read_data(tile_read_data),
                .vector_read_valid(tile_vector_read_valid),
                .vector_read_ready(tile_vector_read_ready),
                .vector_read_plane(result_plane_q),
                .vector_read_y(result_y_q),
                .vector_read_x(result_x_q),
                .vector_read_data_valid(tile_vector_read_data_valid),
                .vector_read_data(tile_vector_read_data),
                .protocol_error(tile_protocol_error),
                .perf_descriptors(unused_descriptors),
                .perf_product_terms(unused_terms),
                .perf_destination_updates(unused_updates),
                .perf_relation_stalls(unused_relation_stalls)
            );
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            job_tag_q <= '0;
            job_input_head_q <= '0;
            job_output_tile_q <= '0;
            job_accumulate_q <= 1'b0;
            job_emit_results_q <= 1'b1;
            weight_lane_q <= '0;
            weight_out_q <= '0;
            token_plane_q <= '0;
            token_y_q <= '0;
            token_x_q <= '0;
            result_plane_q <= '0;
            result_y_q <= '0;
            result_x_q <= '0;
            result_out_q <= '0;
            result_data_q <= '0;
            result_vector_data_q <= '0;
            protocol_error_q <= 1'b0;
            perf_jobs <= '0;
            perf_token_requests <= '0;
            perf_token_responses <= '0;
            perf_weight_requests <= '0;
            perf_weight_responses <= '0;
            perf_results <= '0;
            perf_result_jobs <= '0;
        end else begin
            if (tile_protocol_error && state_q != ST_IDLE) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else if (
                (token_rsp_valid && state_q != ST_TOKEN_WAIT)
                || (weight_rsp_valid && state_q != ST_WEIGHT_WAIT)
            ) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                case (state_q)
                    ST_IDLE: begin
                        if (job_fire) begin
                            job_tag_q <= job_tag;
                            job_input_head_q <= job_input_head;
                            job_output_tile_q <= job_output_tile;
                            job_accumulate_q <= job_accumulate;
                            job_emit_results_q <= job_emit_results;
                            weight_lane_q <= '0;
                            weight_out_q <= '0;
                            token_plane_q <= '0;
                            token_y_q <= '0;
                            token_x_q <= '0;
                            result_plane_q <= '0;
                            result_y_q <= '0;
                            result_x_q <= '0;
                            result_out_q <= '0;
                            perf_jobs <= perf_jobs + 1'b1;
                            state_q <= ST_WEIGHT_REQ;
                        end
                    end

                    ST_WEIGHT_REQ: begin
                        if (weight_req_fire) begin
                            perf_weight_requests <=
                                perf_weight_requests + 1'b1;
                            state_q <= ST_WEIGHT_WAIT;
                        end
                    end

                    ST_WEIGHT_WAIT: begin
                        if (weight_rsp_fire) begin
                            if (!weight_rsp_matches) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                perf_weight_responses <=
                                    perf_weight_responses + 1'b1;
                                if (final_weight) begin
                                    state_q <= ST_START;
                                end else if (
                                    32'(weight_out_q) + 1 < OUT_DIM
                                ) begin
                                    weight_out_q <= weight_out_q + 1'b1;
                                    state_q <= ST_WEIGHT_REQ;
                                end else begin
                                    weight_out_q <= '0;
                                    weight_lane_q <= weight_lane_q + 1'b1;
                                    state_q <= ST_WEIGHT_REQ;
                                end
                            end
                        end
                    end

                    ST_START: begin
                        if (tile_projection_start) begin
                            token_plane_q <= '0;
                            token_y_q <= '0;
                            token_x_q <= '0;
                            state_q <= ST_PLANE_START;
                        end
                    end

                    ST_PLANE_START: begin
                        if (tile_plane_start)
                            state_q <= ST_TOKEN_REQ;
                    end

                    ST_TOKEN_REQ: begin
                        if (token_req_fire) begin
                            perf_token_requests <=
                                perf_token_requests + 1'b1;
                            state_q <= ST_TOKEN_WAIT;
                        end
                    end

                    ST_TOKEN_WAIT: begin
                        if (token_rsp_fire) begin
                            if (!token_rsp_matches) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                perf_token_responses <=
                                    perf_token_responses + 1'b1;
                                if (final_token) begin
                                    state_q <= ST_STREAM_DRAIN;
                                end else if (
                                    32'(token_x_q) + 1 < WIDTH
                                ) begin
                                    token_x_q <= token_x_q + 1'b1;
                                    state_q <= ST_TOKEN_REQ;
                                end else if (
                                    32'(token_y_q) + 1 < HEIGHT
                                ) begin
                                    token_x_q <= '0;
                                    token_y_q <= token_y_q + 1'b1;
                                    state_q <= ST_TOKEN_REQ;
                                end else begin
                                    token_x_q <= '0;
                                    token_y_q <= '0;
                                    token_plane_q <= token_plane_q + 1'b1;
                                    state_q <= ST_PLANE_START;
                                end
                            end
                        end
                    end

                    ST_STREAM_DRAIN: begin
                        if (tile_stream_idle)
                            state_q <= USE_SCORE_ACTIVE_FRONT
                                ? ST_RELATION_SEAL : ST_CLOSE;
                    end

                    ST_RELATION_SEAL: begin
                        if (tile_relation_seal)
                            state_q <= ST_CLOSE;
                    end

                    ST_CLOSE: begin
                        if (tile_projection_close)
                            state_q <= ST_RUN_DONE;
                    end

                    ST_RUN_DONE: begin
                        if (tile_projection_done) begin
                            if (job_emit_results_q) begin
                                result_plane_q <= '0;
                                result_y_q <= '0;
                                result_x_q <= '0;
                                result_out_q <= '0;
                                state_q <= ST_READ_REQ;
                            end else begin
                                state_q <= ST_WEIGHT_RELEASE;
                            end
                        end
                    end

                    ST_READ_REQ: begin
                        if (tile_read_valid || tile_vector_read_valid)
                            state_q <= ST_READ_WAIT;
                    end

                    ST_READ_WAIT: begin
                        if ((!VECTOR_RESULT_MODE && tile_read_data_valid)
                            || (VECTOR_RESULT_MODE
                                && tile_vector_read_data_valid)) begin
                            if (VECTOR_RESULT_MODE)
                                result_vector_data_q <= tile_vector_read_data;
                            else
                                result_data_q <= tile_read_data;
                            state_q <= ST_RESULT_OUT;
                        end
                    end

                    ST_RESULT_OUT: begin
                        if (result_fire) begin
                            perf_results <= perf_results + 1'b1;
                            if (final_result) begin
                                perf_result_jobs <= perf_result_jobs + 1'b1;
                                state_q <= ST_WEIGHT_RELEASE;
                            end else if (!VECTOR_RESULT_MODE &&
                                32'(result_out_q) + 1 < OUT_DIM
                            ) begin
                                result_out_q <= result_out_q + 1'b1;
                                state_q <= ST_READ_REQ;
                            end else if (
                                32'(result_x_q) + 1 < WIDTH
                            ) begin
                                result_out_q <= '0;
                                result_x_q <= result_x_q + 1'b1;
                                state_q <= ST_READ_REQ;
                            end else if (
                                32'(result_y_q) + 1 < HEIGHT
                            ) begin
                                result_out_q <= '0;
                                result_x_q <= '0;
                                result_y_q <= result_y_q + 1'b1;
                                state_q <= ST_READ_REQ;
                            end else begin
                                result_out_q <= '0;
                                result_x_q <= '0;
                                result_y_q <= '0;
                                result_plane_q <= result_plane_q + 1'b1;
                                state_q <= ST_READ_REQ;
                            end
                        end
                    end

                    ST_WEIGHT_RELEASE: begin
                        if (tile_weight_context_release)
                            state_q <= ST_JOB_DONE;
                    end

                    ST_JOB_DONE: begin
                        if (job_done_fire)
                            state_q <= ST_IDLE;
                    end

                    default: state_q <= ST_ERROR;
                endcase
            end
        end
    end

    logic unused_projection_busy;
    logic [31:0] unused_total_weights;
    logic [31:0] unused_total_results;
    assign unused_projection_busy = tile_projection_busy;
    assign unused_total_weights = TOTAL_WEIGHTS;
    assign unused_total_results = TOTAL_RESULTS;
endmodule

`default_nettype wire
