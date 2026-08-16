`timescale 1ns/1ps
`default_nettype none

// Tagged Local5 T450 head/output-tile engine with a window-persistent exact
// relation memo. The first output tile builds score/Shiftmax5 relations from
// Q/K tokens. Later output tiles replay resident relations and fall back to
// the identical score path on a miss. Projection weights remain job-local.
module qfit_local5_memo_tagged_t450_job_engine #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 32,
    parameter int TAG_W = 24,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter bit VECTOR_RESULT_MODE = 1'b0,
    parameter int GATE_W = 9,
    parameter int MAX_HEADS = 24,
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
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int MEMO_HEAD_W =
        (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS),
    parameter int PTR_W = $clog2(513)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         job_valid,
    output logic                         job_ready,
    input  logic [TAG_W-1:0]             job_tag,
    input  logic [HEAD_W-1:0]            job_input_head,
    input  logic [OUTPUT_TILE_W-1:0]     job_output_tile,
    input  logic                         job_decode_required,
    input  logic                         job_cache_release,
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
    output logic [31:0]                  perf_result_jobs,
    output logic [31:0]                  perf_memo_hits,
    output logic [31:0]                  perf_memo_fallbacks,
    output logic [31:0]                  perf_memo_resident_builds,
    output logic [31:0]                  perf_cache_release_intents,
    output logic [31:0]                  perf_replay_records
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;
    localparam int TOTAL_WEIGHTS = HEAD_DIM * OUT_DIM;
    localparam int TOTAL_RESULTS = HEIGHT * WIDTH * TIME_PLANES * OUT_DIM;
    localparam int SCORE_TAG_W = PLANE_W + Y_W + X_W;
    localparam int ACC_VEC_W = OUT_DIM * ACC_W;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_WEIGHT_REQ,
        ST_WEIGHT_WAIT,
        ST_PROJECTION_START,
        ST_WINDOW_START,
        ST_PATH_SELECT,
        ST_REPLAY_ISSUE,
        ST_REPLAY_WAIT,
        ST_HEAD_START,
        ST_PLANE_START,
        ST_TOKEN_REQ,
        ST_TOKEN_WAIT,
        ST_PLANE_DRAIN,
        ST_DESCRIPTOR_DRAIN,
        ST_CLOSE,
        ST_RUN_DONE,
        ST_READ_REQ,
        ST_READ_WAIT,
        ST_RESULT_OUT,
        ST_WEIGHT_RELEASE,
        ST_JOB_DONE,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] job_tag_q;
    logic [HEAD_W-1:0] job_input_head_q;
    logic [OUTPUT_TILE_W-1:0] job_output_tile_q;
    logic job_decode_required_q;
    logic job_cache_release_q;
    logic job_accumulate_q;
    logic job_emit_results_q;
    logic use_replay_q;
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
    logic memo_head_done_seen_q;
    logic memo_head_resident_seen_q;

    logic score_in_ready;
    logic score_out_valid;
    logic score_out_ready;
    logic [SCORE_TAG_W-1:0] score_in_tag;
    logic [SCORE_TAG_W-1:0] score_out_tag;
    logic [5*16-1:0] score_out_q7;
    logic [5*GATE_W-1:0] score_out_gate;
    logic [31:0] score_out_k;
    logic [4:0] score_out_mask;
    logic [PLANE_W-1:0] score_out_plane;
    logic [Y_W-1:0] score_out_y;
    logic [X_W-1:0] score_out_x;

    logic memo_window_start;
    logic memo_head_start;
    logic memo_head_ready;
    logic memo_head_done;
    logic memo_head_resident;
    logic memo_head_critical;
    logic memo_head_overflow;
    logic [31:0] memo_head_service_cycles;
    logic [PTR_W-1:0] memo_head_record_count;
    logic memo_plane_start;
    logic memo_in_ready;
    logic memo_plane_idle;
    logic memo_replay_start;
    logic memo_replay_cmd_ready;
    logic memo_replay_done;
    logic memo_replay_miss;
    logic memo_weight_ready;
    logic memo_weight_release;
    logic memo_weight_release_ready;
    logic memo_projection_start;
    logic memo_projection_close;
    logic memo_projection_close_ready;
    logic memo_projection_busy;
    logic memo_projection_done;
    logic memo_read_valid;
    logic memo_read_ready;
    logic memo_read_data_valid;
    logic signed [ACC_W-1:0] memo_read_data;
    logic memo_vector_read_valid;
    logic memo_vector_read_ready;
    logic memo_vector_read_data_valid;
    logic [ACC_VEC_W-1:0] memo_vector_read_data;
    logic memo_descriptor_valid;
    logic memo_descriptor_ready;
    logic memo_descriptor_last;
    logic memo_descriptor_stream_idle;
    logic memo_protocol_error;

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
    logic final_token_in_plane;
    logic final_result;
    logic window_reset_required;

    logic [8:0] unused_descriptor_source_id;
    logic [Y_W-1:0] unused_descriptor_y;
    logic [X_W-1:0] unused_descriptor_x;
    logic [HEAD_DIM-1:0] unused_descriptor_k;
    logic [5*GATE_W-1:0] unused_descriptor_gates;
    logic [4:0] unused_descriptor_mask;
    logic [31:0] unused_speculative_writes;
    logic [31:0] unused_discarded_writes;
    logic [31:0] unused_committed_records;
    logic [31:0] unused_capacity_misses;
    logic [31:0] unused_descriptors;
    logic [31:0] unused_product_terms;
    logic [31:0] unused_destination_updates;
    logic [15:0] unused_score_cycles;
    logic [3:0] unused_score_route;

    initial begin
        if (HEIGHT != 15 || WIDTH != 15 || TIME_PLANES != 2)
            $error("memo tagged Local5 job engine requires T450");
        if (HEAD_DIM != 32)
            $error("memo tagged Local5 job engine requires HEAD_DIM=32");
        if (HEAD_W < MEMO_HEAD_W)
            $error("job head id must cover relation memo directory");
    end

    assign job_ready = state_q == ST_IDLE && !memo_protocol_error;
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
        && (!weight_rsp_matches || memo_weight_ready);
    assign weight_rsp_fire = weight_rsp_valid && weight_rsp_ready;
    assign final_weight = 32'(weight_lane_q) + 1 == HEAD_DIM
                       && 32'(weight_out_q) + 1 == OUT_DIM;

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
        && (!token_rsp_matches || score_in_ready);
    assign token_rsp_fire = token_rsp_valid && token_rsp_ready;
    assign final_token_in_plane = 32'(token_y_q) + 1 == HEIGHT
                               && 32'(token_x_q) + 1 == WIDTH;

    assign score_in_tag = {token_plane_q, token_y_q, token_x_q};
    assign {score_out_plane, score_out_y, score_out_x} = score_out_tag;
    assign score_out_ready = !use_replay_q && memo_in_ready;

    assign memo_window_start = state_q == ST_WINDOW_START;
    assign memo_head_start = state_q == ST_HEAD_START && memo_head_ready;
    assign memo_plane_start = state_q == ST_PLANE_START
                           && memo_plane_idle && !score_out_valid;
    assign memo_replay_start = state_q == ST_REPLAY_ISSUE
                            && memo_replay_cmd_ready;
    assign memo_projection_start = state_q == ST_PROJECTION_START;
    assign memo_projection_close = state_q == ST_CLOSE
                                && memo_projection_close_ready;
    assign memo_weight_release = state_q == ST_WEIGHT_RELEASE
                              && memo_weight_release_ready;
    assign memo_read_valid = !VECTOR_RESULT_MODE
                          && state_q == ST_READ_REQ && memo_read_ready;
    assign memo_vector_read_valid = VECTOR_RESULT_MODE
                                 && state_q == ST_READ_REQ
                                 && memo_vector_read_ready;
    assign window_reset_required = job_decode_required_q
                                && job_output_tile_q == '0
                                && job_input_head_q == '0;

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
    assign protocol_error = protocol_error_q || memo_protocol_error;

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2),
        .TAG_W(SCORE_TAG_W),
        .SCORE_W(16),
        .GATE_W(GATE_W)
    ) u_score (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(state_q == ST_TOKEN_WAIT
                  && token_rsp_valid && token_rsp_matches),
        .in_ready(score_in_ready),
        .in_tag(score_in_tag),
        .in_q(token_rsp_q),
        .in_k(token_rsp_k),
        .in_valid_mask(token_rsp_valid_mask),
        .out_valid(score_out_valid),
        .out_ready(score_out_ready),
        .out_tag(score_out_tag),
        .out_score_q7(score_out_q7),
        .out_gate_q17(score_out_gate),
        .out_k_self(score_out_k),
        .out_valid_mask(score_out_mask),
        .perf_service_cycles(unused_score_cycles),
        .perf_route_direct_mask(unused_score_route)
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
        .ENABLE_VECTOR_READ(VECTOR_RESULT_MODE),
        .MAX_HEADS(MAX_HEADS),
        .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
        .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
    ) u_memo_projection (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(memo_window_start),
        .head_start(memo_head_start),
        .head_ready(memo_head_ready),
        .head_index(MEMO_HEAD_W'(job_input_head_q)),
        .head_done(memo_head_done),
        .head_resident(memo_head_resident),
        .head_critical(memo_head_critical),
        .head_overflow(memo_head_overflow),
        .head_service_cycles(memo_head_service_cycles),
        .head_record_count(memo_head_record_count),
        .plane_start(memo_plane_start),
        .plane_id(token_plane_q),
        .in_valid(score_out_valid && !use_replay_q),
        .in_ready(memo_in_ready),
        .in_y(score_out_y),
        .in_x(score_out_x),
        .in_candidate_valid(score_out_mask),
        .in_k_self(score_out_k),
        .in_direction_gates(score_out_gate),
        .plane_idle(memo_plane_idle),
        .use_replay(use_replay_q),
        .replay_start(memo_replay_start),
        .replay_cmd_ready(memo_replay_cmd_ready),
        .replay_head_index(MEMO_HEAD_W'(job_input_head_q)),
        .replay_done(memo_replay_done),
        .replay_miss(memo_replay_miss),
        .weight_valid(state_q == ST_WEIGHT_WAIT
                      && weight_rsp_valid && weight_rsp_matches),
        .weight_ready(memo_weight_ready),
        .weight_lane(weight_rsp_lane),
        .weight_out(weight_rsp_out),
        .weight_data(weight_rsp_data),
        .weight_last(final_weight),
        .weight_context_release(memo_weight_release),
        .weight_context_release_ready(memo_weight_release_ready),
        .projection_start(memo_projection_start),
        .projection_accumulate(job_accumulate_q),
        .projection_close(memo_projection_close),
        .projection_close_ready(memo_projection_close_ready),
        .projection_busy(memo_projection_busy),
        .projection_done(memo_projection_done),
        .read_valid(memo_read_valid),
        .read_ready(memo_read_ready),
        .read_plane(result_plane_q),
        .read_y(result_y_q),
        .read_x(result_x_q),
        .read_out(result_out_q),
        .read_data_valid(memo_read_data_valid),
        .read_data(memo_read_data),
        .vector_read_valid(memo_vector_read_valid),
        .vector_read_ready(memo_vector_read_ready),
        .vector_read_plane(result_plane_q),
        .vector_read_y(result_y_q),
        .vector_read_x(result_x_q),
        .vector_read_data_valid(memo_vector_read_data_valid),
        .vector_read_data(memo_vector_read_data),
        .descriptor_valid(memo_descriptor_valid),
        .descriptor_ready(memo_descriptor_ready),
        .descriptor_source_id(unused_descriptor_source_id),
        .descriptor_y(unused_descriptor_y),
        .descriptor_x(unused_descriptor_x),
        .descriptor_k(unused_descriptor_k),
        .descriptor_gates(unused_descriptor_gates),
        .descriptor_valid_mask(unused_descriptor_mask),
        .descriptor_last(memo_descriptor_last),
        .descriptor_stream_idle(memo_descriptor_stream_idle),
        .protocol_error(memo_protocol_error),
        .perf_speculative_writes(unused_speculative_writes),
        .perf_discarded_writes(unused_discarded_writes),
        .perf_committed_records(unused_committed_records),
        .perf_replay_reads(perf_replay_records),
        .perf_capacity_misses(unused_capacity_misses),
        .perf_descriptors(unused_descriptors),
        .perf_product_terms(unused_product_terms),
        .perf_destination_updates(unused_destination_updates)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            job_tag_q <= '0;
            job_input_head_q <= '0;
            job_output_tile_q <= '0;
            job_decode_required_q <= 1'b0;
            job_cache_release_q <= 1'b0;
            job_accumulate_q <= 1'b0;
            job_emit_results_q <= 1'b1;
            use_replay_q <= 1'b0;
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
            memo_head_done_seen_q <= 1'b0;
            memo_head_resident_seen_q <= 1'b0;
            perf_jobs <= '0;
            perf_token_requests <= '0;
            perf_token_responses <= '0;
            perf_weight_requests <= '0;
            perf_weight_responses <= '0;
            perf_results <= '0;
            perf_result_jobs <= '0;
            perf_memo_hits <= '0;
            perf_memo_fallbacks <= '0;
            perf_memo_resident_builds <= '0;
            perf_cache_release_intents <= '0;
        end else begin
            if (memo_head_start) begin
                memo_head_done_seen_q <= 1'b0;
                memo_head_resident_seen_q <= 1'b0;
            end else if (memo_head_done) begin
                memo_head_done_seen_q <= 1'b1;
                memo_head_resident_seen_q <= memo_head_resident;
            end
            if (memo_protocol_error && state_q != ST_IDLE) begin
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
                            if (32'(job_input_head) >= MAX_HEADS
                                || job_decode_required
                                   != (job_output_tile == '0)) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                job_tag_q <= job_tag;
                                job_input_head_q <= job_input_head;
                                job_output_tile_q <= job_output_tile;
                                job_decode_required_q <= job_decode_required;
                                job_cache_release_q <= job_cache_release;
                                job_accumulate_q <= job_accumulate;
                                job_emit_results_q <= job_emit_results;
                                use_replay_q <= !job_decode_required;
                                weight_lane_q <= '0;
                                weight_out_q <= '0;
                                result_plane_q <= '0;
                                result_y_q <= '0;
                                result_x_q <= '0;
                                result_out_q <= '0;
                                perf_jobs <= perf_jobs + 1'b1;
                                if (job_cache_release)
                                    perf_cache_release_intents <=
                                        perf_cache_release_intents + 1'b1;
                                state_q <= ST_WEIGHT_REQ;
                            end
                        end
                    end

                    ST_WEIGHT_REQ: begin
                        if (weight_req_fire) begin
                            perf_weight_requests <= perf_weight_requests + 1'b1;
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
                                    state_q <= ST_PROJECTION_START;
                                end else if (32'(weight_out_q) + 1 < OUT_DIM) begin
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

                    ST_PROJECTION_START: begin
                        if (memo_projection_start) begin
                            if (window_reset_required)
                                state_q <= ST_WINDOW_START;
                            else
                                state_q <= ST_PATH_SELECT;
                        end
                    end

                    ST_WINDOW_START: state_q <= ST_PATH_SELECT;

                    ST_PATH_SELECT: begin
                        if (use_replay_q)
                            state_q <= ST_REPLAY_ISSUE;
                        else
                            state_q <= ST_HEAD_START;
                    end

                    ST_REPLAY_ISSUE: begin
                        if (memo_replay_start)
                            state_q <= ST_REPLAY_WAIT;
                    end

                    ST_REPLAY_WAIT: begin
                        if (memo_replay_done) begin
                            if (memo_replay_miss) begin
                                use_replay_q <= 1'b0;
                                perf_memo_fallbacks <=
                                    perf_memo_fallbacks + 1'b1;
                                state_q <= ST_HEAD_START;
                            end else begin
                                perf_memo_hits <= perf_memo_hits + 1'b1;
                                state_q <= ST_DESCRIPTOR_DRAIN;
                            end
                        end
                    end

                    ST_HEAD_START: begin
                        if (memo_head_start) begin
                            token_plane_q <= '0;
                            token_y_q <= '0;
                            token_x_q <= '0;
                            state_q <= ST_PLANE_START;
                        end
                    end

                    ST_PLANE_START: begin
                        if (memo_plane_start)
                            state_q <= ST_TOKEN_REQ;
                    end

                    ST_TOKEN_REQ: begin
                        if (token_req_fire) begin
                            perf_token_requests <= perf_token_requests + 1'b1;
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
                                if (final_token_in_plane) begin
                                    state_q <= ST_PLANE_DRAIN;
                                end else if (32'(token_x_q) + 1 < WIDTH) begin
                                    token_x_q <= token_x_q + 1'b1;
                                    state_q <= ST_TOKEN_REQ;
                                end else begin
                                    token_x_q <= '0;
                                    token_y_q <= token_y_q + 1'b1;
                                    state_q <= ST_TOKEN_REQ;
                                end
                            end
                        end
                    end

                    ST_PLANE_DRAIN: begin
                        if (memo_plane_idle && !score_out_valid) begin
                            if (32'(token_plane_q) + 1 < TIME_PLANES) begin
                                token_plane_q <= token_plane_q + 1'b1;
                                token_y_q <= '0;
                                token_x_q <= '0;
                                state_q <= ST_PLANE_START;
                            end else if (memo_head_done_seen_q) begin
                                if (memo_head_resident_seen_q)
                                    perf_memo_resident_builds <=
                                        perf_memo_resident_builds + 1'b1;
                                state_q <= ST_DESCRIPTOR_DRAIN;
                            end
                        end
                    end

                    ST_DESCRIPTOR_DRAIN: begin
                        if (memo_descriptor_stream_idle)
                            state_q <= ST_CLOSE;
                    end

                    ST_CLOSE: begin
                        if (memo_projection_close)
                            state_q <= ST_RUN_DONE;
                    end

                    ST_RUN_DONE: begin
                        if (memo_projection_done) begin
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
                        if (memo_read_valid || memo_vector_read_valid)
                            state_q <= ST_READ_WAIT;
                    end

                    ST_READ_WAIT: begin
                        if ((!VECTOR_RESULT_MODE && memo_read_data_valid)
                            || (VECTOR_RESULT_MODE
                                && memo_vector_read_data_valid)) begin
                            if (VECTOR_RESULT_MODE)
                                result_vector_data_q <= memo_vector_read_data;
                            else
                                result_data_q <= memo_read_data;
                            state_q <= ST_RESULT_OUT;
                        end
                    end

                    ST_RESULT_OUT: begin
                        if (result_fire) begin
                            perf_results <= perf_results + 1'b1;
                            if (final_result) begin
                                perf_result_jobs <= perf_result_jobs + 1'b1;
                                state_q <= ST_WEIGHT_RELEASE;
                            end else if (!VECTOR_RESULT_MODE
                                         && 32'(result_out_q) + 1 < OUT_DIM) begin
                                result_out_q <= result_out_q + 1'b1;
                                state_q <= ST_READ_REQ;
                            end else if (32'(result_x_q) + 1 < WIDTH) begin
                                result_out_q <= '0;
                                result_x_q <= result_x_q + 1'b1;
                                state_q <= ST_READ_REQ;
                            end else if (32'(result_y_q) + 1 < HEIGHT) begin
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
                        if (memo_weight_release)
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

    logic [31:0] unused_total_weights;
    logic [31:0] unused_total_results;
    logic unused_projection_busy;
    logic unused_descriptor_signals;
    logic unused_head_metadata;
    logic unused_cache_release;
    assign unused_total_weights = TOTAL_WEIGHTS;
    assign unused_total_results = TOTAL_RESULTS;
    assign unused_projection_busy = memo_projection_busy;
    assign unused_descriptor_signals = memo_descriptor_valid
                                     ^ memo_descriptor_ready
                                     ^ memo_descriptor_last;
    assign unused_head_metadata = memo_head_critical
                                ^ memo_head_overflow
                                ^ ^memo_head_service_cycles
                                ^ ^memo_head_record_count;
    assign unused_cache_release = job_cache_release_q;
    logic [5*16-1:0] unused_score_q7_sink;
    assign unused_score_q7_sink = score_out_q7;
endmodule

`default_nettype wire
