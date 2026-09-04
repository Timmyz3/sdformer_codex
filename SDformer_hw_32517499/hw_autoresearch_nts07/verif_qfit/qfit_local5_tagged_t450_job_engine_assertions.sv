`timescale 1ns/1ps
`default_nettype none

module qfit_local5_tagged_t450_job_engine_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int TAG_W = 24,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int TOKEN_ID_W = 9,
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter bit VECTOR_RESULT_MODE = 1'b0
) (
    input logic clk_core,
    input logic rst_core,
    input logic token_req_valid,
    input logic token_req_ready,
    input logic [TAG_W-1:0] token_req_tag,
    input logic [HEAD_W-1:0] token_req_input_head,
    input logic [TOKEN_ID_W-1:0] token_req_token_id,
    input logic token_req_plane,
    input logic [3:0] token_req_y,
    input logic [3:0] token_req_x,
    input logic token_rsp_valid,
    input logic token_rsp_ready,
    input logic [TAG_W-1:0] token_rsp_tag,
    input logic [HEAD_W-1:0] token_rsp_input_head,
    input logic [TOKEN_ID_W-1:0] token_rsp_token_id,
    input logic [31:0] token_rsp_q,
    input logic [5*32-1:0] token_rsp_k,
    input logic [4:0] token_rsp_valid_mask,
    input logic token_rsp_error,
    input logic weight_req_valid,
    input logic weight_req_ready,
    input logic [TAG_W-1:0] weight_req_tag,
    input logic [HEAD_W-1:0] weight_req_input_head,
    input logic [OUTPUT_TILE_W-1:0] weight_req_output_tile,
    input logic [4:0] weight_req_lane,
    input logic [OUT_W-1:0] weight_req_out,
    input logic weight_rsp_valid,
    input logic weight_rsp_ready,
    input logic [TAG_W-1:0] weight_rsp_tag,
    input logic [HEAD_W-1:0] weight_rsp_input_head,
    input logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile,
    input logic [4:0] weight_rsp_lane,
    input logic [OUT_W-1:0] weight_rsp_out,
    input logic signed [7:0] weight_rsp_data,
    input logic weight_rsp_error,
    input logic result_valid,
    input logic result_ready,
    input logic [TAG_W-1:0] result_tag,
    input logic [HEAD_W-1:0] result_input_head,
    input logic [OUTPUT_TILE_W-1:0] result_output_tile,
    input logic result_plane,
    input logic [3:0] result_y,
    input logic [3:0] result_x,
    input logic [OUT_W-1:0] result_out,
    input logic signed [31:0] result_data,
    input logic result_last,
    input logic result_vector_valid,
    input logic result_vector_ready,
    input logic [OUT_DIM*32-1:0] result_vector_data,
    input logic job_done_valid,
    input logic job_done_ready,
    input logic [TAG_W-1:0] job_done_tag,
    input logic [HEAD_W-1:0] job_done_input_head,
    input logic job_done_error,
    input logic protocol_error,
    input logic [4:0] state_q,
    input logic [31:0] perf_jobs,
    input logic [31:0] perf_token_requests,
    input logic [31:0] perf_token_responses,
    input logic [31:0] perf_weight_requests,
    input logic [31:0] perf_weight_responses,
    input logic [31:0] perf_results,
    input logic [31:0] perf_result_jobs
);
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;
    localparam int TOTAL_WEIGHTS = HEAD_DIM * OUT_DIM;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;
    localparam logic [4:0] ST_WEIGHT_WAIT = 5'd2;
    localparam logic [4:0] ST_TOKEN_WAIT = 5'd6;

    property p_token_request_stable;
        @(posedge clk_core) disable iff (rst_core)
            token_req_valid && !token_req_ready
            |=> token_req_valid
                && $stable({token_req_tag, token_req_input_head,
                            token_req_token_id, token_req_plane,
                            token_req_y, token_req_x});
    endproperty

    property p_weight_request_stable;
        @(posedge clk_core) disable iff (rst_core)
            weight_req_valid && !weight_req_ready
            |=> weight_req_valid
                && $stable({weight_req_tag, weight_req_input_head,
                            weight_req_output_tile, weight_req_lane,
                            weight_req_out});
    endproperty

    property p_token_response_stable;
        @(posedge clk_core) disable iff (rst_core)
            token_rsp_valid && !token_rsp_ready
            |=> token_rsp_valid
                && $stable({token_rsp_tag, token_rsp_input_head,
                            token_rsp_token_id, token_rsp_q, token_rsp_k,
                            token_rsp_valid_mask, token_rsp_error});
    endproperty

    property p_weight_response_stable;
        @(posedge clk_core) disable iff (rst_core)
            weight_rsp_valid && !weight_rsp_ready
            |=> weight_rsp_valid
                && $stable({weight_rsp_tag, weight_rsp_input_head,
                            weight_rsp_output_tile, weight_rsp_lane,
                            weight_rsp_out, weight_rsp_data,
                            weight_rsp_error});
    endproperty

    property p_bad_token_response_reports_error;
        @(posedge clk_core) disable iff (rst_core)
            token_rsp_valid && token_rsp_ready
            && (token_rsp_error
                || token_rsp_tag != token_req_tag
                || token_rsp_input_head != token_req_input_head
                || token_rsp_token_id != token_req_token_id)
            |=> protocol_error;
    endproperty

    property p_bad_weight_response_reports_error;
        @(posedge clk_core) disable iff (rst_core)
            weight_rsp_valid && weight_rsp_ready
            && (weight_rsp_error
                || weight_rsp_tag != weight_req_tag
                || weight_rsp_input_head != weight_req_input_head
                || weight_rsp_output_tile != weight_req_output_tile
                || weight_rsp_lane != weight_req_lane
                || weight_rsp_out != weight_req_out)
            |=> protocol_error;
    endproperty

    property p_unsolicited_token_response_reports_error;
        @(posedge clk_core) disable iff (rst_core)
            token_rsp_valid && state_q != ST_TOKEN_WAIT
            |=> protocol_error;
    endproperty

    property p_unsolicited_weight_response_reports_error;
        @(posedge clk_core) disable iff (rst_core)
            weight_rsp_valid && state_q != ST_WEIGHT_WAIT
            |=> protocol_error;
    endproperty

    property p_result_stable;
        @(posedge clk_core) disable iff (rst_core)
            result_valid && !result_ready
            |=> result_valid
                && $stable({result_tag, result_input_head,
                            result_output_tile, result_plane,
                            result_y, result_x, result_out,
                            result_data, result_last});
    endproperty

    property p_vector_result_stable;
        @(posedge clk_core) disable iff (rst_core)
            result_vector_valid && !result_vector_ready
            |=> result_vector_valid
                && $stable({result_tag, result_input_head,
                            result_output_tile, result_plane,
                            result_y, result_x, result_last,
                            result_vector_data});
    endproperty

    property p_vector_result_geometry;
        @(posedge clk_core) disable iff (rst_core)
            result_vector_valid
            |-> 32'(result_plane) < TIME_PLANES
                && 32'(result_y) < HEIGHT
                && 32'(result_x) < WIDTH
                && result_last == (
                    32'(result_plane) + 1 == TIME_PLANES
                    && 32'(result_y) + 1 == HEIGHT
                    && 32'(result_x) + 1 == WIDTH
                );
    endproperty

    property p_job_done_stable;
        @(posedge clk_core) disable iff (rst_core)
            job_done_valid && !job_done_ready
            |=> job_done_valid
                && $stable({job_done_tag, job_done_input_head,
                            job_done_error});
    endproperty

    property p_token_geometry;
        @(posedge clk_core) disable iff (rst_core)
            token_req_valid
            |-> 32'(token_req_plane) < TIME_PLANES
                && 32'(token_req_y) < HEIGHT
                && 32'(token_req_x) < WIDTH
                && 32'(token_req_token_id)
                    == 32'(token_req_plane) * HEIGHT * WIDTH
                     + 32'(token_req_y) * WIDTH
                     + 32'(token_req_x);
    endproperty

    property p_weight_geometry;
        @(posedge clk_core) disable iff (rst_core)
            weight_req_valid
            |-> 32'(weight_req_lane) < HEAD_DIM
                && 32'(weight_req_out) < OUT_DIM;
    endproperty

    property p_result_geometry;
        @(posedge clk_core) disable iff (rst_core)
            result_valid
            |-> 32'(result_plane) < TIME_PLANES
                && 32'(result_y) < HEIGHT
                && 32'(result_x) < WIDTH
                && 32'(result_out) < OUT_DIM
                && result_last == (
                    32'(result_plane) + 1 == TIME_PLANES
                    && 32'(result_y) + 1 == HEIGHT
                    && 32'(result_x) + 1 == WIDTH
                    && 32'(result_out) + 1 == OUT_DIM
                );
    endproperty

    property p_completed_job_has_full_ledger;
        @(posedge clk_core) disable iff (rst_core)
            job_done_valid && !job_done_error
            |-> perf_token_responses == perf_jobs * TOTAL_TOKENS
                && perf_weight_responses == perf_jobs * TOTAL_WEIGHTS
                && perf_results == perf_result_jobs
                    * (VECTOR_RESULT_MODE ? TOTAL_TOKENS : TOTAL_RESULTS)
                && perf_result_jobs <= perf_jobs;
    endproperty

    property p_response_ledgers_never_lead;
        @(posedge clk_core) disable iff (rst_core)
            perf_token_responses <= perf_token_requests
            && perf_weight_responses <= perf_weight_requests;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
            $past(protocol_error) |-> protocol_error;
    endproperty

    property p_error_stops_external_work;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error
            |-> !token_req_valid && !weight_req_valid
                && !result_valid && !result_vector_valid;
    endproperty

    assert property (p_token_request_stable);
    assert property (p_weight_request_stable);
    assert property (p_token_response_stable);
    assert property (p_weight_response_stable);
    assert property (p_bad_token_response_reports_error);
    assert property (p_bad_weight_response_reports_error);
    assert property (p_unsolicited_token_response_reports_error);
    assert property (p_unsolicited_weight_response_reports_error);
    assert property (p_result_stable);
    assert property (p_vector_result_stable);
    assert property (p_vector_result_geometry);
    assert property (p_job_done_stable);
    assert property (p_token_geometry);
    assert property (p_weight_geometry);
    assert property (p_result_geometry);
    assert property (p_completed_job_has_full_ledger);
    assert property (p_response_ledgers_never_lead);
    assert property (p_protocol_error_sticky);
    assert property (p_error_stops_external_work);
endmodule

bind qfit_local5_tagged_t450_job_engine
    qfit_local5_tagged_t450_job_engine_assertions #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .TAG_W(TAG_W),
        .HEAD_W(HEAD_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .TOKEN_ID_W(TOKEN_ID_W), .OUT_W(OUT_W),
        .VECTOR_RESULT_MODE(VECTOR_RESULT_MODE)
    ) u_qfit_local5_tagged_t450_job_engine_assertions (
        .clk_core(clk_core), .rst_core(rst_core),
        .token_req_valid(token_req_valid), .token_req_ready(token_req_ready),
        .token_req_tag(token_req_tag),
        .token_req_input_head(token_req_input_head),
        .token_req_token_id(token_req_token_id),
        .token_req_plane(token_req_plane), .token_req_y(token_req_y),
        .token_req_x(token_req_x),
        .token_rsp_valid(token_rsp_valid), .token_rsp_ready(token_rsp_ready),
        .token_rsp_tag(token_rsp_tag),
        .token_rsp_input_head(token_rsp_input_head),
        .token_rsp_token_id(token_rsp_token_id),
        .token_rsp_q(token_rsp_q), .token_rsp_k(token_rsp_k),
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
        .result_valid(result_valid), .result_ready(result_ready),
        .result_tag(result_tag), .result_input_head(result_input_head),
        .result_output_tile(result_output_tile), .result_plane(result_plane),
        .result_y(result_y), .result_x(result_x), .result_out(result_out),
        .result_data(result_data), .result_last(result_last),
        .result_vector_valid(result_vector_valid),
        .result_vector_ready(result_vector_ready),
        .result_vector_data(result_vector_data),
        .job_done_valid(job_done_valid), .job_done_ready(job_done_ready),
        .job_done_tag(job_done_tag),
        .job_done_input_head(job_done_input_head),
        .job_done_error(job_done_error), .protocol_error(protocol_error),
        .state_q(state_q),
        .perf_jobs(perf_jobs),
        .perf_token_requests(perf_token_requests),
        .perf_token_responses(perf_token_responses),
        .perf_weight_requests(perf_weight_requests),
        .perf_weight_responses(perf_weight_responses),
        .perf_results(perf_results),
        .perf_result_jobs(perf_result_jobs)
    );

`default_nettype wire
