`timescale 1ns/1ps
`default_nettype none

// Optional lane-local product-cache front end for the existing TCFM5 backend.
module qfit_cached_tcfm5_projection_top #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int CACHE_WAYS = 4,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W =
        (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,

    input  logic                       run_start,
    output logic                       run_busy,
    output logic                       run_done,

    input  logic                       term_valid,
    output logic                       term_ready,
    input  logic [PLANE_W-1:0]         term_source_plane,
    input  logic [Y_W-1:0]             term_source_y,
    input  logic [X_W-1:0]             term_source_x,
    input  logic [LANE_W-1:0]          term_lane,
    input  logic [GATE_W-1:0]          term_gate,
    input  logic [4:0]                 term_destination_mask,
    input  logic                       term_window_last,
    input  logic                       window_close,
    output logic                       window_close_ready,

    input  logic                       read_valid,
    output logic                       read_ready,
    input  logic [PLANE_W-1:0]         read_plane,
    input  logic [Y_W-1:0]             read_y,
    input  logic [X_W-1:0]             read_x,
    input  logic [OUT_W-1:0]           read_out,
    output logic                       read_data_valid,
    output logic signed [ACC_W-1:0]    read_data,

    output logic                       protocol_error,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates,
    output logic [31:0]                perf_cache_hits,
    output logic [31:0]                perf_cache_misses,
    output logic [31:0]                perf_tag_compares,
    output logic [31:0]                perf_lru_writes,
    output logic [31:0]                perf_product_reads,
    output logic [31:0]                perf_product_writes,
    output logic [31:0]                perf_product_starts,
    output logic [31:0]                perf_weight_reads
);
    logic cache_weight_ready;
    logic backend_weight_ready;
    logic cache_epoch_start_ready;
    logic cache_epoch_close_ready;
    logic cache_epoch_done;
    logic cache_epoch_active;
    logic cache_protocol_error;
    logic backend_protocol_error;
    logic cache_out_valid;
    logic cache_out_ready;
    logic cache_in_ready;
    logic [LANE_W-1:0] cache_out_lane;
    logic [GATE_W-1:0] cache_out_gate;
    logic [PLANE_W-1:0] cache_out_plane;
    logic [Y_W-1:0] cache_out_y;
    logic [X_W-1:0] cache_out_x;
    logic [4:0] cache_out_mask;
    logic cache_out_last;
    logic [OUT_DIM*ACC_W-1:0] cache_out_product;
    logic cache_close_fire;
    logic backend_close_ready;
    logic backend_close_pending_q;
    logic close_inflight_q;
    logic wrapper_protocol_error_q;
    logic [31:0] unused_cache_accepted;
    logic [31:0] unused_cache_output_stalls;

    assign weight_ready = cache_weight_ready && backend_weight_ready;
    assign term_ready = cache_in_ready;
    assign window_close_ready = cache_epoch_close_ready
                              && !close_inflight_q;
    assign cache_close_fire = window_close && window_close_ready;
    assign protocol_error = cache_protocol_error
                         || backend_protocol_error
                         || wrapper_protocol_error_q;

    qfit_lane_product_cache_leaf #(
        .LANES(HEAD_DIM),
        .WAYS(CACHE_WAYS),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .PLANE_W(PLANE_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .DEST_MASK_W(5)
    ) u_product_cache (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(weight_valid && backend_weight_ready),
        .weight_ready(cache_weight_ready),
        .weight_lane(weight_lane), .weight_out(weight_out),
        .weight_data(weight_data), .weight_last(weight_last),
        .epoch_start_valid(run_start),
        .epoch_start_ready(cache_epoch_start_ready),
        .epoch_close_valid(cache_close_fire),
        .epoch_close_ready(cache_epoch_close_ready),
        .epoch_active(cache_epoch_active), .epoch_done(cache_epoch_done),
        .in_valid(term_valid), .in_ready(cache_in_ready),
        .in_lane(term_lane), .in_gate(term_gate),
        .in_source_plane(term_source_plane),
        .in_source_y(term_source_y), .in_source_x(term_source_x),
        .in_destination_mask(term_destination_mask),
        .in_window_last(term_window_last),
        .out_valid(cache_out_valid), .out_ready(cache_out_ready),
        .out_lane(cache_out_lane), .out_gate(cache_out_gate),
        .out_source_plane(cache_out_plane),
        .out_source_y(cache_out_y), .out_source_x(cache_out_x),
        .out_destination_mask(cache_out_mask),
        .out_window_last(cache_out_last),
        .out_product(cache_out_product),
        .protocol_error(cache_protocol_error),
        .perf_accepted_terms(unused_cache_accepted),
        .perf_cache_hits(perf_cache_hits),
        .perf_cache_misses(perf_cache_misses),
        .perf_tag_compares(perf_tag_compares),
        .perf_lru_writes(perf_lru_writes),
        .perf_product_reads(perf_product_reads),
        .perf_product_writes(perf_product_writes),
        .perf_product_starts(perf_product_starts),
        .perf_weight_reads(perf_weight_reads),
        .perf_output_stalls(unused_cache_output_stalls)
    );

    qfit_tcfm5_projection_top #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W), .W_W(W_W), .ACC_W(ACC_W),
        .USE_PRECOMPUTED_PRODUCT(1'b1),
        .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
        .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
    ) u_backend (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(weight_valid && cache_weight_ready),
        .weight_ready(backend_weight_ready),
        .weight_lane(weight_lane), .weight_out(weight_out),
        .weight_data(weight_data), .weight_last(weight_last),
        .weight_context_release(1'b0), .weight_context_release_ready(),
        .run_start(run_start), .run_accumulate(1'b0),
        .run_busy(run_busy), .run_done(run_done),
        .term_valid(cache_out_valid), .term_ready(cache_out_ready),
        .term_source_plane(cache_out_plane),
        .term_source_y(cache_out_y), .term_source_x(cache_out_x),
        .term_lane(cache_out_lane), .term_gate(cache_out_gate),
        .term_destination_mask(cache_out_mask),
        .term_product(cache_out_product),
        .term_window_last(cache_out_last),
        .window_close(backend_close_pending_q),
        .window_close_ready(backend_close_ready),
        .read_valid(read_valid), .read_ready(read_ready),
        .read_plane(read_plane), .read_y(read_y), .read_x(read_x),
        .read_out(read_out), .read_data_valid(read_data_valid),
        .read_data(read_data),
        .vector_read_valid(1'b0), .vector_read_ready(),
        .vector_read_plane('0), .vector_read_y('0),
        .vector_read_x('0), .vector_read_data_valid(),
        .vector_read_data(), .protocol_error(backend_protocol_error),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            backend_close_pending_q <= 1'b0;
            close_inflight_q <= 1'b0;
            wrapper_protocol_error_q <= 1'b0;
        end else begin
            if (run_start) begin
                wrapper_protocol_error_q <= !cache_epoch_start_ready;
                backend_close_pending_q <= 1'b0;
                close_inflight_q <= 1'b0;
            end
            if (cache_close_fire)
                close_inflight_q <= 1'b1;
            if (cache_epoch_done)
                backend_close_pending_q <= 1'b1;
            if (backend_close_pending_q && backend_close_ready) begin
                backend_close_pending_q <= 1'b0;
                close_inflight_q <= 1'b0;
            end
            if (window_close && !window_close_ready)
                wrapper_protocol_error_q <= 1'b1;
        end
    end

    initial begin
        if (CACHE_WAYS < 2)
            $fatal(1, "cached TCFM5 requires CACHE_WAYS>=2");
    end
endmodule

`default_nettype wire
