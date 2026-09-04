`timescale 1ns/1ps
`default_nettype none

// Continuous Local5 deployment slice from raw Q/K candidates through
// score/Shiftmax5, inverse-stencil relation build, and projection.
module qfit_local5_score_active_projection_tile #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int RELATION_READ_LATENCY = 1,
    parameter int RELATION_MEMORY_IMPL = 0,
    parameter int RELATION_SCHED_MODE = 0,
    // 0: topology-colored TCFM5; 1: equal-capacity Linear5 baseline.
    parameter int BACKEND_KIND = 0,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter bit GROUP_EQUAL_GATES = 1'b1,
    parameter int PRODUCT_CACHE_WAYS = 0,
    // 0: sealed residual score leaf; 1: exact Q==0 / ident-K bypass.
    parameter bit ARCH_QSILENT = 1'b0,
    parameter bit ARCH_IDENTK = 1'b1,
    parameter bit ARCH_QSILENT_OVERLAP = 1'b1,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W =
        (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int META_W = PLANE_W + Y_W + X_W
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,
    input  logic                       weight_context_release,
    output logic                       weight_context_release_ready,

    input  logic                       projection_start,
    input  logic                       projection_accumulate,
    input  logic                       projection_close,
    output logic                       projection_close_ready,
    output logic                       projection_busy,
    output logic                       projection_done,

    input  logic                       relation_start,
    input  logic                       relation_seal,
    output logic                       relation_seal_ready,
    output logic                       relation_active,
    output logic                       relation_done,

    input  logic                       row_valid,
    output logic                       row_ready,
    input  logic [PLANE_W-1:0]         row_plane,
    input  logic [Y_W-1:0]             row_destination_y,
    input  logic [X_W-1:0]             row_destination_x,
    input  logic [HEAD_DIM-1:0]        row_q,
    input  logic [5*HEAD_DIM-1:0]      row_candidate_k,
    input  logic [4:0]                 row_candidate_valid,

    input  logic                       read_valid,
    output logic                       read_ready,
    input  logic [PLANE_W-1:0]         read_plane,
    input  logic [Y_W-1:0]             read_y,
    input  logic [X_W-1:0]             read_x,
    input  logic [OUT_W-1:0]           read_out,
    output logic                       read_data_valid,
    output logic signed [ACC_W-1:0]    read_data,

    output logic                       protocol_error,
    output logic [31:0]                perf_score_rows,
    output logic [31:0]                perf_score_service_cycles,
    output logic [31:0]                perf_score_direct_rows,
    output logic [31:0]                perf_relation_writes,
    output logic [31:0]                perf_active_source_reads,
    output logic [31:0]                perf_dense_reads_avoided,
    output logic [31:0]                perf_memory_wait_cycles,
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates,
    output logic [31:0]                perf_qsilent_rows,
    output logic [31:0]                perf_identk_rows,
    output logic [31:0]                perf_overlap_accepts,
    output logic [31:0]                perf_cache_hits,
    output logic [31:0]                perf_cache_misses,
    output logic [31:0]                perf_tag_compares,
    output logic [31:0]                perf_lru_writes,
    output logic [31:0]                perf_product_reads,
    output logic [31:0]                perf_product_writes,
    output logic [31:0]                perf_product_starts,
    output logic [31:0]                perf_weight_reads
);
    logic score_in_ready;
    logic score_out_valid;
    logic score_out_ready;
    logic [META_W-1:0] score_in_tag;
    logic [META_W-1:0] score_out_tag;
    logic [5*SCORE_W-1:0] score_out_q7;
    logic [5*GATE_W-1:0] score_out_gate;
    logic [HEAD_DIM-1:0] score_out_k_self;
    logic [4:0] score_out_valid_mask;
    logic [15:0] score_service_cycles;
    logic [3:0] score_route_direct_mask;
    logic [4:0] row_active_candidate_mask;
    logic [4:0] meta_mask_q [0:1];
    logic meta_wr_q;
    logic meta_rd_q;
    logic [1:0] meta_count_q;
    logic protocol_error_q;
    logic backend_protocol_error;
    logic backend_relation_ready;
    logic backend_relation_seal;
    logic row_fire;
    logic score_out_fire;

    initial begin
        if (HEIGHT != 15 || WIDTH != 15 || TIME_PLANES != 2)
            $error("Local5 score/projection tile currently requires T450");
        if (HEAD_DIM != 32)
            $error("Local5 score/projection tile requires HEAD_DIM=32");
        if (BACKEND_KIND < 0 || BACKEND_KIND > 1)
            $error("unsupported Local5 projection backend");
    end

    always_comb begin
        score_in_tag = {
            row_plane,
            row_destination_y,
            row_destination_x
        };
        for (int candidate = 0; candidate < 5; candidate = candidate + 1)
            row_active_candidate_mask[candidate] =
                row_candidate_valid[candidate]
                && (row_candidate_k[candidate*HEAD_DIM +: HEAD_DIM] != '0);
    end

    logic [31:0] score_qsilent_rows;
    logic [31:0] score_identk_rows;
    assign row_ready = relation_active
                     && !relation_seal
                     && (meta_count_q < 2'd2)
                     && score_in_ready;
    assign row_fire = row_valid && row_ready;
    assign score_out_ready = (meta_count_q != 2'd0)
                           && backend_relation_ready;
    assign score_out_fire = score_out_valid && score_out_ready;
    assign relation_seal_ready = relation_active
                               && (meta_count_q == 2'd0)
                               && score_in_ready
                               && !score_out_valid
                               && !row_valid;
    assign backend_relation_seal = relation_seal && relation_seal_ready;
    assign protocol_error = protocol_error_q || backend_protocol_error;

    qfit_local5_qsilent_score_leaf #(
        .ENABLE_QSILENT(ARCH_QSILENT),
        .ENABLE_IDENTK(ARCH_IDENTK),
        .ENABLE_OVERLAP(ARCH_QSILENT_OVERLAP),
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2),
        .TAG_W(META_W),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W)
    ) u_score (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(row_valid && row_ready),
        .in_ready(score_in_ready),
        .in_tag(score_in_tag),
        .in_q(row_q),
        .in_k(row_candidate_k),
        .in_valid_mask(row_candidate_valid),
        .out_valid(score_out_valid),
        .out_ready(score_out_ready),
        .out_tag(score_out_tag),
        .out_score_q7(score_out_q7),
        .out_gate_q17(score_out_gate),
        .out_k_self(score_out_k_self),
        .out_valid_mask(score_out_valid_mask),
        .perf_service_cycles(score_service_cycles),
        .perf_route_direct_mask(score_route_direct_mask),
        .perf_qsilent_rows(score_qsilent_rows),
        .perf_identk_rows(score_identk_rows),
        .perf_overlap_accepts(perf_overlap_accepts)
    );

    assign perf_qsilent_rows = score_qsilent_rows;
    assign perf_identk_rows = score_identk_rows;

    qfit_local5_active_projection_tile #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .RELATION_READ_LATENCY(RELATION_READ_LATENCY),
        .RELATION_MEMORY_IMPL(RELATION_MEMORY_IMPL),
        .RELATION_SCHED_MODE(RELATION_SCHED_MODE),
        .BACKEND_KIND(BACKEND_KIND),
        .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
        .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL),
        .GROUP_EQUAL_GATES(GROUP_EQUAL_GATES),
        .PRODUCT_CACHE_WAYS(PRODUCT_CACHE_WAYS)
    ) u_projection (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .weight_valid(weight_valid),
        .weight_ready(weight_ready),
        .weight_lane(weight_lane),
        .weight_out(weight_out),
        .weight_data(weight_data),
        .weight_last(weight_last),
        .weight_context_release(weight_context_release),
        .weight_context_release_ready(weight_context_release_ready),
        .projection_start(projection_start),
        .projection_accumulate(projection_accumulate),
        .projection_close(projection_close),
        .projection_close_ready(projection_close_ready),
        .projection_busy(projection_busy),
        .projection_done(projection_done),
        .relation_start(relation_start),
        .relation_seal(backend_relation_seal),
        .relation_active(relation_active),
        .relation_done(relation_done),
        .relation_valid(score_out_valid && (meta_count_q != 2'd0)),
        .relation_ready(backend_relation_ready),
        .relation_plane(score_out_tag[META_W-1 -: PLANE_W]),
        .relation_destination_y(
            score_out_tag[X_W +: Y_W]
        ),
        .relation_destination_x(score_out_tag[X_W-1:0]),
        .relation_candidate_valid(score_out_valid_mask),
        .relation_active_candidate_mask(meta_mask_q[meta_rd_q]),
        .relation_k_self(score_out_k_self),
        .relation_direction_gates(score_out_gate),
        .read_valid(read_valid),
        .read_ready(read_ready),
        .read_plane(read_plane),
        .read_y(read_y),
        .read_x(read_x),
        .read_out(read_out),
        .read_data_valid(read_data_valid),
        .read_data(read_data),
        .protocol_error(backend_protocol_error),
        .perf_relation_writes(perf_relation_writes),
        .perf_active_source_reads(perf_active_source_reads),
        .perf_dense_reads_avoided(perf_dense_reads_avoided),
        .perf_memory_wait_cycles(perf_memory_wait_cycles),
        .perf_descriptors(perf_descriptors),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates),
        .perf_cache_hits(perf_cache_hits),
        .perf_cache_misses(perf_cache_misses),
        .perf_tag_compares(perf_tag_compares),
        .perf_lru_writes(perf_lru_writes),
        .perf_product_reads(perf_product_reads),
        .perf_product_writes(perf_product_writes),
        .perf_product_starts(perf_product_starts),
        .perf_weight_reads(perf_weight_reads)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            meta_mask_q[0] <= '0;
            meta_mask_q[1] <= '0;
            meta_wr_q <= 1'b0;
            meta_rd_q <= 1'b0;
            meta_count_q <= 2'd0;
            protocol_error_q <= 1'b0;
            perf_score_rows <= '0;
            perf_score_service_cycles <= '0;
            perf_score_direct_rows <= '0;
        end else begin
            if (projection_start) begin
                protocol_error_q <= 1'b0;
                perf_score_rows <= '0;
                perf_score_service_cycles <= '0;
                perf_score_direct_rows <= '0;
            end
            if (row_fire) begin
                if (meta_count_q == 2'd2)
                    protocol_error_q <= 1'b1;
                meta_mask_q[meta_wr_q] <= row_active_candidate_mask;
                meta_wr_q <= meta_wr_q + 1'b1;
            end
            if (score_out_valid && (meta_count_q == 2'd0))
                protocol_error_q <= 1'b1;
            if (score_out_fire) begin
                meta_rd_q <= meta_rd_q + 1'b1;
                perf_score_rows <= perf_score_rows + 1'b1;
                perf_score_service_cycles <=
                    perf_score_service_cycles + 32'(score_service_cycles);
                if (score_route_direct_mask != 0)
                    perf_score_direct_rows <= perf_score_direct_rows + 1'b1;
            end
            unique case ({row_fire, score_out_fire})
                2'b10: meta_count_q <= meta_count_q + 2'd1;
                2'b01: meta_count_q <= meta_count_q - 2'd1;
                default: ;
            endcase
            if (relation_seal && !relation_seal_ready)
                protocol_error_q <= 1'b1;
        end
    end

    logic [5*SCORE_W-1:0] unused_score_q7;
    assign unused_score_q7 = score_out_q7;
endmodule

`default_nettype wire
