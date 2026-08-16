`timescale 1ns/1ps
`default_nettype none

// Q/K预装时同步构造TTB active mask，并把both-K-zero精确折叠到0/1/2三类。
module h67_ttb8_metadata_builder #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int BUNDLE_SIZE = 8,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int BUNDLE_COUNT = (PAIRS + BUNDLE_SIZE - 1) / BUNDLE_SIZE,
    parameter int BUNDLE_ID_W = (BUNDLE_COUNT <= 1) ? 1 : $clog2(BUNDLE_COUNT),
    parameter int COUNT_W = $clog2(2 * PAIRS + 1),
    parameter int LANE_W = (BUNDLE_SIZE <= 1) ? 1 : $clog2(BUNDLE_SIZE)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       row_load_start,
    input  logic                       window_start,

    input  logic                       load_valid,
    output logic                       load_ready,
    input  logic [PAIR_ID_W-1:0]       load_pair_id,
    input  logic [2*HEAD_DIM-1:0]      load_q_pair,
    input  logic [2*HEAD_DIM-1:0]      load_k_pair,
    output logic                       row_loaded,

    output logic                       scan_valid,
    input  logic                       scan_ready,
    output logic [BUNDLE_ID_W-1:0]     scan_bundle_id,
    output logic [BUNDLE_SIZE-1:0]     scan_active_mask,
    output logic                       scan_last,
    output logic                       scan_done,

    output logic [COUNT_W-1:0]         zk_count0,
    output logic [COUNT_W-1:0]         zk_count1,
    output logic [COUNT_W-1:0]         zk_count2,
    output logic [31:0]                perf_preclassified_pairs,
    output logic [31:0]                perf_active_pairs,
    output logic [31:0]                perf_metadata_bits,
    output logic                       protocol_error
);
    localparam int NEXT_W = $clog2(PAIRS + 1);

    logic [BUNDLE_SIZE-1:0] active_mask_mem [0:BUNDLE_COUNT-1];
    logic [BUNDLE_SIZE-1:0] build_mask_q;
    logic [NEXT_W-1:0] next_pair_q;
    logic [BUNDLE_ID_W-1:0] scan_index_q;
    logic scan_active_q;
    logic row_loaded_q;
    logic scan_done_q;
    logic protocol_error_q;
    logic [COUNT_W-1:0] zk_count0_q;
    logic [COUNT_W-1:0] zk_count1_q;
    logic [COUNT_W-1:0] zk_count2_q;
    logic [COUNT_W+1:0] seed_token_total;

    logic both_k_zero;
    logic [5:0] qcount0;
    logic [5:0] qcount1;
    logic [1:0] zk_class0;
    logic [1:0] zk_class1;
    logic [LANE_W-1:0] load_lane;
    logic [BUNDLE_ID_W-1:0] load_bundle;
    logic load_last_lane;
    logic [BUNDLE_SIZE-1:0] mask_with_current;
    logic load_id_legal;
    logic load_fire;

    function automatic [5:0] popcount32(input logic [HEAD_DIM-1:0] value);
        integer lane;
        begin
            popcount32 = '0;
            for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
                popcount32 = popcount32 + 6'(value[lane]);
        end
    endfunction

    function automatic [1:0] zero_k_class(input logic [5:0] qcount);
        begin
            if (qcount <= 8)
                zero_k_class = 2;
            else if (qcount <= 23)
                zero_k_class = 1;
            else
                zero_k_class = 0;
        end
    endfunction

    assign both_k_zero = load_k_pair == 0;
    assign qcount0 = popcount32(load_q_pair[HEAD_DIM-1:0]);
    assign qcount1 = popcount32(load_q_pair[2*HEAD_DIM-1:HEAD_DIM]);
    assign zk_class0 = zero_k_class(qcount0);
    assign zk_class1 = zero_k_class(qcount1);
    assign load_lane = LANE_W'(32'(load_pair_id) % BUNDLE_SIZE);
    assign load_bundle = BUNDLE_ID_W'(32'(load_pair_id) / BUNDLE_SIZE);
    assign load_last_lane = load_lane == LANE_W'(BUNDLE_SIZE - 1)
                         || 32'(load_pair_id) == 32'(PAIRS - 1);
    assign mask_with_current = both_k_zero
        ? build_mask_q
        : build_mask_q | (BUNDLE_SIZE'(1) << load_lane);
    assign load_id_legal = 32'(next_pair_q) < 32'(PAIRS)
                        && 32'(load_pair_id) == 32'(next_pair_q);
    assign load_ready = !row_loaded_q && !scan_active_q;
    assign load_fire = load_valid && load_ready;

    assign row_loaded = row_loaded_q;
    assign scan_valid = scan_active_q;
    assign scan_bundle_id = scan_index_q;
    assign scan_active_mask = active_mask_mem[scan_index_q];
    assign scan_last = scan_valid && 32'(scan_index_q) == 32'(BUNDLE_COUNT - 1);
    assign scan_done = scan_done_q;
    assign zk_count0 = zk_count0_q;
    assign zk_count1 = zk_count1_q;
    assign zk_count2 = zk_count2_q;
    assign seed_token_total = (COUNT_W+2)'(zk_count0_q)
                            + (COUNT_W+2)'(zk_count1_q)
                            + (COUNT_W+2)'(zk_count2_q);
    assign perf_preclassified_pairs = 32'(next_pair_q);
    assign perf_active_pairs = 32'(PAIRS) - 32'(seed_token_total >> 1);
    assign perf_metadata_bits = row_loaded_q
                              ? 32'(BUNDLE_COUNT * BUNDLE_SIZE) : 0;
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            build_mask_q <= '0;
            next_pair_q <= '0;
            scan_index_q <= '0;
            scan_active_q <= 1'b0;
            row_loaded_q <= 1'b0;
            scan_done_q <= 1'b0;
            protocol_error_q <= 1'b0;
            zk_count0_q <= '0;
            zk_count1_q <= '0;
            zk_count2_q <= '0;
        end else begin
            scan_done_q <= 1'b0;

            if (row_load_start) begin
                if (scan_active_q)
                    protocol_error_q <= 1'b1;
                else begin
                    build_mask_q <= '0;
                    next_pair_q <= '0;
                    row_loaded_q <= 1'b0;
                    protocol_error_q <= 1'b0;
                    zk_count0_q <= '0;
                    zk_count1_q <= '0;
                    zk_count2_q <= '0;
                end
            end

            if (load_fire) begin
                if (!load_id_legal) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    next_pair_q <= next_pair_q + 1'b1;
                    if (both_k_zero) begin
                        case (zk_class0)
                            0: zk_count0_q <= zk_count0_q + 1'b1;
                            1: zk_count1_q <= zk_count1_q + 1'b1;
                            default: zk_count2_q <= zk_count2_q + 1'b1;
                        endcase
                        case (zk_class1)
                            0: zk_count0_q <= zk_count0_q
                                + COUNT_W'(zk_class0 == 0) + 1'b1;
                            1: zk_count1_q <= zk_count1_q
                                + COUNT_W'(zk_class0 == 1) + 1'b1;
                            default: zk_count2_q <= zk_count2_q
                                + COUNT_W'(zk_class0 == 2) + 1'b1;
                        endcase
                    end

                    if (load_last_lane) begin
                        active_mask_mem[load_bundle] <= mask_with_current;
                        build_mask_q <= '0;
                    end else begin
                        build_mask_q <= mask_with_current;
                    end
                    if (32'(load_pair_id) == 32'(PAIRS - 1))
                        row_loaded_q <= 1'b1;
                end
            end else if (load_valid && !load_ready) begin
                protocol_error_q <= 1'b1;
            end

            if (window_start) begin
                if (!row_loaded_q || scan_active_q) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    scan_index_q <= '0;
                    scan_active_q <= 1'b1;
                end
            end else if (scan_valid && scan_ready) begin
                if (scan_last) begin
                    scan_active_q <= 1'b0;
                    scan_done_q <= 1'b1;
                end else begin
                    scan_index_q <= scan_index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
