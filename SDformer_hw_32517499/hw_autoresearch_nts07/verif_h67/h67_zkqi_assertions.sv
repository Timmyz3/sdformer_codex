`timescale 1ns/1ps
`default_nettype none

module h67_ttb8_metadata_builder_assertions #(
    parameter int PAIRS = 225,
    parameter int BUNDLE_SIZE = 8,
    parameter int BUNDLE_COUNT = (PAIRS + BUNDLE_SIZE - 1) / BUNDLE_SIZE,
    parameter int BUNDLE_ID_W = (BUNDLE_COUNT <= 1) ? 1 : $clog2(BUNDLE_COUNT),
    parameter int COUNT_W = $clog2(2 * PAIRS + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic row_loaded,
    input logic scan_valid,
    input logic scan_ready,
    input logic [BUNDLE_ID_W-1:0] scan_bundle_id,
    input logic [BUNDLE_SIZE-1:0] scan_active_mask,
    input logic [COUNT_W-1:0] zk_count0,
    input logic [COUNT_W-1:0] zk_count1,
    input logic [COUNT_W-1:0] zk_count2,
    input logic [31:0] perf_active_pairs
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        row_loaded |->
        (32'(zk_count0) + 32'(zk_count1) + 32'(zk_count2)
         + 2 * perf_active_pairs == 2 * PAIRS));

    assert property (@(posedge clk_core) disable iff (rst_core)
        scan_valid && !scan_ready |=>
        scan_valid && $stable(scan_bundle_id) && $stable(scan_active_mask));
endmodule

module h67_active_bundle_fifo_assertions #(
    parameter int PAIRS = 225,
    parameter int BUNDLE_SIZE = 8,
    parameter int DEPTH = 32,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int OCC_W = $clog2(DEPTH + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic pair_valid,
    input logic pair_ready,
    input logic [PAIR_ID_W-1:0] pair_id,
    input logic [OCC_W-1:0] occupancy,
    input logic [OCC_W-1:0] max_occupancy
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        occupancy <= DEPTH && max_occupancy <= DEPTH);

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        pair_valid |-> 32'(pair_id) < 32'(PAIRS));

    assert property (@(posedge clk_core) disable iff (rst_core || window_start)
        pair_valid && !pair_ready |=> pair_valid && $stable(pair_id));
endmodule

module h67_pair_bitmap_metadata_builder_assertions #(
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int COUNT_W = $clog2(2 * PAIRS + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic row_loaded,
    input logic pair_valid,
    input logic pair_ready,
    input logic [PAIR_ID_W-1:0] pair_id,
    input logic scan_done,
    input logic [COUNT_W-1:0] zk_count0,
    input logic [COUNT_W-1:0] zk_count1,
    input logic [COUNT_W-1:0] zk_count2,
    input logic [31:0] perf_active_pairs
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        row_loaded |->
        (32'(zk_count0) + 32'(zk_count1) + 32'(zk_count2)
         + 2 * perf_active_pairs == 2 * PAIRS));

    assert property (@(posedge clk_core) disable iff (rst_core)
        pair_valid && !pair_ready |=> pair_valid && $stable(pair_id));

    assert property (@(posedge clk_core) disable iff (rst_core)
        pair_valid |-> 32'(pair_id) < 32'(PAIRS));

    assert property (@(posedge clk_core) disable iff (rst_core)
        scan_done |-> !pair_valid);
endmodule

module h67_zkqi_row_top_assertions #(
    parameter int PAIRS = 225,
    parameter int HEAD_DIM = 32,
    parameter int TOKEN_W = $clog2(2 * PAIRS + 1),
    parameter int FIFO_OCC_W = 1
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic out_last,
    input logic [TOKEN_W-1:0] out_token_id,
    input logic [HEAD_DIM-1:0] out_k_bits,
    input logic [8:0] out_gate_q17,
    input logic window_done,
    input logic [31:0] perf_original_tokens,
    input logic [FIFO_OCC_W-1:0] perf_fifo_occupancy,
    input logic [FIFO_OCC_W-1:0] perf_fifo_max_occupancy
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready |=>
        out_valid && $stable(out_last) && $stable(out_token_id)
        && $stable(out_k_bits) && $stable(out_gate_q17));

    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_k_bits != 0 && 32'(out_token_id) < 32'(2 * PAIRS));

    assert property (@(posedge clk_core) disable iff (rst_core)
        window_done |-> perf_original_tokens == 32'(2 * PAIRS));

    assert property (@(posedge clk_core) disable iff (rst_core)
        perf_fifo_occupancy <= perf_fifo_max_occupancy);
endmodule

bind h67_ttb8_metadata_builder h67_ttb8_metadata_builder_assertions #(
    .PAIRS(PAIRS),
    .BUNDLE_SIZE(BUNDLE_SIZE),
    .BUNDLE_COUNT(BUNDLE_COUNT),
    .BUNDLE_ID_W(BUNDLE_ID_W),
    .COUNT_W(COUNT_W)
) u_h67_ttb8_metadata_builder_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .row_loaded(row_loaded),
    .scan_valid(scan_valid),
    .scan_ready(scan_ready),
    .scan_bundle_id(scan_bundle_id),
    .scan_active_mask(scan_active_mask),
    .zk_count0(zk_count0),
    .zk_count1(zk_count1),
    .zk_count2(zk_count2),
    .perf_active_pairs(perf_active_pairs)
);

bind h67_active_bundle_fifo h67_active_bundle_fifo_assertions #(
    .PAIRS(PAIRS),
    .BUNDLE_SIZE(BUNDLE_SIZE),
    .DEPTH(DEPTH),
    .PAIR_ID_W(PAIR_ID_W),
    .OCC_W(OCC_W)
) u_h67_active_bundle_fifo_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .window_start(window_start),
    .pair_valid(pair_valid),
    .pair_ready(pair_ready),
    .pair_id(pair_id),
    .occupancy(occupancy),
    .max_occupancy(max_occupancy)
);

bind h67_pair_bitmap_metadata_builder
    h67_pair_bitmap_metadata_builder_assertions #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .COUNT_W(COUNT_W)
    ) u_h67_pair_bitmap_metadata_builder_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_loaded(row_loaded),
        .pair_valid(pair_valid),
        .pair_ready(pair_ready),
        .pair_id(pair_id),
        .scan_done(scan_done),
        .zk_count0(zk_count0),
        .zk_count1(zk_count1),
        .zk_count2(zk_count2),
        .perf_active_pairs(perf_active_pairs)
    );

bind h67_zkqi_row_shiftmax_top h67_zkqi_row_top_assertions #(
    .PAIRS(PAIRS),
    .HEAD_DIM(HEAD_DIM),
    .TOKEN_W(TOKEN_W),
    .FIFO_OCC_W(FIFO_OCC_W)
) u_h67_zkqi_row_top_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_last(out_last),
    .out_token_id(out_token_id),
    .out_k_bits(out_k_bits),
    .out_gate_q17(out_gate_q17),
    .window_done(window_done),
    .perf_original_tokens(perf_original_tokens),
    .perf_fifo_occupancy(perf_fifo_occupancy),
    .perf_fifo_max_occupancy(perf_fifo_max_occupancy)
);

`default_nettype wire
