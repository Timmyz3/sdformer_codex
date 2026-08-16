`timescale 1ns/1ps
`default_nettype none

module qfit_lane_product_cache_assertions #(
    parameter int PRODUCT_W = 64,
    parameter int WAYS = 4,
    parameter bit NO_REPLACE = 1'b0,
    parameter int WAY_W = (WAYS <= 1) ? 1 : $clog2(WAYS),
    parameter int LANE_W = 2,
    parameter int GATE_W = 9,
    parameter int PLANE_W = 1,
    parameter int Y_W = 3,
    parameter int X_W = 3,
    parameter int DEST_MASK_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic epoch_done,
    input logic out_valid,
    input logic out_ready,
    input logic [LANE_W-1:0] out_lane,
    input logic [GATE_W-1:0] out_gate,
    input logic [PLANE_W-1:0] out_source_plane,
    input logic [Y_W-1:0] out_source_y,
    input logic [X_W-1:0] out_source_x,
    input logic [DEST_MASK_W-1:0] out_destination_mask,
    input logic out_window_last,
    input logic [PRODUCT_W-1:0] out_product,
    input logic out_hit_q,
    input logic [WAY_W-1:0] out_hit_way_q,
    input logic [WAYS-1:0] product_bank_read_valid,
    input logic [WAYS-1:0] product_bank_access,
    input logic [WAYS-1:0] product_bank_write,
    input logic in_fire,
    input logic in_ready,
    input logic input_contract_valid,
    input logic lookup_hit,
    input logic cache_insert,
    input logic closing_q,
    input logic [31:0] perf_accepted_terms,
    input logic [31:0] perf_cache_hits,
    input logic [31:0] perf_cache_misses,
    input logic [31:0] perf_product_reads,
    input logic [31:0] perf_product_writes,
    input logic [31:0] perf_product_starts
);
    property p_output_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
            out_valid && !out_ready
            |=> out_valid
                && $stable({
                    out_lane,
                    out_gate,
                    out_source_plane,
                    out_source_y,
                    out_source_x,
                    out_destination_mask,
                    out_window_last,
                    out_product
                });
    endproperty

    property p_hit_miss_partition;
        @(posedge clk_core) disable iff (rst_core)
            perf_cache_hits + perf_cache_misses
                == perf_accepted_terms;
    endproperty

    property p_done_has_no_output;
        @(posedge clk_core) disable iff (rst_core)
            epoch_done |-> !out_valid;
    endproperty

    property p_hit_uses_completed_sync_read;
        @(posedge clk_core) disable iff (rst_core)
            out_valid
            && out_hit_q
            && !$past(out_valid && !out_ready)
            |-> product_bank_read_valid[out_hit_way_q];
    endproperty

    property p_hit_activity_partition;
        @(posedge clk_core) disable iff (rst_core)
            in_fire && input_contract_valid && lookup_hit
            |=> perf_product_reads == $past(perf_product_reads) + 1'b1
                && perf_product_writes == $past(perf_product_writes)
                && perf_product_starts == $past(perf_product_starts);
    endproperty

    property p_miss_activity_partition;
        @(posedge clk_core) disable iff (rst_core)
            in_fire && input_contract_valid && !lookup_hit
            |=> perf_product_reads == $past(perf_product_reads)
                && perf_product_writes
                    == $past(perf_product_writes)
                        + 32'($past(cache_insert))
                && perf_product_starts == $past(perf_product_starts) + 1'b1;
    endproperty

    property p_hit_reads_exactly_one_bank;
        @(posedge clk_core) disable iff (rst_core)
            in_fire && input_contract_valid && lookup_hit
            |-> $onehot(product_bank_access) && product_bank_write == '0;
    endproperty

    property p_miss_writes_exactly_one_bank;
        @(posedge clk_core) disable iff (rst_core)
            in_fire && input_contract_valid && !lookup_hit
            |-> cache_insert
                ? ($onehot(product_bank_write)
                    && product_bank_access == product_bank_write)
                : (NO_REPLACE
                    && product_bank_write == '0
                    && product_bank_access == '0);
    endproperty

    property p_valid_accept_produces_output;
        @(posedge clk_core) disable iff (rst_core)
            in_fire && input_contract_valid |=> out_valid;
    endproperty

    property p_close_stops_input;
        @(posedge clk_core) disable iff (rst_core)
            closing_q |-> !in_ready;
    endproperty

    assert property (p_output_stable_under_stall);
    assert property (p_hit_miss_partition);
    assert property (p_done_has_no_output);
    assert property (p_hit_uses_completed_sync_read);
    assert property (p_hit_activity_partition);
    assert property (p_miss_activity_partition);
    assert property (p_hit_reads_exactly_one_bank);
    assert property (p_miss_writes_exactly_one_bank);
    assert property (p_valid_accept_produces_output);
    assert property (p_close_stops_input);
endmodule

bind qfit_lane_product_cache_leaf qfit_lane_product_cache_assertions #(
    .PRODUCT_W(OUT_DIM * ACC_W),
    .WAYS(WAYS),
    .NO_REPLACE(NO_REPLACE),
    .WAY_W((WAYS <= 1) ? 1 : $clog2(WAYS)),
    .LANE_W(LANE_W),
    .GATE_W(GATE_W),
    .PLANE_W(PLANE_W),
    .Y_W(Y_W),
    .X_W(X_W),
    .DEST_MASK_W(DEST_MASK_W)
) u_qfit_lane_product_cache_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .epoch_done(epoch_done),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_lane(out_lane),
    .out_gate(out_gate),
    .out_source_plane(out_source_plane),
    .out_source_y(out_source_y),
    .out_source_x(out_source_x),
    .out_destination_mask(out_destination_mask),
    .out_window_last(out_window_last),
    .out_product(out_product),
    .out_hit_q(out_hit_q),
    .out_hit_way_q(out_hit_way_q),
    .product_bank_read_valid(product_bank_read_valid),
    .product_bank_access(product_bank_access),
    .product_bank_write(product_bank_write),
    .in_fire(in_fire),
    .in_ready(in_ready),
    .input_contract_valid(input_contract_valid),
    .lookup_hit(lookup_hit),
    .cache_insert(cache_insert),
    .closing_q(closing_q),
    .perf_accepted_terms(perf_accepted_terms),
    .perf_cache_hits(perf_cache_hits),
    .perf_cache_misses(perf_cache_misses),
    .perf_product_reads(perf_product_reads),
    .perf_product_writes(perf_product_writes),
    .perf_product_starts(perf_product_starts)
);

`default_nettype wire
