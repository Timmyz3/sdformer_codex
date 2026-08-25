`timescale 1ns/1ps
`default_nettype none

// Standalone lossless K4 source-fold plus same-address-forwarding accumulator.
//
// M125 converts one row mask into canonical groups of at most four signed-INT8
// source vectors.  M123 consumes the resulting signed19 vectors at II=1 and
// forwards the just-computed value across consecutive updates to the same
// (block,row), avoiding undefined macro read-during-write behavior.  This
// wrapper also isolates every externally visible handshake and lane-memory
// enable while synchronous reset is asserted.
module m126_block_phased_k4_forwarding_accumulator_island (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         window_start_valid,
    output logic                         window_start_ready,
    output logic                         window_start_accept,

    input  logic                         weight_fill_valid,
    output logic                         weight_fill_ready,
    input  logic [2:0]                   weight_fill_block,
    input  logic [3:0]                   weight_fill_source,
    input  logic [1:0]                   weight_fill_beat,
    input  logic [255:0]                 weight_fill_data,
    output logic                         weight_fill_accept,

    input  logic                         row_valid,
    output logic                         row_ready,
    input  logic [2:0]                   row_block,
    input  logic [8:0]                   row_offset,
    input  logic [15:0]                  row_source_mask,
    input  logic [15:0]                  row_negate_mask,
    output logic                         row_accept,
    output logic                         row_done,

    input  logic                         window_end_valid,
    output logic                         window_end_ready,
    output logic                         window_end_accept,

    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [2:0]                   commit_block,
    output logic [8:0]                   commit_row,
    output logic [1823:0]                commit_data,
    output logic                         commit_last,
    output logic                         window_done,

    output logic                         lane_mem_rd_en,
    output logic [11:0]                  lane_mem_rd_addr,
    input  logic [18:0]                  lane_mem_rd_data [0:95],
    output logic                         lane_mem_wr_en,
    output logic [11:0]                  lane_mem_wr_addr,
    output logic [18:0]                  lane_mem_wr_data [0:95],

    output logic                         observed_fold_update_accept,
    output logic                         observed_accumulator_update_accept,
    output logic [2:0]                   observed_fold_update_block,
    output logic [8:0]                   observed_fold_update_row,
    output logic [1823:0]                observed_fold_update_delta,
    output logic [15:0]                  observed_fold_selected_mask,
    output logic [15:0]                  observed_fold_remaining_mask,
    output logic [15:0]                  observed_cache_valid,
    output logic [2:0]                   observed_resident_block,
    output logic                         observed_resident_block_valid,
    output logic                         fold_protocol_error,
    output logic                         accumulator_protocol_error,
    output logic                         protocol_error,
    output logic                         window_active,
    output logic                         busy
);
    logic wrapper_fault_q;
    logic wrapper_illegal_request;

    logic fold_weight_fill_valid;
    logic fold_weight_fill_ready;
    logic fold_weight_fill_accept;
    logic fold_row_valid;
    logic fold_row_ready;
    logic fold_row_accept;
    logic fold_update_valid;
    logic fold_update_ready;
    logic [2:0] fold_update_block;
    logic [8:0] fold_update_row;
    logic [1823:0] fold_update_delta;
    logic [15:0] fold_update_selected_mask;
    logic fold_update_accept;
    logic fold_row_done;
    logic fold_busy;

    logic accumulator_start_valid;
    logic accumulator_start_ready;
    logic accumulator_start_accept;
    logic accumulator_update_ready;
    logic accumulator_update_accept;
    logic accumulator_end_valid;
    logic accumulator_end_ready;
    logic accumulator_end_accept;
    logic accumulator_commit_valid;
    logic accumulator_window_done;
    logic accumulator_window_active;
    logic accumulator_busy;
    logic internal_lane_mem_rd_en;
    logic [11:0] internal_lane_mem_rd_addr;
    logic internal_lane_mem_wr_en;
    logic [11:0] internal_lane_mem_wr_addr;
    logic [18:0] internal_lane_mem_wr_data [0:95];

    always_comb begin : wrapper_request_audit
        wrapper_illegal_request = 1'b0;
        if ((window_start_valid && (weight_fill_valid || row_valid
                                    || window_end_valid))
                || (weight_fill_valid && (row_valid || window_end_valid))
                || (row_valid && window_end_valid))
            wrapper_illegal_request = 1'b1;
        if (window_start_valid && (accumulator_window_active || fold_busy))
            wrapper_illegal_request = 1'b1;
        if ((weight_fill_valid || row_valid) && !accumulator_window_active)
            wrapper_illegal_request = 1'b1;
        if (window_end_valid
                && (!accumulator_window_active || fold_busy))
            wrapper_illegal_request = 1'b1;
    end

    assign protocol_error = !rst_core
                          && (wrapper_fault_q || wrapper_illegal_request
                              || fold_protocol_error
                              || accumulator_protocol_error);

    assign accumulator_start_valid = window_start_valid && !rst_core
                                   && !wrapper_fault_q
                                   && !wrapper_illegal_request
                                   && !fold_protocol_error;
    assign window_start_ready = !rst_core && !protocol_error
                              && accumulator_start_ready;
    assign window_start_accept = !rst_core && accumulator_start_accept;

    assign fold_weight_fill_valid = weight_fill_valid && !rst_core
                                  && accumulator_window_active
                                  && !wrapper_fault_q
                                  && !wrapper_illegal_request
                                  && !accumulator_protocol_error;
    assign weight_fill_ready = !rst_core && !protocol_error
                             && accumulator_window_active
                             && fold_weight_fill_ready;
    assign weight_fill_accept = !rst_core && fold_weight_fill_accept;

    assign fold_row_valid = row_valid && !rst_core
                          && accumulator_window_active
                          && !wrapper_fault_q
                          && !wrapper_illegal_request
                          && !accumulator_protocol_error;
    assign row_ready = !rst_core && !protocol_error
                     && accumulator_window_active && fold_row_ready;
    assign row_accept = !rst_core && fold_row_accept;
    assign row_done = !rst_core && fold_row_done;

    assign fold_update_ready = !rst_core && !wrapper_fault_q
                             && !fold_protocol_error
                             && !accumulator_protocol_error
                             && accumulator_update_ready;
    assign observed_fold_update_accept = !rst_core && fold_update_accept;
    assign observed_accumulator_update_accept = !rst_core
                                               && accumulator_update_accept;
    assign observed_fold_update_block = fold_update_block;
    assign observed_fold_update_row = fold_update_row;
    assign observed_fold_update_delta = fold_update_delta;
    assign observed_fold_selected_mask = fold_update_selected_mask;

    assign accumulator_end_valid = window_end_valid && !rst_core
                                 && !fold_busy && !wrapper_fault_q
                                 && !wrapper_illegal_request
                                 && !fold_protocol_error;
    assign window_end_ready = !rst_core && !protocol_error && !fold_busy
                            && accumulator_end_ready;
    assign window_end_accept = !rst_core && accumulator_end_accept;

    assign commit_valid = !rst_core && accumulator_commit_valid;
    assign window_done = !rst_core && accumulator_window_done;
    assign window_active = !rst_core && accumulator_window_active;
    assign busy = !rst_core && (fold_busy || accumulator_busy);

    // Reset isolation closes the M123 review's phantom-accept and reset-edge
    // physical-write counterexample without changing its reset-free datapath.
    assign lane_mem_rd_en = !rst_core && internal_lane_mem_rd_en;
    assign lane_mem_rd_addr = internal_lane_mem_rd_addr;
    assign lane_mem_wr_en = !rst_core && internal_lane_mem_wr_en;
    assign lane_mem_wr_addr = internal_lane_mem_wr_addr;
    always_comb begin : reset_isolated_write_data
        for (int lane = 0; lane < 96; lane++)
            lane_mem_wr_data[lane] = internal_lane_mem_wr_data[lane];
    end

    always_ff @(posedge clk_core) begin
        if (rst_core)
            wrapper_fault_q <= 1'b0;
        else if (wrapper_illegal_request)
            wrapper_fault_q <= 1'b1;
    end

    m125_block_phased_k4_row_fold fold (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .weight_fill_valid(fold_weight_fill_valid),
        .weight_fill_ready(fold_weight_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(fold_weight_fill_accept),
        .row_valid(fold_row_valid),
        .row_ready(fold_row_ready),
        .row_block(row_block),
        .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(fold_row_accept),
        .update_valid(fold_update_valid),
        .update_ready(fold_update_ready),
        .update_block(fold_update_block),
        .update_row(fold_update_row),
        .update_delta(fold_update_delta),
        .update_selected_mask(fold_update_selected_mask),
        .update_accept(fold_update_accept),
        .row_done(fold_row_done),
        .observed_remaining_mask(observed_fold_remaining_mask),
        .observed_cache_valid(observed_cache_valid),
        .observed_resident_block_valid(observed_resident_block_valid),
        .observed_resident_block(observed_resident_block),
        .protocol_error(fold_protocol_error),
        .busy(fold_busy)
    );

    m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter accumulator (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(accumulator_start_valid),
        .window_start_ready(accumulator_start_ready),
        .window_start_accept(accumulator_start_accept),
        .update_valid(fold_update_valid && !rst_core
                      && !wrapper_fault_q && !fold_protocol_error),
        .update_ready(accumulator_update_ready),
        .update_block(fold_update_block),
        .update_row(fold_update_row),
        .update_delta(fold_update_delta),
        .update_accept(accumulator_update_accept),
        .window_end_valid(accumulator_end_valid),
        .window_end_ready(accumulator_end_ready),
        .window_end_accept(accumulator_end_accept),
        .commit_valid(accumulator_commit_valid),
        .commit_ready(commit_ready && !rst_core),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(accumulator_window_done),
        .lane_mem_rd_en(internal_lane_mem_rd_en),
        .lane_mem_rd_addr(internal_lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(internal_lane_mem_wr_en),
        .lane_mem_wr_addr(internal_lane_mem_wr_addr),
        .lane_mem_wr_data(internal_lane_mem_wr_data),
        .protocol_error(accumulator_protocol_error),
        .window_active(accumulator_window_active),
        .busy(accumulator_busy)
    );

endmodule

`default_nettype wire
