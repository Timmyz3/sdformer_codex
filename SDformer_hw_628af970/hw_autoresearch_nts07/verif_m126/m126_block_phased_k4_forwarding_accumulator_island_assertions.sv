`timescale 1ns/1ps
`default_nettype none

module m126_block_phased_k4_forwarding_accumulator_island_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic window_start_valid,
    input logic window_start_ready,
    input logic window_start_accept,
    input logic weight_fill_valid,
    input logic weight_fill_ready,
    input logic weight_fill_accept,
    input logic row_valid,
    input logic row_ready,
    input logic row_accept,
    input logic row_done,
    input logic window_end_valid,
    input logic window_end_ready,
    input logic window_end_accept,
    input logic commit_valid,
    input logic commit_ready,
    input logic [2:0] commit_block,
    input logic [8:0] commit_row,
    input logic [1823:0] commit_data,
    input logic commit_last,
    input logic window_done,
    input logic lane_mem_rd_en,
    input logic lane_mem_wr_en,
    input logic observed_fold_update_accept,
    input logic observed_accumulator_update_accept,
    input logic [15:0] observed_fold_selected_mask,
    input logic [15:0] observed_fold_remaining_mask,
    input logic protocol_error
);
`ifdef SVA_RUNTIME_ENABLED
    ap_start_handshake: assert property (@(posedge clk_core)
        window_start_accept == (window_start_valid && window_start_ready));
    ap_fill_handshake: assert property (@(posedge clk_core)
        weight_fill_accept == (weight_fill_valid && weight_fill_ready));
    ap_row_handshake: assert property (@(posedge clk_core)
        row_accept == (row_valid && row_ready));
    ap_end_handshake: assert property (@(posedge clk_core)
        window_end_accept == (window_end_valid && window_end_ready));

    ap_fold_bounded: assert property (@(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept
        |-> $countones(observed_fold_selected_mask) inside {[1:4]});
    ap_fold_accumulator_handshake_match: assert property (
        @(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept
        == observed_accumulator_update_accept);
    ap_fold_selected_subset: assert property (
        @(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept
        |-> (observed_fold_selected_mask
             & ~observed_fold_remaining_mask) == 0);
    ap_fold_write_conservation: assert property (
        @(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept |=> lane_mem_wr_en);
    ap_no_spurious_write: assert property (
        @(posedge clk_core) disable iff (rst_core)
        lane_mem_wr_en |-> $past(observed_fold_update_accept));
    ap_row_done_clear: assert property (@(posedge clk_core) disable iff (rst_core)
        row_done |-> observed_fold_remaining_mask == 0);

    ap_commit_stable_on_stall: assert property (
        @(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready
        |=> commit_valid
            && $stable({commit_block, commit_row, commit_data, commit_last}));
    ap_fault_external_quarantine: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error
        |-> !window_start_ready && !weight_fill_ready && !row_ready
            && !window_end_ready && !commit_valid);
    ap_reset_isolation: assert property (@(posedge clk_core)
        rst_core
        |-> !window_start_ready && !window_start_accept
            && !weight_fill_ready && !weight_fill_accept
            && !row_ready && !row_accept && !row_done
            && !window_end_ready && !window_end_accept
            && !commit_valid && !window_done
            && !lane_mem_rd_en && !lane_mem_wr_en
            && !observed_fold_update_accept
            && !observed_accumulator_update_accept);

    cp_four_consecutive_same_row_folds: cover property (
        @(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept
        ##1 observed_fold_update_accept
        ##1 observed_fold_update_accept
        ##1 observed_fold_update_accept);
    cp_full_k4_to_write: cover property (@(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept
        && $countones(observed_fold_selected_mask) == 4
        ##1 lane_mem_wr_en);
    cp_tail_to_write: cover property (@(posedge clk_core) disable iff (rst_core)
        observed_fold_update_accept
        && $countones(observed_fold_selected_mask) inside {[1:3]}
        ##1 lane_mem_wr_en);
    cp_commit_stall_release: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready ##1 commit_valid && commit_ready);
    cp_reset_with_prior_update: cover property (@(posedge clk_core)
        !rst_core && observed_fold_update_accept ##1 rst_core
        && !lane_mem_wr_en);
`endif
endmodule

`default_nettype wire
