`timescale 1ns/1ps
`default_nettype none

module m123_w384_signed19_forwarding_lane_sliced_accumulator_assertions #(
    parameter int VECTOR_BITS = 1824
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start_valid,
    input logic window_start_ready,
    input logic window_start_accept,
    input logic update_valid,
    input logic update_ready,
    input logic update_accept,
    input logic [2:0] update_block,
    input logic [8:0] update_row,
    input logic window_end_valid,
    input logic window_end_ready,
    input logic window_end_accept,
    input logic commit_valid,
    input logic commit_ready,
    input logic [2:0] commit_block,
    input logic [8:0] commit_row,
    input logic [VECTOR_BITS-1:0] commit_data,
    input logic commit_last,
    input logic window_done,
    input logic lane_mem_rd_en,
    input logic [11:0] lane_mem_rd_addr,
    input logic lane_mem_wr_en,
    input logic [11:0] lane_mem_wr_addr,
    input logic protocol_error,
    input logic window_active,
    input logic busy
);
`ifdef SVA_RUNTIME_ENABLED
    ap_start_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_start_accept == (window_start_valid && window_start_ready));
    ap_update_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept == (update_valid && update_ready));
    ap_end_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_end_accept == (window_end_valid && window_end_ready));
    ap_same_address_forward_suppresses_macro_read: assert property (
        @(posedge clk_core) disable iff (rst_core)
        update_accept && $past(update_accept)
            && update_block == $past(update_block)
            && update_row == $past(update_row)
        |-> !lane_mem_rd_en);
    ap_every_accepted_update_writes_next_cycle: assert property (
        @(posedge clk_core) disable iff (rst_core)
        update_accept |=> lane_mem_wr_en || protocol_error);
    ap_request_collision: assert property (@(posedge clk_core) disable iff (rst_core)
        (window_start_valid && update_valid)
        || (window_start_valid && window_end_valid)
        || (update_valid && window_end_valid)
        |-> !window_start_accept && !update_accept && !window_end_accept);
    ap_fault_quarantine: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !window_start_ready && !update_ready
            && !window_end_ready && !commit_valid);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);
    ap_commit_stable_on_stall: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready |=> commit_valid
            && $stable({commit_block, commit_row, commit_data, commit_last}));
    ap_commit_last_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_last |-> commit_block == 7 && commit_row == 383);
    ap_window_done_has_last_accept: assert property (@(posedge clk_core) disable iff (rst_core)
        window_done |-> $past(commit_valid && commit_ready && commit_last));
    ap_read_address_range: assert property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_rd_en |-> lane_mem_rd_addr < 3072);
    ap_write_address_range: assert property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_wr_en |-> lane_mem_wr_addr < 3072);
    ap_no_commit_while_window_active: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid |-> !window_active);

    cp_update_ii1: cover property (@(posedge clk_core) disable iff (rst_core)
        update_accept ##1 update_accept);
    cp_same_address_forward_chain: cover property (
        @(posedge clk_core) disable iff (rst_core)
        update_accept
        ##1 update_accept && update_block == $past(update_block)
            && update_row == $past(update_row) && !lane_mem_rd_en
        ##1 update_accept && update_block == $past(update_block)
            && update_row == $past(update_row) && !lane_mem_rd_en);
    cp_read_write_overlap: cover property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_rd_en && lane_mem_wr_en);
    cp_commit_stall: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready ##1 commit_valid && commit_ready);
    cp_full_commit: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_ready && commit_last ##1 window_done);
    cp_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error ##1 protocol_error);
`endif
endmodule

`default_nettype wire
