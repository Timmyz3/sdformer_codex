`timescale 1ns/1ps
`default_nettype none

module m111_independent_assertions #(
    parameter int VECTOR_BITS = 2304
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
    input logic [7:0] mem_rd_en,
    input logic [8:0] mem_rd_addr [0:7],
    input logic [7:0] mem_wr_en,
    input logic [8:0] mem_wr_addr [0:7],
    input logic protocol_error,
    input logic window_active,
    input logic busy
);
    ap_start_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_start_accept == (window_start_valid && window_start_ready));
    ap_update_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept == (update_valid && update_ready));
    ap_end_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_end_accept == (window_end_valid && window_end_ready));
    ap_single_global_read: assert property (@(posedge clk_core) disable iff (rst_core)
        $onehot0(mem_rd_en));
    ap_single_global_write: assert property (@(posedge clk_core) disable iff (rst_core)
        $onehot0(mem_wr_en));
    ap_update_has_one_read: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept |-> $onehot(mem_rd_en));
    ap_update_read_bank: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept |-> mem_rd_en[update_block]);
    ap_no_write_during_commit_phase: assert property (@(posedge clk_core) disable iff (rst_core)
        busy && !window_active && !protocol_error |-> !(|mem_wr_en));
    ap_commit_stable_on_stall: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready |=> commit_valid
            && $stable({commit_block, commit_row, commit_data, commit_last}));
    ap_last_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_last |-> commit_block == 3'd7 && commit_row == 9'd383);
    ap_done_after_last: assert property (@(posedge clk_core) disable iff (rst_core)
        window_done |-> $past(commit_valid && commit_ready && commit_last));
    ap_no_commit_while_active: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid |-> !window_active);
    ap_fault_quarantine: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !window_start_ready && !update_ready
            && !window_end_ready && !commit_valid);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);

    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_addr_range
            ap_read_addr_range: assert property (@(posedge clk_core) disable iff (rst_core)
                mem_rd_en[bank] |-> mem_rd_addr[bank] < 384);
            ap_write_addr_range: assert property (@(posedge clk_core) disable iff (rst_core)
                mem_wr_en[bank] |-> mem_wr_addr[bank] < 384);
        end
    endgenerate

    cp_ii1: cover property (@(posedge clk_core) disable iff (rst_core)
        update_accept ##1 update_accept);
    cp_dual_port_overlap: cover property (@(posedge clk_core) disable iff (rst_core)
        (|mem_rd_en) && (|mem_wr_en));
    cp_stall_release: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready ##1 commit_valid && commit_ready);
    cp_complete: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_ready && commit_last ##1 window_done);
    cp_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error ##1 protocol_error);
endmodule

`default_nettype wire
