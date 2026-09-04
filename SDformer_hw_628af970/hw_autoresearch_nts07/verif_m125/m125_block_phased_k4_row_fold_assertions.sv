`timescale 1ns/1ps
`default_nettype none

module m125_block_phased_k4_row_fold_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic weight_fill_valid,
    input logic weight_fill_ready,
    input logic weight_fill_accept,
    input logic row_valid,
    input logic row_ready,
    input logic row_accept,
    input logic update_valid,
    input logic update_ready,
    input logic update_accept,
    input logic [2:0] update_block,
    input logic [8:0] update_row,
    input logic [1823:0] update_delta,
    input logic [15:0] update_selected_mask,
    input logic [15:0] observed_remaining_mask,
    input logic row_done,
    input logic protocol_error
);
`ifdef SVA_RUNTIME_ENABLED
    ap_fill_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        weight_fill_accept == (weight_fill_valid && weight_fill_ready));
    ap_row_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        row_accept == (row_valid && row_ready));
    ap_update_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept == (update_valid && update_ready));
    ap_selected_nonempty_bounded: assert property (
        @(posedge clk_core) disable iff (rst_core)
        update_valid |-> $countones(update_selected_mask) inside {[1:4]});
    ap_selected_subset: assert property (@(posedge clk_core) disable iff (rst_core)
        update_valid
        |-> (update_selected_mask & ~observed_remaining_mask) == 0);
    ap_select_and_clear: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept
        |=> observed_remaining_mask
            == ($past(observed_remaining_mask)
                & ~$past(update_selected_mask)));
    ap_strict_progress: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept |=> $countones(observed_remaining_mask)
                           < $past($countones(observed_remaining_mask)));
    ap_update_stable_on_stall: assert property (
        @(posedge clk_core) disable iff (rst_core)
        update_valid && !update_ready
        |=> update_valid
            && $stable({update_block, update_row, update_delta,
                        update_selected_mask}));
    ap_fault_quarantine: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !weight_fill_ready && !row_ready && !update_valid);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);

    cp_full_k4: cover property (@(posedge clk_core) disable iff (rst_core)
        update_accept && $countones(update_selected_mask) == 4);
    cp_tail_k1: cover property (@(posedge clk_core) disable iff (rst_core)
        update_accept && $countones(update_selected_mask) == 1);
    cp_two_fold_same_row: cover property (@(posedge clk_core) disable iff (rst_core)
        update_accept ##1 update_accept
            && update_block == $past(update_block)
            && update_row == $past(update_row));
    cp_update_stall_release: cover property (@(posedge clk_core) disable iff (rst_core)
        update_valid && !update_ready ##1 update_valid && update_ready);
    cp_empty_row: cover property (@(posedge clk_core) disable iff (rst_core)
        row_accept && observed_remaining_mask == 0 ##1 row_done);
    cp_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error ##1 protocol_error);
`endif
endmodule

`default_nettype wire
