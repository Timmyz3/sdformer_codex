`timescale 1ns/1ps
`default_nettype none

module m128_descriptor_streamed_k4_row_fold_assertions #(
    parameter int SOURCES = 16,
    parameter int UPDATE_BITS = 1824
) (
    input logic clk_core,
    input logic rst_core,
    input logic weight_fill_accept,
    input logic weight_fill_ready,
    input logic group_valid,
    input logic group_ready,
    input logic group_accept,
    input logic [3:0] group_source_valid,
    input logic update_valid,
    input logic update_ready,
    input logic update_accept,
    input logic [2:0] update_block,
    input logic [8:0] update_row,
    input logic [UPDATE_BITS-1:0] update_delta,
    input logic [SOURCES-1:0] update_selected_mask,
    input logic update_last,
    input logic row_done,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_group_accept_definition:
        assert property (group_accept == (group_valid && group_ready));
    ap_fill_accept_requires_ready:
        assert property (weight_fill_accept |-> weight_fill_ready);
    ap_update_accept_definition:
        assert property (update_accept == (update_valid && update_ready));
    ap_no_accept_while_faulted:
        assert property (protocol_error
                         |-> !(weight_fill_accept || group_accept
                              || update_accept));
    ap_nonempty_descriptor:
        assert property (group_accept |-> group_source_valid != 0);
    ap_update_stable_under_stall:
        assert property (update_valid && !update_ready
                         |=> update_valid
                             && $stable({update_block, update_row,
                                         update_delta,
                                         update_selected_mask,
                                         update_last}));
    ap_last_update_generates_done:
        assert property (update_accept && update_last |=> row_done);
    ap_nonlast_update_no_done:
        assert property (update_accept && !update_last |=> !row_done);

    cp_cross_row_replace:
        cover property (group_accept && update_accept && update_last
                        ##1 group_accept && update_accept && update_last
                        ##1 group_accept && update_accept && update_last
                        ##1 group_accept && update_accept && update_last);
    cp_k4_descriptor:
        cover property (group_accept && group_source_valid == 4'b1111);
    cp_tail_descriptor:
        cover property (group_accept && group_source_valid != 4'b1111);
    cp_update_stall_release:
        cover property (update_valid && !update_ready
                        ##1 update_valid && update_ready);
    cp_reset_quiesce:
        cover property (@(posedge clk_core) disable iff (1'b0) rst_core
                        && !weight_fill_accept && !group_accept
                        && !update_valid && !protocol_error);
endmodule

`default_nettype wire
