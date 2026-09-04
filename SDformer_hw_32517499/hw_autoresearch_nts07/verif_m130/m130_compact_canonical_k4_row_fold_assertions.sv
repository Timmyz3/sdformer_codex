`timescale 1ns/1ps
`default_nettype none

module m130_compact_canonical_k4_row_fold_assertions #(
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
    input logic [2:0] group_block,
    input logic [8:0] group_row,
    input logic [1:0] group_source_count_m1,
    input logic [3:0] group_source [0:3],
    input logic [3:0] group_negate,
    input logic group_last,
    input logic update_valid,
    input logic update_ready,
    input logic update_accept,
    input logic [2:0] update_block,
    input logic [8:0] update_row,
    input logic [UPDATE_BITS-1:0] update_delta,
    input logic [SOURCES-1:0] update_selected_mask,
    input logic update_last,
    input logic done_valid,
    input logic [2:0] done_block,
    input logic [8:0] done_row,
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
                              || update_accept || done_valid));
    ap_k2_strictly_increasing:
        assert property (group_accept && group_source_count_m1 >= 1
                         |-> group_source[1] > group_source[0]);
    ap_k3_strictly_increasing:
        assert property (group_accept && group_source_count_m1 >= 2
                         |-> group_source[2] > group_source[1]);
    ap_k4_strictly_increasing:
        assert property (group_accept && group_source_count_m1 == 3
                         |-> group_source[3] > group_source[2]);
    ap_k1_padding_zero:
        assert property (group_accept && group_source_count_m1 == 0
                         |-> group_source[1] == 0
                             && group_source[2] == 0
                             && group_source[3] == 0
                             && group_negate[3:1] == 0);
    ap_k2_padding_zero:
        assert property (group_accept && group_source_count_m1 == 1
                         |-> group_source[2] == 0
                             && group_source[3] == 0
                             && group_negate[3:2] == 0);
    ap_k3_padding_zero:
        assert property (group_accept && group_source_count_m1 == 2
                         |-> group_source[3] == 0
                             && group_negate[3] == 0);
    ap_update_stable_under_stall:
        assert property (update_valid && !update_ready
                         |=> update_valid
                             && $stable({update_block, update_row,
                                         update_delta,
                                         update_selected_mask,
                                         update_last}));
    ap_done_exact_definition:
        assert property (done_valid == (update_accept && update_last));
    ap_done_tag_matches_update:
        assert property (done_valid
                         |-> done_block == update_block
                             && done_row == update_row);

    cp_cross_row_replace:
        cover property (group_accept && group_last
                        && update_accept && update_last
                        ##1 group_accept && group_last
                            && update_accept && update_last
                        ##1 group_accept && group_last
                            && update_accept && update_last);
    cp_multidescriptor_row:
        cover property (group_accept && !group_last
                        ##1 group_accept && group_last);
    cp_k1_descriptor:
        cover property (group_accept && group_source_count_m1 == 0);
    cp_k4_descriptor:
        cover property (group_accept && group_source_count_m1 == 3);
    cp_update_stall_release:
        cover property (update_valid && !update_ready
                        ##1 update_valid && update_ready);
    cp_tagged_done_overlaps_next_group:
        cover property (done_valid && group_accept
                        && {done_block, done_row}
                           != {group_block, group_row});
    cp_reset_quiesce:
        cover property (@(posedge clk_core) disable iff (1'b0) rst_core
                        && !weight_fill_accept && !group_accept
                        && !update_valid && !done_valid && !protocol_error);
endmodule

`default_nettype wire
