`timescale 1ns/1ps
`default_nettype none

module m148_destination_tagged_mosaic_k4_packer_assertions #(
    parameter int MASK_BITS = 128,
    parameter int SEQUENCE_BITS = 32,
    parameter int ROW_BITS = 9
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         row_valid,
    input logic                         row_ready,
    input logic [SEQUENCE_BITS-1:0]     row_sequence,
    input logic [ROW_BITS-1:0]          row_id,
    input logic [MASK_BITS-1:0]         row_event_mask,
    input logic                         row_accept,
    input logic                         descriptor_valid,
    input logic                         descriptor_ready,
    input logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input logic [ROW_BITS-1:0]          descriptor_row,
    input logic [1:0]                   descriptor_count_m1,
    input logic [2:0]                   descriptor_destination [0:3],
    input logic [3:0]                   descriptor_source [0:3],
    input logic [3:0]                   descriptor_tuple_valid,
    input logic                         descriptor_last,
    input logic                         descriptor_accept,
    input logic                         done_valid,
    input logic [SEQUENCE_BITS-1:0]     done_sequence,
    input logic [ROW_BITS-1:0]          done_row,
    input logic                         observed_active,
    input logic [MASK_BITS-1:0]         observed_remaining_mask,
    input logic [7:0]                   observed_work_popcount,
    input logic [SEQUENCE_BITS-1:0]     observed_next_sequence,
    input logic                         protocol_error,
    input logic                         busy
);
    logic [6:0] tuple_index [0:3];
    always_comb begin
        for (int pick = 0; pick < 4; pick++)
            tuple_index[pick] = {descriptor_destination[pick],
                                 descriptor_source[pick]};
    end

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_row_accept_definition:
        assert property (row_accept == (row_valid && row_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_row_payload_stable_under_stall:
        assert property (row_valid && !row_ready
            |=> row_valid
                && $stable({row_sequence, row_id, row_event_mask}));
    ap_descriptor_stable_under_stall:
        assert property (descriptor_valid && !descriptor_ready
            |=> descriptor_valid
                && $stable({descriptor_sequence, descriptor_row,
                            descriptor_count_m1,
                            descriptor_destination[0],
                            descriptor_destination[1],
                            descriptor_destination[2],
                            descriptor_destination[3],
                            descriptor_source[0], descriptor_source[1],
                            descriptor_source[2], descriptor_source[3],
                            descriptor_tuple_valid, descriptor_last}));
    ap_tuple_valid_shape:
        assert property (descriptor_valid
            |-> descriptor_tuple_valid inside {
                4'b0001, 4'b0011, 4'b0111, 4'b1111});
    ap_count_matches_valids:
        assert property (descriptor_valid
            |-> descriptor_count_m1
                == (descriptor_tuple_valid[3] ? 2'd3
                    : descriptor_tuple_valid[2] ? 2'd2
                    : descriptor_tuple_valid[1] ? 2'd1 : 2'd0));
    ap_tuple_order_01:
        assert property (descriptor_valid && descriptor_tuple_valid[1]
                         |-> tuple_index[0] < tuple_index[1]);
    ap_tuple_order_12:
        assert property (descriptor_valid && descriptor_tuple_valid[2]
                         |-> tuple_index[1] < tuple_index[2]);
    ap_tuple_order_23:
        assert property (descriptor_valid && descriptor_tuple_valid[3]
                         |-> tuple_index[2] < tuple_index[3]);
    ap_padding_one:
        assert property (descriptor_valid && !descriptor_tuple_valid[1]
            |-> {descriptor_destination[1], descriptor_source[1],
                 descriptor_destination[2], descriptor_source[2],
                 descriptor_destination[3], descriptor_source[3]} == 0);
    ap_padding_two:
        assert property (descriptor_valid && descriptor_tuple_valid[1]
                         && !descriptor_tuple_valid[2]
            |-> {descriptor_destination[2], descriptor_source[2],
                 descriptor_destination[3], descriptor_source[3]} == 0);
    ap_padding_three:
        assert property (descriptor_valid && descriptor_tuple_valid[2]
                         && !descriptor_tuple_valid[3]
            |-> {descriptor_destination[3], descriptor_source[3]} == 0);
    ap_last_definition:
        assert property (descriptor_valid
                         |-> descriptor_last
                             == (observed_work_popcount <= 4));
    ap_zero_row_has_no_descriptor:
        assert property (row_accept && row_event_mask == 0
                         |-> !descriptor_valid);
    ap_done_definition:
        assert property (done_valid
            |-> (row_accept && row_event_mask == 0)
                || (descriptor_accept && descriptor_last));
    ap_done_identity:
        assert property (descriptor_accept && descriptor_last
            |-> done_valid && done_sequence == descriptor_sequence
                && done_row == descriptor_row);
    ap_active_has_remaining_work:
        assert property (observed_active
                         |-> observed_remaining_mask != 0);
    ap_busy_definition:
        assert property (busy == observed_active);
    ap_protocol_error_sticky:
        assert property (protocol_error |=> protocol_error);

    cp_zero_row:
        cover property (row_accept && row_event_mask == 0 && done_valid);
    cp_fallthrough_first_descriptor:
        cover property (row_accept && descriptor_accept);
    cp_full_four_tuple:
        cover property (descriptor_accept
                        && descriptor_tuple_valid == 4'b1111);
    cp_tail_one_tuple:
        cover property (descriptor_accept && descriptor_last
                        && descriptor_tuple_valid == 4'b0001);
    cp_tail_two_tuple:
        cover property (descriptor_accept && descriptor_last
                        && descriptor_tuple_valid == 4'b0011);
    cp_tail_three_tuple:
        cover property (descriptor_accept && descriptor_last
                        && descriptor_tuple_valid == 4'b0111);
    cp_descriptor_stall:
        cover property (descriptor_valid && !descriptor_ready
                        ##1 descriptor_valid && descriptor_ready);
    cp_cross_destination_descriptor:
        cover property (descriptor_accept && descriptor_tuple_valid[1]
                        && descriptor_destination[0]
                           != descriptor_destination[1]);
endmodule

`default_nettype wire
