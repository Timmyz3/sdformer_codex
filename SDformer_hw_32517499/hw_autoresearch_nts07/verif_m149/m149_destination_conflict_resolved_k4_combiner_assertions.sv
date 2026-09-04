`timescale 1ns/1ps
`default_nettype none

module m149_destination_conflict_resolved_k4_combiner_assertions #(
    parameter int LANES = 96,
    parameter int SEQUENCE_BITS = 32,
    parameter int ROW_BITS = 9
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         descriptor_valid,
    input logic                         descriptor_ready,
    input logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input logic [ROW_BITS-1:0]          descriptor_row,
    input logic                         descriptor_last,
    input logic [3:0]                   descriptor_tuple_valid,
    input logic [2:0]                   tuple_destination [0:3],
    input logic                         tuple_negate [0:3],
    input logic signed [7:0]            tuple_vector [0:3][0:LANES-1],
    input logic                         descriptor_accept,
    input logic                         result_valid,
    input logic                         result_ready,
    input logic [SEQUENCE_BITS-1:0]     result_sequence,
    input logic [ROW_BITS-1:0]          result_row,
    input logic                         result_last,
    input logic [3:0]                   result_group_valid,
    input logic [2:0]                   result_destination [0:3],
    input logic signed [10:0]           result_vector [0:3][0:LANES-1],
    input logic                         result_accept,
    input logic                         protocol_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_result_accept_definition:
        assert property (result_accept == (result_valid && result_ready));
    ap_descriptor_payload_stable_under_stall:
        assert property (descriptor_valid && !descriptor_ready
                         && !protocol_error
            |=> descriptor_valid
                && $stable({descriptor_sequence, descriptor_row,
                            descriptor_last, descriptor_tuple_valid,
                            tuple_destination[0], tuple_destination[1],
                            tuple_destination[2], tuple_destination[3],
                            tuple_negate[0], tuple_negate[1],
                            tuple_negate[2], tuple_negate[3]}));
    ap_result_payload_stable_under_stall:
        assert property (result_valid && !result_ready
            |=> result_valid
                && $stable({result_sequence, result_row, result_last,
                            result_group_valid,
                            result_destination[0], result_destination[1],
                            result_destination[2], result_destination[3]}));
    ap_result_follows_accept:
        assert property (descriptor_accept
                         |=> result_valid
                             && result_sequence
                                == $past(descriptor_sequence)
                             && result_row == $past(descriptor_row)
                             && result_last == $past(descriptor_last));
    ap_group0_owns_tuple0:
        assert property (result_valid |-> result_group_valid[0]);
    ap_no_duplicate_01:
        assert property (result_valid && result_group_valid[1]
            |-> result_destination[1] != result_destination[0]);
    ap_no_duplicate_02:
        assert property (result_valid && result_group_valid[2]
            |-> result_destination[2] != result_destination[0]);
    ap_no_duplicate_12:
        assert property (result_valid && result_group_valid[1]
                         && result_group_valid[2]
            |-> result_destination[2] != result_destination[1]);
    ap_no_duplicate_03:
        assert property (result_valid && result_group_valid[3]
            |-> result_destination[3] != result_destination[0]);
    ap_no_duplicate_13:
        assert property (result_valid && result_group_valid[1]
                         && result_group_valid[3]
            |-> result_destination[3] != result_destination[1]);
    ap_no_duplicate_23:
        assert property (result_valid && result_group_valid[2]
                         && result_group_valid[3]
            |-> result_destination[3] != result_destination[2]);
    ap_fault_is_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_busy_definition:
        assert property (result_valid |-> busy);

    cp_all_same_destination:
        cover property (result_accept
                        && result_group_valid == 4'b0001);
    cp_two_plus_two:
        cover property (result_accept
                        && $countones(result_group_valid) == 2);
    cp_two_plus_one_plus_one:
        cover property (result_accept
                        && $countones(result_group_valid) == 3);
    cp_all_distinct:
        cover property (result_accept
                        && result_group_valid == 4'b1111);
    cp_result_stall:
        cover property (result_valid && !result_ready
                        ##1 result_valid && result_ready);
    cp_back_to_back_accept:
        cover property (descriptor_accept ##1 descriptor_accept);

    generate
        for (genvar tuple = 0; tuple < 4; tuple++) begin : g_tuple
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                ap_tuple_lane_stable_under_stall:
                    assert property (descriptor_valid && !descriptor_ready
                                     && !protocol_error
                        |=> $stable(tuple_vector[tuple][lane]));
                ap_result_lane_stable_under_stall:
                    assert property (result_valid && !result_ready
                        |=> $stable(result_vector[tuple][lane]));
            end
        end
    endgenerate
endmodule

`default_nettype wire
