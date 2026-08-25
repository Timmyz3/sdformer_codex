`timescale 1ns/1ps
`default_nettype none

module m154_four_bank_destination_vector_supplier_assertions #(
    parameter int LANES = 96,
    parameter int SEQUENCE_BITS = 32,
    parameter int PARTITION_BITS = 9
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         context_open_valid,
    input logic                         context_open_ready,
    input logic                         context_open_accept,
    input logic                         descriptor_valid,
    input logic                         descriptor_ready,
    input logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input logic [1:0]                   descriptor_operator,
    input logic [PARTITION_BITS-1:0]    descriptor_partition,
    input logic [8:0]                   descriptor_row,
    input logic [3:0]                   descriptor_source,
    input logic [3:0]                   descriptor_destination_valid,
    input logic [2:0]                   descriptor_destination [0:3],
    input logic [3:0]                   descriptor_negate,
    input logic                         descriptor_last,
    input logic                         descriptor_accept,
    input logic                         context_close_valid,
    input logic                         context_close_ready,
    input logic                         context_close_accept,
    input logic [3:0]                   bank_rd_en,
    input logic [4:0]                   bank_rd_addr [0:3],
    input logic                         result_valid,
    input logic                         result_ready,
    input logic [SEQUENCE_BITS-1:0]     result_sequence,
    input logic [1:0]                   result_operator,
    input logic [PARTITION_BITS-1:0]    result_partition,
    input logic [8:0]                   result_row,
    input logic [3:0]                   result_source,
    input logic [3:0]                   result_destination_valid,
    input logic [2:0]                   result_destination [0:3],
    input logic [3:0]                   result_negate,
    input logic                         result_last,
    input logic signed [7:0]            result_vector [0:3][0:LANES-1],
    input logic                         result_accept,
    input logic                         context_active,
    input logic                         protocol_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_context_open_accept_definition:
        assert property (context_open_accept
                         == (context_open_valid && context_open_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_context_close_accept_definition:
        assert property (context_close_accept
                         == (context_close_valid && context_close_ready));
    ap_result_accept_definition:
        assert property (result_accept == (result_valid && result_ready));
    ap_read_count_matches_descriptor:
        assert property (descriptor_accept
            |-> $countones(bank_rd_en)
                == $countones(descriptor_destination_valid));
    ap_no_read_without_accept:
        assert property ((|bank_rd_en) |-> descriptor_accept);
    ap_result_metadata_stable_under_stall:
        assert property (result_valid && !result_ready
            |=> result_valid
                && $stable({result_sequence, result_operator,
                            result_partition, result_row, result_source,
                            result_destination_valid, result_negate,
                            result_last, result_destination[0],
                            result_destination[1], result_destination[2],
                            result_destination[3]}));
    ap_fault_is_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_pending_result_survives_fault:
        assert property (result_valid && !result_ready && protocol_error
                         |=> result_valid);
    ap_busy_when_context_active:
        assert property (context_active |-> busy);

    generate
        for (genvar bank = 0; bank < 4; bank++) begin : g_bank
            ap_bank_address_matches_tuple:
                assert property (descriptor_accept && bank_rd_en[bank]
                    |-> ((descriptor_destination_valid[0]
                          && descriptor_destination[0][1:0] == bank
                          && bank_rd_addr[bank]
                             == {descriptor_destination[0][2],
                                 descriptor_source})
                         || (descriptor_destination_valid[1]
                             && descriptor_destination[1][1:0] == bank
                             && bank_rd_addr[bank]
                                == {descriptor_destination[1][2],
                                    descriptor_source})
                         || (descriptor_destination_valid[2]
                             && descriptor_destination[2][1:0] == bank
                             && bank_rd_addr[bank]
                                == {descriptor_destination[2][2],
                                    descriptor_source})
                         || (descriptor_destination_valid[3]
                             && descriptor_destination[3][1:0] == bank
                             && bank_rd_addr[bank]
                                == {descriptor_destination[3][2],
                                    descriptor_source})));
        end
        for (genvar tuple = 0; tuple < 4; tuple++) begin : g_tuple
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                ap_result_vector_stable_under_stall:
                    assert property (result_valid && !result_ready
                                     |=> $stable(result_vector[tuple][lane]));
            end
        end
    endgenerate

    cp_all_four_banks:
        cover property (descriptor_accept && bank_rd_en == 4'b1111);
    cp_low_destination_row:
        cover property (descriptor_accept
                        && descriptor_destination[0] == 3'd0
                        && descriptor_destination[3] == 3'd3);
    cp_high_destination_row:
        cover property (descriptor_accept
                        && descriptor_destination[0] == 3'd4
                        && descriptor_destination[3] == 3'd7);
    cp_back_to_back_descriptor:
        cover property (descriptor_accept ##1 descriptor_accept);
    cp_result_stall:
        cover property (result_valid && !result_ready
                        ##1 result_valid && result_ready);
    cp_protocol_fault_with_pending_result:
        cover property (result_valid && !result_ready && protocol_error);
endmodule

`default_nettype wire
