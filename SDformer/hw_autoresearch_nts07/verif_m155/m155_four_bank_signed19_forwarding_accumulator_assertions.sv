`timescale 1ns/1ps
`default_nettype none

module m155_four_bank_signed19_forwarding_accumulator_assertions #(
    parameter int LANES = 96,
    parameter int ACC_BITS = 19
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         window_start_valid,
    input logic                         window_start_ready,
    input logic                         window_start_accept,
    input logic                         update_valid,
    input logic                         update_ready,
    input logic [8:0]                   update_row,
    input logic [3:0]                   update_group_valid,
    input logic [2:0]                   update_destination [0:3],
    input logic [3:0]                   update_negate,
    input logic signed [10:0]           update_vector [0:3][0:LANES-1],
    input logic                         update_accept,
    input logic                         window_end_valid,
    input logic                         window_end_ready,
    input logic                         window_end_accept,
    input logic                         window_done,
    input logic [3:0]                   acc_rd_en,
    input logic [9:0]                   acc_rd_addr [0:3],
    input logic [3:0]                   acc_wr_en,
    input logic [9:0]                   acc_wr_addr [0:3],
    input logic signed [ACC_BITS-1:0]   acc_wr_data [0:3][0:LANES-1],
    input logic [3:0]                   same_address_forward,
    input logic                         window_active,
    input logic                         protocol_error,
    input logic                         overflow_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_start_accept_definition:
        assert property (window_start_accept
                         == (window_start_valid && window_start_ready));
    ap_update_accept_definition:
        assert property (update_accept == (update_valid && update_ready));
    ap_end_accept_definition:
        assert property (window_end_accept
                         == (window_end_valid && window_end_ready));
    ap_done_follows_end:
        assert property (window_end_accept |=> window_done);
    ap_fault_is_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_overflow_suppresses_all_writes:
        assert property (overflow_error |-> acc_wr_en == 4'b0000);
    ap_forward_suppresses_macro_read:
        assert property ((same_address_forward & acc_rd_en) == 4'b0000);
    ap_no_write_before_window:
        assert property ((|acc_wr_en) |-> window_active);

    generate
        for (genvar bank = 0; bank < 4; bank++) begin : g_bank
            ap_forward_address_matches_write:
                assert property (same_address_forward[bank]
                                 |-> acc_rd_addr[bank]
                                     == acc_wr_addr[bank]);
        end
    endgenerate

    cp_full4_accept:
        cover property (update_accept && update_group_valid == 4'b1111);
    cp_all_bank_write:
        cover property (acc_wr_en == 4'b1111);
    cp_all_bank_forward:
        cover property (same_address_forward == 4'b1111);
    cp_read_write_overlap:
        cover property ((|acc_rd_en) && (|acc_wr_en));
    cp_window_done:
        cover property (window_done);
    cp_protocol_fault:
        cover property (protocol_error);
    cp_overflow_fault:
        cover property (overflow_error);
endmodule

`default_nettype wire
