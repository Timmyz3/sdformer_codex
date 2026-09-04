`timescale 1ns/1ps
`default_nettype none

module m194_fc2_same_token_pair_bank_head_selector_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int COUNT_BITS = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic pair_valid,
    input logic pair_ready,
    input logic [1:0] window_valid,
    input logic [TAG_BITS-1:0] window_token_tag [0:1],
    input logic [COUNT_BITS-1:0] window_bank_count [0:1][0:7],
    input logic [CHANNEL_BITS-1:0] window_head_channel [0:1][0:7],
    input logic pair_accept,
    input logic issue_valid,
    input logic issue_ready,
    input logic [TAG_BITS-1:0] issue_token_tag,
    input logic [3:0] issue_source_count,
    input logic [7:0] issue_bank_valid,
    input logic [7:0] issue_selected_window,
    input logic [CHANNEL_BITS-1:0] issue_source_channel [0:7],
    input logic issue_pair_last,
    input logic issue_accept,
    input logic protocol_error,
    input logic busy
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_pair_handshake:
        assert property (pair_accept == (pair_valid && pair_ready));
    ap_issue_handshake:
        assert property (issue_accept == (issue_valid && issue_ready));
    ap_issue_nonempty:
        assert property (issue_valid |-> issue_bank_valid != 0
            && issue_source_count == $countones(issue_bank_valid));
    ap_same_token_admission:
        assert property (pair_accept && window_valid == 2'b11
            |-> window_token_tag[0] == window_token_tag[1]);
    ap_busy_identity:
        assert property (busy == issue_valid);
    ap_protocol_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_closes_input:
        assert property (protocol_error |=> !pair_ready);
    ap_hold_issue_on_stall:
        assert property (issue_valid && !issue_ready |=>
            $stable({issue_valid, issue_token_tag, issue_source_count,
                     issue_bank_valid, issue_selected_window,
                     issue_pair_last, issue_source_channel[0],
                     issue_source_channel[1], issue_source_channel[2],
                     issue_source_channel[3], issue_source_channel[4],
                     issue_source_channel[5], issue_source_channel[6],
                     issue_source_channel[7]}));
    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank
            ap_channel_bank_identity:
                assert property (issue_valid && issue_bank_valid[bank]
                    |-> issue_source_channel[bank][2:0] == bank[2:0]);
            ap_invalid_channel_zero:
                assert property (issue_valid && !issue_bank_valid[bank]
                    |-> issue_source_channel[bank] == 0);
        end
    endgenerate

    cp_window0_only:
        cover property (pair_accept && window_valid == 2'b01);
    cp_window1_only:
        cover property (pair_accept && window_valid == 2'b10);
    cp_both_windows:
        cover property (pair_accept && window_valid == 2'b11);
    cp_bank_fallthrough:
        cover property (issue_valid && issue_bank_valid[7]
            && issue_selected_window[7]);
    cp_all_banks:
        cover property (issue_accept && issue_bank_valid == 8'hff);
    cp_partial_banks:
        cover property (issue_accept && issue_bank_valid != 8'hff);
    cp_pair_last:
        cover property (issue_accept && issue_pair_last);
    cp_pair_not_last:
        cover property (issue_accept && !issue_pair_last);
    cp_stall_then_accept:
        cover property (issue_valid && !issue_ready ##1 issue_accept);
    cp_same_cycle_replace:
        cover property (issue_accept && pair_accept);
    cp_cross_token_attack:
        cover property (pair_valid && window_valid == 2'b11
            && window_token_tag[0] != window_token_tag[1]
            && protocol_error);
    cp_bad_channel_attack:
        cover property (pair_valid && protocol_error);
`endif
endmodule

`default_nettype wire
