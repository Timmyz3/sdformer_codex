`timescale 1ns/1ps
`default_nettype none

module m189_fc2_k7_bank_compacting_accumulator_assertions #(
    parameter int LANES = 96,
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic rst_core,
    input logic issue_valid,
    input logic issue_ready,
    input logic issue_accept,
    input logic [7:0] issue_bank_valid,
    input logic result_valid,
    input logic result_ready,
    input logic result_accept,
    input logic [TAG_BITS-1:0] result_tag,
    input logic result_last,
    input logic [3:0] result_source_count,
    input logic [7:0] result_bank_mask,
    input logic signed [23:0] result_accumulator [0:LANES-1],
    input logic [8*LANES-1:0] accepted_weight_bank_active_mask,
    input logic [7*LANES-1:0] accepted_compacted_lane_active_mask,
    input logic protocol_error,
    input logic numeric_overflow,
    input logic busy
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_issue_handshake:
        assert property (issue_accept == (issue_valid && issue_ready));
    ap_result_handshake:
        assert property (result_accept == (result_valid && result_ready));
    ap_busy_identity:
        assert property (busy == result_valid);
    ap_protocol_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_overflow_sticky:
        assert property (numeric_overflow |=> numeric_overflow);
    ap_fault_closes_issue:
        assert property ((protocol_error || numeric_overflow) |=> !issue_ready);
    ap_accept_legal_k7:
        assert property (issue_accept |->
            ($countones(issue_bank_valid) >= 1
             && $countones(issue_bank_valid) <= 7));
    ap_bank_activity_zero_without_accept:
        assert property (!issue_accept |->
            accepted_weight_bank_active_mask == '0);
    ap_compacted_activity_zero_without_accept:
        assert property (!issue_accept |->
            accepted_compacted_lane_active_mask == '0);
    ap_hold_header_on_stall:
        assert property (result_valid && !result_ready |=>
            $stable({result_valid, result_tag, result_last,
                     result_source_count, result_bank_mask}));
    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_hold_lane
            ap_hold_accumulator_on_stall:
                assert property (result_valid && !result_ready |=>
                    $stable(result_accumulator[lane]));
        end
    endgenerate

    cp_one_source:   cover property (issue_accept && $countones(issue_bank_valid) == 1);
    cp_two_source:   cover property (issue_accept && $countones(issue_bank_valid) == 2);
    cp_three_source: cover property (issue_accept && $countones(issue_bank_valid) == 3);
    cp_four_source:  cover property (issue_accept && $countones(issue_bank_valid) == 4);
    cp_five_source:  cover property (issue_accept && $countones(issue_bank_valid) == 5);
    cp_six_source:   cover property (issue_accept && $countones(issue_bank_valid) == 6);
    cp_seven_source: cover property (issue_accept && $countones(issue_bank_valid) == 7);
    cp_hole_at_low_bank: cover property (issue_accept && issue_bank_valid == 8'hfe);
    cp_hole_at_high_bank: cover property (issue_accept && issue_bank_valid == 8'h7f);
    cp_nonprefix_sparse_mask: cover property (issue_accept && issue_bank_valid == 8'h81);
    cp_same_cycle_result_replace: cover property (result_accept && issue_accept);
    cp_stall_then_accept:
        cover property (result_valid && !result_ready ##1 result_accept);
    cp_overflow_preserves_pending_result:
        cover property (numeric_overflow && result_valid);
    cp_empty_mask_attack: cover property (issue_valid && issue_bank_valid == 8'h00);
    cp_full_mask_attack: cover property (issue_valid && issue_bank_valid == 8'hff);
`endif
endmodule

`default_nettype wire
