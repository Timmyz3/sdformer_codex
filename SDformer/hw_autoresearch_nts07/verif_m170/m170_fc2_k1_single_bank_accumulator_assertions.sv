`timescale 1ns/1ps
`default_nettype none

module m170_fc2_k1_single_bank_accumulator_assertions #(
    parameter int LANES = 96,
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic rst_core,
    input logic issue_valid,
    input logic issue_ready,
    input logic issue_accept,
    input logic issue_slot_valid,
    input logic result_valid,
    input logic result_ready,
    input logic result_accept,
    input logic [TAG_BITS-1:0] result_tag,
    input logic result_last,
    input logic [2:0] result_source_count,
    input logic [7:0] result_bank_mask,
    input logic signed [23:0] result_accumulator [0:LANES-1],
    input logic [LANES-1:0] accepted_weight_active_mask,
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
    ap_accept_nonempty:
        assert property (issue_accept |-> issue_slot_valid);
    ap_activity_zero_without_accept:
        assert property (!issue_accept |-> accepted_weight_active_mask == '0);
    ap_activity_full_on_accept:
        assert property (issue_accept |-> &accepted_weight_active_mask);
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

    cp_accept:
        cover property (issue_accept);
    cp_same_cycle_result_replace:
        cover property (result_accept && issue_accept);
    cp_stall_then_accept:
        cover property (result_valid && !result_ready ##1 result_accept);
    cp_overflow_preserves_pending_result:
        cover property (numeric_overflow && result_valid);
    cp_protocol_fault_preserves_pending_result:
        cover property (protocol_error && result_valid);
`endif
endmodule

`default_nettype wire
