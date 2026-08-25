`timescale 1ns/1ps
`default_nettype none

module qfit_atlif_rank3_exact96_core_assertions #(
    parameter int TAG_W = 48
) (
    input logic clk_core,
    input logic rst_core,
    input logic request_valid,
    input logic request_ready,
    input logic request_legal,
    input logic result_valid,
    input logic result_ready,
    input logic [TAG_W-1:0] result_tag,
    input logic [2:0] result_beat,
    input logic [(32*24)-1:0] result_values,
    input logic done,
    input logic [TAG_W-1:0] done_tag,
    input logic protocol_error,
    input logic busy,
    input logic arithmetic_active,
    input logic stage_select,
    input logic [2:0] phase_cycle,
    input logic [95:0] multiplier_active_mask
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_accepted_request_is_legal: assert property (
        request_valid && request_ready |-> request_legal);
    ap_no_ready_after_error: assert property (protocol_error |-> !request_ready);
    ap_result_stable_under_stall: assert property (
        result_valid && !result_ready
        |=> result_valid && $stable({result_tag, result_beat, result_values}));
    ap_done_has_matching_last_retirement: assert property (
        done |-> $past(result_valid && result_ready && result_beat == 4)
                 && done_tag == $past(result_tag));
    ap_arithmetic_uses_all_slots: assert property (
        arithmetic_active |-> &multiplier_active_mask);
    ap_no_inactive_multiplier_slots: assert property (
        !arithmetic_active |-> multiplier_active_mask == '0);
    ap_stage1_always_computes: assert property (
        busy && !stage_select |-> arithmetic_active);
    ap_stage1_advances: assert property (
        arithmetic_active && !stage_select && phase_cycle < 4
        |=> !stage_select && phase_cycle == $past(phase_cycle)+1'b1);
    ap_stage_boundary_no_bubble: assert property (
        arithmetic_active && !stage_select && phase_cycle == 4
        |=> arithmetic_active && stage_select && phase_cycle == 0);
    ap_stage2_advances_if_not_stalled: assert property (
        arithmetic_active && stage_select && phase_cycle < 4
        |=> phase_cycle == $past(phase_cycle)+1'b1);
    ap_stage2_stalls_with_result: assert property (
        busy && stage_select && result_valid && !result_ready
        |=> $stable(phase_cycle));

    cp_stage_boundary: cover property (
        arithmetic_active && !stage_select && phase_cycle == 4
        ##1 arithmetic_active && stage_select && phase_cycle == 0);
    cp_output_stall: cover property (result_valid && !result_ready ##1
                                     result_valid && !result_ready ##1
                                     result_valid && result_ready);
    cp_done: cover property (done);
`endif
endmodule

bind qfit_atlif_rank3_exact96_core
    qfit_atlif_rank3_exact96_core_assertions #(.TAG_W(TAG_W))
    m27_sva (.*);

`default_nettype wire
