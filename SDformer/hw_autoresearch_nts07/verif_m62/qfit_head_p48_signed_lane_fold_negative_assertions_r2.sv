`timescale 1ns/1ps
`default_nettype none

module qfit_head_p48_signed_lane_fold_negative_assertions_r2 #(
    parameter int PIXELS = 48,
    parameter int OUTPUTS = 2,
    parameter int SOURCE_SLOTS = 8,
    parameter int ACC_W = 13,
    parameter int TAG_W = 48,
    parameter int LANES = PIXELS * OUTPUTS
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic command_accept,
    input logic event_valid,
    input logic event_ready,
    input logic event_accept,
    input logic output_valid,
    input logic output_ready,
    input logic output_accept,
    input logic [TAG_W-1:0] output_tag,
    input logic [LANES*ACC_W-1:0] output_acc,
    input logic [15:0] output_source_issues,
    input logic [15:0] output_signed_events,
    input logic protocol_error,
    input logic mask_overlap,
    input logic invalid_slot_mask,
    input logic reserved_negative_weight,
    input logic event_has_source,
    input logic [15:0] event_signed_count,
    input logic accumulator_overflow,
    input logic legal_case_active,
    input logic [3:0] legal_case_id,
    input logic attack_case_active,
    input logic [3:0] attack_case_id
);
    logic malformed_event;

    assign malformed_event = !event_has_source || event_signed_count == 0
        || mask_overlap || invalid_slot_mask || reserved_negative_weight
        || accumulator_overflow;

    initial $display("M62_R2_NEGATIVE_ASSERTION_MODULE_ACTIVE=1");

    ap_command_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core)
        command_accept == (command_valid && command_ready));
    ap_event_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept == (event_valid && event_ready));
    ap_output_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_acc,
                                     output_source_issues,
                                     output_signed_events}));

    // The frozen RTL accepts an active-group malformed event, then enters its
    // sticky fail-closed state.  This is deliberately not a pre-accept reject.
    ap_accepted_malformed_faults_next: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && malformed_event |=> protocol_error);
    ap_directed_attack_is_accepted_malformed: assert property (
        @(posedge clk_core) disable iff (rst_core)
        event_accept && attack_case_active |-> malformed_event);
    ap_directed_legal_is_not_malformed: assert property (
        @(posedge clk_core) disable iff (rst_core)
        event_accept && legal_case_active |-> !malformed_event);
    ap_directed_legal_does_not_fault: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active |=> !protocol_error);
    ap_fault_is_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error |=> protocol_error);
    ap_fault_closes_all_interfaces: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error
        |-> !command_ready && !event_ready && !output_valid
            && !command_accept && !event_accept && !output_accept);

    ap_overlap_case_cause: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 0
        |-> mask_overlap && event_signed_count != 0);
    ap_invalid_slot_case_cause: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 1
        |-> invalid_slot_mask && event_signed_count != 0);
    ap_reserved_weight_case_cause: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 2
        |-> reserved_negative_weight && event_signed_count != 0);
    ap_no_signed_work_case_cause: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 3
        |-> event_has_source && event_signed_count == 0);
    ap_overflow_case_cause: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 4
        |-> accumulator_overflow && event_signed_count != 0);

    cp_legal_full8_0: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active && legal_case_id == 0);
    cp_legal_full8_1: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active && legal_case_id == 1);
    cp_legal_full8_2: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active && legal_case_id == 2);
    cp_legal_full8_3: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active && legal_case_id == 3);
    cp_legal_full8_4: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active && legal_case_id == 4);
    cp_legal_full8_5: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && legal_case_active && legal_case_id == 5);
    cp_near_positive_limit: cover property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid && legal_case_active && legal_case_id == 3
        && $signed(output_acc[0 +: ACC_W]) == 13'sd4094);
    cp_near_negative_limit: cover property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid && legal_case_active && legal_case_id == 4
        && $signed(output_acc[0 +: ACC_W]) == -13'sd4095);
    cp_five_cycle_stall_case: cover property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid && !output_ready && legal_case_active
        && legal_case_id == 5);

    cp_attack_overlap: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 0
        && mask_overlap);
    cp_attack_invalid_slot: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 1
        && invalid_slot_mask);
    cp_attack_reserved_negative_128: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 2
        && reserved_negative_weight);
    cp_attack_no_signed_work: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 3
        && event_has_source && event_signed_count == 0);
    cp_attack_accumulator_overflow: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept && attack_case_active && attack_case_id == 4
        && accumulator_overflow);
endmodule

`default_nettype wire
