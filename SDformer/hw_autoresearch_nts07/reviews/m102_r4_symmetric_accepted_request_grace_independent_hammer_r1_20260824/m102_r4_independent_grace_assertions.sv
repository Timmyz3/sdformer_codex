`timescale 1ns/1ps
`default_nettype none

module m102_r4_independent_grace_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic request_valid,
    input logic request_ready,
    input logic accepted_grace_match,
    input logic request_semantically_valid,
    input logic request_violation,
    input logic request_fault,
    input logic m82_beat_accept,
    input logic m82_output_valid,
    input logic m82_output_accept,
    input logic output_valid,
    input logic output_ready,
    input logic output_accept,
    input logic protocol_error,
    input logic phase_load_ready,
    input logic grace_hold_event,
    input logic glitch_return_event,
    input logic mutation_event,
    input logic phase_reload_event,
    input logic reset_recovery_event
);
    ap_output_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_accept == (output_valid && output_ready));

    ap_exact_grace_is_not_reaccepted_or_faulted: assert property (
        @(posedge clk_core) disable iff (rst_core || request_fault)
        request_valid && accepted_grace_match
        |-> !request_semantically_valid && !request_violation
            && !request_ready && !m82_beat_accept && !protocol_error);

    ap_named_grace_hold: assert property (@(posedge clk_core)
        disable iff (rst_core)
        grace_hold_event |-> request_valid && accepted_grace_match
            && !request_ready && !m82_beat_accept && !protocol_error
            && output_valid);

    ap_between_edge_low_high_is_unobservable: assert property (
        @(posedge clk_core) disable iff (rst_core)
        glitch_return_event |-> request_valid && accepted_grace_match
            && !request_violation && !protocol_error && !m82_beat_accept);

    ap_changed_identity_quarantines_same_cycle: assert property (
        @(posedge clk_core) disable iff (rst_core)
        mutation_event |-> request_valid && request_violation
            && protocol_error && !request_ready && !output_valid
            && !output_accept && m82_output_valid && !m82_output_accept);

    ap_fault_is_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core)
        request_fault |=> request_fault);

    ap_registered_fault_quarantines_output: assert property (
        @(posedge clk_core) disable iff (rst_core)
        request_fault |-> protocol_error && !request_ready
            && !output_valid && !output_accept);

    ap_faulted_phase_reload_is_blocked: assert property (
        @(posedge clk_core) disable iff (rst_core)
        phase_reload_event |-> request_fault && protocol_error
            && !phase_load_ready && m82_output_valid);

    ap_named_reset_recovery: assert property (@(posedge clk_core)
        reset_recovery_event |-> !rst_core && !request_fault
            && !protocol_error && !m82_output_valid
            && !output_valid && !output_accept);

    cp_grace_hold: cover property (@(posedge clk_core)
        grace_hold_event && accepted_grace_match && output_valid);
    cp_glitch_return: cover property (@(posedge clk_core)
        glitch_return_event && accepted_grace_match && !protocol_error);
    cp_identity_mutation: cover property (@(posedge clk_core)
        mutation_event && request_violation && protocol_error
        && m82_output_valid && !output_accept);
    cp_fault_sticky: cover property (@(posedge clk_core)
        request_fault && protocol_error && m82_output_valid);
    cp_phase_reload_block: cover property (@(posedge clk_core)
        phase_reload_event && !phase_load_ready && request_fault);
    cp_reset_recovery: cover property (@(posedge clk_core)
        reset_recovery_event && !request_fault && !protocol_error);
endmodule

`default_nettype wire
