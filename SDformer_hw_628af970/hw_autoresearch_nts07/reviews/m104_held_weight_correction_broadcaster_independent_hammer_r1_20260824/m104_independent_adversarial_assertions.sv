`timescale 1ns/1ps
`default_nettype none

module m104_independent_adversarial_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic load_ready,
    input logic event_valid,
    input logic event_ready,
    input logic event_accept,
    input logic event_last_for_key,
    input logic output_valid,
    input logic output_ready,
    input logic output_accept,
    input logic [31:0] output_tag,
    input logic [1151:0] output_values,
    input logic held_valid,
    input logic collecting,
    input logic protocol_error,
    input logic illegal_request,
    input logic request_fault,
    input logic output_valid_q
);
    ap_same_cycle_illegal_quarantines_old_output: assert property (
        @(posedge clk_core) disable iff (rst_core)
        illegal_request && output_valid_q
        |-> protocol_error && !load_ready && !event_ready
            && !output_valid && !output_accept);

    ap_ready_release_does_not_retire_on_illegal: assert property (
        @(posedge clk_core) disable iff (rst_core)
        illegal_request && output_valid_q && output_ready
        |-> !output_valid && !output_accept);

    ap_fault_is_reset_only: assert property (
        @(posedge clk_core) disable iff (rst_core)
        request_fault |=> request_fault);

    ap_reset_clears_all_live_state: assert property (
        @(posedge clk_core)
        rst_core |=> !request_fault && !output_valid_q
                    && !held_valid && !collecting);

    ap_legal_stall_holds_payload: assert property (
        @(posedge clk_core) disable iff (rst_core)
        output_valid && !output_ready && !illegal_request
        |=> protocol_error
            || (output_valid && $stable({output_tag, output_values})));

    ap_unaccepted_last_does_not_release_key: assert property (
        @(posedge clk_core) disable iff (rst_core)
        event_valid && event_last_for_key && !event_ready
            && held_valid && !protocol_error
        |=> held_valid || protocol_error);

    ap_accepted_last_releases_key: assert property (
        @(posedge clk_core) disable iff (rst_core)
        event_accept && event_last_for_key |=> !held_valid);

    cp_illegal_plus_ready_release: cover property (
        @(posedge clk_core) disable iff (rst_core)
        illegal_request && output_valid_q && output_ready
            && !output_valid && !output_accept);
    cp_legal_stalled_last_then_release: cover property (
        @(posedge clk_core) disable iff (rst_core)
        event_valid && event_last_for_key && !event_ready ##1
        event_accept && event_last_for_key);
    cp_reset_recovery: cover property (
        @(posedge clk_core) request_fault ##1 rst_core ##1 !request_fault);
endmodule

`default_nettype wire
