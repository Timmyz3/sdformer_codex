`timescale 1ns/1ps
`default_nettype none

module m106_independent_adversarial_assertions #(
    parameter int ROW_W = 6,
    parameter int BASE_W = 12,
    parameter int CONTEXT_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic event_valid,
    input logic event_ready,
    input logic event_accept,
    input logic window_close_valid,
    input logic window_close_ready,
    input logic window_close_accept,
    input logic service_valid,
    input logic service_ready,
    input logic service_is_event,
    input logic [3:0] service_source,
    input logic [2:0] service_block,
    input logic [1:0] service_load_beat,
    input logic [ROW_W-1:0] service_row_offset,
    input logic [BASE_W-1:0] service_destination_row,
    input logic service_negate,
    input logic service_last_for_key,
    input logic [CONTEXT_W-1:0] service_context,
    input logic service_accept,
    input logic protocol_error,
    input logic illegal_request,
    input logic fill_available,
    input logic drain_active,
    input logic fill_bank,
    input logic drain_bank,
    input logic accepted_event_grace,
    input logic accepted_close_grace,
    input logic accepted_event_grace_match,
    input logic accepted_close_grace_match
);
    ap_event_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept == (event_valid && event_ready));

    ap_close_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        window_close_accept == (window_close_valid
                                  && window_close_ready));

    ap_service_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        service_accept == (service_valid && service_ready));

    ap_ping_pong_banks_are_distinct: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        drain_active && fill_available |-> fill_bank != drain_bank);

    ap_service_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        service_valid && !service_ready
        |=> protocol_error
            || (service_valid
                && $stable({service_is_event, service_source, service_block,
                            service_load_beat, service_row_offset,
                            service_destination_row, service_negate,
                            service_last_for_key, service_context})));

    ap_fault_is_sticky_until_reset: assert property (@(posedge clk_core)
        disable iff (rst_core)
        protocol_error |=> protocol_error);

    ap_fault_quarantines_every_interface: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !event_ready && !window_close_ready
            && !service_valid && !service_accept);

    // These four assertions encode the literal frozen accepted-valid policy:
    // an exact held request is not reaccepted, while any mutation before a
    // sampled valid-low edge enters same-cycle fail-closed quarantine.
    ap_exact_held_event_is_not_reaccepted: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_event_grace && event_valid && accepted_event_grace_match
        |-> !event_ready && !event_accept && !illegal_request);

    ap_exact_held_close_is_not_reaccepted: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && accepted_close_grace_match
        |-> !window_close_ready && !window_close_accept
            && !illegal_request);

    ap_held_event_mutation_fails_closed: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_event_grace && event_valid && !accepted_event_grace_match
        |-> protocol_error && illegal_request && !event_ready);

    ap_held_close_mutation_fails_closed: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && !accepted_close_grace_match
        |-> protocol_error && illegal_request && !window_close_ready);

    cp_ping_pong_overlap: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        drain_active && fill_available && fill_bank != drain_bank
            && service_valid && event_accept);

    cp_service_stall: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_valid && !service_ready);

    cp_event_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        accepted_event_grace && event_valid && accepted_event_grace_match
            && !event_ready);

    cp_close_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && accepted_close_grace_match && !window_close_ready);

    cp_fault: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);
endmodule

`default_nettype wire
