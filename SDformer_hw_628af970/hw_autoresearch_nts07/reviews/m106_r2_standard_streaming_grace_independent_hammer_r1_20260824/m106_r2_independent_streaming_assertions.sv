`timescale 1ns/1ps
`default_nettype none

module m106_r2_independent_streaming_assertions (
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
    input logic [5:0] service_row_offset,
    input logic [11:0] service_destination_row,
    input logic service_negate,
    input logic service_last_for_key,
    input logic [15:0] service_context,
    input logic service_accept,
    input logic protocol_error,
    input logic request_collision,
    input logic illegal_request,
    input logic event_semantically_valid,
    input logic close_semantically_valid,
    input logic accepted_event_grace,
    input logic accepted_close_grace,
    input logic accepted_event_grace_match,
    input logic accepted_close_grace_match,
    input logic fill_available,
    input logic drain_active,
    input logic fill_bank,
    input logic drain_bank
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

    ap_exact_held_event_grace_no_reaccept: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_event_grace && event_valid && accepted_event_grace_match
        |-> !event_ready && !event_accept && !illegal_request);

    ap_exact_held_close_grace_no_cross_bank_reaccept: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && accepted_close_grace_match
        |-> !window_close_ready && !window_close_accept
            && !illegal_request);

    // R2 standard streaming rule: a changed, semantically legal payload is a
    // new transaction and may be accepted without an intervening valid-low.
    ap_changed_legal_event_streams: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_event_grace && event_valid && !accepted_event_grace_match
            && event_semantically_valid && !window_close_valid
        |-> event_ready && event_accept && !illegal_request);

    ap_changed_legal_close_streams: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && !accepted_close_grace_match && close_semantically_valid
            && !event_valid
        |-> window_close_ready && window_close_accept && !illegal_request);

    ap_illegal_event_fails_closed: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        event_valid && !event_semantically_valid
            && !accepted_event_grace_match
        |-> protocol_error && illegal_request && !event_ready);

    ap_illegal_close_fails_closed: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        window_close_valid && !close_semantically_valid
            && !accepted_close_grace_match
        |-> protocol_error && illegal_request && !window_close_ready);

    ap_collision_fails_closed: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        request_collision |-> protocol_error && illegal_request
            && !event_ready && !window_close_ready && !service_valid);

    ap_fault_sticky_until_reset: assert property (@(posedge clk_core)
        disable iff (rst_core)
        protocol_error |=> protocol_error);

    ap_fault_quarantines_all_interfaces: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !event_ready && !window_close_ready
            && !service_valid && !service_accept);

    ap_service_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        service_valid && !service_ready
        |=> protocol_error
            || (service_valid
                && $stable({service_is_event, service_source, service_block,
                            service_load_beat, service_row_offset,
                            service_destination_row, service_negate,
                            service_last_for_key, service_context})));

    ap_fill_and_drain_banks_distinct: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fill_available && drain_active |-> fill_bank != drain_bank);

    cp_event_stream_ii1: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        event_accept ##1 event_accept);

    cp_changed_legal_close: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && !accepted_close_grace_match && close_semantically_valid
            && window_close_accept);

    cp_exact_close_cross_bank_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        accepted_close_grace && window_close_valid
            && accepted_close_grace_match && fill_available
            && !window_close_accept);

    cp_stall: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_valid && !service_ready);

    cp_fault: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);
endmodule

`default_nettype wire
