`timescale 1ns/1ps
`default_nettype none

module m106_bounded_bitmap_transpose_scheduler_assertions #(
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
    input logic accepted_event_grace_match,
    input logic accepted_close_grace_match,
    input logic illegal_request,
    input logic [1:0] bank_ready,
    input logic fill_bank,
    input logic drain_bank
);
    logic shadow_loading_q;
    logic [1:0] shadow_expected_beat_q;
    logic [6:0] shadow_key_q;

    always_ff @(posedge clk_core) begin
        if (rst_core || protocol_error) begin
            shadow_loading_q <= 1'b0;
            shadow_expected_beat_q <= '0;
            shadow_key_q <= '0;
        end else if (service_accept) begin
            if (!service_is_event) begin
                if (service_load_beat == 0) begin
                    shadow_loading_q <= 1'b1;
                    shadow_expected_beat_q <= 1;
                    shadow_key_q <= {service_source, service_block};
                end else if (service_load_beat == 2) begin
                    shadow_loading_q <= 1'b0;
                    shadow_expected_beat_q <= '0;
                end else begin
                    shadow_expected_beat_q <= shadow_expected_beat_q + 1'b1;
                end
            end
        end
    end

    ap_event_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_accept == (event_valid && event_ready));
    ap_close_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        window_close_accept == (window_close_valid && window_close_ready));
    ap_service_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        service_accept == (service_valid && service_ready));
    ap_ingress_close_mutual_exclusion: assert property (@(posedge clk_core)
        disable iff (rst_core)
        event_valid && window_close_valid |-> protocol_error);
    ap_fault_is_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error |=> protocol_error);
    ap_fault_quarantines_all_interfaces: assert property (@(posedge clk_core)
        disable iff (rst_core)
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
    ap_load_starts_at_zero: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_accept && !service_is_event && !shadow_loading_q
        |-> service_load_beat == 0);
    ap_load_continues_monotonically: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_accept && !service_is_event && shadow_loading_q
        |-> service_load_beat == shadow_expected_beat_q
            && {service_source, service_block} == shadow_key_q);
    ap_event_only_after_three_loads: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_accept && service_is_event |-> !shadow_loading_q);
    ap_event_grace_blocks_reaccept: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        event_valid && accepted_event_grace_match
        |-> !event_ready && !illegal_request);
    ap_close_grace_blocks_reaccept: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        window_close_valid && accepted_close_grace_match
        |-> !window_close_ready && !illegal_request);

    cp_ping_pong_overlap: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_valid && event_accept && fill_bank != drain_bank);
    cp_event_ii1: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_accept && service_is_event ##1
        service_accept && service_is_event);
    cp_key_turnover_without_idle: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_accept && service_is_event && service_last_for_key ##1
        service_accept && !service_is_event && service_load_beat == 0);
    cp_event_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        event_valid && accepted_event_grace_match && !event_ready);
    cp_close_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        window_close_valid && accepted_close_grace_match
        && !window_close_ready);
    cp_fault: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);
endmodule

`default_nettype wire
