`timescale 1ns/1ps
`default_nettype none

module m119_independent_assertions #(
    parameter int DELTA_BITS = 1824
) (
    input logic clk_core,
    input logic rst_core,
    input logic service_valid,
    input logic service_ready,
    input logic service_is_event,
    input logic [3:0] service_source,
    input logic [2:0] service_block,
    input logic [1:0] service_load_beat,
    input logic [8:0] service_row_offset,
    input logic service_negate,
    input logic service_last_for_key,
    input logic service_accept,
    input logic weight_rd_en,
    input logic [6:0] weight_rd_key,
    input logic [1:0] weight_rd_beat,
    input logic update_valid,
    input logic update_ready,
    input logic [2:0] update_block,
    input logic [8:0] update_row,
    input logic [DELTA_BITS-1:0] update_delta,
    input logic update_accept,
    input logic payload_active,
    input logic tail_bypass_available,
    input logic protocol_error
);
`ifdef SVA_RUNTIME_ENABLED
    ap_service_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        service_accept == (service_valid && service_ready));
    ap_update_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept == (update_valid && update_ready));
    ap_read_exactly_once_per_accepted_load: assert property (
        @(posedge clk_core) disable iff (rst_core)
        weight_rd_en == (service_accept && !service_is_event));
    ap_read_identity_matches_load: assert property (
        @(posedge clk_core) disable iff (rst_core)
        weight_rd_en |-> weight_rd_key == {service_source, service_block}
            && weight_rd_beat == service_load_beat
            && weight_rd_beat < 3);
    ap_update_stable_on_stall: assert property (
        @(posedge clk_core) disable iff (rst_core)
        update_valid && !update_ready |=> update_valid
            && $stable({update_block, update_row, update_delta}));
    ap_fault_quarantines_only_new_service: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !service_ready && !service_accept && !weight_rd_en);
    ap_older_update_can_drain_during_fault: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && update_valid && update_ready |-> update_accept);
    ap_tail_event_creates_exact_update: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        service_accept && service_is_event && tail_bypass_available
        |=> update_valid
            && update_block == $past(service_block)
            && update_row == $past(service_row_offset));

    cp_fixed_latency_tail: cover property (
        @(posedge clk_core) disable iff (rst_core)
        service_accept && !service_is_event && service_load_beat == 0
        ##1 service_accept && !service_is_event && service_load_beat == 1
        ##1 service_accept && !service_is_event && service_load_beat == 2
        ##1 service_accept && service_is_event && tail_bypass_available);
    cp_event_input_backpressure: cover property (
        @(posedge clk_core) disable iff (rst_core)
        service_valid && service_is_event && !service_ready [*3]
        ##1 service_accept && service_is_event && update_accept);
    cp_output_backpressure: cover property (
        @(posedge clk_core) disable iff (rst_core)
        update_valid && !update_ready [*3]
        ##1 update_valid && update_ready);
    cp_fault_with_older_update: cover property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && update_valid && !update_ready [*2]
        ##1 protocol_error && update_valid && update_ready && update_accept);
    cp_signed_boundary_update: cover property (
        @(posedge clk_core) disable iff (rst_core)
        update_accept
            && $signed(update_delta[0 * 19 +: 19]) == 19'sd128
            && $signed(update_delta[1 * 19 +: 19]) == -19'sd127);
`endif
endmodule

`default_nettype wire
