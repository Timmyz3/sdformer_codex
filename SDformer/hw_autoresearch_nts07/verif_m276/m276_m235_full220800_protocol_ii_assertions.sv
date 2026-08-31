`timescale 1ns/1ps
`default_nettype none

module m276_m235_full220800_protocol_ii_assertions #(
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic rst_core,
    input logic request_valid,
    input logic request_ready,
    input logic request_accept,
    input logic [TAG_BITS-1:0] request_tag,
    input logic [21:0] variance_plus_epsilon_uq6p16,
    input logic signed [17:0] mean_sq3p14,
    input logic signed [15:0] gamma_sq1p14,
    input logic signed [15:0] beta_sq1p14,
    input logic result_valid,
    input logic result_ready,
    input logic result_accept,
    input logic protocol_error
);
    ap_request_no_accept_while_backpressured: assert property (
        @(posedge clk_core) disable iff (rst_core)
        request_valid && !request_ready && !protocol_error
        |-> !request_accept);

    ap_request_payload_stable_while_backpressured: assert property (
        @(posedge clk_core) disable iff (rst_core)
        request_valid && !request_ready && !protocol_error
        |=> protocol_error || request_accept ||
            (request_valid && $stable({request_tag,
                                      variance_plus_epsilon_uq6p16,
                                      mean_sq3p14,
                                      gamma_sq1p14,
                                      beta_sq1p14})));

    // A legal request held behind a retiring result must be accepted on the
    // next edge.  This is the no-driver-bubble service-boundary property.
    ap_held_request_accepts_after_result_retire: assert property (
        @(posedge clk_core) disable iff (rst_core)
        result_accept && request_valid && !protocol_error
        |=> request_accept);

    cp_request_backpressure: cover property (
        @(posedge clk_core) disable iff (rst_core)
        request_valid && !request_ready && !protocol_error);

    cp_held_request_turnaround: cover property (
        @(posedge clk_core) disable iff (rst_core)
        result_accept && request_valid && !protocol_error
        ##1 request_accept);

    cp_result_backpressure: cover property (
        @(posedge clk_core) disable iff (rst_core)
        result_valid && !result_ready && !protocol_error);

    cp_illegal_pending_result_atomic: cover property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && result_ready && !request_ready && !result_valid);
endmodule

`default_nettype wire
