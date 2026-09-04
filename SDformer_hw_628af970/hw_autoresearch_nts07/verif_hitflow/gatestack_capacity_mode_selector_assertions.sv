`timescale 1ns/1ps
`default_nettype none

module gatestack_capacity_mode_selector_assertions #(
    parameter int TAG_W = 32,
    parameter int SIZE_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic response_valid,
    input logic response_ready,
    input logic [TAG_W-1:0] response_tag,
    input logic response_is_csr,
    input logic [1:0] response_reason,
    input logic [SIZE_W-1:0] response_csr_bits
);

    property p_response_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        response_valid && !response_ready |=> response_valid &&
            $stable({response_tag, response_is_csr,
                     response_reason, response_csr_bits});
    endproperty

    property p_csr_reason_consistent;
        @(posedge clk_core) disable iff (rst_core)
        response_valid |-> (response_is_csr == (response_reason == 2'd0));
    endproperty

    property p_reason_known;
        @(posedge clk_core) disable iff (rst_core)
        response_valid |-> (response_reason inside {2'd0, 2'd1, 2'd2});
    endproperty

    assert property (p_response_stable_under_stall);
    assert property (p_csr_reason_consistent);
    assert property (p_reason_known);

endmodule

`default_nettype wire
