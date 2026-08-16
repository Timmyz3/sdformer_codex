`timescale 1ns/1ps
`default_nettype none

module gatestack_backend_done_guard_assertions #(
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic checked_done_valid,
    input logic checked_done_ready,
    input logic [TAG_W-1:0] checked_done_execution_tag,
    input logic checked_done_error,
    input logic protocol_error
);
    property p_done_stable;
        @(posedge clk_core) disable iff (rst_core)
        checked_done_valid && !checked_done_ready |=> checked_done_valid &&
            $stable({checked_done_execution_tag, checked_done_error});
    endproperty
    assert property (p_done_stable);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
