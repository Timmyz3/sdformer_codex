`timescale 1ns/1ps
`default_nettype none

module gatestack_context_abort_controller_assertions #(
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic group_accept_pulse,
    input logic group_active,
    input logic abort_done_valid,
    input logic abort_done_ready,
    input logic [TAG_W-1:0] abort_done_tag,
    input logic abort_done_error,
    input logic fabric_reset_pulse,
    input logic admission_blocked,
    input logic protocol_error
);
    property p_no_double_group;
        @(posedge clk_core) disable iff (rst_core)
        group_active |-> !group_accept_pulse;
    endproperty
    assert property (p_no_double_group);

    property p_abort_stable;
        @(posedge clk_core) disable iff (rst_core)
        abort_done_valid && !abort_done_ready |=> abort_done_valid &&
            $stable({abort_done_tag, abort_done_error});
    endproperty
    assert property (p_abort_stable);

    property p_abort_blocks_admission;
        @(posedge clk_core) disable iff (rst_core)
        abort_done_valid |-> admission_blocked;
    endproperty
    assert property (p_abort_blocks_admission);

    property p_reset_is_pulse;
        @(posedge clk_core) disable iff (rst_core)
        fabric_reset_pulse |=> !fabric_reset_pulse;
    endproperty
    assert property (p_reset_is_pulse);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
