`timescale 1ns/1ps
`default_nettype none

module hitflow_dptme_assertions #(
    parameter int LANES = 32,
    parameter int SLOTS = 10,
    parameter int ACC_W = 24,
    parameter int TAG_W = 48
) (
    input logic                           clk_core,
    input logic                           rst_core,
    input logic                           step_valid,
    input logic                           step_ready,
    input logic                           step_last,
    input logic                           protocol_error,
    input logic                           out_valid,
    input logic                           out_ready,
    input logic [(SLOTS*LANES)-1:0]       out_events,
    input logic [(SLOTS*LANES*ACC_W)-1:0] out_hidden,
    input logic [SLOTS-1:0]               out_slot_valid,
    input logic [TAG_W-1:0]               out_tag
);

    property p_output_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            out_valid && !out_ready |=> out_valid &&
            $stable(out_events) && $stable(out_hidden) &&
            $stable(out_slot_valid) && $stable(out_tag);
    endproperty

    property p_protocol_error_is_rejected;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |-> step_valid && !step_ready;
    endproperty

    property p_accepted_step_has_no_protocol_error;
        @(posedge clk_core) disable iff (rst_core)
            step_valid && step_ready |-> !protocol_error;
    endproperty

    property p_last_step_produces_output;
        @(posedge clk_core) disable iff (rst_core)
            step_valid && step_ready && step_last |=> out_valid;
    endproperty

    assert property (p_output_stable_under_backpressure);
    assert property (p_protocol_error_is_rejected);
    assert property (p_accepted_step_has_no_protocol_error);
    assert property (p_last_step_produces_output);

endmodule

`default_nettype wire
