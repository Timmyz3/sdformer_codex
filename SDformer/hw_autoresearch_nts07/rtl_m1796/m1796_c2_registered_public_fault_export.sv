`timescale 1ns/1ps
`default_nettype none

// Synthesizable M1796 public-fault boundary.  It is split out so directed
// verification can attack every event input without hierarchical writes.
module m1796_c2_registered_public_fault_export (
    input  logic clk_core,
    input  logic rst_core,
    input  logic core_fault_sample_enable,
    input  logic core_fault_event_raw,
    input  logic adapter_fault_sample_enable,
    input  logic adapter_fault_event_raw,
    input  logic core_req_valid,
    input  logic core_req_accept,
    input  logic adapter_req_accept,
    input  logic core_rsp_valid,
    input  logic core_rsp_accept,
    input  logic adapter_rsp_accept,
    output logic protocol_error
);
    logic core_fault_event, adapter_fault_event;
    logic req_accept_mismatch, rsp_accept_mismatch;

    // These validity terms are functional ownership qualifiers, not X
    // coercion.  A payload has no protocol meaning while its valid is zero.
    assign core_fault_event = core_fault_sample_enable
        && core_fault_event_raw;
    assign adapter_fault_event = adapter_fault_sample_enable
        && adapter_fault_event_raw;
    assign req_accept_mismatch = core_req_valid
        && (core_req_accept != adapter_req_accept);
    assign rsp_accept_mismatch = core_rsp_valid
        && (core_rsp_accept != adapter_rsp_accept);

    always_ff @(posedge clk_core) begin
        if (rst_core)
            protocol_error <= 1'b0;
        else if (core_fault_event || adapter_fault_event
                || req_accept_mismatch || rsp_accept_mismatch)
            protocol_error <= 1'b1;
    end
endmodule

`default_nettype wire
