`timescale 1ns/1ps
`default_nettype none

// Logic-only Synopsys top for the M7 ATLIF slice.  The 3,840-bit L16 hidden
// vector remains internal because it is an implementation state/debug view,
// not an accelerator interface.  Parameter/activation SRAMs and the output
// serializer are deliberately not hidden by this wrapper and remain a paper-
// PPA admission requirement.
module hitflow_dptme_paper_top #(
    parameter int LANES = 16,
    parameter int SLOTS = 10,
    parameter int PACK_GROUPS = 5,
    parameter int X_W = 8,
    parameter int W_W = 8,
    parameter int ACC_W = 24,
    parameter int TAG_W = 48
) (
    input  logic                               clk_core,
    input  logic                               rst_core,
    input  logic                               step_valid,
    output logic                               step_ready,
    input  logic                               mode_t2,
    input  logic                               step_first,
    input  logic                               step_last,
    input  logic [PACK_GROUPS-1:0]             group_valid,
    input  logic [(PACK_GROUPS*LANES*X_W)-1:0] x_groups,
    input  logic [(SLOTS*W_W)-1:0]             weight_slots,
    input  logic [(SLOTS*ACC_W)-1:0]           bias_slots,
    input  logic [(SLOTS*ACC_W)-1:0]           threshold_slots,
    input  logic [TAG_W-1:0]                   step_tag,
    output logic                               out_valid,
    input  logic                               out_ready,
    output logic [(SLOTS*LANES)-1:0]           out_events,
    output logic [SLOTS-1:0]                   out_slot_valid,
    output logic [TAG_W-1:0]                   out_tag,
    output logic                               protocol_error
);
    logic [(SLOTS*LANES*ACC_W)-1:0] hidden_state_debug;

    hitflow_dptme_array #(
        .LANES(LANES),
        .SLOTS(SLOTS),
        .PACK_GROUPS(PACK_GROUPS),
        .X_W(X_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .TAG_W(TAG_W)
    ) u_dptme (
        .clk_core,
        .rst_core,
        .step_valid,
        .step_ready,
        .mode_t2,
        .step_first,
        .step_last,
        .group_valid,
        .x_groups,
        .weight_slots,
        .bias_slots,
        .threshold_slots,
        .step_tag,
        .out_valid,
        .out_ready,
        .out_events,
        .out_hidden(hidden_state_debug),
        .out_slot_valid,
        .out_tag,
        .protocol_error
    );
endmodule

`default_nettype wire
