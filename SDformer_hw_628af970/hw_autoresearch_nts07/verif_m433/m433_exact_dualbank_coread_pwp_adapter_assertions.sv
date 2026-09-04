`timescale 1ns/1ps
`default_nettype none

module m433_exact_dualbank_coread_pwp_adapter_assertions #(
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic reset_n,
    input logic config_reload,
    input logic request_valid,
    input logic request_ready,
    input logic request_accept,
    input logic [TAG_BITS-1:0] low_tag,
    input logic low_tile,
    input logic [4:0] low_center_id,
    input logic [2:0] low_output_block,
    input logic request_narrow,
    input logic [767:0] low_data,
    input logic [TAG_BITS-1:0] high_tag,
    input logic high_tile,
    input logic [4:0] high_center_id,
    input logic [2:0] high_output_block,
    input logic [511:0] high_data,
    input logic contribution_valid,
    input logic contribution_ready,
    input logic contribution_accept,
    input logic [TAG_BITS-1:0] contribution_tag,
    input logic contribution_tile,
    input logic [4:0] contribution_center_id,
    input logic [2:0] contribution_output_block,
    input logic contribution_narrow,
    input logic [1151:0] contribution_data,
    input logic protocol_error,
    input logic busy,
    input logic debug_output_full,
    input logic [31:0] debug_protocol_faults
);
    logic wide_metadata_legal;
    logic wide_padding_legal;
    logic narrow_high_side_zero;
    logic request_payload_legal;
    logic illegal_request;
    logic illegal_reload;

    assign wide_metadata_legal =
        high_tag == low_tag && high_tile == low_tile &&
        high_center_id == low_center_id &&
        high_output_block == low_output_block;
    assign wide_padding_legal = high_data[511:384] == 0;
    assign narrow_high_side_zero = high_tag == 0 && high_tile == 0 &&
        high_center_id == 0 && high_output_block == 0 && high_data == 0;
    assign request_payload_legal = request_narrow ?
        narrow_high_side_zero :
        (wide_metadata_legal && wide_padding_legal);
    assign illegal_request = request_valid && !request_payload_legal;
    assign illegal_reload = config_reload && (busy || request_valid);

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n);

    ap_request_accept: assert property (
        request_accept == (request_valid && request_ready));
    ap_contribution_accept: assert property (
        contribution_accept ==
            (contribution_valid && contribution_ready));
    ap_single_entry_busy: assert property (busy == debug_output_full);
    ap_fault_sticky: assert property (protocol_error |=> protocol_error);
    ap_fault_quiescent: assert property (
        protocol_error |-> !request_ready && !request_accept &&
            !contribution_valid && !contribution_accept);
    ap_illegal_atomic_suppression: assert property (
        illegal_request || illegal_reload |->
            !request_ready && !request_accept &&
            !contribution_valid && !contribution_accept);
    ap_illegal_sets_fault: assert property (
        illegal_request || illegal_reload |=> protocol_error);
    ap_stable_under_stall: assert property (
        contribution_valid && !contribution_ready &&
            !illegal_request && !illegal_reload
        |=> illegal_request || illegal_reload || protocol_error ||
            (contribution_valid &&
            $stable({contribution_tag,contribution_tile,
                     contribution_center_id,contribution_output_block,
                     contribution_narrow,contribution_data})));
    ap_reload_empty_only: assert property (
        config_reload && !busy && !request_valid |->
            !request_ready && !request_accept && !contribution_valid);
    ap_fault_count_bound: assert property (debug_protocol_faults <= 1);

    cp_narrow: cover property (
        request_accept && request_narrow ##1
        contribution_accept && contribution_narrow);
    cp_wide: cover property (
        request_accept && !request_narrow ##1
        contribution_accept && !contribution_narrow);
    cp_pop_push: cover property (request_accept && contribution_accept);
    cp_ii1_request: cover property (request_accept ##1 request_accept);
    cp_long_stall: cover property (
        contribution_valid && !contribution_ready [*8] ##1
        contribution_accept);
    cp_protocol_fault: cover property (protocol_error);
endmodule

`default_nettype wire
