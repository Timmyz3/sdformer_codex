`timescale 1ns/1ps
`default_nettype none

module m452_independent_m451_assertions #(
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
    input logic request_fuse_correction,
    input logic correction_subtract,
    input logic [TAG_BITS-1:0] correction_tag,
    input logic correction_tile,
    input logic [2:0] correction_output_block,
    input logic [767:0] correction_data,
    input logic contribution_valid,
    input logic contribution_ready,
    input logic contribution_accept,
    input logic [TAG_BITS-1:0] contribution_tag,
    input logic contribution_tile,
    input logic [4:0] contribution_center_id,
    input logic [2:0] contribution_output_block,
    input logic contribution_narrow,
    input logic contribution_fused,
    input logic [1247:0] contribution_data,
    input logic protocol_error,
    input logic busy,
    input logic [31:0] debug_protocol_faults
);
    logic pwp_legal;
    logic correction_legal;
    logic illegal_request;
    logic illegal_reload;

    always_comb begin
        pwp_legal = request_narrow ?
            (high_tag == '0 && high_tile == 1'b0 &&
             high_center_id == '0 && high_output_block == '0 &&
             high_data == '0) :
            (high_tag == low_tag && high_tile == low_tile &&
             high_center_id == low_center_id &&
             high_output_block == low_output_block &&
             high_data[511:384] == '0);
        correction_legal = request_fuse_correction ?
            (correction_tag == low_tag &&
             correction_tile == low_tile &&
             correction_output_block == low_output_block) :
            (correction_subtract == 1'b0 && correction_tag == '0 &&
             correction_tile == 1'b0 &&
             correction_output_block == '0 && correction_data == '0);
    end
    assign illegal_request = request_valid &&
        !(pwp_legal && correction_legal);
    assign illegal_reload = config_reload && (busy || request_valid);

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n);

    ap_accept_definition: assert property (
        request_accept == (request_valid && request_ready));
    ap_retire_definition: assert property (
        contribution_accept == (contribution_valid && contribution_ready));
    ap_fault_sticky: assert property (protocol_error |=> protocol_error);
    ap_fault_quiet: assert property (
        protocol_error |-> !request_ready && !request_accept &&
            !contribution_valid && !contribution_accept);
    ap_illegal_atomic: assert property (
        illegal_request || illegal_reload |->
            !request_ready && !request_accept &&
            !contribution_valid && !contribution_accept);
    ap_illegal_faults: assert property (
        illegal_request || illegal_reload |=> protocol_error);
    ap_stall_stable: assert property (
        contribution_valid && !contribution_ready &&
            !illegal_request && !illegal_reload
        |=> protocol_error || illegal_request || illegal_reload ||
            (contribution_valid &&
             $stable({contribution_tag,contribution_tile,
                      contribution_center_id,contribution_output_block,
                      contribution_narrow,contribution_fused,
                      contribution_data})));
    ap_output_known: assert property (
        contribution_valid |->
            !$isunknown({contribution_tag,contribution_tile,
                         contribution_center_id,contribution_output_block,
                         contribution_narrow,contribution_fused,
                         contribution_data}));
    ap_empty_reload_quiet: assert property (
        config_reload && !busy && !request_valid |->
            !request_ready && !request_accept && !contribution_valid);
    ap_single_fault_count: assert property (debug_protocol_faults <= 1);

    cp_plain: cover property (
        request_accept && !request_fuse_correction ##1
        contribution_accept && !contribution_fused);
    cp_fused_add: cover property (
        request_accept && request_fuse_correction && !correction_subtract ##1
        contribution_accept && contribution_fused);
    cp_fused_sub: cover property (
        request_accept && request_fuse_correction && correction_subtract ##1
        contribution_accept && contribution_fused);
    cp_narrow: cover property (request_accept && request_narrow);
    cp_wide: cover property (request_accept && !request_narrow);
    cp_pop_push: cover property (request_accept && contribution_accept);
    cp_ii1: cover property (request_accept ##1 request_accept);
    cp_stall12: cover property (
        contribution_valid && !contribution_ready [*12] ##1
        contribution_accept);
    cp_fault: cover property (protocol_error);
endmodule

`default_nettype wire
