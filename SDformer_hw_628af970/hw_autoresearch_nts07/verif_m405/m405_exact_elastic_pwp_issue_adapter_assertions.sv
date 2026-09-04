`timescale 1ns/1ps
`default_nettype none

module m405_exact_elastic_pwp_issue_adapter_assertions #(
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic reset_n,
    input logic config_reload,
    input logic low_valid,
    input logic low_ready,
    input logic low_accept,
    input logic high_valid,
    input logic high_ready,
    input logic high_accept,
    input logic contribution_valid,
    input logic contribution_ready,
    input logic contribution_accept,
    input logic [TAG_BITS-1:0] contribution_tag,
    input logic contribution_tile,
    input logic [4:0] contribution_center_id,
    input logic [2:0] contribution_output_block,
    input logic contribution_narrow,
    input logic contribution_part_high,
    input logic contribution_last,
    input logic [1151:0] contribution_data,
    input logic protocol_error,
    input logic busy,
    input logic [1:0] debug_completed_fifo_count
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n);

    ap_low_accept: assert property (
        low_accept == (low_valid && low_ready));
    ap_high_accept: assert property (
        high_accept == (high_valid && high_ready));
    ap_contribution_accept: assert property (
        contribution_accept ==
            (contribution_valid && contribution_ready));
    ap_fault_sticky: assert property (
        protocol_error |=> protocol_error);
    ap_fault_suppresses_output: assert property (
        protocol_error |-> !contribution_valid && !low_ready && !high_ready);
    ap_stable_under_stall: assert property (
        contribution_valid && !contribution_ready
        |=> protocol_error || (contribution_valid &&
            $stable({contribution_tag,contribution_tile,
                     contribution_center_id,contribution_output_block,
                     contribution_narrow,contribution_part_high,
                     contribution_last,contribution_data})));
    ap_high_is_wide_final: assert property (
        contribution_valid && contribution_part_high
        |-> !contribution_narrow && contribution_last);
    ap_narrow_is_single_final: assert property (
        contribution_valid && contribution_narrow
        |-> !contribution_part_high && contribution_last);
    ap_wide_low_is_not_final: assert property (
        contribution_valid && !contribution_narrow
            && !contribution_part_high |-> !contribution_last);
    ap_fifo_bound: assert property (debug_completed_fifo_count <= 2);
    ap_reload_busy_fails: assert property (
        config_reload && busy |=> protocol_error);

    cp_narrow: cover property (
        contribution_accept && contribution_narrow);
    cp_wide_pair: cover property (
        contribution_accept && !contribution_narrow
            && !contribution_part_high ##1
        contribution_accept && contribution_part_high);
    cp_output_stall: cover property (
        contribution_valid && !contribution_ready ##[1:8]
        contribution_accept);
    cp_fault: cover property (protocol_error);
endmodule

`default_nettype wire
