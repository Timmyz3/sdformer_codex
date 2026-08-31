`timescale 1ns/1ps
`default_nettype none

module m405_q32_elastic_selected_slice_assertions (
    input logic clk_core,
    input logic reset_n,
    input logic config_valid,
    input logic config_ready,
    input logic config_accept,
    input logic phase_release_valid,
    input logic phase_release_ready,
    input logic phase_release_accept,
    input logic row_valid,
    input logic row_ready,
    input logic row_accept,
    input logic result_valid,
    input logic result_ready,
    input logic result_accept,
    input logic pwp_low_valid,
    input logic pwp_low_ready,
    input logic pwp_low_accept,
    input logic pwp_high_valid,
    input logic pwp_high_ready,
    input logic pwp_high_accept,
    input logic contribution_valid,
    input logic contribution_ready,
    input logic contribution_accept,
    input logic [23:0] contribution_tag,
    input logic contribution_tile,
    input logic [4:0] contribution_center_id,
    input logic [2:0] contribution_output_block,
    input logic contribution_narrow,
    input logic contribution_part_high,
    input logic contribution_last,
    input logic [1151:0] contribution_data,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n);

    ap_config_accept: assert property (
        config_accept == (config_valid && config_ready));
    ap_release_accept: assert property (
        phase_release_accept ==
            (phase_release_valid && phase_release_ready));
    ap_row_accept: assert property (
        row_accept == (row_valid && row_ready));
    ap_result_accept: assert property (
        result_accept == (result_valid && result_ready));
    ap_low_accept: assert property (
        pwp_low_accept == (pwp_low_valid && pwp_low_ready));
    ap_high_accept: assert property (
        pwp_high_accept == (pwp_high_valid && pwp_high_ready));
    ap_contribution_accept: assert property (
        contribution_accept ==
            (contribution_valid && contribution_ready));
    ap_fault_sticky: assert property (protocol_error |=> protocol_error);
    ap_global_fail_closed: assert property (
        protocol_error |-> !config_ready && !config_accept
            && !phase_release_ready && !phase_release_accept
            && !row_ready && !row_accept
            && !result_valid && !result_accept
            && !pwp_low_ready && !pwp_low_accept
            && !pwp_high_ready && !pwp_high_accept
            && !contribution_valid && !contribution_accept);
    ap_contribution_stable: assert property (
        contribution_valid && !contribution_ready
        |=> protocol_error || (contribution_valid &&
            $stable({contribution_tag,contribution_tile,
                     contribution_center_id,contribution_output_block,
                     contribution_narrow,contribution_part_high,
                     contribution_last,contribution_data})));

    cp_legal_pwp_after_rows: cover property (
        pwp_low_accept ##[1:4] contribution_accept);
    cp_phase_release: cover property (phase_release_accept);
    cp_global_fault: cover property (protocol_error);
endmodule

`default_nettype wire
