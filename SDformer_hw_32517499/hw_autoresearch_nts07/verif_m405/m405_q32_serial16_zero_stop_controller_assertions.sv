`timescale 1ns/1ps
`default_nettype none

module m405_q32_serial16_zero_stop_controller_assertions #(
    parameter int TAG_BITS = 24
) (
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
    input logic [TAG_BITS-1:0] result_tag,
    input logic [11:0] result_row_id,
    input logic [15:0] result_original,
    input logic [4:0] result_center_id,
    input logic [4:0] result_distance,
    input logic result_use_pwp,
    input logic result_last,
    input logic configuration_live,
    input logic protocol_error,
    input logic debug_pass1_pending,
    input logic [31:0] debug_source_rows,
    input logic [31:0] debug_pass0_tasks,
    input logic [31:0] debug_pass1_tasks,
    input logic [31:0] debug_early_stops,
    input logic [31:0] debug_results
);
    function automatic logic [4:0] popcount16(input logic [15:0] value);
        integer index;
        logic [4:0] count;
        begin
            count = '0;
            for (index = 0; index < 16; index = index + 1)
                count = count + value[index];
            popcount16 = count;
        end
    endfunction

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n);

    ap_config_accept: assert property (
        config_accept == (config_valid && config_ready));
    ap_phase_release_accept: assert property (
        phase_release_accept ==
            (phase_release_valid && phase_release_ready));
    ap_row_accept: assert property (
        row_accept == (row_valid && row_ready));
    ap_result_accept: assert property (
        result_accept == (result_valid && result_ready));
    ap_fault_sticky: assert property (protocol_error |=> protocol_error);
    ap_fault_suppresses: assert property (
        protocol_error |-> !config_ready && !phase_release_ready
            && !row_ready && !result_valid);
    ap_result_stable: assert property (
        result_valid && !result_ready
        |=> protocol_error || (result_valid &&
            $stable({result_tag,result_row_id,result_original,
                     result_center_id,result_distance,result_use_pwp,
                     result_last})));
    ap_pass0_conservation: assert property (
        debug_source_rows == debug_pass0_tasks);
    ap_result_use_strict: assert property (
        result_valid && result_use_pwp
        |-> ({1'b0,result_distance} + 6'd1)
             < {1'b0,popcount16(result_original)});
    ap_pop_lt2_fallback: assert property (
        result_valid && popcount16(result_original) < 2
        |-> !result_use_pwp);
    ap_registered_pass1_task: assert property (
        debug_pass1_pending |=> protocol_error ||
            !debug_pass1_pending);
    ap_result_count_bound: assert property (
        debug_results <= debug_source_rows);
    ap_early_bound: assert property (
        debug_early_stops <= debug_source_rows);
    ap_last_keeps_config_live: assert property (
        result_accept && result_last |=> configuration_live);
    ap_release_clears_config_live: assert property (
        phase_release_accept |=> !configuration_live);

    cp_zero: cover property (
        result_accept && result_original == 0 && !result_use_pwp);
    cp_pop1: cover property (
        result_accept && popcount16(result_original) == 1
            && !result_use_pwp);
    cp_early: cover property (
        result_accept && popcount16(result_original) >= 2
            && result_distance == 0);
    cp_pass1: cover property (debug_pass1_pending);
    cp_stall: cover property (
        result_valid && !result_ready ##[1:8] result_accept);
    cp_fault: cover property (protocol_error);
    cp_phase_release: cover property (phase_release_accept);
endmodule

`default_nettype wire
