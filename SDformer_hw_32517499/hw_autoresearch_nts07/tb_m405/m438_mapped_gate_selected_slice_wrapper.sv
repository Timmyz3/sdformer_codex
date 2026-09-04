`timescale 1ns/1ps
`default_nettype none

// Testbench-only compatibility shell for direct mapped-gate activity replay.
// The gate DUT is u_gate; none of this wrapper or its scoreboards is included
// in the M438 SAIF scope.  The scoreboards replace only the 320 debug DFFs
// proven unobservable and removed by M416/M431 DC.
module m405_q32_elastic_selected_slice #(
    parameter int TAG_BITS = 24,
    parameter int ROWS_PER_PHASE = 3000
) (
    input  logic                    clk_core,
    input  logic                    reset_n,
    input  logic                    config_valid,
    output logic                    config_ready,
    output logic                    config_accept,
    input  logic [1:0]              config_beat_index,
    input  logic                    config_commit,
    input  logic [TAG_BITS-1:0]     config_tag,
    input  logic [255:0]            config_data,
    input  logic                    phase_release_valid,
    output logic                    phase_release_ready,
    output logic                    phase_release_accept,
    input  logic                    row_valid,
    output logic                    row_ready,
    output logic                    row_accept,
    input  logic [11:0]             row_id,
    input  logic [15:0]             row_original,
    input  logic                    row_last,
    output logic                    result_valid,
    input  logic                    result_ready,
    output logic                    result_accept,
    output logic [TAG_BITS-1:0]     result_tag,
    output logic [11:0]             result_row_id,
    output logic [15:0]             result_original,
    output logic [4:0]              result_center_id,
    output logic [4:0]              result_distance,
    output logic                    result_use_pwp,
    output logic                    result_last,
    input  logic                    pwp_low_valid,
    output logic                    pwp_low_ready,
    output logic                    pwp_low_accept,
    input  logic [TAG_BITS-1:0]     pwp_low_tag,
    input  logic                    pwp_low_tile,
    input  logic [4:0]              pwp_low_center_id,
    input  logic [2:0]              pwp_low_output_block,
    input  logic [767:0]            pwp_low_data,
    input  logic                    pwp_high_valid,
    output logic                    pwp_high_ready,
    output logic                    pwp_high_accept,
    input  logic [TAG_BITS-1:0]     pwp_high_tag,
    input  logic                    pwp_high_tile,
    input  logic [4:0]              pwp_high_center_id,
    input  logic [2:0]              pwp_high_output_block,
    input  logic [511:0]            pwp_high_data,
    output logic                    contribution_valid,
    input  logic                    contribution_ready,
    output logic                    contribution_accept,
    output logic [TAG_BITS-1:0]     contribution_tag,
    output logic                    contribution_tile,
    output logic [4:0]              contribution_center_id,
    output logic [2:0]              contribution_output_block,
    output logic                    contribution_narrow,
    output logic                    contribution_part_high,
    output logic                    contribution_last,
    output logic [1151:0]           contribution_data,
    output logic                    protocol_error,
    output logic                    busy
);
    logic [511:0] centers_w;
    logic [255:0] narrow_bitmap_w;
    logic [TAG_BITS-1:0] configured_tag_w;
    logic configuration_live_w;

    logic [31:0] matcher_debug_source, matcher_debug_pass0;
    logic [31:0] matcher_debug_pass1, matcher_debug_early;
    logic [31:0] matcher_debug_results;
    logic [31:0] adapter_debug_low, adapter_debug_high;
    logic [31:0] adapter_debug_narrow, adapter_debug_wide;
    logic [31:0] adapter_debug_contributions;

    integer pass0_distance;
    integer candidate_distance;

    m405_q32_elastic_selected_slice_mapped_gate u_gate (
        .clk_core, .reset_n,
        .config_valid, .config_ready, .config_accept,
        .config_beat_index, .config_commit, .config_tag, .config_data,
        .phase_release_valid, .phase_release_ready, .phase_release_accept,
        .row_valid, .row_ready, .row_accept, .row_id, .row_original,
        .row_last, .result_valid, .result_ready, .result_accept,
        .result_tag, .result_row_id, .result_original, .result_center_id,
        .result_distance, .result_use_pwp, .result_last,
        .pwp_low_valid, .pwp_low_ready, .pwp_low_accept, .pwp_low_tag,
        .pwp_low_tile, .pwp_low_center_id, .pwp_low_output_block,
        .pwp_low_data, .pwp_high_valid, .pwp_high_ready,
        .pwp_high_accept, .pwp_high_tag, .pwp_high_tile,
        .pwp_high_center_id, .pwp_high_output_block, .pwp_high_data,
        .contribution_valid, .contribution_ready, .contribution_accept,
        .contribution_tag, .contribution_tile, .contribution_center_id,
        .contribution_output_block, .contribution_narrow,
        .contribution_part_high, .contribution_last, .contribution_data,
        .protocol_error, .busy
    );

    // These nets survived mapping and are observed only by the existing TB.
    assign centers_w = u_gate.centers_w;
    assign narrow_bitmap_w = u_gate.narrow_bitmap_w;
    assign configured_tag_w = u_gate.configured_tag_w;
    assign configuration_live_w = u_gate.configuration_live_w;

    function automatic integer popcount16(input logic [15:0] value);
        integer count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 16; bit_index++)
                count += value[bit_index];
            popcount16 = count;
        end
    endfunction

    function automatic integer hamming16(
        input logic [15:0] lhs,
        input logic [15:0] rhs
    );
        hamming16 = popcount16(lhs ^ rhs);
    endfunction

    always_comb begin
        pass0_distance = 16;
        for (int center = 0; center < 16; center++) begin
            candidate_distance = hamming16(
                row_original, centers_w[center*16 +: 16]);
            if (candidate_distance < pass0_distance)
                pass0_distance = candidate_distance;
        end
    end

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            matcher_debug_source <= '0;
            matcher_debug_pass0 <= '0;
            matcher_debug_pass1 <= '0;
            matcher_debug_early <= '0;
            matcher_debug_results <= '0;
            adapter_debug_low <= '0;
            adapter_debug_high <= '0;
            adapter_debug_narrow <= '0;
            adapter_debug_wide <= '0;
            adapter_debug_contributions <= '0;
        end else begin
            if (row_accept) begin
                matcher_debug_source <= matcher_debug_source + 1'b1;
                matcher_debug_pass0 <= matcher_debug_pass0 + 1'b1;
                if (popcount16(row_original) >= 2) begin
                    if (pass0_distance == 0)
                        matcher_debug_early <= matcher_debug_early + 1'b1;
                    else
                        matcher_debug_pass1 <= matcher_debug_pass1 + 1'b1;
                end
            end
            if (result_accept)
                matcher_debug_results <= matcher_debug_results + 1'b1;
            if (pwp_low_accept) begin
                adapter_debug_low <= adapter_debug_low + 1'b1;
                if (narrow_bitmap_w[
                        pwp_low_center_id*8+pwp_low_output_block])
                    adapter_debug_narrow <= adapter_debug_narrow + 1'b1;
                else
                    adapter_debug_wide <= adapter_debug_wide + 1'b1;
            end
            if (pwp_high_accept)
                adapter_debug_high <= adapter_debug_high + 1'b1;
            if (contribution_accept)
                adapter_debug_contributions <=
                    adapter_debug_contributions + 1'b1;
        end
    end
endmodule

`default_nettype wire
