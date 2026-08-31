`timescale 1ns/1ps
`default_nettype none

// Non-contribution integration shell: one 96-byte config owner feeds the q32
// matcher and the exact bitmap lookup used by the elastic PWP adapter. The
// revised M384 descriptor controller remains a separately admitted control cut.
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
    logic matcher_error_w, matcher_busy_w;
    logic adapter_error_w, adapter_busy_w;
    logic global_fault_q;
    logic matcher_config_ready_w, matcher_config_accept_w;
    logic matcher_release_valid_w, matcher_release_ready_w;
    logic matcher_release_accept_w;
    logic matcher_row_ready_w, matcher_row_accept_w;
    logic matcher_result_valid_w, matcher_result_accept_w;
    logic adapter_low_valid_w;
    logic adapter_low_ready_w, adapter_low_accept_w;
    logic adapter_high_valid_w, adapter_high_ready_w, adapter_high_accept_w;
    logic adapter_contribution_valid_w, adapter_contribution_accept_w;
    logic global_safe_w;
    logic pwp_low_identity_legal_w, release_legal_w, shell_violation_w;
    logic adapter_config_reload_w;
    logic selected_narrow_w;

    logic matcher_debug_pending;
    logic [31:0] matcher_debug_source, matcher_debug_pass0;
    logic [31:0] matcher_debug_pass1, matcher_debug_early;
    logic [31:0] matcher_debug_results;
    logic [1:0] adapter_debug_fifo;
    logic [31:0] adapter_debug_low, adapter_debug_high;
    logic [31:0] adapter_debug_narrow, adapter_debug_wide;
    logic [31:0] adapter_debug_contributions;

    assign selected_narrow_w = narrow_bitmap_w[
        pwp_low_center_id * 8 + pwp_low_output_block];
    assign pwp_low_identity_legal_w = configuration_live_w
        && pwp_low_tag == configured_tag_w;
    assign release_legal_w = matcher_release_ready_w && !adapter_busy_w;
    assign shell_violation_w =
        (pwp_low_valid && !pwp_low_identity_legal_w)
        || (phase_release_valid && !release_legal_w);
    assign global_safe_w = !global_fault_q && !matcher_error_w
        && !adapter_error_w && !shell_violation_w;
    assign config_ready = matcher_config_ready_w && global_safe_w;
    assign config_accept = matcher_config_accept_w && global_safe_w;
    assign matcher_release_valid_w = phase_release_valid && global_safe_w
        && !adapter_busy_w;
    assign phase_release_ready = release_legal_w && global_safe_w;
    assign phase_release_accept = matcher_release_accept_w && global_safe_w;
    assign row_ready = matcher_row_ready_w && global_safe_w;
    assign row_accept = matcher_row_accept_w && global_safe_w;
    assign result_valid = matcher_result_valid_w && global_safe_w;
    assign result_accept = matcher_result_accept_w && global_safe_w;

    assign adapter_low_valid_w = pwp_low_valid && global_safe_w
        && pwp_low_identity_legal_w;
    assign adapter_high_valid_w = pwp_high_valid && global_safe_w;
    assign pwp_low_ready = adapter_low_ready_w && global_safe_w
        && pwp_low_identity_legal_w;
    assign pwp_low_accept = adapter_low_accept_w && global_safe_w;
    assign pwp_high_ready = adapter_high_ready_w && global_safe_w;
    assign pwp_high_accept = adapter_high_accept_w && global_safe_w;
    assign contribution_valid = adapter_contribution_valid_w
        && global_safe_w;
    assign contribution_accept = adapter_contribution_accept_w
        && global_safe_w;
    // Beat zero is the only reload boundary. The adapter rejects it if any
    // prior PWP transaction remains buffered or partially emitted.
    assign adapter_config_reload_w = config_accept
        && config_beat_index == 0;
    assign protocol_error = matcher_error_w || adapter_error_w
        || global_fault_q || shell_violation_w;
    assign busy = matcher_busy_w || adapter_busy_w;

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            global_fault_q <= 1'b0;
        end else if (matcher_error_w || adapter_error_w
                || shell_violation_w) begin
            global_fault_q <= 1'b1;
        end
    end

    m405_q32_serial16_zero_stop_controller #(
        .TAG_BITS(TAG_BITS), .ROWS_PER_PHASE(ROWS_PER_PHASE)
    ) u_matcher (
        .clk_core, .reset_n,
        .config_valid(config_valid && global_safe_w),
        .config_ready(matcher_config_ready_w),
        .config_accept(matcher_config_accept_w),
        .config_beat_index, .config_commit, .config_tag, .config_data,
        .phase_release_valid(matcher_release_valid_w),
        .phase_release_ready(matcher_release_ready_w),
        .phase_release_accept(matcher_release_accept_w),
        .row_valid(row_valid && global_safe_w),
        .row_ready(matcher_row_ready_w),
        .row_accept(matcher_row_accept_w), .row_id, .row_original,
        .row_last, .result_valid(matcher_result_valid_w),
        .result_ready(result_ready && global_safe_w),
        .result_accept(matcher_result_accept_w),
        .result_tag, .result_row_id, .result_original, .result_center_id,
        .result_distance, .result_use_pwp, .result_last,
        .configured_centers_q32(centers_w),
        .configured_narrow_bitmap(narrow_bitmap_w),
        .configured_tag(configured_tag_w),
        .configuration_live(configuration_live_w),
        .protocol_error(matcher_error_w), .busy(matcher_busy_w),
        .debug_pass1_pending(matcher_debug_pending),
        .debug_source_rows(matcher_debug_source),
        .debug_pass0_tasks(matcher_debug_pass0),
        .debug_pass1_tasks(matcher_debug_pass1),
        .debug_early_stops(matcher_debug_early),
        .debug_results(matcher_debug_results)
    );

    m405_exact_elastic_pwp_issue_adapter #(.TAG_BITS(TAG_BITS)) u_adapter (
        .clk_core, .reset_n, .config_reload(adapter_config_reload_w),
        .low_valid(adapter_low_valid_w), .low_ready(adapter_low_ready_w),
        .low_accept(adapter_low_accept_w), .low_tag(pwp_low_tag),
        .low_tile(pwp_low_tile), .low_center_id(pwp_low_center_id),
        .low_output_block(pwp_low_output_block),
        .low_narrow(selected_narrow_w), .low_data(pwp_low_data),
        .high_valid(adapter_high_valid_w), .high_ready(adapter_high_ready_w),
        .high_accept(adapter_high_accept_w), .high_tag(pwp_high_tag),
        .high_tile(pwp_high_tile), .high_center_id(pwp_high_center_id),
        .high_output_block(pwp_high_output_block),
        .high_data(pwp_high_data),
        .contribution_valid(adapter_contribution_valid_w),
        .contribution_ready(contribution_ready && global_safe_w),
        .contribution_accept(adapter_contribution_accept_w), .contribution_tag,
        .contribution_tile, .contribution_center_id,
        .contribution_output_block, .contribution_narrow,
        .contribution_part_high, .contribution_last, .contribution_data,
        .protocol_error(adapter_error_w), .busy(adapter_busy_w),
        .debug_completed_fifo_count(adapter_debug_fifo),
        .debug_low_accepts(adapter_debug_low),
        .debug_high_accepts(adapter_debug_high),
        .debug_narrow_blocks(adapter_debug_narrow),
        .debug_wide_blocks(adapter_debug_wide),
        .debug_contributions(adapter_debug_contributions)
    );
endmodule

`default_nettype wire
