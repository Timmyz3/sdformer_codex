`timescale 1ns/1ps
`default_nettype none

// q32 exact nearest-center controller.  One registered row scratch holds an
// optional second 16-center task, so there is no source reread, descriptor
// scratch traffic, or output reorder. Ties are resolved by global center ID.
module m405_q32_serial16_zero_stop_controller #(
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

    output logic [511:0]            configured_centers_q32,
    output logic [255:0]            configured_narrow_bitmap,
    output logic [TAG_BITS-1:0]     configured_tag,
    output logic                    configuration_live,

    output logic                    protocol_error,
    output logic                    busy,
    output logic                    debug_pass1_pending,
    output logic [31:0]             debug_source_rows,
    output logic [31:0]             debug_pass0_tasks,
    output logic [31:0]             debug_pass1_tasks,
    output logic [31:0]             debug_early_stops,
    output logic [31:0]             debug_results
);
    logic fault_q;
    logic [1:0] config_expected_q;
    logic config_sequence_q;
    logic config_live_q;
    logic phase_active_q;
    logic [TAG_BITS-1:0] phase_tag_q;
    logic [511:0] centers_q;
    logic [255:0] narrow_bitmap_q;
    logic [11:0] expected_row_q;

    // Frozen 43-bit minimum per-row scratch, plus one valid state bit.
    logic pass1_pending_q;
    logic [15:0] scratch_original_q;
    logic [11:0] scratch_row_id_q;
    logic [4:0] scratch_population_q;
    logic scratch_last_q;
    logic [3:0] scratch_pass0_id_q;
    logic [4:0] scratch_pass0_distance_q;

    logic result_valid_q;
    logic [TAG_BITS-1:0] result_tag_q;
    logic [11:0] result_row_id_q;
    logic [15:0] result_original_q;
    logic [4:0] result_center_id_q;
    logic [4:0] result_distance_q;
    logic result_use_pwp_q;
    logic result_last_q;

    logic [31:0] source_rows_q, pass0_tasks_q, pass1_tasks_q;
    logic [31:0] early_stops_q, results_q;

    logic [4:0] row_population_w;
    logic [4:0] pass0_best_distance_w, pass1_best_distance_w;
    logic [3:0] pass0_best_id_w, pass1_best_id_w;
    logic [4:0] global_best_distance_w, global_best_id_w;
    logic global_use_pwp_w;
    logic row_shape_legal_w;
    logic safe_w;

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

    always_comb begin : pass0_tournament
        row_population_w = popcount16(row_original);
        pass0_best_distance_w = 5'd31;
        pass0_best_id_w = '0;
        for (integer center = 0; center < 16; center = center + 1) begin
            if (popcount16(row_original ^
                    centers_q[center*16 +: 16]) < pass0_best_distance_w) begin
                pass0_best_distance_w = popcount16(row_original ^
                    centers_q[center*16 +: 16]);
                pass0_best_id_w = center[3:0];
            end
        end
    end

    always_comb begin : pass1_tournament
        pass1_best_distance_w = 5'd31;
        pass1_best_id_w = '0;
        for (integer center = 0; center < 16; center = center + 1) begin
            if (popcount16(scratch_original_q ^
                    centers_q[(center+16)*16 +: 16]) <
                    pass1_best_distance_w) begin
                pass1_best_distance_w = popcount16(scratch_original_q ^
                    centers_q[(center+16)*16 +: 16]);
                pass1_best_id_w = center[3:0];
            end
        end
        if (pass1_best_distance_w < scratch_pass0_distance_q) begin
            global_best_distance_w = pass1_best_distance_w;
            global_best_id_w = {1'b1, pass1_best_id_w};
        end else begin
            // Equality deliberately keeps pass0: every pass0 global ID is
            // lower than every pass1 global ID.
            global_best_distance_w = scratch_pass0_distance_q;
            global_best_id_w = {1'b0, scratch_pass0_id_q};
        end
        global_use_pwp_w = ({1'b0, global_best_distance_w} + 6'd1)
            < {1'b0, scratch_population_q};
    end

    assign safe_w = !fault_q;
    assign config_ready = safe_w && !config_live_q && !phase_active_q
        && !pass1_pending_q && !result_valid_q;
    assign config_accept = config_valid && config_ready;
    assign phase_release_ready = safe_w && config_live_q
        && !phase_active_q && !pass1_pending_q && !result_valid_q;
    assign phase_release_accept = phase_release_valid
        && phase_release_ready;
    assign result_valid = safe_w && result_valid_q;
    assign result_accept = result_valid && result_ready;
    assign row_ready = safe_w && phase_active_q && !pass1_pending_q
        && (!result_valid_q || result_ready);
    assign row_accept = row_valid && row_ready;
    assign row_shape_legal_w = row_id == expected_row_q
        && row_id < ROWS_PER_PHASE
        && row_last == (row_id == ROWS_PER_PHASE-1);

    assign result_tag = result_valid ? result_tag_q : '0;
    assign result_row_id = result_valid ? result_row_id_q : '0;
    assign result_original = result_valid ? result_original_q : '0;
    assign result_center_id = result_valid ? result_center_id_q : '0;
    assign result_distance = result_valid ? result_distance_q : '0;
    assign result_use_pwp = result_valid && result_use_pwp_q;
    assign result_last = result_valid && result_last_q;

    assign configured_centers_q32 = centers_q;
    assign configured_narrow_bitmap = narrow_bitmap_q;
    assign configured_tag = phase_tag_q;
    assign configuration_live = config_live_q;
    assign protocol_error = fault_q;
    assign busy = config_live_q || phase_active_q || pass1_pending_q
        || result_valid_q || config_sequence_q;
    assign debug_pass1_pending = pass1_pending_q;
    assign debug_source_rows = source_rows_q;
    assign debug_pass0_tasks = pass0_tasks_q;
    assign debug_pass1_tasks = pass1_tasks_q;
    assign debug_early_stops = early_stops_q;
    assign debug_results = results_q;

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            fault_q <= 1'b0;
            config_expected_q <= '0;
            config_sequence_q <= 1'b0;
            config_live_q <= 1'b0;
            phase_active_q <= 1'b0;
            phase_tag_q <= '0;
            centers_q <= '0;
            narrow_bitmap_q <= '0;
            expected_row_q <= '0;
            pass1_pending_q <= 1'b0;
            scratch_original_q <= '0;
            scratch_row_id_q <= '0;
            scratch_population_q <= '0;
            scratch_last_q <= 1'b0;
            scratch_pass0_id_q <= '0;
            scratch_pass0_distance_q <= '0;
            result_valid_q <= 1'b0;
            result_tag_q <= '0;
            result_row_id_q <= '0;
            result_original_q <= '0;
            result_center_id_q <= '0;
            result_distance_q <= '0;
            result_use_pwp_q <= 1'b0;
            result_last_q <= 1'b0;
            source_rows_q <= '0;
            pass0_tasks_q <= '0;
            pass1_tasks_q <= '0;
            early_stops_q <= '0;
            results_q <= '0;
        end else begin
            if ((config_valid && !config_ready) ||
                    (phase_release_valid && !phase_release_ready) ||
                    (row_valid && !row_ready && !pass1_pending_q &&
                     !result_valid_q))
                fault_q <= 1'b1;

            if (phase_release_accept)
                config_live_q <= 1'b0;

            if (result_accept) begin
                result_valid_q <= 1'b0;
                results_q <= results_q + 1'b1;
            end

            if (config_accept) begin
                if (config_beat_index != config_expected_q
                        || config_commit != (config_beat_index == 2)
                        || (config_sequence_q && config_tag != phase_tag_q)) begin
                    fault_q <= 1'b1;
                end else begin
                    case (config_beat_index)
                        2'd0: begin
                            centers_q[255:0] <= config_data;
                            phase_tag_q <= config_tag;
                            config_sequence_q <= 1'b1;
                            config_expected_q <= 2'd1;
                        end
                        2'd1: begin
                            centers_q[511:256] <= config_data;
                            config_expected_q <= 2'd2;
                        end
                        2'd2: begin
                            narrow_bitmap_q <= config_data;
                            config_sequence_q <= 1'b0;
                            config_expected_q <= '0;
                            config_live_q <= 1'b1;
                            phase_active_q <= 1'b1;
                            expected_row_q <= '0;
                        end
                        default: fault_q <= 1'b1;
                    endcase
                end
            end

            if (row_accept) begin
                source_rows_q <= source_rows_q + 1'b1;
                pass0_tasks_q <= pass0_tasks_q + 1'b1;
                if (!row_shape_legal_w) begin
                    fault_q <= 1'b1;
                end else begin
                    expected_row_q <= expected_row_q + 1'b1;
                    if (row_last)
                        phase_active_q <= 1'b0;
                    if (row_population_w >= 2
                            && pass0_best_distance_w > 0) begin
                        pass1_pending_q <= 1'b1;
                        scratch_original_q <= row_original;
                        scratch_row_id_q <= row_id;
                        scratch_population_q <= row_population_w;
                        scratch_last_q <= row_last;
                        scratch_pass0_id_q <= pass0_best_id_w;
                        scratch_pass0_distance_q <= pass0_best_distance_w;
                    end else begin
                        result_valid_q <= 1'b1;
                        result_tag_q <= phase_tag_q;
                        result_row_id_q <= row_id;
                        result_original_q <= row_original;
                        result_center_id_q <= {1'b0, pass0_best_id_w};
                        result_distance_q <= pass0_best_distance_w;
                        result_use_pwp_q <= row_population_w >= 2
                            && ({1'b0, pass0_best_distance_w} + 6'd1
                                < {1'b0, row_population_w});
                        result_last_q <= row_last;
                        if (row_population_w >= 2
                                && pass0_best_distance_w == 0)
                            early_stops_q <= early_stops_q + 1'b1;
                    end
                end
            end else if (pass1_pending_q) begin
                // The registered pass0 decision occupies exactly this next
                // task cycle. No source or descriptor scratch transaction is
                // issued in parallel or inserted between the two passes.
                pass1_tasks_q <= pass1_tasks_q + 1'b1;
                pass1_pending_q <= 1'b0;
                result_valid_q <= 1'b1;
                result_tag_q <= phase_tag_q;
                result_row_id_q <= scratch_row_id_q;
                result_original_q <= scratch_original_q;
                result_center_id_q <= global_best_id_w;
                result_distance_q <= global_best_distance_w;
                result_use_pwp_q <= global_use_pwp_w;
                result_last_q <= scratch_last_q;
            end
        end
    end
endmodule

`default_nettype wire
