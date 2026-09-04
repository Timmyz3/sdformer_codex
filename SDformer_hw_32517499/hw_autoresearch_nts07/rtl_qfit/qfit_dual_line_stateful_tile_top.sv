`timescale 1ns/1ps
`default_nettype none

// Stateful one-tile integration contract for the Local/Motion engine.
// Counts, selector bits, and emitted source bits are all derived from the same
// masked bitmap.  A Motion reuse is legal only when the full state identity,
// valid extent, and exact Local seed match the previous committed tile.
module qfit_dual_line_stateful_tile_top #(
    parameter int TILE_BITS = 256,
    parameter int OUT_LANES = 16,
    parameter int TAG_W = 24,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int OUT_W = (OUT_LANES <= 1) ? 1 : $clog2(OUT_LANES),
    parameter int COUNT_W = $clog2(TILE_BITS + 1),
    parameter int PERF_W = 64
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         weight_epoch_clear,
    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [INDEX_W-1:0]           weight_source,
    input  logic [OUT_W-1:0]             weight_lane,
    input  logic signed [W_W-1:0]        weight_data,
    input  logic                         weight_last,
    output logic                         weights_loaded,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [TAG_W-1:0]             request_state_key,
    input  logic [COUNT_W-1:0]           request_valid_bits,
    input  logic [TILE_BITS-1:0]         request_current_bits,
    input  logic [OUT_LANES*ACC_W-1:0]  request_local_seed_acc,
    input  logic                         request_sequence_boundary,
    input  logic                         request_force_refresh,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [TAG_W-1:0]             output_state_key,
    output logic                         output_use_motion,
    output logic                         output_force_local,
    output logic [COUNT_W-1:0]           output_source_count,
    output logic [OUT_LANES*ACC_W-1:0]  output_acc,

    output logic                         protocol_error,
    output logic [PERF_W-1:0]            perf_requests,
    output logic [PERF_W-1:0]            perf_state_hits,
    output logic [PERF_W-1:0]            perf_state_misses,
    output logic [PERF_W-1:0]            perf_local_tiles,
    output logic [PERF_W-1:0]            perf_motion_tiles,
    output logic [PERF_W-1:0]            perf_invalid_valid_bits,
    output logic [PERF_W-1:0]            perf_weight_segment_reads,
    output logic [PERF_W-1:0]            perf_accumulator_updates
);
    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_SELECT = 2'd1,
        ST_EXECUTE = 2'd2
    } state_t;

    state_t control_q;
    logic state_valid_q;
    logic [TAG_W-1:0] state_key_q;
    logic [COUNT_W-1:0] state_valid_bits_q;
    logic [TILE_BITS-1:0] state_bits_q;
    logic [OUT_LANES*ACC_W-1:0] state_seed_q;
    logic [OUT_LANES*ACC_W-1:0] state_acc_q;

    logic [TAG_W-1:0] request_key_q;
    logic [COUNT_W-1:0] valid_bits_q;
    logic [TILE_BITS-1:0] current_bits_q;
    logic [TILE_BITS-1:0] previous_bits_q;
    logic [OUT_LANES*ACC_W-1:0] local_seed_q;
    logic prior_state_valid_q;
    logic sequence_boundary_q;
    logic force_refresh_q;
    logic invalid_valid_bits_q;
    logic [COUNT_W-1:0] current_count_q;
    logic [COUNT_W-1:0] positive_count_q;
    logic [COUNT_W-1:0] negative_count_q;
    logic selector_issued_q;

    logic selector_request_valid;
    logic selector_request_ready;
    logic selector_decision_valid;
    logic selector_decision_ready;
    logic [TAG_W-1:0] selector_decision_tag;
    logic selector_use_motion;
    logic selector_seed_previous;
    logic [COUNT_W:0] selector_work_count;
    logic [COUNT_W:0] selector_local_work_count;
    logic [COUNT_W:0] selector_transition_work_count;
    logic selector_force_local;
    logic selector_counts_legal;
    logic selector_protocol_error;
    logic [PERF_W-1:0] unused_selector_decisions;
    logic [PERF_W-1:0] unused_selector_local;
    logic [PERF_W-1:0] unused_selector_motion;
    logic [PERF_W-1:0] unused_selector_local_work;
    logic [PERF_W-1:0] unused_selector_transition_work;
    logic [PERF_W-1:0] unused_selector_selected_work;

    logic executor_command_valid;
    logic executor_command_ready;
    logic [OUT_LANES*ACC_W-1:0] executor_seed;
    logic executor_output_valid;
    logic executor_output_ready;
    logic [TAG_W-1:0] executor_output_tag;
    logic executor_output_use_motion;
    logic [COUNT_W-1:0] executor_output_source_count;
    logic [OUT_LANES*ACC_W-1:0] executor_output_acc;
    logic executor_protocol_error;
    logic [PERF_W-1:0] unused_executor_commands;
    logic [PERF_W-1:0] unused_executor_local;
    logic [PERF_W-1:0] unused_executor_motion;
    logic [PERF_W-1:0] unused_executor_positive;
    logic [PERF_W-1:0] unused_executor_negative;

    logic request_fire;
    logic decision_fire;
    logic output_fire;
    logic valid_bits_legal;
    logic [TILE_BITS-1:0] request_mask;
    logic [TILE_BITS-1:0] masked_current;
    logic state_identity_match;
    logic [TILE_BITS-1:0] selected_previous;
    logic [COUNT_W-1:0] current_count;
    logic [COUNT_W-1:0] positive_count;
    logic [COUNT_W-1:0] negative_count;

    always_comb begin
        request_mask = '0;
        if (valid_bits_legal) begin
            for (integer bit_index = 0; bit_index < TILE_BITS; bit_index = bit_index + 1)
                request_mask[bit_index] = COUNT_W'(bit_index) < request_valid_bits;
        end else begin
            request_mask = '1;
        end
    end

    assign valid_bits_legal = request_valid_bits <= COUNT_W'(TILE_BITS);
    assign masked_current = request_current_bits & request_mask;
    assign state_identity_match = state_valid_q
                               && state_key_q == request_state_key
                               && state_valid_bits_q == request_valid_bits
                               && state_seed_q == request_local_seed_acc
                               && valid_bits_legal;
    assign selected_previous = state_identity_match
        ? (state_bits_q & request_mask) : '0;

    always_comb begin
        current_count = '0;
        positive_count = '0;
        negative_count = '0;
        for (integer bit_index = 0; bit_index < TILE_BITS; bit_index = bit_index + 1) begin
            current_count = current_count + COUNT_W'(masked_current[bit_index]);
            positive_count = positive_count
                + COUNT_W'(masked_current[bit_index] && !selected_previous[bit_index]);
            negative_count = negative_count
                + COUNT_W'(!masked_current[bit_index] && selected_previous[bit_index]);
        end
    end

    // Never admit a descriptor in the same cycle that the weight epoch is
    // being cleared or written.  Otherwise an old state/Acc32 could be paired
    // with a subsequently replaced weight image.
    assign request_ready = control_q == ST_IDLE
                        && weights_loaded
                        && !weight_epoch_clear
                        && !weight_valid;
    assign request_fire = request_valid && request_ready;
    assign selector_request_valid = control_q == ST_SELECT && !selector_issued_q;
    assign selector_decision_ready = control_q == ST_SELECT
                                  && executor_command_ready;
    assign decision_fire = selector_decision_valid && selector_decision_ready;
    assign executor_command_valid = decision_fire;
    assign executor_seed = selector_use_motion ? state_acc_q : local_seed_q;

    assign output_valid = control_q == ST_EXECUTE && executor_output_valid;
    assign executor_output_ready = control_q == ST_EXECUTE && output_ready;
    assign output_fire = output_valid && output_ready;
    assign output_state_key = executor_output_tag;
    assign output_use_motion = executor_output_use_motion;
    assign output_force_local = !executor_output_use_motion;
    assign output_source_count = executor_output_source_count;
    assign output_acc = executor_output_acc;
    assign protocol_error = selector_protocol_error
                         || executor_protocol_error
                         || perf_invalid_valid_bits != '0;

    qfit_dual_line_tile_selector #(
        .TAG_W(TAG_W), .COUNT_W(COUNT_W), .PERF_W(PERF_W)
    ) u_selector (
        .clk_core(clk_core), .rst_core(rst_core),
        .request_valid(selector_request_valid),
        .request_ready(selector_request_ready),
        .request_tag(request_key_q),
        .request_valid_bits(valid_bits_q),
        .request_current_nonzero(current_count_q),
        .request_positive_transitions(positive_count_q),
        .request_negative_transitions(negative_count_q),
        .request_prior_state_valid(prior_state_valid_q),
        .request_sequence_boundary(sequence_boundary_q),
        .request_force_refresh(force_refresh_q || invalid_valid_bits_q),
        .decision_valid(selector_decision_valid),
        .decision_ready(selector_decision_ready),
        .decision_tag(selector_decision_tag),
        .decision_use_motion(selector_use_motion),
        .decision_seed_previous(selector_seed_previous),
        .decision_work_count(selector_work_count),
        .decision_local_work_count(selector_local_work_count),
        .decision_transition_work_count(selector_transition_work_count),
        .decision_force_local(selector_force_local),
        .decision_counts_legal(selector_counts_legal),
        .protocol_error(selector_protocol_error),
        .perf_decisions(unused_selector_decisions),
        .perf_local_decisions(unused_selector_local),
        .perf_motion_decisions(unused_selector_motion),
        .perf_local_work(unused_selector_local_work),
        .perf_transition_work(unused_selector_transition_work),
        .perf_selected_work(unused_selector_selected_work)
    );

    qfit_dual_line_tile_executor #(
        .TILE_BITS(TILE_BITS), .OUT_LANES(OUT_LANES), .TAG_W(TAG_W),
        .W_W(W_W), .ACC_W(ACC_W), .INDEX_W(INDEX_W),
        .OUT_W(OUT_W), .COUNT_W(COUNT_W), .PERF_W(PERF_W)
    ) u_executor (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_epoch_clear(weight_epoch_clear),
        .weight_valid(weight_valid), .weight_ready(weight_ready),
        .weight_source(weight_source), .weight_lane(weight_lane),
        .weight_data(weight_data), .weight_last(weight_last),
        .weights_loaded(weights_loaded),
        .command_valid(executor_command_valid),
        .command_ready(executor_command_ready),
        .command_tag(selector_decision_tag),
        .command_use_motion(selector_use_motion),
        .command_current_bits(current_bits_q),
        .command_previous_bits(previous_bits_q),
        .command_seed_acc(executor_seed),
        .output_valid(executor_output_valid),
        .output_ready(executor_output_ready),
        .output_tag(executor_output_tag),
        .output_use_motion(executor_output_use_motion),
        .output_source_count(executor_output_source_count),
        .output_acc(executor_output_acc),
        .protocol_error(executor_protocol_error),
        .perf_commands(unused_executor_commands),
        .perf_local_commands(unused_executor_local),
        .perf_motion_commands(unused_executor_motion),
        .perf_weight_segment_reads(perf_weight_segment_reads),
        .perf_accumulator_updates(perf_accumulator_updates),
        .perf_positive_sources(unused_executor_positive),
        .perf_negative_sources(unused_executor_negative)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            control_q <= ST_IDLE;
            state_valid_q <= 1'b0;
            state_key_q <= '0;
            state_valid_bits_q <= '0;
            state_bits_q <= '0;
            state_seed_q <= '0;
            state_acc_q <= '0;
            request_key_q <= '0;
            valid_bits_q <= '0;
            current_bits_q <= '0;
            previous_bits_q <= '0;
            local_seed_q <= '0;
            prior_state_valid_q <= 1'b0;
            sequence_boundary_q <= 1'b0;
            force_refresh_q <= 1'b0;
            invalid_valid_bits_q <= 1'b0;
            current_count_q <= '0;
            positive_count_q <= '0;
            negative_count_q <= '0;
            selector_issued_q <= 1'b0;
            perf_requests <= '0;
            perf_state_hits <= '0;
            perf_state_misses <= '0;
            perf_local_tiles <= '0;
            perf_motion_tiles <= '0;
            perf_invalid_valid_bits <= '0;
        end else begin
            if (request_fire) begin
                request_key_q <= request_state_key;
                valid_bits_q <= valid_bits_legal
                    ? request_valid_bits : COUNT_W'(TILE_BITS);
                current_bits_q <= masked_current;
                previous_bits_q <= selected_previous;
                local_seed_q <= request_local_seed_acc;
                prior_state_valid_q <= state_identity_match;
                sequence_boundary_q <= request_sequence_boundary;
                force_refresh_q <= request_force_refresh;
                invalid_valid_bits_q <= !valid_bits_legal;
                current_count_q <= current_count;
                positive_count_q <= positive_count;
                negative_count_q <= negative_count;
                selector_issued_q <= 1'b0;
                perf_requests <= perf_requests + PERF_W'(1);
                if (state_identity_match)
                    perf_state_hits <= perf_state_hits + PERF_W'(1);
                else
                    perf_state_misses <= perf_state_misses + PERF_W'(1);
                if (!valid_bits_legal)
                    perf_invalid_valid_bits <= perf_invalid_valid_bits + PERF_W'(1);
                control_q <= ST_SELECT;
            end

            if (decision_fire)
                control_q <= ST_EXECUTE;

            if (selector_request_valid && selector_request_ready)
                selector_issued_q <= 1'b1;

            if (output_fire) begin
                state_valid_q <= 1'b1;
                state_key_q <= request_key_q;
                state_valid_bits_q <= valid_bits_q;
                state_bits_q <= current_bits_q;
                state_seed_q <= local_seed_q;
                state_acc_q <= executor_output_acc;
                if (executor_output_use_motion)
                    perf_motion_tiles <= perf_motion_tiles + PERF_W'(1);
                else
                    perf_local_tiles <= perf_local_tiles + PERF_W'(1);
                control_q <= ST_IDLE;
            end

            if (weight_epoch_clear && control_q == ST_IDLE)
                state_valid_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
