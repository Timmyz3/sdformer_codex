`timescale 1ns/1ps
`default_nettype none

// Shared exact Local/Motion tile executor.
//
// One selected source index reads one resident INT8 weight segment and updates
// OUT_LANES Acc32 destinations in parallel.  Local seeds the accumulators with
// bias/residual state and emits current one-bits.  Motion seeds them with the
// previous exact output and emits signed XOR-frontier updates.  The seed is an
// explicit transaction input so sequence/reset ownership stays outside this
// leaf and can be audited by the system scheduler.
module qfit_dual_line_tile_executor #(
    parameter int TILE_BITS = 256,
    parameter int OUT_LANES = 16,
    parameter int TAG_W = 24,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int OUT_W = (OUT_LANES <= 1) ? 1 : $clog2(OUT_LANES),
    parameter int COUNT_W = $clog2(TILE_BITS + 1),
    parameter int PERF_W = 32
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

    input  logic                         command_valid,
    output logic                         command_ready,
    input  logic [TAG_W-1:0]             command_tag,
    input  logic                         command_use_motion,
    input  logic [TILE_BITS-1:0]         command_current_bits,
    input  logic [TILE_BITS-1:0]         command_previous_bits,
    input  logic [OUT_LANES*ACC_W-1:0]  command_seed_acc,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [TAG_W-1:0]             output_tag,
    output logic                         output_use_motion,
    output logic [COUNT_W-1:0]           output_source_count,
    output logic [OUT_LANES*ACC_W-1:0]  output_acc,

    output logic                         protocol_error,
    output logic [PERF_W-1:0]            perf_commands,
    output logic [PERF_W-1:0]            perf_local_commands,
    output logic [PERF_W-1:0]            perf_motion_commands,
    output logic [PERF_W-1:0]            perf_weight_segment_reads,
    output logic [PERF_W-1:0]            perf_accumulator_updates,
    output logic [PERF_W-1:0]            perf_positive_sources,
    output logic [PERF_W-1:0]            perf_negative_sources
);
    logic signed [W_W-1:0] weight_q [0:TILE_BITS-1][0:OUT_LANES-1];
    logic signed [ACC_W-1:0] acc_q [0:OUT_LANES-1];
    logic [INDEX_W-1:0] expected_source_q;
    logic [OUT_W-1:0] expected_lane_q;
    logic weights_loaded_q;
    logic protocol_error_q;

    logic stream_command_valid;
    logic stream_command_ready;
    logic stream_source_valid;
    logic stream_source_ready;
    logic [TAG_W-1:0] stream_source_tag;
    logic [INDEX_W-1:0] stream_source_index;
    logic stream_source_negative;
    logic stream_source_use_motion;
    logic stream_source_last;
    logic stream_done_valid;
    logic stream_done_ready;
    logic [TAG_W-1:0] stream_done_tag;
    logic stream_done_use_motion;
    logic [COUNT_W-1:0] stream_done_source_count;
    logic [PERF_W-1:0] stream_perf_sources;
    logic [PERF_W-1:0] unused_stream_positive;
    logic [PERF_W-1:0] unused_stream_negative;
    logic [PERF_W-1:0] unused_stream_commands;
    logic [PERF_W-1:0] unused_stream_local;
    logic [PERF_W-1:0] unused_stream_motion;

    logic weight_fire;
    logic weight_contract_valid;
    logic command_fire;
    logic source_fire;
    logic output_fire;
    logic expected_last;

    initial begin
        if (ACC_W < W_W)
            $error("ACC_W must be at least W_W");
    end

    assign weights_loaded = weights_loaded_q;
    assign protocol_error = protocol_error_q;
    assign expected_last = expected_source_q == INDEX_W'(TILE_BITS - 1)
                        && expected_lane_q == OUT_W'(OUT_LANES - 1);
    assign weight_ready = stream_command_ready
                       && !command_valid
                       && !weight_epoch_clear
                       && !weights_loaded_q;
    assign weight_fire = weight_valid && weight_ready;
    assign weight_contract_valid = 32'(weight_source) < TILE_BITS
                                && 32'(weight_lane) < OUT_LANES
                                && weight_source == expected_source_q
                                && weight_lane == expected_lane_q
                                && weight_last == expected_last;

    assign command_ready = stream_command_ready
                        && weights_loaded_q
                        && !weight_valid
                        && !weight_epoch_clear;
    assign command_fire = command_valid && command_ready;
    assign stream_command_valid = command_fire;

    assign stream_source_ready = 1'b1;
    assign source_fire = stream_source_valid && stream_source_ready;
    assign stream_done_ready = output_ready;
    assign output_valid = stream_done_valid;
    assign output_fire = output_valid && output_ready;
    assign output_tag = stream_done_tag;
    assign output_use_motion = stream_done_use_motion;
    assign output_source_count = stream_done_source_count;

    generate
        for (genvar lane = 0; lane < OUT_LANES; lane = lane + 1) begin : g_output
            assign output_acc[lane*ACC_W +: ACC_W] = acc_q[lane];
        end
    endgenerate

    qfit_dual_line_source_streamer #(
        .TILE_BITS(TILE_BITS),
        .TAG_W(TAG_W),
        .INDEX_W(INDEX_W),
        .COUNT_W(COUNT_W),
        .PERF_W(PERF_W)
    ) u_source_streamer (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .command_valid(stream_command_valid),
        .command_ready(stream_command_ready),
        .command_tag(command_tag),
        .command_use_motion(command_use_motion),
        .command_current_bits(command_current_bits),
        .command_previous_bits(command_previous_bits),
        .source_valid(stream_source_valid),
        .source_ready(stream_source_ready),
        .source_tag(stream_source_tag),
        .source_index(stream_source_index),
        .source_negative(stream_source_negative),
        .source_use_motion(stream_source_use_motion),
        .source_last(stream_source_last),
        .done_valid(stream_done_valid),
        .done_ready(stream_done_ready),
        .done_tag(stream_done_tag),
        .done_use_motion(stream_done_use_motion),
        .done_source_count(stream_done_source_count),
        .perf_commands(unused_stream_commands),
        .perf_local_commands(unused_stream_local),
        .perf_motion_commands(unused_stream_motion),
        .perf_sources(stream_perf_sources),
        .perf_positive_sources(unused_stream_positive),
        .perf_negative_sources(unused_stream_negative)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            expected_source_q <= '0;
            expected_lane_q <= '0;
            weights_loaded_q <= 1'b0;
            protocol_error_q <= 1'b0;
            perf_commands <= '0;
            perf_local_commands <= '0;
            perf_motion_commands <= '0;
            perf_weight_segment_reads <= '0;
            perf_accumulator_updates <= '0;
            perf_positive_sources <= '0;
            perf_negative_sources <= '0;
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                acc_q[lane] <= '0;
        end else begin
            if (weight_epoch_clear) begin
                if (!stream_command_ready || command_valid || weight_valid)
                    protocol_error_q <= 1'b1;
                else begin
                    expected_source_q <= '0;
                    expected_lane_q <= '0;
                    weights_loaded_q <= 1'b0;
                end
            end

            if (weight_fire) begin
                if (!weight_contract_valid) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    weight_q[weight_source][weight_lane] <= weight_data;
                    if (expected_last) begin
                        weights_loaded_q <= 1'b1;
                    end else if (expected_lane_q == OUT_W'(OUT_LANES - 1)) begin
                        expected_lane_q <= '0;
                        expected_source_q <= expected_source_q + INDEX_W'(1);
                    end else begin
                        expected_lane_q <= expected_lane_q + OUT_W'(1);
                    end
                end
            end

            if (command_fire) begin
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    acc_q[lane] <= command_seed_acc[lane*ACC_W +: ACC_W];
            end

            if (source_fire) begin
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    if (stream_source_negative)
                        acc_q[lane] <= acc_q[lane]
                            - {{(ACC_W-W_W){weight_q[stream_source_index][lane][W_W-1]}},
                               weight_q[stream_source_index][lane]};
                    else
                        acc_q[lane] <= acc_q[lane]
                            + {{(ACC_W-W_W){weight_q[stream_source_index][lane][W_W-1]}},
                               weight_q[stream_source_index][lane]};
                end
                perf_weight_segment_reads <= perf_weight_segment_reads + PERF_W'(1);
                perf_accumulator_updates <= perf_accumulator_updates
                    + PERF_W'(OUT_LANES);
                if (stream_source_negative)
                    perf_negative_sources <= perf_negative_sources + PERF_W'(1);
                else
                    perf_positive_sources <= perf_positive_sources + PERF_W'(1);
            end

            if (output_fire) begin
                perf_commands <= perf_commands + PERF_W'(1);
                if (output_use_motion)
                    perf_motion_commands <= perf_motion_commands + PERF_W'(1);
                else
                    perf_local_commands <= perf_local_commands + PERF_W'(1);
            end
        end
    end
endmodule

`default_nettype wire
