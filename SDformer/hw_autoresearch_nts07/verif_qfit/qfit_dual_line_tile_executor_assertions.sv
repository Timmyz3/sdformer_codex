`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_tile_executor_assertions #(
    parameter int TAG_W = 24,
    parameter int COUNT_W = 9,
    parameter int OUTPUT_W = 512,
    parameter int PERF_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic weights_loaded,
    input logic command_valid,
    input logic command_ready,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic output_use_motion,
    input logic [COUNT_W-1:0] output_source_count,
    input logic [OUTPUT_W-1:0] output_acc,
    input logic [PERF_W-1:0] perf_commands,
    input logic [PERF_W-1:0] perf_local_commands,
    input logic [PERF_W-1:0] perf_motion_commands,
    input logic [PERF_W-1:0] perf_weight_segment_reads,
    input logic [PERF_W-1:0] perf_positive_sources,
    input logic [PERF_W-1:0] perf_negative_sources
);
    property p_output_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        output_valid && !output_ready |=> output_valid
            && $stable({output_tag, output_use_motion,
                        output_source_count, output_acc});
    endproperty
    assert property (p_output_stable_under_stall);

    property p_command_requires_weights;
        @(posedge clk_core) disable iff (rst_core)
        command_valid && command_ready |-> weights_loaded;
    endproperty
    assert property (p_command_requires_weights);

    property p_command_accounting;
        @(posedge clk_core) disable iff (rst_core)
        perf_commands == perf_local_commands + perf_motion_commands;
    endproperty
    assert property (p_command_accounting);

    property p_source_accounting;
        @(posedge clk_core) disable iff (rst_core)
        perf_weight_segment_reads == perf_positive_sources + perf_negative_sources;
    endproperty
    assert property (p_source_accounting);

    cover property (@(posedge clk_core) disable iff (rst_core)
        output_valid && output_ready && !output_use_motion && output_source_count == '0);
    cover property (@(posedge clk_core) disable iff (rst_core)
        output_valid && output_ready && output_use_motion && output_source_count == '0);
endmodule

`default_nettype wire
