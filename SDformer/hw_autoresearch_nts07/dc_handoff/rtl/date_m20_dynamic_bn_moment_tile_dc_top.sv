`timescale 1ns/1ps

// Standalone DC wrapper.  It deliberately has no dependency on the active
// DATE dual-line synthesis filelists or tops.
module date_m20_dynamic_bn_moment_tile_dc_top #(
    parameter int IN_W = 32,
    parameter int MAX_REDUCTION_POPULATION = 4194304,
    localparam int LANES = 16,
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1),
    localparam int POP_GROWTH_W =
        (MAX_REDUCTION_POPULATION <= 1) ? 0 : $clog2(MAX_REDUCTION_POPULATION),
    localparam int SUM_W = IN_W + POP_GROWTH_W,
    localparam int SUMSQ_W = (2 * IN_W) - 1 + POP_GROWTH_W
) (
    input  logic                              clk_core,
    input  logic                              rst_core,
    input  logic                              in_valid,
    output logic                              in_ready,
    input  logic                              in_first,
    input  logic                              in_last,
    input  logic [COUNT_W-1:0]                reduction_population,
    input  logic [(LANES*IN_W)-1:0]           in_values,
    output logic                              request_legal,
    output logic                              busy,
    output logic [COUNT_W-1:0]                accepted_count,
    output logic [COUNT_W-1:0]                active_population,
    output logic                              protocol_error,
    output logic                              result_valid,
    input  logic                              result_ready,
    output logic [COUNT_W-1:0]                result_count,
    output logic [(LANES*SUM_W)-1:0]          result_sum,
    output logic [(LANES*SUMSQ_W)-1:0]        result_sumsq
);
    qfit_dynamic_bn_moment_tile #(
        .LANES(LANES), .IN_W(IN_W),
        .MAX_REDUCTION_POPULATION(MAX_REDUCTION_POPULATION)
    ) u_moment_tile (
        .clk_core, .rst_core, .in_valid, .in_ready, .in_first, .in_last,
        .reduction_population, .in_values, .request_legal, .busy,
        .accepted_count, .active_population, .protocol_error,
        .result_valid, .result_ready, .result_count, .result_sum,
        .result_sumsq
    );
endmodule
