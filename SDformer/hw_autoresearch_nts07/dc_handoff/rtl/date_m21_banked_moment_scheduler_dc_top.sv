`timescale 1ns/1ps

// Standalone logic-only DC wrapper for the M21 banked raw-moment scheduler.
// The register banks are intentionally inferred here; this wrapper does not
// claim an SRAM macro implementation or complete dynamic-BN datapath.
module date_m21_banked_moment_scheduler_dc_top #(
    parameter int IN_W = 32,
    parameter int TAG_W = 48,
    parameter int MAX_REDUCTION_POPULATION = 4194304,
    parameter int MAX_LANE_TILES = 16,
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1),
    localparam int TILE_W = (MAX_LANE_TILES <= 2) ? 1 : $clog2(MAX_LANE_TILES),
    localparam int ACTIVE_W = $clog2(MAX_LANE_TILES + 1),
    localparam int GROWTH_W =
        (MAX_REDUCTION_POPULATION <= 1) ? 0 : $clog2(MAX_REDUCTION_POPULATION),
    localparam int SUM_W = IN_W + GROWTH_W,
    localparam int SUMSQ_W = (2*IN_W)-1 + GROWTH_W
) (
    input  logic                              clk_core,
    input  logic                              rst_core,
    input  logic                              operator_start_valid,
    output logic                              operator_start_ready,
    input  logic [COUNT_W-1:0]                operator_reduction_population,
    input  logic [ACTIVE_W-1:0]               operator_active_lane_tiles,
    input  logic [TAG_W-1:0]                  operator_start_tag,
    output logic                              operator_start_legal,
    output logic                              operator_active,
    output logic [COUNT_W-1:0]                active_reduction_population,
    output logic [ACTIVE_W-1:0]               active_lane_tiles,
    output logic [TAG_W-1:0]                  active_tag,
    input  logic                              packet_valid,
    output logic                              packet_ready,
    input  logic [TILE_W-1:0]                 packet_lane_tile_id,
    input  logic                              packet_first,
    input  logic                              packet_last,
    input  logic [(96*IN_W)-1:0]              packet_values,
    output logic                              packet_legal,
    output logic [COUNT_W-1:0]                packet_accepted_count,
    output logic                              result_valid,
    input  logic                              result_ready,
    output logic [TAG_W-1:0]                  result_tag,
    output logic [TILE_W-1:0]                 result_lane_tile_id,
    output logic [2:0]                        result_slice_id,
    output logic [COUNT_W-1:0]                result_count,
    output logic [(16*SUM_W)-1:0]             result_sum,
    output logic [(16*SUMSQ_W)-1:0]           result_sumsq,
    output logic                              operator_done,
    output logic [TAG_W-1:0]                  operator_done_tag,
    output logic                              protocol_error,
    output logic [2:0]                        fifo_level,
    output logic [2:0]                        serializer_slice
);
    qfit_dynamic_bn_banked_moment_scheduler #(
        .IN_W(IN_W), .TAG_W(TAG_W),
        .MAX_REDUCTION_POPULATION(MAX_REDUCTION_POPULATION),
        .MAX_LANE_TILES(MAX_LANE_TILES)
    ) u_scheduler (.*);
endmodule
