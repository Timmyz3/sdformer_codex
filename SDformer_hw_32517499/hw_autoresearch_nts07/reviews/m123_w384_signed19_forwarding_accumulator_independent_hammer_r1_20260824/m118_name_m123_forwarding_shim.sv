`timescale 1ns/1ps
`default_nettype none

// Review-only name shim: allows the frozen M120 wrapper to elaborate without
// editing production while replacing its M118 accumulator instance by M123.
module m118_w384_signed19_lane_sliced_accumulator_adapter (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         window_start_valid,
    output logic                         window_start_ready,
    output logic                         window_start_accept,
    input  logic                         update_valid,
    output logic                         update_ready,
    input  logic [2:0]                   update_block,
    input  logic [8:0]                   update_row,
    input  logic [1823:0]                update_delta,
    output logic                         update_accept,
    input  logic                         window_end_valid,
    output logic                         window_end_ready,
    output logic                         window_end_accept,
    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [2:0]                   commit_block,
    output logic [8:0]                   commit_row,
    output logic [1823:0]                commit_data,
    output logic                         commit_last,
    output logic                         window_done,
    output logic                         lane_mem_rd_en,
    output logic [11:0]                  lane_mem_rd_addr,
    input  logic [18:0]                  lane_mem_rd_data [0:95],
    output logic                         lane_mem_wr_en,
    output logic [11:0]                  lane_mem_wr_addr,
    output logic [18:0]                  lane_mem_wr_data [0:95],
    output logic                         protocol_error,
    output logic                         window_active,
    output logic                         busy
);
    m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter replacement (.*);
endmodule

`default_nettype wire
