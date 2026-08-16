`timescale 1ns/1ps
`default_nettype none

// 生产物理代理顶层：只暴露部署数据/控制接口，验证用perf计数器在综合时被裁除。
module h67_zkqi_row_shiftmax_physical_top #(
    parameter bit ZK_BYPASS_ENABLE = 1'b1,
    parameter bit BUNDLE_SKIP_ENABLE = 1'b1,
    parameter int ROW_MEMORY_IMPL = 1,
    parameter int DIRECTORY_MEMORY_IMPL = 1
) (
    input  logic          clk_core,
    input  logic          rst_core,
    input  logic          row_load_start,
    input  logic          row_load_valid,
    output logic          row_load_ready,
    input  logic [7:0]    row_load_pair_id,
    input  logic [63:0]   row_load_q_pair,
    input  logic [63:0]   row_load_k_pair,
    output logic          row_loaded,
    input  logic          window_start,
    input  logic          window_seal,
    input  logic          descriptor_issue_enable,
    input  logic          cfg_preserve_mean,
    input  logic [7:0]    cfg_threshold_q8,
    output logic          seal_ready,
    output logic          window_done,
    output logic          out_valid,
    input  logic          out_ready,
    output logic          out_last,
    output logic [8:0]    out_token_id,
    output logic [31:0]   out_k_bits,
    output logic [8:0]    out_gate_q17,
    output logic [7:0]    out_threshold_q8,
    output logic          protocol_error
);
    logic [31:0] unused_perf [0:16];
    logic unused_fifo_occupancy;
    logic unused_fifo_max_occupancy;

    h67_zkqi_row_shiftmax_top #(
        .ZK_BYPASS_ENABLE(ZK_BYPASS_ENABLE),
        .BUNDLE_SKIP_ENABLE(BUNDLE_SKIP_ENABLE),
        .ROW_MEMORY_IMPL(ROW_MEMORY_IMPL),
        .DIRECTORY_MEMORY_IMPL(DIRECTORY_MEMORY_IMPL)
    ) u_core (
        .clk_core,
        .rst_core,
        .row_load_start,
        .row_load_valid,
        .row_load_ready,
        .row_load_pair_id,
        .row_load_q_pair,
        .row_load_k_pair,
        .row_loaded,
        .window_start,
        .window_seal,
        .descriptor_issue_enable,
        .cfg_preserve_mean,
        .cfg_threshold_q8,
        .seal_ready,
        .window_done,
        .out_valid,
        .out_ready,
        .out_last,
        .out_token_id,
        .out_k_bits,
        .out_gate_q17,
        .out_threshold_q8,
        .protocol_error,
        .perf_score_pairs(unused_perf[0]),
        .perf_score_slots(unused_perf[1]),
        .perf_original_tokens(unused_perf[2]),
        .perf_equal_pairs(unused_perf[3]),
        .perf_seeded_tokens(unused_perf[4]),
        .perf_active_entries(unused_perf[5]),
        .perf_class_transactions(unused_perf[6]),
        .perf_exp_transactions(unused_perf[7]),
        .perf_emitted_tokens(unused_perf[8]),
        .perf_row_read_transactions(unused_perf[9]),
        .perf_row_read_bits(unused_perf[10]),
        .perf_preload_cycles(unused_perf[11]),
        .perf_total_cycles(unused_perf[12]),
        .perf_score_stall_cycles(unused_perf[13]),
        .perf_output_stall_cycles(unused_perf[14]),
        .perf_preclassified_pairs(unused_perf[15]),
        .perf_metadata_bits(unused_perf[16]),
        .perf_fifo_occupancy(unused_fifo_occupancy),
        .perf_fifo_max_occupancy(unused_fifo_max_occupancy)
    );
endmodule

`default_nettype wire
