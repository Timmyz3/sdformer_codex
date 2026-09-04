`timescale 1ns/1ps
`default_nettype none

// Parameter-free paper configurations for matched DC/Formality runs.
// Both designs include the same MSSB5 score front; only quotienting changes.
module h67_fixed2s_mssb5_dc_top (
    input  logic          clk_core,
    input  logic          rst_core,
    input  logic          window_start,
    input  logic          window_seal,
    input  logic          descriptor_issue_enable,
    input  logic          cfg_preserve_mean,
    input  logic [7:0]    cfg_threshold_q8,
    output logic          seal_ready,
    output logic          window_done,
    input  logic          pair_valid,
    output logic          pair_ready,
    input  logic [7:0]    pair_id,
    input  logic [63:0]   q_pair,
    input  logic [63:0]   k_pair,
    output logic          out_valid,
    input  logic          out_ready,
    output logic          out_last,
    output logic [8:0]    out_token_id,
    output logic [31:0]   out_k_bits,
    output logic [8:0]    out_gate_q17,
    output logic [7:0]    out_threshold_q8,
    output logic          protocol_error,
    output logic [31:0]   perf_pairs,
    output logic [31:0]   perf_slots,
    output logic [31:0]   perf_equal_pairs,
    output logic [31:0]   perf_quotient_descriptors,
    output logic [31:0]   perf_original_tokens,
    output logic [31:0]   perf_active_entries,
    output logic [31:0]   perf_class_transactions,
    output logic [31:0]   perf_exp_transactions,
    output logic [31:0]   perf_emitted_tokens,
    output logic [31:0]   perf_k_read_transactions,
    output logic [31:0]   perf_k_read_bits,
    output logic [31:0]   perf_total_cycles,
    output logic [31:0]   perf_pair_stall_cycles,
    output logic [31:0]   perf_descriptor_stall_cycles,
    output logic [31:0]   perf_output_stall_cycles,
    output logic [5:0]    perf_fifo_occupancy,
    output logic [5:0]    perf_fifo_max_occupancy
);
    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .HEAD_DIM(32),
        .PAIRS(225),
        .SLOT_FIFO_DEPTH(32),
        .QUOTIENT_ENABLE(1'b0),
        .MSSB5_SCORE_FRONT(1'b1),
        .MEMORY_IMPL(0)
    ) u_core (.*);
endmodule

module h67_rqtb2s_mssb5_dc_top (
    input  logic          clk_core,
    input  logic          rst_core,
    input  logic          window_start,
    input  logic          window_seal,
    input  logic          descriptor_issue_enable,
    input  logic          cfg_preserve_mean,
    input  logic [7:0]    cfg_threshold_q8,
    output logic          seal_ready,
    output logic          window_done,
    input  logic          pair_valid,
    output logic          pair_ready,
    input  logic [7:0]    pair_id,
    input  logic [63:0]   q_pair,
    input  logic [63:0]   k_pair,
    output logic          out_valid,
    input  logic          out_ready,
    output logic          out_last,
    output logic [8:0]    out_token_id,
    output logic [31:0]   out_k_bits,
    output logic [8:0]    out_gate_q17,
    output logic [7:0]    out_threshold_q8,
    output logic          protocol_error,
    output logic [31:0]   perf_pairs,
    output logic [31:0]   perf_slots,
    output logic [31:0]   perf_equal_pairs,
    output logic [31:0]   perf_quotient_descriptors,
    output logic [31:0]   perf_original_tokens,
    output logic [31:0]   perf_active_entries,
    output logic [31:0]   perf_class_transactions,
    output logic [31:0]   perf_exp_transactions,
    output logic [31:0]   perf_emitted_tokens,
    output logic [31:0]   perf_k_read_transactions,
    output logic [31:0]   perf_k_read_bits,
    output logic [31:0]   perf_total_cycles,
    output logic [31:0]   perf_pair_stall_cycles,
    output logic [31:0]   perf_descriptor_stall_cycles,
    output logic [31:0]   perf_output_stall_cycles,
    output logic [5:0]    perf_fifo_occupancy,
    output logic [5:0]    perf_fifo_max_occupancy
);
    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .HEAD_DIM(32),
        .PAIRS(225),
        .SLOT_FIFO_DEPTH(32),
        .QUOTIENT_ENABLE(1'b1),
        .MSSB5_SCORE_FRONT(1'b1),
        .MEMORY_IMPL(0)
    ) u_core (.*);
endmodule

`default_nettype wire
