`timescale 1ns/1ps
`default_nettype none

module h67_temporal_quotient_scs_frontend #(
    parameter int HEAD_DIM = 32,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 9,
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 162,
    parameter int COUNT_W = $clog2(2 * MAX_DESCRIPTORS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       window_seal,
    output logic                       seal_ready,
    output logic                       window_done,
    input  logic                       pair_valid,
    output logic                       pair_ready,
    input  logic [PAIR_ID_W-1:0]       pair_id,
    input  logic [2*HEAD_DIM-1:0]      q_pair,
    input  logic [2*HEAD_DIM-1:0]      k_pair,
    output logic                       class_valid,
    input  logic                       class_ready,
    output logic [CLASS_W-1:0]         class_score,
    output logic [COUNT_W-1:0]         class_multiplicity,
    output logic                       class_last,
    output logic                       active_valid,
    input  logic                       active_ready,
    output logic [PAIR_ID_W-1:0]       active_pair_id,
    output logic signed [SCORE_W-1:0]  active_score_q7,
    output logic [1:0]                 active_temporal_mask,
    output logic [1:0]                 active_k_mask,
    output logic                       active_last,
    output logic signed [SCORE_W-1:0]  row_max_q7,
    output logic                       protocol_error,
    output logic [31:0]                perf_pairs,
    output logic [31:0]                perf_quotient_descriptors,
    output logic [31:0]                perf_original_tokens,
    output logic [31:0]                perf_active_entries,
    output logic [31:0]                perf_equal_pairs
);
    logic quotient_valid;
    logic quotient_ready;
    logic quotient_input_ready;
    logic [PAIR_ID_W-1:0] quotient_pair_id;
    logic signed [SCORE_W-1:0] quotient_score;
    logic [1:0] quotient_temporal_mask;
    logic [1:0] quotient_active_mask;
    logic [31:0] quotient_descriptors;
    logic directory_ready;
    logic directory_error;

    h67_temporal_score_quotient #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W)
    ) u_quotient (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(pair_valid && quotient_ready),
        .in_ready(quotient_input_ready),
        .in_pair_id(pair_id),
        .in_q_pair(q_pair),
        .in_k_pair(k_pair),
        .out_valid(quotient_valid),
        .out_ready(quotient_ready),
        .out_pair_id(quotient_pair_id),
        .out_score_q7(quotient_score),
        .out_temporal_mask(quotient_temporal_mask),
        .out_active_mask(quotient_active_mask),
        .out_last(),
        .perf_pairs(perf_pairs),
        .perf_descriptors(quotient_descriptors),
        .perf_equal_pairs(perf_equal_pairs)
    );

    h67_temporal_weighted_scs_directory #(
        .MAX_SCORE(MAX_SCORE),
        .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .COUNT_W(COUNT_W),
        .CLASS_W(CLASS_W)
    ) u_directory (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .window_seal(window_seal),
        .window_ready(directory_ready),
        .window_done(window_done),
        .in_valid(quotient_valid),
        .in_ready(quotient_ready),
        .in_pair_id(quotient_pair_id),
        .in_score_q7(quotient_score),
        .in_temporal_mask(quotient_temporal_mask),
        .in_active_mask(quotient_active_mask),
        .class_valid(class_valid),
        .class_ready(class_ready),
        .class_score(class_score),
        .class_multiplicity(class_multiplicity),
        .class_last(class_last),
        .active_valid(active_valid),
        .active_ready(active_ready),
        .active_pair_id(active_pair_id),
        .active_score_q7(active_score_q7),
        .active_temporal_mask(active_temporal_mask),
        .active_k_mask(active_k_mask),
        .active_last(active_last),
        .row_max_q7(row_max_q7),
        .protocol_error(directory_error),
        .perf_quotient_descriptors(perf_quotient_descriptors),
        .perf_original_tokens(perf_original_tokens),
        .perf_active_entries(perf_active_entries)
    );

    assign pair_ready = quotient_input_ready && quotient_ready;
    assign seal_ready = directory_ready == 1'b0
                     && quotient_input_ready
                     && !quotient_valid;
    assign protocol_error = directory_error
                          || (window_seal && !seal_ready)
                          || (window_done
                              && quotient_descriptors
                                 != perf_quotient_descriptors);
endmodule

`default_nettype wire
