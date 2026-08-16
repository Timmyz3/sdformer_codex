`default_nettype none

module h68_castling_deploy_top #(
    parameter int HEAD_DIM    = 32,
    parameter int MAX_TOKENS  = 162,
    parameter int GATE_W      = 9,
    parameter int THRESHOLD_W = 8,
    parameter int TOKEN_W     = $clog2(MAX_TOKENS + 1),
    parameter int ACTIVE_MEM_DEPTH = MAX_TOKENS,
    parameter int SCORE_CLASS_DEPTH = 3
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_frame,
    input  logic                         cfg_enable_score_fold,
    input  logic                         cfg_preserve_mean,
    input  logic [THRESHOLD_W-1:0]       cfg_threshold_q8,

    output logic                         row_req_valid,
    input  logic                         row_req_ready,
    output logic [1:0]                   row_stage,
    output logic [2:0]                   row_block,
    output logic [4:0]                   row_head,
    output logic [9:0]                   row_window,
    output logic [TOKEN_W-1:0]           row_n_tokens,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic                         in_last,
    input  logic [HEAD_DIM-1:0]          in_q_bits,
    input  logic [HEAD_DIM-1:0]          in_k_bits,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic                         out_last,
    output logic [TOKEN_W-1:0]           out_token_idx,
    output logic [HEAD_DIM-1:0]          out_k_bits,
    output logic [GATE_W-1:0]            out_gate_q8,
    output logic [THRESHOLD_W-1:0]       out_threshold_q8,

    output logic                         busy,
    output logic                         done,
    output logic [15:0]                  perf_rows_issued,
    output logic [TOKEN_W-1:0]           perf_row_tokens_loaded,
    output logic [TOKEN_W-1:0]           perf_row_kzero_folded,
    output logic                         perf_score_range_error
);
    logic scheduler_busy;
    logic scheduler_done;
    logic row_engine_busy;
    logic row_engine_done;
    logic row_req_accept;
    logic [TOKEN_W-1:0] unused_entries_emitted;
    logic [1:0] unused_fold_classes;
    logic [15:0] unused_exp_transactions;

    assign row_req_accept = row_req_valid && row_req_ready && !row_engine_busy;

    ttx_descriptor_scheduler #(
        .TOKEN_W(TOKEN_W),
        .WINDOW_W(10),
        .HEAD_W(5)
    ) u_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .start_frame(start_frame),
        .row_req_valid(row_req_valid),
        .row_req_ready(row_req_ready && !row_engine_busy),
        .row_stage(row_stage),
        .row_block(row_block),
        .row_head(row_head),
        .row_window(row_window),
        .row_n_tokens(row_n_tokens),
        .row_done(row_engine_done),
        .busy(scheduler_busy),
        .done(scheduler_done),
        .perf_rows_issued(perf_rows_issued)
    );

    h67_score_class_row_engine #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .ENABLE_MOTION_XOR(1'b0),
        .ACTIVE_MEM_DEPTH(ACTIVE_MEM_DEPTH),
        .SCORE_CLASS_DEPTH(SCORE_CLASS_DEPTH),
        .GATE_W(GATE_W),
        .THRESHOLD_W(THRESHOLD_W),
        .TOKEN_W(TOKEN_W)
    ) u_deploy_row_engine (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_start(row_req_accept),
        .cfg_n_tokens(row_n_tokens),
        .cfg_preserve_mean(cfg_preserve_mean),
        .cfg_enable_score_fold(cfg_enable_score_fold),
        .cfg_threshold_q8(cfg_threshold_q8),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_last(in_last),
        .in_time_sel(1'b0),
        .in_q_bits(in_q_bits),
        .in_k_pair_bits({in_k_bits, in_k_bits}),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last),
        .out_token_idx(out_token_idx),
        .out_k_bits(out_k_bits),
        .out_gate_q8(out_gate_q8),
        .out_threshold_q8(out_threshold_q8),
        .busy(row_engine_busy),
        .done(row_engine_done),
        .perf_tokens_loaded(perf_row_tokens_loaded),
        .perf_kzero_folded(perf_row_kzero_folded),
        .perf_entries_emitted(unused_entries_emitted),
        .perf_fold_classes(unused_fold_classes),
        .perf_exp_transactions(unused_exp_transactions),
        .perf_score_range_error(perf_score_range_error)
    );

    assign busy = scheduler_busy || row_engine_busy;
    assign done = scheduler_done;
endmodule

`default_nettype wire
