`timescale 1ns/1ps
`default_nettype none

// 在Shiftmax归一化域合并等分数时间对，再在gated-K边界无损展开。
module h67_temporal_quotient_shiftmax_gate_top #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int TOKEN_W = $clog2(2 * PAIRS + 1),
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 2 * PAIRS,
    parameter int COUNT_W = $clog2(2 * PAIRS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       window_seal,
    input  logic                       cfg_preserve_mean,
    input  logic [THRESHOLD_W-1:0]     cfg_threshold_q8,
    output logic                       seal_ready,
    output logic                       window_done,

    input  logic                       pair_valid,
    output logic                       pair_ready,
    input  logic [PAIR_ID_W-1:0]       pair_id,
    input  logic [2*HEAD_DIM-1:0]      q_pair,
    input  logic [2*HEAD_DIM-1:0]      k_pair,

    output logic                       out_valid,
    input  logic                       out_ready,
    output logic                       out_last,
    output logic [TOKEN_W-1:0]         out_token_id,
    output logic [HEAD_DIM-1:0]        out_k_bits,
    output logic [GATE_W-1:0]          out_gate_q17,
    output logic [THRESHOLD_W-1:0]     out_threshold_q8,

    output logic                       protocol_error,
    output logic [31:0]                perf_pairs,
    output logic [31:0]                perf_quotient_descriptors,
    output logic [31:0]                perf_original_tokens,
    output logic [31:0]                perf_active_entries,
    output logic [31:0]                perf_equal_pairs,
    output logic [31:0]                perf_class_transactions,
    output logic [31:0]                perf_exp_transactions,
    output logic [31:0]                perf_emitted_tokens
);
    logic frontend_pair_ready;
    logic frontend_done;
    logic frontend_error;
    logic class_valid;
    logic [CLASS_W-1:0] class_score;
    logic [COUNT_W-1:0] class_multiplicity;
    logic class_last;
    logic active_valid;
    logic active_ready;
    logic [PAIR_ID_W-1:0] active_pair_id;
    logic signed [SCORE_W-1:0] active_score_q7;
    logic [1:0] active_temporal_mask;
    logic [1:0] active_k_mask;
    logic active_last;
    logic signed [SCORE_W-1:0] row_max_q7;

    logic [2*HEAD_DIM-1:0] k_pair_store [0:PAIRS-1];
    logic [PAIRS-1:0] pair_seen_q;
    logic pair_in_range;
    logic pair_legal;
    logic protocol_error_q;

    logic signed [SCORE_W-1:0] class_score_q7;
    logic signed [SCORE_W-1:0] class_delta_q7;
    logic [15:0] class_exp_q8;
    logic [31:0] class_sum_term;
    logic [31:0] row_sum_q8_q;

    logic emit_valid_q;
    logic [1:0] emit_mask_q;
    logic [PAIR_ID_W-1:0] emit_pair_id_q;
    logic signed [SCORE_W-1:0] emit_score_q7_q;
    logic emit_active_last_q;
    logic emit_time_sel;
    logic signed [SCORE_W-1:0] emit_delta_q7;
    logic [15:0] emit_exp_q8;
    logic [GATE_W-1:0] emit_gate_q17;
    logic [31:0] class_transactions_q;
    logic [31:0] emitted_tokens_q;
    logic [THRESHOLD_W-1:0] threshold_q8_q;
    logic preserve_mean_q;
    logic class_phase_done_q;

    assign pair_in_range = 32'(pair_id) < 32'(PAIRS);
    assign pair_ready = frontend_pair_ready && pair_legal;

    always_comb begin
        pair_legal = 1'b0;
        if (pair_in_range)
            pair_legal = !pair_seen_q[pair_id];
    end

    h67_temporal_quotient_scs_frontend #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .MAX_SCORE(MAX_SCORE),
        .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
        .COUNT_W(COUNT_W),
        .CLASS_W(CLASS_W)
    ) u_frontend (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .window_seal(window_seal),
        .seal_ready(seal_ready),
        .window_done(frontend_done),
        .pair_valid(pair_valid && pair_legal),
        .pair_ready(frontend_pair_ready),
        .pair_id(pair_id),
        .q_pair(q_pair),
        .k_pair(k_pair),
        .class_valid(class_valid),
        .class_ready(1'b1),
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
        .protocol_error(frontend_error),
        .perf_pairs(perf_pairs),
        .perf_quotient_descriptors(perf_quotient_descriptors),
        .perf_original_tokens(perf_original_tokens),
        .perf_active_entries(perf_active_entries),
        .perf_equal_pairs(perf_equal_pairs)
    );

    assign class_score_q7 = $signed(
        {{(SCORE_W-CLASS_W){1'b0}}, class_score}
    );
    assign class_delta_q7 = class_score_q7 - row_max_q7;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(7)
    ) u_class_exp (
        .delta_q7(class_delta_q7),
        .exp_q8(class_exp_q8)
    );

    assign class_sum_term = 32'(class_exp_q8) * 32'(class_multiplicity);

    assign active_ready = !emit_valid_q;
    assign emit_time_sel = !emit_mask_q[0];
    assign emit_delta_q7 = emit_score_q7_q - row_max_q7;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(7)
    ) u_emit_exp (
        .delta_q7(emit_delta_q7),
        .exp_q8(emit_exp_q8)
    );

    ttx_gate_quant_q17 #(
        .TOKEN_W(TOKEN_W),
        .GATE_W(GATE_W),
        .GATE_FRAC(7)
    ) u_gate_quant (
        .exp_q8(emit_exp_q8),
        .row_sum_q8(row_sum_q8_q),
        .n_tokens(TOKEN_W'(2 * PAIRS)),
        .preserve_mean(preserve_mean_q),
        .gate_q17(emit_gate_q17)
    );

    assign out_valid = emit_valid_q;
    assign out_token_id = TOKEN_W'(2 * 32'(emit_pair_id_q)
                                + 32'(emit_time_sel));
    assign out_k_bits = emit_time_sel
                      ? k_pair_store[emit_pair_id_q][2*HEAD_DIM-1:HEAD_DIM]
                      : k_pair_store[emit_pair_id_q][HEAD_DIM-1:0];
    assign out_gate_q17 = emit_gate_q17;
    assign out_threshold_q8 = threshold_q8_q;
    assign out_last = emit_valid_q && emit_active_last_q
                   && (emit_mask_q == 2'b01 || emit_mask_q == 2'b10);
    assign window_done = frontend_done && !emit_valid_q;
    assign protocol_error = frontend_error || protocol_error_q;
    assign perf_class_transactions = class_transactions_q;
    assign perf_exp_transactions = class_transactions_q + perf_active_entries;
    assign perf_emitted_tokens = emitted_tokens_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pair_seen_q <= '0;
            protocol_error_q <= 1'b0;
            row_sum_q8_q <= '0;
            emit_valid_q <= 1'b0;
            emit_mask_q <= '0;
            emit_pair_id_q <= '0;
            emit_score_q7_q <= '0;
            emit_active_last_q <= 1'b0;
            class_transactions_q <= '0;
            emitted_tokens_q <= '0;
            threshold_q8_q <= '0;
            preserve_mean_q <= 1'b0;
            class_phase_done_q <= 1'b0;
        end else begin
            if (window_start) begin
                pair_seen_q <= '0;
                protocol_error_q <= 1'b0;
                row_sum_q8_q <= '0;
                emit_valid_q <= 1'b0;
                emit_mask_q <= '0;
                class_transactions_q <= '0;
                emitted_tokens_q <= '0;
                threshold_q8_q <= cfg_threshold_q8;
                preserve_mean_q <= cfg_preserve_mean;
                class_phase_done_q <= 1'b0;
            end else begin
                if (pair_valid && frontend_pair_ready && !pair_legal)
                    protocol_error_q <= 1'b1;

                if (pair_valid && pair_ready) begin
                    pair_seen_q[pair_id] <= 1'b1;
                    k_pair_store[pair_id] <= k_pair;
                end

                if (class_valid) begin
                    row_sum_q8_q <= row_sum_q8_q + class_sum_term;
                    class_transactions_q <= class_transactions_q + 1'b1;
                    if (class_last)
                        class_phase_done_q <= 1'b1;
                end

                if (active_valid && active_ready) begin
                    if (!class_phase_done_q
                        || active_k_mask == 0
                        || (active_k_mask & ~active_temporal_mask) != 0) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        emit_valid_q <= 1'b1;
                        emit_mask_q <= active_k_mask;
                        emit_pair_id_q <= active_pair_id;
                        emit_score_q7_q <= active_score_q7;
                        emit_active_last_q <= active_last;
                    end
                end

                if (out_valid && out_ready) begin
                    emitted_tokens_q <= emitted_tokens_q + 1'b1;
                    if (emit_mask_q[0]) begin
                        emit_mask_q[0] <= 1'b0;
                        if (!emit_mask_q[1])
                            emit_valid_q <= 1'b0;
                    end else begin
                        emit_mask_q[1] <= 1'b0;
                        emit_valid_q <= 1'b0;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
