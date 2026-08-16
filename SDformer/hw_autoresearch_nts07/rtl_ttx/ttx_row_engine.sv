`default_nettype none

module ttx_row_engine #(
    parameter int HEAD_DIM    = 32,
    parameter int MAX_TOKENS  = 162,
    parameter int SCORE_W     = 16,
    parameter int SCORE_FRAC  = 7,
    parameter int GATE_W      = 9,
    parameter int GATE_FRAC   = 7,
    parameter int THRESHOLD_W = 8,
    parameter int ALPHA0_Q8   = 5,
    parameter int TOKEN_W     = $clog2(MAX_TOKENS + 1),
    parameter int ACTIVE_W    = $clog2(MAX_TOKENS + 1),
    parameter int ACTIVE_MEM_DEPTH = MAX_TOKENS,
    parameter int CLASS_MEM_DEPTH = HEAD_DIM + 1,
    parameter int CLASS_W     = $clog2(HEAD_DIM + 1)
)(
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         cfg_start,
    input  logic [TOKEN_W-1:0]           cfg_n_tokens,
    input  logic                         cfg_preserve_mean,
    input  logic                         cfg_enable_zfold,
    input  logic [THRESHOLD_W-1:0]       cfg_threshold_q8,

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
    output logic [TOKEN_W-1:0]           perf_tokens_loaded,
    output logic [TOKEN_W-1:0]           perf_kzero_folded,
    output logic [TOKEN_W-1:0]           perf_entries_emitted,
    output logic [CLASS_W-1:0]           perf_fold_classes,
    output logic [15:0]                  perf_exp_transactions
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD,
        ST_SUM_ACTIVE,
        ST_SUM_FOLD,
        ST_EMIT,
        ST_DONE
    } state_t;

    localparam int ADDR_W = $clog2(MAX_TOKENS);
    localparam int ACTIVE_ENTRY_W = SCORE_W + HEAD_DIM + TOKEN_W;

    state_t state_q;

    logic [TOKEN_W-1:0] n_tokens_q;
    logic preserve_mean_q;
    logic enable_zfold_q;
    logic [THRESHOLD_W-1:0] threshold_q8_q;

    logic [TOKEN_W-1:0] load_idx_q;
    logic [ACTIVE_W-1:0] active_count_q;
    logic [ACTIVE_W-1:0] scan_idx_q;
    logic [ACTIVE_W-1:0] emit_idx_q;
    logic [CLASS_W-1:0] class_idx_q;

    logic [ACTIVE_ENTRY_W-1:0] active_entry_mem [0:ACTIVE_MEM_DEPTH-1];
    logic [TOKEN_W-1:0] zero_k_hist [0:CLASS_MEM_DEPTH-1];

    logic signed [SCORE_W-1:0] row_max_q;
    logic [31:0] row_sum_q;
    logic [TOKEN_W-1:0] tokens_loaded_q;
    logic [TOKEN_W-1:0] folded_tokens_q;
    logic [TOKEN_W-1:0] emitted_entries_q;
    logic [CLASS_W-1:0] fold_classes_q;
    logic [15:0] exp_transactions_q;

    logic [CLASS_W-1:0] score_q_active_w;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [CLASS_W-1:0] score_k_active_unused;
    logic [CLASS_W-1:0] score_overlap_unused;
    logic [CLASS_W-1:0] score_same_zero_unused;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [SCORE_W-1:0] input_score_w;
    logic signed [SCORE_W-1:0] active_delta_w;
    logic [15:0] active_exp_w;
    logic [CLASS_W-1:0] class_q_active_w;
    logic signed [SCORE_W-1:0] class_score_w;
    logic signed [SCORE_W-1:0] class_delta_w;
    logic [15:0] class_exp_w;
    logic [31:0] class_sum_term_w;
    logic [GATE_W-1:0] gate_w;
    logic input_k_zero_w;
    logic fold_input_w;
    logic [ADDR_W-1:0] active_read_addr_w;
    logic [ACTIVE_ENTRY_W-1:0] active_read_entry_w;
    logic signed [SCORE_W-1:0] active_read_score_w;
    logic [HEAD_DIM-1:0] active_read_k_w;
    logic [TOKEN_W-1:0] active_read_token_w;
    logic [TOKEN_W-1:0] hist_scan_count_w;
    logic [TOKEN_W-1:0] hist_input_count_w;

    integer hist_idx;
    integer active_mem_idx;
    integer hist_read_idx;

    ttx_tx_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC),
        .ALPHA0_Q8(ALPHA0_Q8),
        .COUNT_W(CLASS_W)
    ) u_score (
        .q_bits(in_q_bits),
        .k_bits(in_k_bits),
        .q_active(score_q_active_w),
        .k_active(score_k_active_unused),
        .overlap(score_overlap_unused),
        .same_zero(score_same_zero_unused),
        .score_q7(input_score_w)
    );

    assign active_read_addr_w = (state_q == ST_EMIT)
                              ? emit_idx_q[ADDR_W-1:0] : scan_idx_q[ADDR_W-1:0];
    always_comb begin
        active_read_entry_w = '0;
        for (active_mem_idx = 0; active_mem_idx < ACTIVE_MEM_DEPTH; active_mem_idx = active_mem_idx + 1) begin
            if (active_read_addr_w == ADDR_W'(active_mem_idx)) begin
                active_read_entry_w = active_entry_mem[active_mem_idx];
            end
        end

        hist_scan_count_w = '0;
        hist_input_count_w = '0;
        for (hist_read_idx = 0; hist_read_idx < CLASS_MEM_DEPTH; hist_read_idx = hist_read_idx + 1) begin
            if (class_idx_q == CLASS_W'(hist_read_idx)) begin
                hist_scan_count_w = zero_k_hist[hist_read_idx];
            end
            if (score_q_active_w == CLASS_W'(hist_read_idx)) begin
                hist_input_count_w = zero_k_hist[hist_read_idx];
            end
        end
    end
    assign active_read_token_w = active_read_entry_w[TOKEN_W-1:0];
    assign active_read_k_w = active_read_entry_w[TOKEN_W +: HEAD_DIM];
    assign active_read_score_w = $signed(active_read_entry_w[TOKEN_W+HEAD_DIM +: SCORE_W]);
    assign active_delta_w = active_read_score_w - row_max_q;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC)
    ) u_active_exp (
        .delta_q7(active_delta_w),
        .exp_q8(active_exp_w)
    );

    assign class_q_active_w = CLASS_W'(class_idx_q);

    ttx_zero_k_class_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC),
        .ALPHA0_Q8(ALPHA0_Q8),
        .COUNT_W(CLASS_W)
    ) u_class_score (
        .q_active(class_q_active_w),
        .score_q7(class_score_w)
    );

    assign class_delta_w = class_score_w - row_max_q;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC)
    ) u_class_exp (
        .delta_q7(class_delta_w),
        .exp_q8(class_exp_w)
    );

    assign class_sum_term_w = hist_scan_count_w * class_exp_w;

    ttx_gate_quant_q17 #(
        .TOKEN_W(TOKEN_W),
        .GATE_W(GATE_W),
        .GATE_FRAC(GATE_FRAC)
    ) u_gate_quant (
        .exp_q8(active_exp_w),
        .row_sum_q8(row_sum_q),
        .n_tokens(n_tokens_q),
        .preserve_mean(preserve_mean_q),
        .gate_q17(gate_w)
    );

    assign input_k_zero_w = ~(|in_k_bits);
    assign fold_input_w = enable_zfold_q && input_k_zero_w;

    always_comb begin
        in_ready = (state_q == ST_LOAD);
        out_valid = (state_q == ST_EMIT) && (active_count_q != 0);
        out_last = out_valid && (emit_idx_q == active_count_q - 1'b1);
        out_token_idx = active_read_token_w;
        out_k_bits = active_read_k_w;
        out_gate_q8 = gate_w;
        out_threshold_q8 = threshold_q8_q;
        busy = (state_q != ST_IDLE);
        done = (state_q == ST_DONE);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            n_tokens_q <= '0;
            preserve_mean_q <= 1'b1;
            enable_zfold_q <= 1'b1;
            threshold_q8_q <= '0;
            load_idx_q <= '0;
            active_count_q <= '0;
            scan_idx_q <= '0;
            emit_idx_q <= '0;
            class_idx_q <= '0;
            row_max_q <= '0;
            row_sum_q <= '0;
            tokens_loaded_q <= '0;
            folded_tokens_q <= '0;
            emitted_entries_q <= '0;
            fold_classes_q <= '0;
            exp_transactions_q <= '0;
            for (hist_idx = 0; hist_idx < CLASS_MEM_DEPTH; hist_idx = hist_idx + 1) begin
                zero_k_hist[hist_idx] <= '0;
            end
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    if (cfg_start) begin
                        state_q <= ST_LOAD;
                        n_tokens_q <= (cfg_n_tokens == 0 || cfg_n_tokens > TOKEN_W'(MAX_TOKENS))
                                      ? TOKEN_W'(MAX_TOKENS) : cfg_n_tokens;
                        preserve_mean_q <= cfg_preserve_mean;
                        enable_zfold_q <= cfg_enable_zfold;
                        threshold_q8_q <= cfg_threshold_q8;
                        load_idx_q <= '0;
                        active_count_q <= '0;
                        scan_idx_q <= '0;
                        emit_idx_q <= '0;
                        class_idx_q <= '0;
                        row_max_q <= -$signed({1'b0, {SCORE_W-1{1'b1}}});
                        row_sum_q <= '0;
                        tokens_loaded_q <= '0;
                        folded_tokens_q <= '0;
                        emitted_entries_q <= '0;
                        fold_classes_q <= '0;
                        exp_transactions_q <= '0;
                        for (hist_idx = 0; hist_idx < CLASS_MEM_DEPTH; hist_idx = hist_idx + 1) begin
                            zero_k_hist[hist_idx] <= '0;
                        end
                    end
                end

                ST_LOAD: begin
                    if (in_valid && in_ready) begin
                        tokens_loaded_q <= tokens_loaded_q + 1'b1;
                        if (tokens_loaded_q == 0 || input_score_w > row_max_q) begin
                            row_max_q <= input_score_w;
                        end

                        if (fold_input_w) begin
                            zero_k_hist[score_q_active_w] <= hist_input_count_w + 1'b1;
                            folded_tokens_q <= folded_tokens_q + 1'b1;
                        end else begin
                            active_entry_mem[active_count_q[ADDR_W-1:0]]
                                <= {input_score_w, in_k_bits, load_idx_q};
                            active_count_q <= active_count_q + 1'b1;
                        end

                        load_idx_q <= load_idx_q + 1'b1;
                        if (in_last || load_idx_q == n_tokens_q - 1'b1) begin
                            n_tokens_q <= load_idx_q + 1'b1;
                            scan_idx_q <= '0;
                            class_idx_q <= '0;
                            row_sum_q <= '0;
                            state_q <= ST_SUM_ACTIVE;
                        end
                    end
                end

                ST_SUM_ACTIVE: begin
                    if (active_count_q == 0) begin
                        class_idx_q <= '0;
                        state_q <= (folded_tokens_q == 0) ? ST_DONE : ST_SUM_FOLD;
                    end else begin
                        row_sum_q <= row_sum_q + {16'd0, active_exp_w};
                        exp_transactions_q <= exp_transactions_q + 1'b1;
                        if (scan_idx_q == active_count_q - 1'b1) begin
                            scan_idx_q <= '0;
                            class_idx_q <= '0;
                            state_q <= (folded_tokens_q == 0) ? ST_EMIT : ST_SUM_FOLD;
                        end else begin
                            scan_idx_q <= scan_idx_q + 1'b1;
                        end
                    end
                end

                ST_SUM_FOLD: begin
                    if (hist_scan_count_w != 0) begin
                        row_sum_q <= row_sum_q + class_sum_term_w;
                        fold_classes_q <= fold_classes_q + 1'b1;
                        exp_transactions_q <= exp_transactions_q + 1'b1;
                    end
                    if (class_idx_q == CLASS_W'(HEAD_DIM)) begin
                        emit_idx_q <= '0;
                        state_q <= (active_count_q == 0) ? ST_DONE : ST_EMIT;
                    end else begin
                        class_idx_q <= class_idx_q + 1'b1;
                    end
                end

                ST_EMIT: begin
                    if (out_valid && out_ready) begin
                        emitted_entries_q <= emitted_entries_q + 1'b1;
                        exp_transactions_q <= exp_transactions_q + 1'b1;
                        if (out_last) begin
                            state_q <= ST_DONE;
                        end else begin
                            emit_idx_q <= emit_idx_q + 1'b1;
                            scan_idx_q <= scan_idx_q + 1'b1;
                        end
                    end
                end

                ST_DONE: begin
                    state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

    assign perf_tokens_loaded = tokens_loaded_q;
    assign perf_kzero_folded = folded_tokens_q;
    assign perf_entries_emitted = emitted_entries_q;
    assign perf_fold_classes = fold_classes_q;
    assign perf_exp_transactions = exp_transactions_q;
endmodule

`default_nettype wire
