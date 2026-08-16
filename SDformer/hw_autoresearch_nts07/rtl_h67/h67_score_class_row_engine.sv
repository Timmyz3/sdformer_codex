`default_nettype none

module h67_score_class_row_engine #(
    parameter int HEAD_DIM      = 32,
    parameter int MAX_TOKENS    = 162,
    parameter bit ENABLE_MOTION_XOR = 1'b1,
    parameter int SCORE_W       = 16,
    parameter int SCORE_FRAC    = 7,
    parameter int SCORE_CLASS_W = ENABLE_MOTION_XOR ? $clog2(HEAD_DIM + 3) : 2,
    parameter int GATE_W        = 9,
    parameter int GATE_FRAC     = 7,
    parameter int THRESHOLD_W   = 8,
    parameter int TOKEN_W       = $clog2(MAX_TOKENS + 1),
    parameter int ACTIVE_W      = $clog2(MAX_TOKENS + 1),
    parameter int ACTIVE_MEM_DEPTH = MAX_TOKENS,
    parameter int SCORE_CLASS_DEPTH = ENABLE_MOTION_XOR ? HEAD_DIM + 3 : 3,
    parameter bit PIPELINE_FOLD_SCAN = ENABLE_MOTION_XOR,
    parameter int CLASS_COUNT_W = ENABLE_MOTION_XOR ? $clog2(HEAD_DIM + 4) : 2,
    parameter int COUNT_W       = $clog2(HEAD_DIM + 1)
)(
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         cfg_start,
    input  logic [TOKEN_W-1:0]           cfg_n_tokens,
    input  logic                         cfg_preserve_mean,
    input  logic                         cfg_enable_score_fold,
    input  logic [THRESHOLD_W-1:0]       cfg_threshold_q8,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic                         in_last,
    input  logic                         in_time_sel,
    input  logic [HEAD_DIM-1:0]          in_q_bits,
    input  logic [2*HEAD_DIM-1:0]        in_k_pair_bits,

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
    output logic [CLASS_COUNT_W-1:0]     perf_fold_classes,
    output logic [15:0]                  perf_exp_transactions,
    output logic                         perf_score_range_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_CLEAR_ALL,
        ST_LOAD,
        ST_SUM_ACTIVE,
        ST_FIND_FOLD,
        ST_SUM_FOLD,
        ST_EMIT,
        ST_DONE
    } state_t;

    localparam int ADDR_W = $clog2(MAX_TOKENS);
    localparam int MAX_FOLD_SCORE = ENABLE_MOTION_XOR ? HEAD_DIM + 2 : 2;
    localparam int ACTIVE_ENTRY_W = SCORE_W + HEAD_DIM + TOKEN_W;

    state_t state_q;
    logic [TOKEN_W-1:0] n_tokens_q;
    logic preserve_mean_q;
    logic enable_score_fold_q;
    logic [THRESHOLD_W-1:0] threshold_q8_q;

    logic [TOKEN_W-1:0] load_idx_q;
    logic [ACTIVE_W-1:0] active_count_q;
    logic [ACTIVE_W-1:0] scan_idx_q;
    logic [ACTIVE_W-1:0] emit_idx_q;
    logic [SCORE_CLASS_W-1:0] clear_all_idx_q;
    logic hist_initialized_q;
    logic [SCORE_CLASS_DEPTH-1:0] class_present_q;
    logic [CLASS_COUNT_W-1:0] classes_remaining_q;

    logic [ACTIVE_ENTRY_W-1:0] active_entry_mem [0:ACTIVE_MEM_DEPTH-1];
    logic [TOKEN_W-1:0] score_hist [0:SCORE_CLASS_DEPTH-1];

    logic signed [SCORE_W-1:0] row_max_q;
    logic [31:0] row_sum_q;
    logic [TOKEN_W-1:0] tokens_loaded_q;
    logic [TOKEN_W-1:0] folded_tokens_q;
    logic [TOKEN_W-1:0] emitted_entries_q;
    logic [CLASS_COUNT_W-1:0] fold_classes_q;
    logic [15:0] exp_transactions_q;
    logic score_range_error_q;

    logic [HEAD_DIM-1:0] current_k_w;
    logic [HEAD_DIM-1:0] peer_k_w;
    logic [COUNT_W-1:0] overlap_unused;
    logic [COUNT_W-1:0] same_zero_unused;
    logic [COUNT_W-1:0] motion_unused;
    logic signed [SCORE_W-1:0] input_score_w;
    logic [SCORE_CLASS_W-1:0] input_score_class_w;
    logic input_score_in_range_w;
    logic input_k_zero_w;
    logic fold_input_w;

    logic signed [SCORE_W-1:0] active_delta_w;
    logic [15:0] active_exp_w;
    logic [SCORE_CLASS_W-1:0] class_score_code_w;
    logic class_score_found_w;
    logic [SCORE_CLASS_W-1:0] class_score_code_q;
    logic [TOKEN_W-1:0] class_hist_count_q;
    logic signed [SCORE_W-1:0] class_score_w;
    logic signed [SCORE_W-1:0] class_delta_w;
    logic [15:0] class_exp_w;
    logic [31:0] class_sum_term_w;
    logic [SCORE_CLASS_W-1:0] class_eval_code_w;
    logic [TOKEN_W-1:0] class_eval_count_w;
    logic [GATE_W-1:0] gate_w;
    logic [ADDR_W-1:0] active_read_addr_w;
    logic [ACTIVE_ENTRY_W-1:0] active_read_entry_w;
    logic signed [SCORE_W-1:0] active_read_score_w;
    logic [HEAD_DIM-1:0] active_read_k_w;
    logic [TOKEN_W-1:0] active_read_token_w;
    logic [TOKEN_W-1:0] hist_scan_count_w;
    logic [TOKEN_W-1:0] hist_input_count_w;

    integer active_mem_idx;
    integer class_find_idx;
    integer hist_read_idx;

    h67_temporal_pair_adapter #(
        .HEAD_DIM(HEAD_DIM)
    ) u_pair_adapter (
        .k_pair_bits(in_k_pair_bits),
        .time_sel(in_time_sel),
        .k_current_bits(current_k_w),
        .k_peer_bits(peer_k_w)
    );

    h67_motionxor_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .COUNT_W(COUNT_W),
        .ENABLE_MOTION_XOR(ENABLE_MOTION_XOR)
    ) u_score (
        .q_bits(in_q_bits),
        .k_current_bits(current_k_w),
        .k_peer_bits(peer_k_w),
        .overlap(overlap_unused),
        .same_zero(same_zero_unused),
        .motion_xor(motion_unused),
        .score_q7(input_score_w)
    );

    assign input_score_in_range_w = !input_score_w[SCORE_W-1]
                                 && (input_score_w <= $signed(SCORE_W'(MAX_FOLD_SCORE)));
    assign input_score_class_w = input_score_w[SCORE_CLASS_W-1:0];
    assign input_k_zero_w = ~(|current_k_w);
    assign fold_input_w = enable_score_fold_q && input_k_zero_w && input_score_in_range_w;

    assign active_read_addr_w = (state_q == ST_EMIT)
                              ? emit_idx_q[ADDR_W-1:0] : scan_idx_q[ADDR_W-1:0];
    always_comb begin
        active_read_entry_w = '0;
        for (active_mem_idx = 0; active_mem_idx < ACTIVE_MEM_DEPTH; active_mem_idx = active_mem_idx + 1) begin
            if ((state_q == ST_SUM_ACTIVE || state_q == ST_EMIT)
                    && active_read_addr_w == ADDR_W'(active_mem_idx)) begin
                active_read_entry_w = active_entry_mem[active_mem_idx];
            end
        end

        class_score_code_w = '0;
        class_score_found_w = 1'b0;
        for (class_find_idx = 0; class_find_idx < SCORE_CLASS_DEPTH; class_find_idx = class_find_idx + 1) begin
            if (state_q == ST_FIND_FOLD && !class_score_found_w && class_present_q[class_find_idx]) begin
                class_score_code_w = SCORE_CLASS_W'(class_find_idx);
                class_score_found_w = 1'b1;
            end
        end

        hist_scan_count_w = '0;
        hist_input_count_w = '0;
        for (hist_read_idx = 0; hist_read_idx < SCORE_CLASS_DEPTH; hist_read_idx = hist_read_idx + 1) begin
            if (state_q == ST_FIND_FOLD && class_score_code_w == SCORE_CLASS_W'(hist_read_idx)) begin
                hist_scan_count_w = score_hist[hist_read_idx];
            end
            if (state_q == ST_LOAD && input_score_class_w == SCORE_CLASS_W'(hist_read_idx)) begin
                hist_input_count_w = score_hist[hist_read_idx];
            end
        end
    end
    assign active_read_token_w = active_read_entry_w[TOKEN_W-1:0];
    assign active_read_k_w = active_read_entry_w[TOKEN_W +: HEAD_DIM];
    assign active_read_score_w = $signed(active_read_entry_w[TOKEN_W+HEAD_DIM +: SCORE_W]);
    assign active_delta_w = (state_q == ST_SUM_ACTIVE || state_q == ST_EMIT)
                          ? active_read_score_w - row_max_q : '0;
    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC)
    ) u_active_exp (
        .delta_q7(active_delta_w),
        .exp_q8(active_exp_w)
    );

    assign class_eval_code_w = PIPELINE_FOLD_SCAN ? class_score_code_q : class_score_code_w;
    assign class_eval_count_w = PIPELINE_FOLD_SCAN ? class_hist_count_q : hist_scan_count_w;
    assign class_score_w = $signed({1'b0, {(SCORE_W-SCORE_CLASS_W-1){1'b0}}, class_eval_code_w});
    assign class_delta_w = (state_q == ST_SUM_FOLD || (!PIPELINE_FOLD_SCAN && state_q == ST_FIND_FOLD))
                         ? class_score_w - row_max_q : '0;
    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC)
    ) u_class_exp (
        .delta_q7(class_delta_w),
        .exp_q8(class_exp_w)
    );
    assign class_sum_term_w = class_eval_count_w * class_exp_w;

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
            enable_score_fold_q <= 1'b1;
            threshold_q8_q <= '0;
            load_idx_q <= '0;
            active_count_q <= '0;
            scan_idx_q <= '0;
            emit_idx_q <= '0;
            clear_all_idx_q <= '0;
            hist_initialized_q <= 1'b0;
            class_present_q <= '0;
            classes_remaining_q <= '0;
            class_score_code_q <= '0;
            class_hist_count_q <= '0;
            row_max_q <= '0;
            row_sum_q <= '0;
            tokens_loaded_q <= '0;
            folded_tokens_q <= '0;
            emitted_entries_q <= '0;
            fold_classes_q <= '0;
            exp_transactions_q <= '0;
            score_range_error_q <= 1'b0;
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    if (cfg_start) begin
                        if (!hist_initialized_q) begin
                            clear_all_idx_q <= '0;
                            state_q <= ST_CLEAR_ALL;
                        end else begin
                            state_q <= ST_LOAD;
                        end
                        n_tokens_q <= (cfg_n_tokens == 0 || cfg_n_tokens > TOKEN_W'(MAX_TOKENS))
                                      ? TOKEN_W'(MAX_TOKENS) : cfg_n_tokens;
                        preserve_mean_q <= cfg_preserve_mean;
                        enable_score_fold_q <= cfg_enable_score_fold;
                        threshold_q8_q <= cfg_threshold_q8;
                        load_idx_q <= '0;
                        active_count_q <= '0;
                        scan_idx_q <= '0;
                        emit_idx_q <= '0;
                        class_present_q <= '0;
                        classes_remaining_q <= '0;
                        class_score_code_q <= '0;
                        class_hist_count_q <= '0;
                        row_max_q <= -$signed({1'b0, {SCORE_W-1{1'b1}}});
                        row_sum_q <= '0;
                        tokens_loaded_q <= '0;
                        folded_tokens_q <= '0;
                        emitted_entries_q <= '0;
                        fold_classes_q <= '0;
                        exp_transactions_q <= '0;
                        score_range_error_q <= 1'b0;
                    end
                end

                ST_CLEAR_ALL: begin
                    score_hist[clear_all_idx_q] <= '0;
                    if (clear_all_idx_q == SCORE_CLASS_W'(MAX_FOLD_SCORE)) begin
                        hist_initialized_q <= 1'b1;
                        state_q <= ST_LOAD;
                    end else begin
                        clear_all_idx_q <= clear_all_idx_q + 1'b1;
                    end
                end

                ST_LOAD: begin
                    if (in_valid && in_ready) begin
                        tokens_loaded_q <= tokens_loaded_q + 1'b1;
                        if (tokens_loaded_q == 0 || input_score_w > row_max_q) begin
                            row_max_q <= input_score_w;
                        end
                        if (enable_score_fold_q && input_k_zero_w && !input_score_in_range_w) begin
                            score_range_error_q <= 1'b1;
                        end

                        if (fold_input_w) begin
                            score_hist[input_score_class_w] <= hist_input_count_w + 1'b1;
                            if (!class_present_q[input_score_class_w]) begin
                                class_present_q[input_score_class_w] <= 1'b1;
                                classes_remaining_q <= classes_remaining_q + 1'b1;
                            end
                            folded_tokens_q <= folded_tokens_q + 1'b1;
                        end else begin
                            active_entry_mem[active_count_q[ADDR_W-1:0]]
                                <= {input_score_w, current_k_w, load_idx_q};
                            active_count_q <= active_count_q + 1'b1;
                        end

                        load_idx_q <= load_idx_q + 1'b1;
                        if (in_last || load_idx_q == n_tokens_q - 1'b1) begin
                            n_tokens_q <= load_idx_q + 1'b1;
                            scan_idx_q <= '0;
                            row_sum_q <= '0;
                            state_q <= ST_SUM_ACTIVE;
                        end
                    end
                end

                ST_SUM_ACTIVE: begin
                    if (active_count_q == 0) begin
                        state_q <= (folded_tokens_q == 0) ? ST_DONE : ST_FIND_FOLD;
                    end else begin
                        row_sum_q <= row_sum_q + {16'd0, active_exp_w};
                        exp_transactions_q <= exp_transactions_q + 1'b1;
                        if (scan_idx_q == active_count_q - 1'b1) begin
                            scan_idx_q <= '0;
                            state_q <= (folded_tokens_q == 0) ? ST_EMIT : ST_FIND_FOLD;
                        end else begin
                            scan_idx_q <= scan_idx_q + 1'b1;
                        end
                    end
                end

                ST_FIND_FOLD: begin
                    if (!class_score_found_w || classes_remaining_q == 0) begin
                        score_range_error_q <= 1'b1;
                        emit_idx_q <= '0;
                        scan_idx_q <= '0;
                        state_q <= (active_count_q == 0) ? ST_DONE : ST_EMIT;
                    end else begin
                        score_hist[class_score_code_w] <= '0;
                        class_present_q[class_score_code_w] <= 1'b0;
                        if (PIPELINE_FOLD_SCAN) begin
                            class_score_code_q <= class_score_code_w;
                            class_hist_count_q <= hist_scan_count_w;
                            state_q <= ST_SUM_FOLD;
                        end else begin
                            row_sum_q <= row_sum_q + class_sum_term_w;
                            fold_classes_q <= fold_classes_q + 1'b1;
                            exp_transactions_q <= exp_transactions_q + 1'b1;
                            classes_remaining_q <= classes_remaining_q - 1'b1;
                            if (classes_remaining_q == 1) begin
                                emit_idx_q <= '0;
                                scan_idx_q <= '0;
                                state_q <= (active_count_q == 0) ? ST_DONE : ST_EMIT;
                            end
                        end
                    end
                end

                ST_SUM_FOLD: begin
                        row_sum_q <= row_sum_q + class_sum_term_w;
                        fold_classes_q <= fold_classes_q + 1'b1;
                        exp_transactions_q <= exp_transactions_q + 1'b1;
                        classes_remaining_q <= classes_remaining_q - 1'b1;
                        if (classes_remaining_q == 1) begin
                            emit_idx_q <= '0;
                            scan_idx_q <= '0;
                            state_q <= (active_count_q == 0) ? ST_DONE : ST_EMIT;
                        end else begin
                            state_q <= ST_FIND_FOLD;
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
    assign perf_score_range_error = score_range_error_q;
endmodule

`default_nettype wire
