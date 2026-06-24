`default_nettype none

module unibin_h60_core_dc #(
    parameter int HEAD_DIM   = 32,
    parameter int MAX_TOKENS = 162,
    parameter int DATA_W     = 8,
    parameter int SCORE_W    = 16,
    parameter int GATE_W     = 8,
    parameter int COUNT_W    = 8,
    parameter int SCORE_FRAC = 7,
    parameter int ALPHA0_Q8  = 5
)(
    input  logic                         clk_core,
    input  logic                         rst_n_core,

    input  logic                         cfg_start,
    input  logic [COUNT_W-1:0]           cfg_n_tokens,
    input  logic [7:0]                   cfg_mu_q8,
    input  logic                         cfg_preserve_mean,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic                         in_last,
    input  logic [HEAD_DIM-1:0]          in_q_bits,
    input  logic [HEAD_DIM-1:0]          in_k_bits,
    input  logic signed [DATA_W-1:0]     in_k_value,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic                         out_last,
    output logic [COUNT_W-1:0]           out_token_idx,
    output logic [GATE_W-1:0]            out_gate,
    output logic signed [DATA_W+GATE_W-1:0] out_gated_k,

    output logic                         busy,
    output logic                         done,
    output logic [COUNT_W-1:0]           perf_tokens_loaded,
    output logic [COUNT_W-1:0]           perf_empty_tokens,
    output logic [COUNT_W-1:0]           perf_issued_tokens
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD,
        ST_FIND_MAX,
        ST_SUM_EXP,
        ST_EMIT,
        ST_DONE
    } state_t;

    state_t state_q, state_d;

    logic [COUNT_W-1:0] n_tokens_q;
    logic [7:0] mu_q8_q;
    logic preserve_mean_q;

    logic [COUNT_W-1:0] load_idx_q, scan_idx_q, emit_idx_q;
    logic signed [SCORE_W-1:0] score_mem_q [0:MAX_TOKENS-1];
    logic signed [DATA_W-1:0]  k_value_mem_q [0:MAX_TOKENS-1];
    logic                      k_event_mem_q [0:MAX_TOKENS-1];
    logic [15:0]               exp_mem_q [0:MAX_TOKENS-1];

    logic signed [SCORE_W-1:0] row_max_q;
    logic signed [31:0] score_sum_q;
    logic [31:0] row_sum_q;

    logic [COUNT_W-1:0] tokens_loaded_q, empty_tokens_q, issued_tokens_q;

    logic token_empty_w;
    logic signed [SCORE_W-1:0] fused_score_w;
    logic signed [SCORE_W-1:0] centered_score_w;
    logic signed [SCORE_W-1:0] score_mean_w;
    logic [15:0] exp_value_w;
    logic [GATE_W-1:0] gate_w;
    logic signed [DATA_W+GATE_W-1:0] gated_k_w;

    function automatic [COUNT_W-1:0] popcount32(input logic [HEAD_DIM-1:0] bits);
        int i;
        logic [COUNT_W-1:0] count;
        begin
            count = '0;
            for (i = 0; i < HEAD_DIM; i = i + 1) begin
                count = count + {{(COUNT_W-1){1'b0}}, bits[i]};
            end
            popcount32 = count;
        end
    endfunction

    function automatic signed [SCORE_W-1:0] consensus_score(
        input logic [HEAD_DIM-1:0] q_bits,
        input logic [HEAD_DIM-1:0] k_bits,
        input logic [7:0]          mu_q8
    );
        logic [COUNT_W-1:0] q_active;
        logic [COUNT_W-1:0] k_active;
        logic [COUNT_W-1:0] overlap;
        logic [COUNT_W-1:0] same_zero;
        logic [31:0] tx_num_q8;
        logic [31:0] sc_num_q8;
        logic [39:0] mu_sc_q16;
        logic [39:0] fused_num_q8;
        logic [39:0] rounded_q7;
        logic [39:0] score_q7;
        begin
            q_active = popcount32(q_bits);
            k_active = popcount32(k_bits);
            overlap = popcount32(q_bits & k_bits);
            same_zero = HEAD_DIM[COUNT_W-1:0] - q_active - k_active + overlap;

            // Software h60 all-binary deployment equivalent before centering:
            // TX = (overlap + alpha0 * same_zero) / head_dim
            // SC = overlap / head_dim
            // score = TX + mu * SC, quantized to Q(SCORE_FRAC).
            tx_num_q8 = ({24'd0, overlap} << 8) + ({24'd0, same_zero} * ALPHA0_Q8[7:0]);
            sc_num_q8 = ({24'd0, overlap} << 8);
            mu_sc_q16 = {8'd0, sc_num_q8} * {32'd0, mu_q8};
            fused_num_q8 = {8'd0, tx_num_q8} + (mu_sc_q16 >> 8);
            rounded_q7 = fused_num_q8 + (((40'(HEAD_DIM)) << (8 - SCORE_FRAC)) >> 1);
            score_q7 = rounded_q7 / ((40'(HEAD_DIM)) << (8 - SCORE_FRAC));

            if (|score_q7[39:SCORE_W-1]) begin
                consensus_score = {1'b0, {SCORE_W-1{1'b1}}};
            end else begin
                consensus_score = $signed(score_q7[SCORE_W-1:0]);
            end
        end
    endfunction

    function automatic [15:0] exp2_approx_q8(input signed [SCORE_W-1:0] delta);
        logic [SCORE_W-1:0] abs_delta;
        logic [8:0] int_shift;
        logic [3:0] frac_idx;
        logic [4:0] frac_round;
        logic [15:0] frac_value;
        logic [7:0] shift_amt;
        begin
            if (delta >= 0) begin
                exp2_approx_q8 = 16'd256;
            end else begin
                abs_delta = -delta;
                int_shift = abs_delta[SCORE_W-1:SCORE_FRAC];
                frac_round = {1'b0, abs_delta[SCORE_FRAC-1:SCORE_FRAC-4]} + {4'd0, (|abs_delta[SCORE_FRAC-5:0])};
                frac_idx = frac_round[4] ? 4'd15 : frac_round[3:0];
                unique case (frac_idx)
                    4'd0: frac_value = 16'd256;
                    4'd1: frac_value = 16'd245;
                    4'd2: frac_value = 16'd234;
                    4'd3: frac_value = 16'd224;
                    4'd4: frac_value = 16'd215;
                    4'd5: frac_value = 16'd205;
                    4'd6: frac_value = 16'd196;
                    4'd7: frac_value = 16'd188;
                    4'd8: frac_value = 16'd181;
                    4'd9: frac_value = 16'd173;
                    4'd10: frac_value = 16'd165;
                    4'd11: frac_value = 16'd158;
                    4'd12: frac_value = 16'd152;
                    4'd13: frac_value = 16'd145;
                    4'd14: frac_value = 16'd139;
                    default: frac_value = 16'd133;
                endcase
                shift_amt = (int_shift > 9'd8) ? 8'd8 : int_shift[7:0];
                exp2_approx_q8 = frac_value >> shift_amt;
            end
        end
    endfunction

    function automatic [7:0] ceil_log2_u32(input logic [31:0] value);
        int j;
        logic [31:0] probe;
        begin
            ceil_log2_u32 = 8'd0;
            probe = (value <= 1) ? 32'd1 : (value - 1'b1);
            for (j = 0; j < 32; j = j + 1) begin
                if (probe[j]) begin
                    ceil_log2_u32 = j[7:0] + 8'd1;
                end
            end
        end
    endfunction

    function automatic [GATE_W-1:0] gate_from_exp(
        input logic [15:0] exp_value,
        input logic [31:0] row_sum,
        input logic [COUNT_W-1:0] n_tokens,
        input logic preserve_mean
    );
        logic [47:0] scaled;
        logic [47:0] shifted;
        logic [7:0] denom_shift;
        begin
            scaled = exp_value * ((1 << GATE_W) - 1);
            if (preserve_mean) begin
                scaled = scaled * n_tokens;
            end
            if (row_sum == 0) begin
                gate_from_exp = '0;
            end else begin
                denom_shift = ceil_log2_u32(row_sum);
                shifted = scaled >> denom_shift;
                if (shifted > ((1 << GATE_W) - 1)) begin
                    gate_from_exp = {GATE_W{1'b1}};
                end else begin
                    gate_from_exp = shifted[GATE_W-1:0];
                end
            end
        end
    endfunction

    assign token_empty_w = ((in_q_bits | in_k_bits) == '0);
    assign fused_score_w = consensus_score(in_q_bits, in_k_bits, mu_q8_q);
    assign score_mean_w = (n_tokens_q == '0) ? '0 : SCORE_W'(score_sum_q / $signed({24'd0, n_tokens_q}));
    assign centered_score_w = score_mem_q[scan_idx_q] - score_mean_w;
    assign exp_value_w = exp2_approx_q8(score_mem_q[scan_idx_q] - row_max_q);
    assign gate_w = gate_from_exp(exp_mem_q[emit_idx_q], row_sum_q, n_tokens_q, preserve_mean_q);
    assign gated_k_w = k_event_mem_q[emit_idx_q] ? ($signed(k_value_mem_q[emit_idx_q]) * $signed({1'b0, gate_w})) : '0;

    always_comb begin
        state_d = state_q;
        in_ready = 1'b0;
        out_valid = 1'b0;
        out_last = 1'b0;
        done = 1'b0;

        unique case (state_q)
            ST_IDLE: begin
                if (cfg_start) begin
                    state_d = ST_LOAD;
                end
            end
            ST_LOAD: begin
                in_ready = 1'b1;
                if (in_valid && in_ready && (in_last || (load_idx_q == n_tokens_q - 1'b1))) begin
                    state_d = ST_FIND_MAX;
                end
            end
            ST_FIND_MAX: begin
                if (scan_idx_q == n_tokens_q - 1'b1) begin
                    state_d = ST_SUM_EXP;
                end
            end
            ST_SUM_EXP: begin
                if (scan_idx_q == n_tokens_q - 1'b1) begin
                    state_d = ST_EMIT;
                end
            end
            ST_EMIT: begin
                out_valid = 1'b1;
                out_last = (emit_idx_q == n_tokens_q - 1'b1);
                if (out_valid && out_ready && out_last) begin
                    state_d = ST_DONE;
                end
            end
            ST_DONE: begin
                done = 1'b1;
                state_d = ST_IDLE;
            end
            default: begin
                state_d = ST_IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_core) begin
        if (!rst_n_core) begin
            state_q <= ST_IDLE;
            n_tokens_q <= '0;
            mu_q8_q <= 8'd16;
            preserve_mean_q <= 1'b1;
            load_idx_q <= '0;
            scan_idx_q <= '0;
            emit_idx_q <= '0;
            row_max_q <= '0;
            score_sum_q <= '0;
            row_sum_q <= '0;
            tokens_loaded_q <= '0;
            empty_tokens_q <= '0;
            issued_tokens_q <= '0;
        end else begin
            state_q <= state_d;

            if (state_q == ST_IDLE && cfg_start) begin
                n_tokens_q <= (cfg_n_tokens == '0 || cfg_n_tokens > MAX_TOKENS[COUNT_W-1:0]) ? MAX_TOKENS[COUNT_W-1:0] : cfg_n_tokens;
                mu_q8_q <= cfg_mu_q8;
                preserve_mean_q <= cfg_preserve_mean;
                load_idx_q <= '0;
                scan_idx_q <= '0;
                emit_idx_q <= '0;
                row_max_q <= -$signed({1'b0, {SCORE_W-1{1'b1}}});
                score_sum_q <= '0;
                row_sum_q <= '0;
                tokens_loaded_q <= '0;
                empty_tokens_q <= '0;
                issued_tokens_q <= '0;
            end

            if (state_q == ST_LOAD && in_valid && in_ready) begin
                score_mem_q[load_idx_q] <= fused_score_w;
                k_value_mem_q[load_idx_q] <= in_k_value;
                k_event_mem_q[load_idx_q] <= |in_k_bits;
                tokens_loaded_q <= tokens_loaded_q + 1'b1;
                score_sum_q <= score_sum_q + {{(32-SCORE_W){fused_score_w[SCORE_W-1]}}, fused_score_w};
                if (token_empty_w) begin
                    empty_tokens_q <= empty_tokens_q + 1'b1;
                end
                load_idx_q <= load_idx_q + 1'b1;
                if (in_last || (load_idx_q == n_tokens_q - 1'b1)) begin
                    n_tokens_q <= load_idx_q + 1'b1;
                    scan_idx_q <= '0;
                end
            end

            if (state_q == ST_FIND_MAX) begin
                score_mem_q[scan_idx_q] <= centered_score_w;
                if (scan_idx_q == '0 || centered_score_w > row_max_q) begin
                    row_max_q <= centered_score_w;
                end
                if (scan_idx_q == n_tokens_q - 1'b1) begin
                    scan_idx_q <= '0;
                    row_sum_q <= '0;
                end else begin
                    scan_idx_q <= scan_idx_q + 1'b1;
                end
            end

            if (state_q == ST_SUM_EXP) begin
                exp_mem_q[scan_idx_q] <= exp_value_w;
                row_sum_q <= row_sum_q + {16'd0, exp_value_w};
                if (scan_idx_q == n_tokens_q - 1'b1) begin
                    scan_idx_q <= '0;
                    emit_idx_q <= '0;
                end else begin
                    scan_idx_q <= scan_idx_q + 1'b1;
                end
            end

            if (state_q == ST_EMIT && out_valid && out_ready) begin
                issued_tokens_q <= issued_tokens_q + 1'b1;
                emit_idx_q <= emit_idx_q + 1'b1;
            end
        end
    end

    assign out_token_idx = emit_idx_q;
    assign out_gate = gate_w;
    assign out_gated_k = gated_k_w;
    assign busy = (state_q != ST_IDLE);
    assign perf_tokens_loaded = tokens_loaded_q;
    assign perf_empty_tokens = empty_tokens_q;
    assign perf_issued_tokens = issued_tokens_q;
endmodule

`default_nettype wire
