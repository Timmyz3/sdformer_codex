`timescale 1ns/1ps
`default_nettype none

module tb_ttx_row_engine;
    localparam int HEAD_DIM = 8;
    localparam int MAX_TOKENS = 8;
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int SCORE_FRAC = 7;
    localparam int SCORE_W = 16;

    logic clk;
    logic rst_n;
    logic cfg_start;
    logic [TOKEN_W-1:0] cfg_n_tokens;
    logic cfg_preserve_mean;
    logic cfg_enable_zfold;
    logic [7:0] cfg_threshold_q8;
    logic in_valid;
    logic in_ready;
    logic in_last;
    logic [HEAD_DIM-1:0] in_q_bits;
    logic [HEAD_DIM-1:0] in_k_bits;
    logic out_valid;
    logic out_ready;
    logic out_last;
    logic [TOKEN_W-1:0] out_token_idx;
    logic [HEAD_DIM-1:0] out_k_bits;
    logic [8:0] out_gate_q8;
    logic [7:0] out_threshold_q8;
    logic busy;
    logic done;
    logic [TOKEN_W-1:0] perf_tokens_loaded;
    logic [TOKEN_W-1:0] perf_kzero_folded;
    logic [TOKEN_W-1:0] perf_entries_emitted;
    logic [$clog2(HEAD_DIM + 1)-1:0] perf_fold_classes;
    logic [15:0] perf_exp_transactions;

    logic [HEAD_DIM-1:0] q_vector [0:MAX_TOKENS-1];
    logic [HEAD_DIM-1:0] k_vector [0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_TOKENS-1];
    integer expected_score [0:MAX_TOKENS-1];

    logic [3:0] late_k_bits;
    logic signed [31:0] late_weights_flat;
    logic [8:0] late_gate;
    logic [7:0] late_threshold;
    logic signed [10:0] late_weight_sum;
    logic signed [28:0] late_scaled_accum;

    integer errors;
    integer output_count;
    integer token_idx;
    integer expected_outputs;
    integer row_max;
    integer row_sum;
    integer denominator_shift;
    integer numerator;

    ttx_row_engine #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(SCORE_FRAC),
        .TOKEN_W(TOKEN_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_start(cfg_start),
        .cfg_n_tokens(cfg_n_tokens),
        .cfg_preserve_mean(cfg_preserve_mean),
        .cfg_enable_zfold(cfg_enable_zfold),
        .cfg_threshold_q8(cfg_threshold_q8),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_last(in_last),
        .in_q_bits(in_q_bits),
        .in_k_bits(in_k_bits),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last),
        .out_token_idx(out_token_idx),
        .out_k_bits(out_k_bits),
        .out_gate_q8(out_gate_q8),
        .out_threshold_q8(out_threshold_q8),
        .busy(busy),
        .done(done),
        .perf_tokens_loaded(perf_tokens_loaded),
        .perf_kzero_folded(perf_kzero_folded),
        .perf_entries_emitted(perf_entries_emitted),
        .perf_fold_classes(perf_fold_classes),
        .perf_exp_transactions(perf_exp_transactions)
    );

    ttx_late_gate_accum #(
        .HEAD_DIM(4),
        .WEIGHT_W(8),
        .GATE_W(9),
        .THRESHOLD_W(8),
        .SUM_W(11),
        .OUT_W(29)
    ) u_late_gate (
        .k_bits(late_k_bits),
        .weights_flat(late_weights_flat),
        .gate_q8(late_gate),
        .threshold_q8(late_threshold),
        .active_weight_sum(late_weight_sum),
        .scaled_accum(late_scaled_accum)
    );

    always #5 clk = ~clk;

    function automatic integer popcount(input logic [HEAD_DIM-1:0] bits);
        integer idx;
        begin
            popcount = 0;
            for (idx = 0; idx < HEAD_DIM; idx = idx + 1) begin
                popcount = popcount + bits[idx];
            end
        end
    endfunction

    function automatic integer score_q7(
        input logic [HEAD_DIM-1:0] q_bits,
        input logic [HEAD_DIM-1:0] k_bits
    );
        integer q_count;
        integer k_count;
        integer overlap_count;
        integer same_zero_count;
        integer score_shift;
        integer tx_num_q8;
        begin
            q_count = popcount(q_bits);
            k_count = popcount(k_bits);
            overlap_count = popcount(q_bits & k_bits);
            same_zero_count = HEAD_DIM - q_count - k_count + overlap_count;
            score_shift = 8 + $clog2(HEAD_DIM) - SCORE_FRAC;
            tx_num_q8 = (overlap_count << 8) + same_zero_count * 5;
            score_q7 = (tx_num_q8 + (1 << (score_shift - 1))) >> score_shift;
        end
    endfunction

    function automatic integer exp2_q8(input integer delta_q7);
        integer abs_delta;
        integer integer_shift;
        integer fraction_index;
        integer fraction_value;
        begin
            if (delta_q7 >= 0) begin
                exp2_q8 = 256;
            end else begin
                abs_delta = -delta_q7;
                integer_shift = abs_delta >> SCORE_FRAC;
                fraction_index = (abs_delta >> (SCORE_FRAC - 4)) & 15;
                if ((abs_delta & ((1 << (SCORE_FRAC - 4)) - 1)) != 0) begin
                    fraction_index = fraction_index + 1;
                end
                if (fraction_index > 15) begin
                    fraction_index = 15;
                end
                case (fraction_index)
                    0: fraction_value = 256;
                    1: fraction_value = 245;
                    2: fraction_value = 234;
                    3: fraction_value = 224;
                    4: fraction_value = 215;
                    5: fraction_value = 205;
                    6: fraction_value = 196;
                    7: fraction_value = 188;
                    8: fraction_value = 181;
                    9: fraction_value = 173;
                    10: fraction_value = 165;
                    11: fraction_value = 158;
                    12: fraction_value = 152;
                    13: fraction_value = 145;
                    14: fraction_value = 139;
                    default: fraction_value = 133;
                endcase
                if (integer_shift > 8) begin
                    integer_shift = 8;
                end
                exp2_q8 = fraction_value >> integer_shift;
            end
        end
    endfunction

    function automatic integer ceil_log2(input integer value);
        integer probe;
        integer shift_value;
        begin
            probe = (value <= 1) ? 1 : value - 1;
            shift_value = 0;
            while (probe > 0) begin
                probe = probe >> 1;
                shift_value = shift_value + 1;
            end
            ceil_log2 = shift_value;
        end
    endfunction

    function automatic integer round_shift_even(
        input integer value,
        input integer shift_value
    );
        integer quotient;
        integer remainder;
        integer half;
        begin
            if (shift_value == 0) begin
                round_shift_even = value;
            end else begin
                quotient = value >> shift_value;
                remainder = value - (quotient << shift_value);
                half = 1 << (shift_value - 1);
                if ((remainder > half) || ((remainder == half) && quotient[0])) begin
                    quotient = quotient + 1;
                end
                round_shift_even = quotient;
            end
        end
    endfunction

    task automatic build_expected(input integer n_tokens);
        integer idx;
        integer exp_value;
        integer scaled;
        begin
            row_max = -32768;
            row_sum = 0;
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                expected_score[idx] = score_q7(q_vector[idx], k_vector[idx]);
                if (expected_score[idx] > row_max) begin
                    row_max = expected_score[idx];
                end
            end
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                row_sum = row_sum + exp2_q8(expected_score[idx] - row_max);
            end
            denominator_shift = ceil_log2(row_sum);
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                exp_value = exp2_q8(expected_score[idx] - row_max);
                scaled = exp_value * 128 * n_tokens;
                expected_gate[idx] = round_shift_even(scaled, denominator_shift);
                if (expected_gate[idx] > 256) begin
                    expected_gate[idx] = 256;
                end
            end
        end
    endtask

    task automatic run_row(input integer n_tokens, input logic enable_zfold);
        integer idx;
        begin
            cfg_n_tokens = n_tokens[TOKEN_W-1:0];
            cfg_enable_zfold = enable_zfold;
            @(negedge clk);
            cfg_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cfg_start = 1'b0;

            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                while (!in_ready) begin
                    @(negedge clk);
                end
                in_valid = 1'b1;
                in_q_bits = q_vector[idx];
                in_k_bits = k_vector[idx];
                in_last = (idx == n_tokens - 1);
                @(posedge clk);
                @(negedge clk);
                in_valid = 1'b0;
                in_last = 1'b0;
            end

            output_count = 0;
            expected_outputs = 0;
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                if (!enable_zfold || k_vector[idx] != 0) begin
                    expected_outputs = expected_outputs + 1;
                end
            end

            while (!done) begin
                @(negedge clk);
                out_ready = (output_count[0] == 0) ? 1'b1 : ~out_ready;
                if (out_valid && out_ready) begin
                    token_idx = out_token_idx;
                    if (out_k_bits !== k_vector[token_idx]) begin
                        $display("ERROR: token %0d K bits mismatch", token_idx);
                        errors = errors + 1;
                    end
                    if (out_gate_q8 !== expected_gate[token_idx][8:0]) begin
                        $display("ERROR: token %0d gate got=%0d expected=%0d", token_idx, out_gate_q8, expected_gate[token_idx]);
                        errors = errors + 1;
                    end
                    if (out_threshold_q8 !== 8'd64) begin
                        $display("ERROR: threshold metadata mismatch");
                        errors = errors + 1;
                    end
                    output_count = output_count + 1;
                end
            end
            out_ready = 1'b1;
            if (output_count != expected_outputs) begin
                $display("ERROR: output count got=%0d expected=%0d", output_count, expected_outputs);
                errors = errors + 1;
            end
            if (perf_tokens_loaded != n_tokens[TOKEN_W-1:0]) begin
                $display("ERROR: loaded count got=%0d expected=%0d", perf_tokens_loaded, n_tokens);
                errors = errors + 1;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        cfg_start = 1'b0;
        cfg_n_tokens = '0;
        cfg_preserve_mean = 1'b1;
        cfg_enable_zfold = 1'b1;
        cfg_threshold_q8 = 8'd64;
        in_valid = 1'b0;
        in_last = 1'b0;
        in_q_bits = '0;
        in_k_bits = '0;
        out_ready = 1'b1;
        errors = 0;

        late_k_bits = 4'b1011;
        late_weights_flat = {8'sd4, 8'sd3, -8'sd1, 8'sd2};
        late_gate = 9'd2;
        late_threshold = 8'd3;

        q_vector[0] = 8'b00000000; k_vector[0] = 8'b00000000;
        q_vector[1] = 8'b00000001; k_vector[1] = 8'b00000000;
        q_vector[2] = 8'b00000111; k_vector[2] = 8'b00000001;
        q_vector[3] = 8'b00000000; k_vector[3] = 8'b00000010;
        q_vector[4] = 8'b00000101; k_vector[4] = 8'b00000000;
        q_vector[5] = 8'b11111111; k_vector[5] = 8'b11111111;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        if (late_weight_sum !== 11'sd5 || late_scaled_accum !== 29'sd30) begin
            $display("ERROR: late-gate factorization got sum=%0d scaled=%0d", late_weight_sum, late_scaled_accum);
            errors = errors + 1;
        end

        build_expected(6);
        run_row(6, 1'b1);
        if (perf_kzero_folded != 3) begin
            $display("ERROR: zfold count got=%0d expected=3", perf_kzero_folded);
            errors = errors + 1;
        end

        run_row(6, 1'b0);
        if (perf_kzero_folded != 0) begin
            $display("ERROR: dense mode unexpectedly folded tokens");
            errors = errors + 1;
        end

        q_vector[0] = 8'b00000000; k_vector[0] = 8'b00000000;
        q_vector[1] = 8'b00000001; k_vector[1] = 8'b00000000;
        q_vector[2] = 8'b00000011; k_vector[2] = 8'b00000000;
        q_vector[3] = 8'b00000111; k_vector[3] = 8'b00000000;
        build_expected(4);
        run_row(4, 1'b1);

        if (errors == 0) begin
            $display("PASS: TTX row engine, ZAF folding, and FGK late-gate tests passed");
        end else begin
            $display("FAIL: %0d error(s)", errors);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
