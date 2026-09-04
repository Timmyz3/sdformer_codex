`timescale 1ns/1ps
`default_nettype none

module tb_h67_score_class_row_engine #(
    parameter bit ENABLE_MOTION_XOR = 1'b1,
    parameter int MAX_TOKENS = 8
);
    localparam int HEAD_DIM = 32;
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int SCORE_FRAC = 7;
    localparam int CLASS_COUNT_W = ENABLE_MOTION_XOR ? $clog2(HEAD_DIM + 4) : 2;

    logic clk;
    logic rst_n;
    logic cfg_start;
    logic [TOKEN_W-1:0] cfg_n_tokens;
    logic cfg_preserve_mean;
    logic cfg_enable_score_fold;
    logic [7:0] cfg_threshold_q8;
    logic in_valid;
    logic in_ready;
    logic in_last;
    logic in_time_sel;
    logic [HEAD_DIM-1:0] in_q_bits;
    logic [2*HEAD_DIM-1:0] in_k_pair_bits;
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
    logic [CLASS_COUNT_W-1:0] perf_fold_classes;
    logic [15:0] perf_exp_transactions;
    logic perf_score_range_error;

    logic [HEAD_DIM-1:0] q_vector [0:MAX_TOKENS-1];
    logic [HEAD_DIM-1:0] k_current_vector [0:MAX_TOKENS-1];
    logic [HEAD_DIM-1:0] k_peer_vector [0:MAX_TOKENS-1];
    logic time_vector [0:MAX_TOKENS-1];
    integer expected_score [0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_TOKENS-1];

    integer errors;
    integer output_count;
    integer expected_outputs;
    integer expected_folded;
    integer row_max;
    integer row_sum;
    integer denominator_shift;
    integer token_idx;

`ifdef GATE_LEVEL_NETLIST
    h67_score_class_row_engine dut (
`else
    h67_score_class_row_engine #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .ENABLE_MOTION_XOR(ENABLE_MOTION_XOR),
        .TOKEN_W(TOKEN_W)
    ) dut (
`endif
        .clk(clk),
        .rst_n(rst_n),
        .cfg_start(cfg_start),
        .cfg_n_tokens(cfg_n_tokens),
        .cfg_preserve_mean(cfg_preserve_mean),
        .cfg_enable_score_fold(cfg_enable_score_fold),
        .cfg_threshold_q8(cfg_threshold_q8),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_last(in_last),
        .in_time_sel(in_time_sel),
        .in_q_bits(in_q_bits),
        .in_k_pair_bits(in_k_pair_bits),
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
        .perf_exp_transactions(perf_exp_transactions),
        .perf_score_range_error(perf_score_range_error)
    );

    always #5 clk = ~clk;

    function automatic integer popcount32(input logic [31:0] value);
        integer idx;
        begin
            popcount32 = 0;
            for (idx = 0; idx < 32; idx = idx + 1) begin
                popcount32 = popcount32 + value[idx];
            end
        end
    endfunction

    function automatic logic [31:0] low_mask(input integer count);
        begin
            if (count <= 0) low_mask = 32'h00000000;
            else if (count >= 32) low_mask = 32'hffffffff;
            else low_mask = (32'h00000001 << count) - 1'b1;
        end
    endfunction

    function automatic integer round_even_silence(
        input integer count,
        input integer integer_base
    );
        integer quotient;
        integer remainder;
        begin
            quotient = count >> 4;
            remainder = count & 15;
            if ((remainder > 8) || ((remainder == 8) && ((integer_base + quotient) % 2 != 0))) begin
                quotient = quotient + 1;
            end
            round_even_silence = quotient;
        end
    endfunction

    function automatic integer h67_score(
        input logic [31:0] q,
        input logic [31:0] k,
        input logic [31:0] peer
    );
        integer overlap_count;
        integer same_zero_count;
        integer motion_count;
        begin
            overlap_count = popcount32(q & k);
            same_zero_count = popcount32(~q & ~k);
            motion_count = ENABLE_MOTION_XOR ? popcount32(k ^ peer) : 0;
            h67_score = 4 * overlap_count + motion_count
                      + round_even_silence(same_zero_count, 4 * overlap_count + motion_count);
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
                if (fraction_index > 15) fraction_index = 15;
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
                if (integer_shift > 8) integer_shift = 8;
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
                expected_score[idx] = h67_score(q_vector[idx], k_current_vector[idx], k_peer_vector[idx]);
                if (expected_score[idx] > row_max) row_max = expected_score[idx];
            end
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                row_sum = row_sum + exp2_q8(expected_score[idx] - row_max);
            end
            denominator_shift = ceil_log2(row_sum);
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                exp_value = exp2_q8(expected_score[idx] - row_max);
                scaled = exp_value * 128 * n_tokens;
                expected_gate[idx] = round_shift_even(scaled, denominator_shift);
                if (expected_gate[idx] > 256) expected_gate[idx] = 256;
            end
        end
    endtask

    task automatic run_row(input integer n_tokens, input logic enable_fold);
        integer idx;
        begin
            build_expected(n_tokens);
            cfg_n_tokens = n_tokens[TOKEN_W-1:0];
            cfg_enable_score_fold = enable_fold;
            @(negedge clk);
            cfg_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cfg_start = 1'b0;

            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                while (!in_ready) @(negedge clk);
                in_valid = 1'b1;
                in_q_bits = q_vector[idx];
                in_time_sel = time_vector[idx];
                if (time_vector[idx]) begin
                    in_k_pair_bits = {k_current_vector[idx], k_peer_vector[idx]};
                end else begin
                    in_k_pair_bits = {k_peer_vector[idx], k_current_vector[idx]};
                end
                in_last = (idx == n_tokens - 1);
                @(posedge clk);
                @(negedge clk);
                in_valid = 1'b0;
                in_last = 1'b0;
            end

            output_count = 0;
            expected_outputs = 0;
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                if (!enable_fold || k_current_vector[idx] != 0) expected_outputs = expected_outputs + 1;
            end
            expected_folded = enable_fold ? n_tokens - expected_outputs : 0;

            while (!done) begin
                @(negedge clk);
                out_ready = (output_count[0] == 0) ? 1'b1 : ~out_ready;
                if (out_valid && out_ready) begin
                    token_idx = out_token_idx;
                    if (out_k_bits !== k_current_vector[token_idx]) begin
                        $display("ERROR token %0d K mismatch", token_idx);
                        errors = errors + 1;
                    end
                    if (out_gate_q8 !== expected_gate[token_idx][8:0]) begin
                        $display("ERROR token %0d gate got=%0d expected=%0d score=%0d", token_idx, out_gate_q8, expected_gate[token_idx], expected_score[token_idx]);
                        errors = errors + 1;
                    end
                    output_count = output_count + 1;
                end
            end
            out_ready = 1'b1;
            if (output_count != expected_outputs) begin
                $display("ERROR output count got=%0d expected=%0d", output_count, expected_outputs);
                errors = errors + 1;
            end
            if (perf_tokens_loaded != n_tokens[TOKEN_W-1:0]) begin
                $display("ERROR loaded count got=%0d expected=%0d", perf_tokens_loaded, n_tokens);
                errors = errors + 1;
            end
            if (perf_kzero_folded != expected_folded[TOKEN_W-1:0]) begin
                $display("ERROR folded count got=%0d expected=%0d", perf_kzero_folded, expected_folded);
                errors = errors + 1;
            end
            if (perf_score_range_error) begin
                $display("ERROR unexpected score range error");
                errors = errors + 1;
            end
            @(posedge clk);
        end
    endtask

    task automatic prepare_all_active(input integer n_tokens);
        integer idx;
        begin
            for (idx = 0; idx < n_tokens; idx = idx + 1) begin
                q_vector[idx] = 32'h9e3779b9 ^ idx;
                k_current_vector[idx] = 32'h00000001 << (idx % 32);
                k_peer_vector[idx] = 32'h80000000 >> (idx % 32);
                time_vector[idx] = (idx % 2) != 0;
            end
        end
    endtask

    task automatic prepare_all_fold_classes(output integer class_count);
        integer target;
        integer q_active;
        integer peer_active;
        logic found;
        begin
            class_count = ENABLE_MOTION_XOR ? 35 : 3;
            for (target = 0; target < class_count; target = target + 1) begin
                found = 0;
                for (q_active = 0; q_active <= 32; q_active = q_active + 1) begin
                    for (peer_active = 0; peer_active <= 32; peer_active = peer_active + 1) begin
                        if (!found && h67_score(low_mask(q_active), 32'h0, low_mask(peer_active)) == target) begin
                            q_vector[target] = low_mask(q_active);
                            k_current_vector[target] = 32'h00000000;
                            k_peer_vector[target] = low_mask(peer_active);
                            time_vector[target] = 1'b0;
                            found = 1;
                        end
                    end
                end
                if (!found) begin
                    $fatal(1, "unable to construct fold score class %0d", target);
                end
            end
        end
    endtask

    initial begin
        integer init_idx;
        integer all_class_count;
        clk = 1'b0;
        rst_n = 1'b0;
        cfg_start = 1'b0;
        cfg_n_tokens = '0;
        cfg_preserve_mean = 1'b1;
        cfg_enable_score_fold = 1'b1;
        cfg_threshold_q8 = 8'd64;
        in_valid = 1'b0;
        in_last = 1'b0;
        in_time_sel = 1'b0;
        in_q_bits = '0;
        in_k_pair_bits = '0;
        out_ready = 1'b1;
        errors = 0;

        if (MAX_TOKENS < 8) begin
            $fatal(1, "MAX_TOKENS must be at least 8");
        end
        for (init_idx = 0; init_idx < MAX_TOKENS; init_idx = init_idx + 1) begin
            q_vector[init_idx] = 32'h9e3779b9 * (init_idx + 1);
            k_current_vector[init_idx] = ((init_idx % 4) == 0)
                                       ? 32'h00000000
                                       : (32'h7f4a7c15 ^ (32'h01010101 * init_idx));
            k_peer_vector[init_idx] = 32'hd1b54a35 ^ (32'h10204081 * init_idx);
            time_vector[init_idx] = (init_idx % 2) != 0;
        end

        // Tokens 0 and 1 have the same q/current-K class but different peer K.
        // A q_active-only zero-fold would merge them incorrectly.
        q_vector[0] = 32'h00000000; k_current_vector[0] = 32'h00000000; k_peer_vector[0] = 32'h00000000; time_vector[0] = 1'b0;
        q_vector[1] = 32'h00000000; k_current_vector[1] = 32'h00000000; k_peer_vector[1] = 32'hffffffff; time_vector[1] = 1'b1;
        q_vector[2] = 32'h0000ffff; k_current_vector[2] = 32'h00000000; k_peer_vector[2] = 32'h0000000f; time_vector[2] = 1'b0;
        q_vector[3] = 32'h0000ffff; k_current_vector[3] = 32'h0000000f; k_peer_vector[3] = 32'h00000000; time_vector[3] = 1'b1;
        q_vector[4] = 32'h00ff00ff; k_current_vector[4] = 32'h00000003; k_peer_vector[4] = 32'h00000001; time_vector[4] = 1'b0;
        q_vector[5] = 32'hffffffff; k_current_vector[5] = 32'hffffffff; k_peer_vector[5] = 32'h00000000; time_vector[5] = 1'b1;
        q_vector[6] = 32'h00000000; k_current_vector[6] = 32'h80000000; k_peer_vector[6] = 32'h00000000; time_vector[6] = 1'b0;
        q_vector[7] = 32'ha5a5a5a5; k_current_vector[7] = 32'h5a5a5a5a; k_peer_vector[7] = 32'ha5a5a5a5; time_vector[7] = 1'b1;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        run_row(MAX_TOKENS, 1'b1);
        if (MAX_TOKENS == 8 && perf_fold_classes != (ENABLE_MOTION_XOR ? 3 : 2)) begin
            $display("ERROR fold class count got=%0d expected=%0d", perf_fold_classes, ENABLE_MOTION_XOR ? 3 : 2);
            errors = errors + 1;
        end

        run_row(MAX_TOKENS, 1'b0);
        if (perf_kzero_folded != 0) begin
            $display("ERROR fold disabled but folded=%0d", perf_kzero_folded);
            errors = errors + 1;
        end

        prepare_all_active(MAX_TOKENS);
        run_row(MAX_TOKENS, 1'b1);
        if (perf_kzero_folded != 0 || perf_fold_classes != 0) begin
            $display("ERROR all-active row unexpectedly folded tokens/classes");
            errors = errors + 1;
        end

        q_vector[0] = 32'h00000001;
        k_current_vector[0] = 32'h00000001;
        k_peer_vector[0] = 32'h00000000;
        time_vector[0] = 1'b0;
        run_row(1, 1'b1);

        all_class_count = ENABLE_MOTION_XOR ? 35 : 3;
        if (MAX_TOKENS >= all_class_count) begin
            prepare_all_fold_classes(all_class_count);
            run_row(all_class_count, 1'b1);
            if (perf_fold_classes != all_class_count[CLASS_COUNT_W-1:0]) begin
                $display("ERROR all-class row got=%0d expected=%0d", perf_fold_classes, all_class_count);
                errors = errors + 1;
            end
        end

        if (errors == 0) begin
            $display("PASS: score-class row engine preserves denominator and gates, motion=%0d", ENABLE_MOTION_XOR);
        end else begin
            $fatal(1, "FAIL: H67 row engine errors=%0d", errors);
        end
        $finish;
    end

    initial begin
        repeat (200000) @(posedge clk);
        $fatal(1, "FAIL: row engine watchdog timeout");
    end
endmodule

`default_nettype wire
