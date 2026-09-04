`timescale 1ns/1ps
`default_nettype none

module tb_h67_denominator_certificate;
    localparam int HEAD_DIM = 32;
    localparam int PAIRS = 225;
    localparam int TOKENS = 2 * PAIRS;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(TOKENS + 1);

    logic clk_core;
    logic rst_core;
    logic row_load_start;
    logic load_accept;
    logic [PAIR_ID_W-1:0] load_pair_id;
    logic [63:0] load_q_pair;
    logic [63:0] load_k_pair;
    logic certificate_valid;
    logic certificate_pass;
    logic [5:0] denominator_shift;
    logic [5:0] row_qcount_max;
    logic [$clog2(PAIRS+1)-1:0] accepted_pairs;
    logic protocol_error;
    logic qkm_certificate_valid;
    logic qkm_certificate_pass;
    logic [5:0] qkm_denominator_shift;
    logic [7:0] row_score_upper_bound;
    logic [$clog2(PAIRS+1)-1:0] qkm_accepted_pairs;
    logic qkm_protocol_error;

    logic signed [15:0] score0_w;
    logic signed [15:0] score1_w;
    logic [5:0] unused_count [0:5];
    logic signed [15:0] exp_delta_q7;
    logic [15:0] exp_probe_q8;
    logic [31:0] row_sum_probe;
    logic [5:0] baseline_shift;
    logic [15:0] gate_exp_probe;
    logic [8:0] baseline_gate;
    logic [8:0] certified_gate;
    logic used_certificate;
    logic [8:0] qkm_certified_gate;
    logic qkm_used_certificate;

    integer score_mem [0:TOKENS-1];
    integer row_kind;
    integer pair_index;
    integer token_index;
    integer row_max;
    integer row_sum;
    integer tested_rows;
    integer tested_gates;
    integer errors;

    h67_row_qmax_denominator_certificate #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .load_q_pair(load_q_pair),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .denominator_shift(denominator_shift),
        .row_qcount_max(row_qcount_max),
        .accepted_pairs(accepted_pairs),
        .protocol_error(protocol_error)
    );

    h67_row_qkm_denominator_certificate #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W)
    ) qkm_dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .load_q_pair(load_q_pair),
        .load_k_pair(load_k_pair),
        .certificate_valid(qkm_certificate_valid),
        .certificate_pass(qkm_certificate_pass),
        .denominator_shift(qkm_denominator_shift),
        .row_score_upper_bound(row_score_upper_bound),
        .accepted_pairs(qkm_accepted_pairs),
        .protocol_error(qkm_protocol_error)
    );

    h67_motionxor_score_q7 u_score0 (
        .q_bits(load_q_pair[31:0]),
        .k_current_bits(load_k_pair[31:0]),
        .k_peer_bits(load_k_pair[63:32]),
        .overlap(unused_count[0]),
        .same_zero(unused_count[1]),
        .motion_xor(unused_count[2]),
        .score_q7(score0_w)
    );
    h67_motionxor_score_q7 u_score1 (
        .q_bits(load_q_pair[63:32]),
        .k_current_bits(load_k_pair[63:32]),
        .k_peer_bits(load_k_pair[31:0]),
        .overlap(unused_count[3]),
        .same_zero(unused_count[4]),
        .motion_xor(unused_count[5]),
        .score_q7(score1_w)
    );

    ttx_exp2_lut_q8 u_exp_probe (
        .delta_q7(exp_delta_q7),
        .exp_q8(exp_probe_q8)
    );
    ttx_ceil_log2_u32 u_baseline_shift (
        .value(row_sum_probe),
        .shift_amount(baseline_shift)
    );
    ttx_gate_quant_q17 #(
        .TOKEN_W(TOKEN_W)
    ) u_baseline_gate (
        .exp_q8(gate_exp_probe),
        .row_sum_q8(row_sum_probe),
        .n_tokens(TOKEN_W'(TOKENS)),
        .preserve_mean(1'b1),
        .gate_q17(baseline_gate)
    );
    h67_certified_gate_quant_q17 #(
        .TOKEN_W(TOKEN_W)
    ) u_certified_gate (
        .exp_q8(gate_exp_probe),
        .row_sum_q8(row_sum_probe),
        .n_tokens(TOKEN_W'(TOKENS)),
        .preserve_mean(1'b1),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .certified_shift(denominator_shift),
        .used_certificate(used_certificate),
        .gate_q17(certified_gate)
    );
    h67_certified_gate_quant_q17 #(
        .TOKEN_W(TOKEN_W)
    ) u_qkm_certified_gate (
        .exp_q8(gate_exp_probe),
        .row_sum_q8(row_sum_probe),
        .n_tokens(TOKEN_W'(TOKENS)),
        .preserve_mean(1'b1),
        .certificate_valid(qkm_certificate_valid),
        .certificate_pass(qkm_certificate_pass),
        .certified_shift(qkm_denominator_shift),
        .used_certificate(qkm_used_certificate),
        .gate_q17(qkm_certified_gate)
    );

    always #5 clk_core = ~clk_core;

    function automatic [31:0] rotated_mask(
        input integer count,
        input integer rotate
    );
        integer lane;
        integer position;
        begin
            rotated_mask = 32'd0;
            for (lane = 0; lane < count; lane = lane + 1) begin
                position = (lane + rotate) % 32;
                rotated_mask[position] = 1'b1;
            end
        end
    endfunction

    task automatic reset_dut;
        begin
            rst_core = 1'b1;
            row_load_start = 1'b0;
            load_accept = 1'b0;
            load_pair_id = '0;
            load_q_pair = '0;
            load_k_pair = '0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic start_row;
        begin
            @(negedge clk_core);
            row_load_start = 1'b1;
            load_accept = 1'b0;
            @(posedge clk_core);
            #1;
            @(negedge clk_core);
            row_load_start = 1'b0;
        end
    endtask

    task automatic build_pair(
        input integer kind,
        input integer index,
        output logic [63:0] q_pair_value,
        output logic [63:0] k_pair_value
    );
        integer qcount0;
        integer qcount1;
        logic [31:0] q0;
        logic [31:0] q1;
        logic [31:0] k0;
        logic [31:0] k1;
        begin
            if (kind == 0) begin
                qcount0 = index == 0 ? 15 : ((index * 5 + 3) % 16);
                qcount1 = (index * 7 + 1) % 16;
            end else if (kind == 1) begin
                qcount0 = index == 0 ? 16 : ((index * 3 + 2) % 20);
                qcount1 = (index * 9 + 4) % 18;
            end else begin
                qcount0 = (index * (kind + 3) + kind) % 33;
                qcount1 = (index * (kind + 7) + 2 * kind) % 33;
            end
            q0 = rotated_mask(qcount0, index + kind);
            q1 = rotated_mask(qcount1, 3 * index + kind);
            k0 = 32'h9e37_79b9 ^ (32'(index) * 32'h045d_9f3b)
               ^ (32'(kind) * 32'h27d4_eb2d);
            k1 = {k0[12:0], k0[31:13]} ^ 32'ha5a5_5a5a;
            if (index == 0 && (kind == 0 || kind == 1)) begin
                k0 = q0;
                k1 = ~q0;
                q1 = 32'd0;
            end
            q_pair_value = {q1, q0};
            k_pair_value = {k1, k0};
        end
    endtask

    task automatic load_complete_row(input integer kind);
        logic [63:0] q_value;
        logic [63:0] k_value;
        begin
            start_row();
            for (pair_index = 0; pair_index < PAIRS;
                 pair_index = pair_index + 1) begin
                if (((pair_index + kind) % 11) == 0) begin
                    @(negedge clk_core);
                    load_accept = 1'b0;
                    @(posedge clk_core);
                end
                build_pair(kind, pair_index, q_value, k_value);
                @(negedge clk_core);
                load_pair_id = PAIR_ID_W'(pair_index);
                load_q_pair = q_value;
                load_k_pair = k_value;
                load_accept = 1'b1;
                @(posedge clk_core);
                #1;
                score_mem[2 * pair_index] = $signed(score0_w);
                score_mem[2 * pair_index + 1] = $signed(score1_w);
            end
            @(negedge clk_core);
            load_accept = 1'b0;
            #1;
        end
    endtask

    task automatic check_loaded_row(input integer expected_kind);
        begin
            row_max = 0;
            for (token_index = 0; token_index < TOKENS;
                 token_index = token_index + 1) begin
                if (score_mem[token_index] > row_max)
                    row_max = score_mem[token_index];
            end

            row_sum = 0;
            for (token_index = 0; token_index < TOKENS;
                 token_index = token_index + 1) begin
                exp_delta_q7 = 16'(score_mem[token_index] - row_max);
                #1;
                row_sum = row_sum + exp_probe_q8;
            end
            row_sum_probe = 32'(row_sum);
            #1;

            if (!certificate_valid || accepted_pairs != TOKENS/2
                || !qkm_certificate_valid || qkm_accepted_pairs != TOKENS/2) begin
                $error("row %0d did not certify complete row", expected_kind);
                errors = errors + 1;
            end
            if (expected_kind == 0) begin
                if (!certificate_pass || row_qcount_max != 15 || row_max != 93
                    || !qkm_certificate_pass || row_score_upper_bound != 93) begin
                    $error("pass-bound row mismatch pass=%0b qmax=%0d row_max=%0d",
                           certificate_pass, row_qcount_max, row_max);
                    errors = errors + 1;
                end
            end
            if (expected_kind == 1) begin
                if (certificate_pass || row_qcount_max < 16 || row_max < 97
                    || qkm_certificate_pass || row_score_upper_bound < 97) begin
                    $error("fail-bound row mismatch pass=%0b qmax=%0d row_max=%0d",
                           certificate_pass, row_qcount_max, row_max);
                    errors = errors + 1;
                end
            end
            if (certificate_pass && (row_max > 96 || baseline_shift != 17)) begin
                $error("certificate false positive qmax=%0d row_max=%0d shift=%0d sum=%0d",
                       row_qcount_max, row_max, baseline_shift, row_sum);
                errors = errors + 1;
            end
            if (row_score_upper_bound < row_max) begin
                $error("QKM upper bound below actual bound=%0d actual=%0d",
                       row_score_upper_bound, row_max);
                errors = errors + 1;
            end
            if (qkm_certificate_pass
                && (row_score_upper_bound > 96 || baseline_shift != 17)) begin
                $error("QKM certificate false positive bound=%0d actual=%0d shift=%0d",
                       row_score_upper_bound, row_max, baseline_shift);
                errors = errors + 1;
            end

            for (token_index = 0; token_index < TOKENS;
                 token_index = token_index + 1) begin
                exp_delta_q7 = 16'(score_mem[token_index] - row_max);
                #1;
                gate_exp_probe = exp_probe_q8;
                #1;
                if (baseline_gate !== certified_gate) begin
                    $error("gate mismatch kind=%0d token=%0d base=%0d cert=%0d",
                           expected_kind, token_index,
                           baseline_gate, certified_gate);
                    errors = errors + 1;
                end
                if (used_certificate !== certificate_pass) begin
                    $error("certificate select mismatch kind=%0d token=%0d",
                           expected_kind, token_index);
                    errors = errors + 1;
                end
                if (baseline_gate !== qkm_certified_gate) begin
                    $error("QKM gate mismatch kind=%0d token=%0d base=%0d cert=%0d",
                           expected_kind, token_index,
                           baseline_gate, qkm_certified_gate);
                    errors = errors + 1;
                end
                if (qkm_used_certificate !== qkm_certificate_pass) begin
                    $error("QKM certificate select mismatch kind=%0d token=%0d",
                           expected_kind, token_index);
                    errors = errors + 1;
                end
                tested_gates = tested_gates + 2;
            end
            tested_rows = tested_rows + 1;
        end
    endtask

    task automatic check_short_row_rejected;
        logic [63:0] q_value;
        logic [63:0] k_value;
        begin
            reset_dut();
            start_row();
            for (pair_index = 0; pair_index < 17;
                 pair_index = pair_index + 1) begin
                build_pair(0, pair_index, q_value, k_value);
                @(negedge clk_core);
                load_pair_id = PAIR_ID_W'(pair_index);
                load_q_pair = q_value;
                load_k_pair = k_value;
                load_accept = 1'b1;
                @(posedge clk_core);
            end
            @(negedge clk_core);
            load_accept = 1'b0;
            row_load_start = 1'b1;
            @(posedge clk_core);
            #1;
            if (!protocol_error || certificate_valid
                || !qkm_protocol_error || qkm_certificate_valid) begin
                $error("short row did not fail closed");
                errors = errors + 1;
            end
            @(negedge clk_core);
            row_load_start = 1'b0;
        end
    endtask

    task automatic check_out_of_order_rejected;
        logic [63:0] q_value;
        logic [63:0] k_value;
        begin
            reset_dut();
            start_row();
            build_pair(0, 1, q_value, k_value);
            @(negedge clk_core);
            load_pair_id = PAIR_ID_W'(1);
            load_q_pair = q_value;
            load_k_pair = k_value;
            load_accept = 1'b1;
            @(posedge clk_core);
            #1;
            if (!protocol_error || certificate_valid || accepted_pairs != 0
                || !qkm_protocol_error || qkm_certificate_valid
                || qkm_accepted_pairs != 0) begin
                $error("out-of-order row did not fail closed");
                errors = errors + 1;
            end
            @(negedge clk_core);
            load_accept = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        errors = 0;
        tested_rows = 0;
        tested_gates = 0;
        row_sum_probe = '0;
        exp_delta_q7 = '0;
        gate_exp_probe = '0;

        reset_dut();
        for (row_kind = 0; row_kind < 18; row_kind = row_kind + 1) begin
            load_complete_row(row_kind);
            check_loaded_row(row_kind);
        end
        check_short_row_rejected();
        check_out_of_order_rejected();

        if (errors != 0)
            $fatal(1, "FAIL tb_h67_denominator_certificate errors=%0d", errors);
        $display("PASS tb_h67_denominator_certificate rows=%0d gates=%0d errors=0",
                 tested_rows, tested_gates);
        $finish;
    end
endmodule

`default_nettype wire
