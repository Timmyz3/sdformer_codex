`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_quotient_shiftmax_gate_miter #(
    parameter int PAIRS = 40,
    parameter int SEED = 32'h67c0ffee,
    parameter bit PRESERVE_MEAN = 1'b0
);
    localparam int HEAD_DIM = 32;
    localparam int TOKENS = 2 * PAIRS;
    localparam int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(TOKENS + 1);
    localparam int MAX_OUTPUTS = TOKENS;

    logic clk;
    logic rst_core;
    logic rst_n;

    logic base_start;
    logic base_in_valid;
    logic base_in_ready;
    logic base_in_last;
    logic base_time_sel;
    logic [31:0] base_q_bits;
    logic [63:0] base_k_pair;
    logic base_out_valid;
    logic base_out_ready;
    logic base_out_last;
    logic [TOKEN_W-1:0] base_out_token;
    logic [31:0] base_out_k;
    logic [8:0] base_out_gate;
    logic [7:0] base_out_threshold;
    logic base_busy;
    logic base_done;
    logic base_range_error;
    logic [15:0] base_perf_exp;

    logic quot_start;
    logic quot_seal;
    logic quot_seal_ready;
    logic quot_done;
    logic quot_pair_valid;
    logic quot_pair_ready;
    logic [PAIR_ID_W-1:0] quot_pair_id;
    logic [63:0] quot_q_pair;
    logic [63:0] quot_k_pair;
    logic quot_out_valid;
    logic quot_out_ready;
    logic quot_out_last;
    logic [TOKEN_W-1:0] quot_out_token;
    logic [31:0] quot_out_k;
    logic [8:0] quot_out_gate;
    logic [7:0] quot_out_threshold;
    logic quot_error;
    logic [31:0] quot_perf_pairs;
    logic [31:0] quot_perf_descriptors;
    logic [31:0] quot_perf_tokens;
    logic [31:0] quot_perf_active;
    logic [31:0] quot_perf_equal;
    logic [31:0] quot_perf_classes;
    logic [31:0] quot_perf_exp;
    logic [31:0] quot_perf_emitted;

    logic [31:0] q0_mem [0:PAIRS-1];
    logic [31:0] q1_mem [0:PAIRS-1];
    logic [31:0] k0_mem [0:PAIRS-1];
    logic [31:0] k1_mem [0:PAIRS-1];
    logic [TOKEN_W-1:0] base_token_log [0:MAX_OUTPUTS-1];
    logic [31:0] base_k_log [0:MAX_OUTPUTS-1];
    logic [8:0] base_gate_log [0:MAX_OUTPUTS-1];
    logic base_last_log [0:MAX_OUTPUTS-1];
    logic [TOKEN_W-1:0] quot_token_log [0:MAX_OUTPUTS-1];
    logic [31:0] quot_k_log [0:MAX_OUTPUTS-1];
    logic [8:0] quot_gate_log [0:MAX_OUTPUTS-1];
    logic quot_last_log [0:MAX_OUTPUTS-1];

    integer base_count;
    integer quot_count;
    integer expected_active;
    integer cycle_count;
    logic [31:0] rng_state;
    logic base_done_seen;
    logic quot_done_seen;

    function automatic logic [31:0] xorshift32(input logic [31:0] value);
        logic [31:0] next;
        begin
            next = value ^ (value << 13);
            next = next ^ (next >> 17);
            next = next ^ (next << 5);
            xorshift32 = next;
        end
    endfunction

    h67_score_class_row_engine #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(TOKENS),
        .ACTIVE_MEM_DEPTH(TOKENS),
        .GATE_W(9),
        .THRESHOLD_W(8),
        .TOKEN_W(TOKEN_W)
    ) u_baseline (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_start(base_start),
        .cfg_n_tokens(TOKEN_W'(TOKENS)),
        .cfg_preserve_mean(PRESERVE_MEAN),
        .cfg_enable_score_fold(1'b1),
        .cfg_threshold_q8(8'd128),
        .in_valid(base_in_valid),
        .in_ready(base_in_ready),
        .in_last(base_in_last),
        .in_time_sel(base_time_sel),
        .in_q_bits(base_q_bits),
        .in_k_pair_bits(base_k_pair),
        .out_valid(base_out_valid),
        .out_ready(base_out_ready),
        .out_last(base_out_last),
        .out_token_idx(base_out_token),
        .out_k_bits(base_out_k),
        .out_gate_q8(base_out_gate),
        .out_threshold_q8(base_out_threshold),
        .busy(base_busy),
        .done(base_done),
        .perf_tokens_loaded(),
        .perf_kzero_folded(),
        .perf_entries_emitted(),
        .perf_fold_classes(),
        .perf_exp_transactions(base_perf_exp),
        .perf_score_range_error(base_range_error)
    );

    h67_temporal_quotient_shiftmax_gate_top #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .TOKEN_W(TOKEN_W)
    ) u_quotient (
        .clk_core(clk),
        .rst_core(rst_core),
        .window_start(quot_start),
        .window_seal(quot_seal),
        .cfg_preserve_mean(PRESERVE_MEAN),
        .cfg_threshold_q8(8'd128),
        .seal_ready(quot_seal_ready),
        .window_done(quot_done),
        .pair_valid(quot_pair_valid),
        .pair_ready(quot_pair_ready),
        .pair_id(quot_pair_id),
        .q_pair(quot_q_pair),
        .k_pair(quot_k_pair),
        .out_valid(quot_out_valid),
        .out_ready(quot_out_ready),
        .out_last(quot_out_last),
        .out_token_id(quot_out_token),
        .out_k_bits(quot_out_k),
        .out_gate_q17(quot_out_gate),
        .out_threshold_q8(quot_out_threshold),
        .protocol_error(quot_error),
        .perf_pairs(quot_perf_pairs),
        .perf_quotient_descriptors(quot_perf_descriptors),
        .perf_original_tokens(quot_perf_tokens),
        .perf_active_entries(quot_perf_active),
        .perf_equal_pairs(quot_perf_equal),
        .perf_class_transactions(quot_perf_classes),
        .perf_exp_transactions(quot_perf_exp),
        .perf_emitted_tokens(quot_perf_emitted)
    );

    always #5 clk = ~clk;

    always @(negedge clk) begin
        if (rst_core) begin
            cycle_count <= 0;
            base_out_ready <= 1'b0;
            quot_out_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            base_out_ready <= (cycle_count % 5) != 1;
            quot_out_ready <= (cycle_count % 7) != 2;
        end
    end

    always @(posedge clk) begin
        if (!rst_n && base_out_valid)
            $fatal(1, "baseline output during reset");
        if (!rst_n && quot_out_valid)
            $fatal(1, "quotient output during reset");
        if (rst_core) begin
            base_count = 0;
            quot_count = 0;
            base_done_seen = 1'b0;
            quot_done_seen = 1'b0;
        end else begin
            if (base_out_valid && base_out_ready) begin
                base_token_log[base_count] = base_out_token;
                base_k_log[base_count] = base_out_k;
                base_gate_log[base_count] = base_out_gate;
                base_last_log[base_count] = base_out_last;
                if (base_out_threshold != 8'd128)
                    $fatal(1, "baseline threshold mismatch");
                base_count = base_count + 1;
            end
            if (quot_out_valid && quot_out_ready) begin
                quot_token_log[quot_count] = quot_out_token;
                quot_k_log[quot_count] = quot_out_k;
                quot_gate_log[quot_count] = quot_out_gate;
                quot_last_log[quot_count] = quot_out_last;
                if (quot_out_threshold != 8'd128)
                    $fatal(1, "quotient threshold mismatch");
                quot_count = quot_count + 1;
            end
            if (base_done)
                base_done_seen = 1'b1;
            if (quot_done)
                quot_done_seen = 1'b1;
        end
    end

    task automatic drive_baseline;
        begin
            @(negedge clk);
            base_start = 1'b1;
            @(negedge clk);
            base_start = 1'b0;
            for (integer pair = 0; pair < PAIRS; pair = pair + 1) begin
                while (!base_in_ready) @(negedge clk);
                base_in_valid = 1'b1;
                base_in_last = 1'b0;
                base_time_sel = 1'b0;
                base_q_bits = q0_mem[pair];
                base_k_pair = {k1_mem[pair], k0_mem[pair]};
                @(negedge clk);
                base_in_valid = 1'b0;

                while (!base_in_ready) @(negedge clk);
                base_in_valid = 1'b1;
                base_in_last = pair == PAIRS - 1;
                base_time_sel = 1'b1;
                base_q_bits = q1_mem[pair];
                base_k_pair = {k1_mem[pair], k0_mem[pair]};
                @(negedge clk);
                base_in_valid = 1'b0;
                base_in_last = 1'b0;
            end
        end
    endtask

    task automatic drive_quotient;
        begin
            @(negedge clk);
            quot_start = 1'b1;
            @(negedge clk);
            quot_start = 1'b0;
            for (integer pair = 0; pair < PAIRS; pair = pair + 1) begin
                quot_pair_id = PAIR_ID_W'(pair);
                quot_q_pair = {q1_mem[pair], q0_mem[pair]};
                quot_k_pair = {k1_mem[pair], k0_mem[pair]};
                while (!quot_pair_ready) @(negedge clk);
                quot_pair_valid = 1'b1;
                @(negedge clk);
                quot_pair_valid = 1'b0;
            end
            while (!quot_seal_ready) @(negedge clk);
            quot_seal = 1'b1;
            @(negedge clk);
            quot_seal = 1'b0;
        end
    endtask

    initial begin : global_watchdog
        wait (!rst_core);
        repeat (200000) @(posedge clk);
        if (!(base_done_seen && quot_done_seen))
            $fatal(1, "timeout PAIRS=%0d base=%0d quot=%0d", PAIRS,
                base_count, quot_count);
    end

    initial begin
        clk = 1'b0;
        rst_core = 1'b1;
        rst_n = 1'b0;
        base_start = 1'b0;
        base_in_valid = 1'b0;
        base_in_last = 1'b0;
        base_time_sel = 1'b0;
        base_q_bits = '0;
        base_k_pair = '0;
        base_out_ready = 1'b0;
        quot_start = 1'b0;
        quot_seal = 1'b0;
        quot_pair_valid = 1'b0;
        quot_pair_id = '0;
        quot_q_pair = '0;
        quot_k_pair = '0;
        quot_out_ready = 1'b0;
        expected_active = 0;
        cycle_count = 0;
        rng_state = SEED;

        for (integer pair = 0; pair < PAIRS; pair = pair + 1) begin
            rng_state = xorshift32(rng_state);
            q0_mem[pair] = rng_state;
            rng_state = xorshift32(rng_state);
            q1_mem[pair] = rng_state;
            rng_state = xorshift32(rng_state);
            k0_mem[pair] = rng_state;
            rng_state = xorshift32(rng_state);
            k1_mem[pair] = rng_state;
            if ((pair % 11) == 0) begin
                k0_mem[pair] = '0;
                k1_mem[pair] = '0;
            end else if ((pair % 5) == 0) begin
                q1_mem[pair] = q0_mem[pair];
                k1_mem[pair] = k0_mem[pair];
            end else begin
                if ((pair % 7) == 0)
                    k0_mem[pair] = '0;
                if ((pair % 9) == 0)
                    k1_mem[pair] = '0;
            end
            if (k0_mem[pair] != 0)
                expected_active = expected_active + 1;
            if (k1_mem[pair] != 0)
                expected_active = expected_active + 1;
        end

        repeat (4) @(negedge clk);
        rst_core = 1'b0;
        rst_n = 1'b1;
        fork
            drive_baseline();
            drive_quotient();
        join
        wait (base_done_seen && quot_done_seen);
        repeat (2) @(negedge clk);

        if (base_range_error || quot_error)
            $fatal(1, "protocol/range error base=%0d quotient=%0d",
                base_range_error, quot_error);
        if (base_count != expected_active || quot_count != expected_active)
            $fatal(1, "output count mismatch expected=%0d base=%0d quot=%0d",
                expected_active, base_count, quot_count);
        for (integer index = 0; index < expected_active; index = index + 1) begin
            if (base_token_log[index] != quot_token_log[index]
                || base_k_log[index] != quot_k_log[index]
                || base_gate_log[index] != quot_gate_log[index]
                || base_last_log[index] != quot_last_log[index])
                $fatal(1,
                    "miter mismatch index=%0d token=%0d/%0d k=%h/%h gate=%0d/%0d last=%0d/%0d",
                    index, base_token_log[index], quot_token_log[index],
                    base_k_log[index], quot_k_log[index],
                    base_gate_log[index], quot_gate_log[index],
                    base_last_log[index], quot_last_log[index]);
        end
        if (quot_perf_pairs != PAIRS
            || quot_perf_tokens != TOKENS
            || quot_perf_emitted != expected_active
            || quot_perf_equal == 0
            || quot_perf_descriptors > TOKENS
            || quot_perf_active > quot_perf_descriptors
            || quot_perf_classes == 0
            || quot_perf_exp != quot_perf_classes + quot_perf_active
            || quot_perf_exp > base_perf_exp)
            $fatal(1,
                "perf mismatch pairs=%0d tokens=%0d emitted=%0d equal=%0d desc=%0d active=%0d classes=%0d exp=%0d/%0d",
                quot_perf_pairs, quot_perf_tokens, quot_perf_emitted,
                quot_perf_equal, quot_perf_descriptors, quot_perf_active,
                quot_perf_classes, quot_perf_exp, base_perf_exp);

        $display(
            "PASS H67 TESC gated-K miter pairs=%0d tokens=%0d preserve=%0d active=%0d descriptors=%0d equal=%0d classes=%0d exp=%0d baseline_exp=%0d",
            PAIRS, TOKENS, PRESERVE_MEAN, expected_active, quot_perf_descriptors,
            quot_perf_equal, quot_perf_classes, quot_perf_exp, base_perf_exp
        );
        $finish;
    end
endmodule

`default_nettype wire
