`timescale 1ns/1ps
`default_nettype none

module tb_unibin_h60_core_dc;
    localparam int HEAD_DIM = 32;
    localparam int MAX_TOKENS = 162;
    localparam int DATA_W = 8;
    localparam int SCORE_W = 16;
    localparam int GATE_W = 8;
    localparam int COUNT_W = 8;

    logic clk_core;
    logic rst_n_core;
    logic cfg_start;
    logic [COUNT_W-1:0] cfg_n_tokens;
    logic [7:0] cfg_mu_q8;
    logic cfg_preserve_mean;
    logic in_valid;
    logic in_ready;
    logic in_last;
    logic [HEAD_DIM-1:0] in_q_bits;
    logic [HEAD_DIM-1:0] in_k_bits;
    logic signed [DATA_W-1:0] in_k_value;
    logic out_valid;
    logic out_ready;
    logic out_last;
    logic [COUNT_W-1:0] out_token_idx;
    logic [GATE_W-1:0] out_gate;
    logic signed [DATA_W+GATE_W-1:0] out_gated_k;
    logic busy;
    logic done;
    logic [COUNT_W-1:0] perf_tokens_loaded;
    logic [COUNT_W-1:0] perf_empty_tokens;
    logic [COUNT_W-1:0] perf_issued_tokens;

    int out_count;
    int nonzero_gate_count;
    int expected_last_idx;
    int held_token_idx;
    int held_gate;
    int token_i;

    unibin_h60_core_dc #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .DATA_W(DATA_W),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W),
        .COUNT_W(COUNT_W)
    ) dut (
        .clk_core(clk_core),
        .rst_n_core(rst_n_core),
        .cfg_start(cfg_start),
        .cfg_n_tokens(cfg_n_tokens),
        .cfg_mu_q8(cfg_mu_q8),
        .cfg_preserve_mean(cfg_preserve_mean),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_last(in_last),
        .in_q_bits(in_q_bits),
        .in_k_bits(in_k_bits),
        .in_k_value(in_k_value),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last),
        .out_token_idx(out_token_idx),
        .out_gate(out_gate),
        .out_gated_k(out_gated_k),
        .busy(busy),
        .done(done),
        .perf_tokens_loaded(perf_tokens_loaded),
        .perf_empty_tokens(perf_empty_tokens),
        .perf_issued_tokens(perf_issued_tokens)
    );

    always #5 clk_core = ~clk_core;

    task automatic start_frame(
        input logic [COUNT_W-1:0] token_count,
        input int expected_last
    );
        begin
            cfg_n_tokens <= token_count;
            expected_last_idx <= expected_last;
            @(posedge clk_core);
            cfg_start <= 1'b1;
            @(posedge clk_core);
            cfg_start <= 1'b0;
        end
    endtask

    task automatic send_token(
        input logic [HEAD_DIM-1:0] q,
        input logic [HEAD_DIM-1:0] k,
        input logic signed [DATA_W-1:0] kv,
        input logic last
    );
        begin
            @(posedge clk_core);
            in_valid <= 1'b1;
            in_q_bits <= q;
            in_k_bits <= k;
            in_k_value <= kv;
            in_last <= last;
            while (!in_ready) begin
                @(posedge clk_core);
            end
            @(posedge clk_core);
            in_valid <= 1'b0;
            in_q_bits <= '0;
            in_k_bits <= '0;
            in_k_value <= '0;
            in_last <= 1'b0;
        end
    endtask

    initial begin
        $dumpfile("tb_unibin_h60_core_dc.vcd");
        $dumpvars(0, tb_unibin_h60_core_dc);

        clk_core = 1'b0;
        rst_n_core = 1'b0;
        cfg_start = 1'b0;
        cfg_n_tokens = 8'd4;
        cfg_mu_q8 = 8'd16;
        cfg_preserve_mean = 1'b1;
        in_valid = 1'b0;
        in_last = 1'b0;
        in_q_bits = '0;
        in_k_bits = '0;
        in_k_value = '0;
        out_ready = 1'b1;
        out_count = 0;
        nonzero_gate_count = 0;
        expected_last_idx = 3;
        held_token_idx = 0;
        held_gate = 0;

        repeat (4) @(posedge clk_core);
        rst_n_core <= 1'b1;

        start_frame(8'd4, 3);
        send_token(32'h0000_000b, 32'h0000_000d, 8'sd3, 1'b0);
        send_token(32'h0000_0000, 32'h0000_0000, 8'sd5, 1'b0);
        send_token(32'h0000_00ff, 32'h0000_000f, -8'sd2, 1'b0);
        send_token(32'h0000_0001, 32'h0000_0001, 8'sd4, 1'b1);

        wait(done === 1'b1);
        @(posedge clk_core);

        if (perf_tokens_loaded !== 8'd4) begin
            $display("FAIL: perf_tokens_loaded=%0d", perf_tokens_loaded);
            $finish;
        end
        if (perf_empty_tokens !== 8'd1) begin
            $display("FAIL: perf_empty_tokens=%0d", perf_empty_tokens);
            $finish;
        end
        if (perf_issued_tokens !== 8'd4) begin
            $display("FAIL: perf_issued_tokens=%0d", perf_issued_tokens);
            $finish;
        end
        if (out_count != 4) begin
            $display("FAIL: out_count=%0d", out_count);
            $finish;
        end
        if (nonzero_gate_count == 0) begin
            $display("FAIL: no nonzero gates observed");
            $finish;
        end

        out_ready <= 1'b0;
        start_frame(8'd6, 2);
        send_token(32'h0000_0003, 32'h0000_0001, 8'sd7, 1'b0);
        send_token(32'h0000_0000, 32'h0000_0000, 8'sd1, 1'b0);
        send_token(32'h0000_00f0, 32'h0000_00f0, -8'sd3, 1'b1);

        wait(out_valid === 1'b1);
        held_token_idx = out_token_idx;
        held_gate = out_gate;
        repeat (2) @(posedge clk_core);
        if (out_valid !== 1'b1 || out_token_idx !== held_token_idx[COUNT_W-1:0] || out_gate !== held_gate[GATE_W-1:0]) begin
            $display("FAIL: output changed under backpressure valid=%0d idx=%0d/%0d gate=%0d/%0d",
                     out_valid, out_token_idx, held_token_idx, out_gate, held_gate);
            $finish;
        end
        out_ready <= 1'b1;

        wait(done === 1'b1);
        @(posedge clk_core);

        if (perf_tokens_loaded !== 8'd3) begin
            $display("FAIL: early-last perf_tokens_loaded=%0d", perf_tokens_loaded);
            $finish;
        end
        if (perf_empty_tokens !== 8'd1) begin
            $display("FAIL: early-last perf_empty_tokens=%0d", perf_empty_tokens);
            $finish;
        end
        if (perf_issued_tokens !== 8'd3) begin
            $display("FAIL: early-last perf_issued_tokens=%0d", perf_issued_tokens);
            $finish;
        end
        if (out_count != 7) begin
            $display("FAIL: cumulative out_count=%0d", out_count);
            $finish;
        end

        start_frame(8'd162, 161);
        for (token_i = 0; token_i < 162; token_i = token_i + 1) begin
            send_token(
                32'h0000_0001 << (token_i % HEAD_DIM),
                32'h0000_0003 << (token_i % (HEAD_DIM - 1)),
                token_i[0] ? -8'sd2 : 8'sd2,
                (token_i == 161)
            );
        end

        wait(done === 1'b1);
        @(posedge clk_core);

        if (perf_tokens_loaded !== 8'd162) begin
            $display("FAIL: max-token perf_tokens_loaded=%0d", perf_tokens_loaded);
            $finish;
        end
        if (perf_issued_tokens !== 8'd162) begin
            $display("FAIL: max-token perf_issued_tokens=%0d", perf_issued_tokens);
            $finish;
        end
        if (out_count != 169) begin
            $display("FAIL: final cumulative out_count=%0d", out_count);
            $finish;
        end

        $display("PASS: unibin_h60_core_dc directed test passed");
        $finish;
    end

    always_ff @(posedge clk_core) begin
        if (!rst_n_core) begin
            out_count <= 0;
            nonzero_gate_count <= 0;
        end else if (out_valid && out_ready) begin
            out_count <= out_count + 1;
            if (out_gate != '0) begin
                nonzero_gate_count <= nonzero_gate_count + 1;
            end
            if (out_last && out_token_idx != expected_last_idx[COUNT_W-1:0]) begin
                $display("FAIL: out_last at token %0d", out_token_idx);
                $finish;
            end
        end
    end
endmodule

`default_nettype wire
