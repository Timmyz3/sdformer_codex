`timescale 1ns/1ps
`include "../rtl_allbinary/unibin_h60_pkg.vh"

module tb_unibin_h60_modules;
    reg clk;
    reg rst_n;
    reg clear;
    reg enable;
    reg signed [15:0] membrane;
    reg signed [15:0] threshold;
    wire event_out;
    reg signed [15:0] input_current;
    reg [3:0] leak_shift;
    reg soft_reset_en;
    wire state_event_out;
    wire signed [15:0] mem_state;
    wire signed [15:0] mem_candidate;

    reg [`UBIN_HEAD_DIM-1:0] q_bits;
    reg [`UBIN_HEAD_DIM-1:0] k_bits;
    wire [7:0] q_active;
    wire [7:0] k_active;
    wire [7:0] overlap;
    wire [7:0] mismatch;
    wire signed [`UBIN_SCORE_W-1:0] tx_score;
    wire signed [`UBIN_SCORE_W-1:0] sc_score;
    wire signed [`UBIN_SCORE_W-1:0] fused_score;

    reg [63:0] q_bundle;
    reg [63:0] k_bundle;
    wire empty_bundle;
    wire [7:0] active_count;

    reg signed [8*`UBIN_SCORE_W-1:0] scores_flat;
    wire [8*`UBIN_GATE_W-1:0] gates_flat;

    wire signed [`UBIN_DATA_W+`UBIN_GATE_W-1:0] gated_k;

    binary_atlif_unit u_atlif (
        .membrane(membrane),
        .threshold(threshold),
        .event_out(event_out)
    );

    binary_atlif_state_unit u_atlif_state (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .input_current(input_current),
        .threshold(threshold),
        .leak_shift(leak_shift),
        .soft_reset_en(soft_reset_en),
        .event_out(state_event_out),
        .mem_state(mem_state),
        .mem_candidate(mem_candidate)
    );

    binary_popcount_consensus u_score (
        .q_bits(q_bits),
        .k_bits(k_bits),
        .mu_q8(`UBIN_MU_Q8_DEFAULT),
        .q_active(q_active),
        .k_active(k_active),
        .overlap(overlap),
        .mismatch(mismatch),
        .tx_score(tx_score),
        .sc_score(sc_score),
        .fused_score(fused_score)
    );

    ttb_skip_unit #(.BUNDLE_BITS(64)) u_ttb (
        .q_bundle(q_bundle),
        .k_bundle(k_bundle),
        .empty_bundle(empty_bundle),
        .active_count(active_count)
    );

    shiftmax_int8_unit #(.MAX_TOKENS(8)) u_shift (
        .scores_flat(scores_flat),
        .n_tokens(8'd4),
        .preserve_mean(1'b1),
        .gates_flat(gates_flat)
    );

    gated_k_unit u_gated (
        .k_event(1'b1),
        .k_value(8'sd3),
        .gate(8'd64),
        .gated_out(gated_k)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        enable = 1'b0;
        input_current = 16'sd0;
        leak_shift = 4'd0;
        soft_reset_en = 1'b1;
        membrane = 16'sd9;
        threshold = 16'sd10;
        #12;
        rst_n = 1'b1;
        #1;
        if (event_out !== 1'b0) begin
            $display("FAIL: ATLIF below threshold fired");
            $finish;
        end
        membrane = 16'sd10;
        #1;
        if (event_out !== 1'b1) begin
            $display("FAIL: ATLIF threshold event missing");
            $finish;
        end

        input_current = 16'sd4;
        leak_shift = 4'd0;
        soft_reset_en = 1'b1;
        enable = 1'b1;
        @(posedge clk);
        #1;
        if (state_event_out !== 1'b0 || mem_state !== 16'sd4) begin
            $display("FAIL: state ATLIF step1 event=%0d mem=%0d", state_event_out, mem_state);
            $finish;
        end
        @(posedge clk);
        #1;
        if (state_event_out !== 1'b0 || mem_state !== 16'sd8) begin
            $display("FAIL: state ATLIF step2 event=%0d mem=%0d", state_event_out, mem_state);
            $finish;
        end
        @(posedge clk);
        #1;
        if (state_event_out !== 1'b1 || mem_state !== 16'sd2) begin
            $display("FAIL: state ATLIF soft reset event=%0d mem=%0d", state_event_out, mem_state);
            $finish;
        end

        clear = 1'b1;
        @(posedge clk);
        #1;
        clear = 1'b0;
        input_current = 16'sd8;
        @(posedge clk);
        #1;
        input_current = 16'sd0;
        leak_shift = 4'd1;
        @(posedge clk);
        #1;
        if (state_event_out !== 1'b0 || mem_state !== 16'sd4) begin
            $display("FAIL: state ATLIF leak event=%0d mem=%0d", state_event_out, mem_state);
            $finish;
        end
        enable = 1'b0;

        q_bits = 32'h0000000b; // 1011, active=3
        k_bits = 32'h0000000d; // 1101, active=3, overlap=2
        #1;
        if (q_active !== 8'd3 || k_active !== 8'd3 || overlap !== 8'd2 || mismatch !== 8'd2) begin
            $display("FAIL: consensus counts q=%0d k=%0d overlap=%0d mismatch=%0d",
                     q_active, k_active, overlap, mismatch);
            $finish;
        end
        if (tx_score !== 16'sd10 || sc_score !== 16'sd8 || fused_score !== 16'sd11) begin
            $display("FAIL: consensus scores tx=%0d sc=%0d fused=%0d", tx_score, sc_score, fused_score);
            $finish;
        end

        q_bundle = 64'd0;
        k_bundle = 64'd0;
        #1;
        if (empty_bundle !== 1'b1 || active_count !== 8'd0) begin
            $display("FAIL: empty TTB detection");
            $finish;
        end
        q_bundle = 64'h0000_0000_0000_0003;
        k_bundle = 64'h0000_0000_0000_0004;
        #1;
        if (empty_bundle !== 1'b0 || active_count !== 8'd3) begin
            $display("FAIL: non-empty TTB active_count=%0d", active_count);
            $finish;
        end

        scores_flat = {16'sd0, 16'sd0, 16'sd0, 16'sd0, 16'sd0, 16'sd1, 16'sd2, 16'sd3};
        #1;
        if (gates_flat[0 +: 8] == 8'd0) begin
            $display("FAIL: shiftmax gate zero for active token");
            $finish;
        end

        if (gated_k !== 16'sd192) begin
            $display("FAIL: gated_k expected 192 got %0d", gated_k);
            $finish;
        end

        $display("PASS: UniBin-H60 module smoke tests passed");
        $finish;
    end
endmodule
