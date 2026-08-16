`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_slot_restart_reject_2s;
    localparam int HEAD_DIM = 32;
    localparam int PAIRS = 1;
    localparam int TOKEN_W = 2;

    logic clk;
    logic rst_core;
    logic window_start;
    logic window_seal;
    logic seal_ready;
    logic window_done;
    logic pair_valid;
    logic pair_ready;
    logic [0:0] pair_id;
    logic [63:0] q_pair;
    logic [63:0] k_pair;
    logic out_valid;
    logic out_ready;
    logic out_last;
    logic [TOKEN_W-1:0] out_token_id;
    logic [HEAD_DIM-1:0] out_k_bits;
    logic [8:0] out_gate_q17;
    logic [7:0] out_threshold_q8;
    logic protocol_error;
    logic [TOKEN_W-1:0] held_token;
    logic [HEAD_DIM-1:0] held_k;
    logic [8:0] held_gate;
    integer outputs;
    integer watchdog;
    logic [31:0] held_pairs;
    logic [2:0] held_fifo_occupancy;

    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .PAIR_ID_W(1),
        .TOKEN_W(TOKEN_W),
        .MAX_DESCRIPTORS(2),
        .SLOT_FIFO_DEPTH(4),
        .FIFO_OCC_W(3),
        .QUOTIENT_ENABLE(1'b1)
    ) dut (
        .clk_core(clk), .rst_core(rst_core),
        .window_start(window_start), .window_seal(window_seal),
        .descriptor_issue_enable(1'b1),
        .cfg_preserve_mean(1'b0), .cfg_threshold_q8(8'd0),
        .seal_ready(seal_ready), .window_done(window_done),
        .pair_valid(pair_valid), .pair_ready(pair_ready), .pair_id(pair_id),
        .q_pair(q_pair), .k_pair(k_pair),
        .out_valid(out_valid), .out_ready(out_ready), .out_last(out_last),
        .out_token_id(out_token_id), .out_k_bits(out_k_bits),
        .out_gate_q17(out_gate_q17), .out_threshold_q8(out_threshold_q8),
        .protocol_error(protocol_error),
        .perf_pairs(), .perf_slots(), .perf_equal_pairs(),
        .perf_quotient_descriptors(), .perf_original_tokens(),
        .perf_active_entries(), .perf_class_transactions(),
        .perf_exp_transactions(), .perf_emitted_tokens(),
        .perf_k_read_transactions(), .perf_k_read_bits(),
        .perf_total_cycles(), .perf_pair_stall_cycles(),
        .perf_descriptor_stall_cycles(), .perf_output_stall_cycles(),
        .perf_fifo_occupancy(), .perf_fifo_max_occupancy()
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_core && out_valid && out_ready) begin
            if (out_token_id !== TOKEN_W'(outputs))
                $fatal(1, "restart reject changed output order expected=%0d got=%0d",
                    outputs, out_token_id);
            outputs = outputs + 1;
        end
    end

    initial begin
        clk = 1'b0;
        rst_core = 1'b1;
        window_start = 1'b0;
        window_seal = 1'b0;
        pair_valid = 1'b0;
        pair_id = '0;
        q_pair = {32'hffff_ffff, 32'hffff_ffff};
        k_pair = {32'hffff_ffff, 32'hffff_ffff};
        out_ready = 1'b0;
        outputs = 0;
        repeat (4) @(negedge clk);
        rst_core = 1'b0;

        @(negedge clk);
        window_start = 1'b1;
        #1;
        if (!dut.window_start_accept || dut.window_start_reject)
            $fatal(1, "initial window_start was not accepted");
        @(negedge clk);
        window_start = 1'b0;
        pair_valid = 1'b1;
        while (!pair_ready) @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        pair_valid = 1'b0;
        if (dut.perf_pairs != 1)
            $fatal(1, "single pair was not committed");

        watchdog = 0;
        while (!seal_ready && watchdog < 200) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (!seal_ready) $fatal(1, "seal_ready timeout");
        window_seal = 1'b1;
        @(negedge clk);
        window_seal = 1'b0;

        watchdog = 0;
        while (!out_valid && watchdog < 500) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (!out_valid) $fatal(1, "output did not reach held emitter state");
        held_token = out_token_id;
        held_k = out_k_bits;
        held_gate = out_gate_q17;
        held_pairs = dut.perf_pairs;
        held_fifo_occupancy = dut.perf_fifo_occupancy;

        pair_valid = 1'b1;
        window_start = 1'b1;
        #1;
        if (dut.window_start_accept || !dut.window_start_reject)
            $fatal(1, "early restart was not rejected");
        if (!out_valid || out_token_id !== held_token
            || out_k_bits !== held_k || out_gate_q17 !== held_gate)
            $fatal(1, "early restart disturbed held output before clock");
        @(posedge clk);
        #1;
        if (!protocol_error || !out_valid || out_token_id !== held_token
            || out_k_bits !== held_k || out_gate_q17 !== held_gate)
            $fatal(1, "rejected restart was not fail-closed");
        if (dut.encoder_pair_commit || dut.perf_pairs != held_pairs
            || dut.perf_fifo_occupancy != held_fifo_occupancy)
            $fatal(1, "rejected restart accepted an unhandshaken pair");
        @(negedge clk);
        window_start = 1'b0;
        pair_valid = 1'b0;
        out_ready = 1'b1;

        watchdog = 0;
        while (!window_done && watchdog < 500) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (!window_done || outputs != 2)
            $fatal(1, "prior window did not complete after rejected restart outputs=%0d", outputs);

        window_start = 1'b1;
        #1;
        if (!dut.window_start_accept || dut.window_start_reject)
            $fatal(1, "restart after window_done was not accepted");
        @(posedge clk);
        #1;
        if (protocol_error)
            $fatal(1, "legal restart did not clear sticky protocol error");
        @(negedge clk);
        window_start = 1'b0;

        $display("PASS H67 RQTB 2S rejected-restart fail-closed outputs=%0d", outputs);
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk);
        $fatal(1, "restart reject watchdog timeout");
    end
endmodule

`default_nettype wire
