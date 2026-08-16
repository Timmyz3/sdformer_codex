`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_slot_build_restart_reject_2s;
    localparam int HEAD_DIM = 32;
    localparam int PAIRS = 2;
    localparam int TOKEN_W = 2;

    logic clk;
    logic rst_core;
    logic window_start;
    logic window_seal;
    logic seal_ready;
    logic window_done;
    logic pair_valid;
    logic pair_ready;
    logic pair_id;
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
    logic [31:0] held_pairs;
    logic [2:0] held_fifo_occupancy;
    integer outputs;
    integer watchdog;

    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .PAIR_ID_W(1),
        .TOKEN_W(TOKEN_W),
        .MAX_DESCRIPTORS(4),
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
                $fatal(1, "build restart changed output order expected=%0d got=%0d",
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
        pair_id = 1'b0;
        q_pair = {32'hffff_ffff, 32'hffff_ffff};
        k_pair = {32'hffff_ffff, 32'hffff_ffff};
        out_ready = 1'b1;
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

        pair_id = 1'b0;
        pair_valid = 1'b1;
        while (!pair_ready) @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        pair_valid = 1'b0;
        if (dut.perf_pairs != 1)
            $fatal(1, "pair0 was not committed");

        pair_id = 1'b1;
        watchdog = 0;
        while (!dut.encoder_pair_ready && watchdog < 100) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (!dut.encoder_pair_ready)
            $fatal(1, "pair1 never reached internally legal build state");
        held_pairs = dut.perf_pairs;
        held_fifo_occupancy = dut.perf_fifo_occupancy;

        pair_valid = 1'b1;
        window_start = 1'b1;
        #1;
        if (dut.window_start_accept || !dut.window_start_reject)
            $fatal(1, "build-stage restart was not rejected");
        if (pair_ready || dut.encoder_pair_valid)
            $fatal(1, "raw window_start did not block the legal pair1 request");
        @(posedge clk);
        #1;
        if (!protocol_error || dut.encoder_pair_commit
            || dut.perf_pairs != held_pairs
            || dut.perf_fifo_occupancy != held_fifo_occupancy)
            $fatal(1, "build-stage rejected restart accepted pair1 or disturbed FIFO");

        @(negedge clk);
        window_start = 1'b0;
        while (!pair_ready) @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        pair_valid = 1'b0;
        if (dut.perf_pairs != 2)
            $fatal(1, "pair1 did not commit after rejected restart");

        watchdog = 0;
        while (!seal_ready && watchdog < 300) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (!seal_ready) $fatal(1, "seal_ready timeout");
        window_seal = 1'b1;
        @(negedge clk);
        window_seal = 1'b0;

        watchdog = 0;
        while (!window_done && watchdog < 800) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (!window_done || outputs != 4)
            $fatal(1, "old window failed after build-stage reject outputs=%0d", outputs);

        window_start = 1'b1;
        #1;
        if (!dut.window_start_accept || dut.window_start_reject)
            $fatal(1, "legal next window_start was not accepted");
        @(posedge clk);
        #1;
        if (protocol_error)
            $fatal(1, "legal next window_start did not clear protocol_error");

        $display("PASS H67 RQTB 2S build-stage rejected-restart mutation-kill outputs=%0d", outputs);
        $finish;
    end

    initial begin
        repeat (3000) @(posedge clk);
        $fatal(1, "build-stage restart reject watchdog timeout");
    end
endmodule

`default_nettype wire
