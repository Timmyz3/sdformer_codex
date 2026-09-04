`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_banked_accumulator_overflow_atomic;
    localparam int TOKENS = 2;
    localparam int BANKS = 1;
    localparam int PRODUCT_W = 8;
    localparam int ACC_W = 8;
    localparam int OUT_TILE = 1;

    logic clk_core = 1'b0;
    logic rst_core;
    logic flush;
    logic group_start_valid, group_start_ready;
    logic [15:0] group_start_tag;
    logic update_valid, update_ready;
    logic update_token_ids;
    logic [15:0] update_tag;
    logic update_is_bias;
    logic [7:0] update_values;
    logic [7:0] update_bias_values;
    logic final_valid, final_ready;
    /* verilator lint_off UNUSEDSIGNAL */
    logic final_token_ids;
    logic [15:0] final_tag;
    logic [7:0] final_values;
    logic group_finish_valid, group_finish_ready;
    logic [15:0] group_finish_tag;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_updates, count_writes, count_bias_commits;
    logic [31:0] count_bank_stall_cycles, count_final_stall_cycles;
    /* verilator lint_on UNUSEDSIGNAL */
    integer final_handshakes;
    logic [31:0] updates_before_flush, writes_before_flush;
    logic [31:0] bias_before_flush, bank_stalls_before_flush;
    logic [31:0] final_stalls_before_flush;

    /* verilator lint_off BLKSEQ */
    always #1 clk_core = ~clk_core;
    /* verilator lint_on BLKSEQ */

    hitflow_banked_accumulator #(
        .TOKENS(TOKENS), .BANKS(BANKS), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(16), .TOKEN_ID_W(1)
    ) dut (.flush(flush), .*);

    always @(posedge clk_core) begin
        if (!rst_core && final_valid && final_ready) begin
            if (final_tag != 16'h0f11)
                $fatal(1, "post-flush final tag mismatch");
            if ((!final_token_ids && $signed(final_values) != 8'sd30) ||
                (final_token_ids && $signed(final_values) != 8'sd0))
                $fatal(1, "post-flush final value mismatch token=%0d value=%0d",
                       final_token_ids, $signed(final_values));
            final_handshakes <= final_handshakes + 1;
        end
    end

    task automatic send_update(input logic is_bias, input logic signed [7:0] value);
        begin
            update_is_bias = is_bias;
            update_values = value;
            update_bias_values = value;
            do @(negedge clk_core); while (!update_ready);
            update_valid = 1'b1;
            @(posedge clk_core);
            #0.1 update_valid = 1'b0;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        flush = 1'b0;
        group_start_valid = 1'b0;
        group_start_tag = 16'h0f10;
        update_valid = 1'b0;
        update_token_ids = 1'b0;
        update_tag = 16'h0f10;
        update_is_bias = 1'b0;
        update_values = '0;
        update_bias_values = '0;
        final_ready = 1'b1;
        group_finish_valid = 1'b0;
        final_handshakes = 0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        group_start_valid = 1'b1;
        do @(posedge clk_core); while (!group_start_ready);
        #0.1 group_start_valid = 1'b0;

        send_update(1'b0, 8'sd127);
        send_update(1'b1, 8'sd1);
        repeat (3) @(posedge clk_core);

        if (!accumulator_overflow || final_handshakes != 0 || final_valid ||
            count_bias_commits != 1 || protocol_error)
            $fatal(1,
                "overflow quarantine mismatch overflow=%b final_hs=%0d final_valid=%b bias=%0d protocol=%b",
                accumulator_overflow, final_handshakes, final_valid,
                count_bias_commits, protocol_error);

        updates_before_flush = count_updates;
        writes_before_flush = count_writes;
        bias_before_flush = count_bias_commits;
        bank_stalls_before_flush = count_bank_stall_cycles;
        final_stalls_before_flush = count_final_stall_cycles;
        flush = 1'b1;
        @(posedge clk_core);
        #0.1;
        if (accumulator_overflow || !flush || group_start_ready ||
            update_ready || final_valid || group_finish_ready)
            $fatal(1, "flush failed to clear overflow or mask handshakes");
        if (count_updates != updates_before_flush ||
            count_writes != writes_before_flush ||
            count_bias_commits != bias_before_flush ||
            count_bank_stall_cycles != bank_stalls_before_flush ||
            count_final_stall_cycles != final_stalls_before_flush)
            $fatal(1, "flush changed performance counters");
        flush = 1'b0;
        #0.1;
        if (!group_start_ready || final_valid)
            $fatal(1, "post-flush accumulator did not return idle");

        group_start_tag = 16'h0f11;
        update_tag = 16'h0f11;
        group_start_valid = 1'b1;
        do @(posedge clk_core); while (!group_start_ready);
        #0.1 group_start_valid = 1'b0;

        update_token_ids = 1'b0;
        send_update(1'b0, 8'sd10);
        send_update(1'b1, 8'sd20);
        update_token_ids = 1'b1;
        send_update(1'b1, 8'sd0);
        repeat (3) @(posedge clk_core);
        #0.1;

        if (accumulator_overflow || final_handshakes != 2 ||
            !group_finish_ready || protocol_error || count_updates != 5 ||
            count_writes != 5 || count_bias_commits != 3)
            $fatal(1,
                "post-flush recovery mismatch overflow=%b final_hs=%0d finish=%b updates=%0d writes=%0d bias=%0d",
                accumulator_overflow, final_handshakes, group_finish_ready,
                count_updates, count_writes, count_bias_commits);
        group_finish_valid = 1'b1;
        @(posedge clk_core);
        #0.1 group_finish_valid = 1'b0;
        if (!group_start_ready)
            $fatal(1, "post-flush recovery group did not finish");

        $display("RESULT status=PASS quarantined=1 overflow_cleared=1 recovery_finals=2 counters_preserved=1");
        $finish;
    end
endmodule

`default_nettype wire
