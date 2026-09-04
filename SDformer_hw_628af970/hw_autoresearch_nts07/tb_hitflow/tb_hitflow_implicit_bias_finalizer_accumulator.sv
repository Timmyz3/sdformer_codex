`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_implicit_bias_finalizer_accumulator;
    localparam int TOKENS = 8;
    localparam int BANKS = 2;
    localparam int PRODUCT_W = 8;
    localparam int ACC_W = 12;
    localparam int OUT_TILE = 2;
    localparam int TOKEN_ID_W = $clog2(TOKENS);

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic flush = 1'b0;
    logic group_start_valid;
    logic group_start_ready;
    logic [31:0] group_start_tag;
    logic [BANKS-1:0] update_valid;
    logic [BANKS-1:0] update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] update_token_ids;
    logic [31:0] update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] update_values;
    logic finalize_start_valid;
    logic finalize_start_ready;
    logic [31:0] finalize_start_tag;
    logic [(OUT_TILE*ACC_W)-1:0] finalize_bias_values;
    logic [BANKS-1:0] final_valid;
    logic [BANKS-1:0] final_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [31:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic finalize_done_valid;
    logic finalize_done_ready;
    logic [31:0] finalize_done_tag;
    logic protocol_error;
    logic accumulator_overflow;
    logic [31:0] count_updates;
    logic [31:0] count_product_writes;
    logic [31:0] count_final_reads;
    logic [31:0] count_final_emits;
    logic [31:0] count_update_stall_cycles;
    logic [31:0] count_final_stall_cycles;

    logic signed [ACC_W-1:0] expected0 [0:TOKENS-1];
    logic signed [ACC_W-1:0] expected1 [0:TOKENS-1];
    logic [TOKENS-1:0] expected_emit;
    logic [TOKENS-1:0] seen;
    integer phase;
    integer cycle_count;
    integer finalize_cycle;
    integer last_emit_cycle;
    integer emitted_in_phase;

    always #5 clk_core = ~clk_core;

    hitflow_implicit_bias_finalizer_accumulator #(
        .TOKENS(TOKENS), .BANKS(BANKS), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(32),
        .COUNTER_W(32), .TOKEN_ID_W(TOKEN_ID_W)
    ) dut (.*);

    task automatic start_group(input logic [31:0] tag);
        begin
            @(negedge clk_core);
            group_start_tag = tag;
            group_start_valid = 1'b1;
            do @(posedge clk_core); while (!group_start_ready);
            @(negedge clk_core);
            group_start_valid = 1'b0;
            update_tag = tag;
        end
    endtask

    task automatic send_update(
        input int token,
        input int signed value0,
        input int signed value1
    );
        int bank;
        begin
            bank = token % BANKS;
            @(negedge clk_core);
            update_token_ids = '0;
            update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] =
                TOKEN_ID_W'(token);
            update_values[0 +: PRODUCT_W] = PRODUCT_W'(value0);
            update_values[PRODUCT_W +: PRODUCT_W] = PRODUCT_W'(value1);
            update_valid = '0;
            update_valid[bank] = 1'b1;
            do @(posedge clk_core); while (!update_ready[bank]);
            @(negedge clk_core);
            update_valid = '0;
        end
    endtask

    task automatic start_finalize(
        input logic [31:0] tag,
        input int signed bias0,
        input int signed bias1
    );
        begin
            @(negedge clk_core);
            finalize_start_tag = tag;
            finalize_bias_values[0 +: ACC_W] = ACC_W'(bias0);
            finalize_bias_values[ACC_W +: ACC_W] = ACC_W'(bias1);
            finalize_start_valid = 1'b1;
            do @(posedge clk_core); while (!finalize_start_ready);
            @(negedge clk_core);
            finalize_start_valid = 1'b0;
            finalize_cycle = cycle_count;
        end
    endtask

    task automatic wait_done;
        begin
            while (!finalize_done_valid)
                @(posedge clk_core);
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            finalize_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            finalize_done_ready = 1'b0;
        end
    endtask

    always_comb begin
        if (phase == 1) begin
            final_ready = '1;
        end else if (phase == 2) begin
            final_ready[0] = (cycle_count % 3) != 0;
            final_ready[1] = (cycle_count % 4) != 1;
        end else begin
            final_ready = '0;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            seen <= '0;
            emitted_in_phase <= 0;
            last_emit_cycle <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    int token;
                    logic signed [ACC_W-1:0] got0;
                    logic signed [ACC_W-1:0] got1;
                    token = 32'(final_token_ids[
                        (bank*TOKEN_ID_W) +: TOKEN_ID_W]);
                    got0 = $signed(final_values[
                        (bank*OUT_TILE*ACC_W) +: ACC_W]);
                    got1 = $signed(final_values[
                        (bank*OUT_TILE*ACC_W)+ACC_W +: ACC_W]);
                    if (token >= TOKENS || !expected_emit[token])
                        $fatal(1, "unexpected final token=%0d phase=%0d",
                               token, phase);
                    if (seen[token])
                        $fatal(1, "duplicate final token=%0d", token);
                    if (got0 !== expected0[token] || got1 !== expected1[token])
                        $fatal(1,
                            "final mismatch token=%0d got=(%0d,%0d) exp=(%0d,%0d)",
                            token, got0, got1, expected0[token], expected1[token]);
                    seen[token] <= 1'b1;
                    emitted_in_phase <= emitted_in_phase + 1;
                    last_emit_cycle <= cycle_count;
                end
            end
        end
    end

    initial begin
        group_start_valid = 1'b0;
        group_start_tag = '0;
        update_valid = '0;
        update_token_ids = '0;
        update_tag = '0;
        update_values = '0;
        finalize_start_valid = 1'b0;
        finalize_start_tag = '0;
        finalize_bias_values = '0;
        finalize_done_ready = 1'b0;
        phase = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        phase = 1;
        seen = '0;
        emitted_in_phase = 0;
        for (int token = 0; token < TOKENS; token = token + 1) begin
            expected0[token] = 12'sd10;
            expected1[token] = -12'sd20;
        end
        expected0[0] = 12'sd17;
        expected1[0] = -12'sd26;
        expected0[3] = 12'sd6;
        expected1[3] = -12'sd12;
        expected0[4] = 12'sd110;
        expected1[4] = -12'sd120;
        expected0[7] = 12'sd11;
        expected1[7] = -12'sd18;
        expected_emit = '1;
        start_group(32'h1234_0001);
        send_update(0, 5, -7);
        send_update(0, 2, 1);
        send_update(3, -4, 8);
        send_update(4, 100, -100);
        send_update(7, 1, 2);
        repeat (3) @(posedge clk_core);

        @(negedge clk_core);
        finalize_start_tag = 32'hdead_beef;
        finalize_start_valid = 1'b1;
        @(posedge clk_core);
        if (finalize_start_ready || !protocol_error)
            $fatal(1, "wrong-tag finalize was not rejected");
        @(negedge clk_core);
        finalize_start_valid = 1'b0;

        start_finalize(32'h1234_0001, 10, -20);
        wait_done();
        if (seen != expected_emit)
            $fatal(1, "phase1 final coverage seen=%b", seen);
        if ((last_emit_cycle - finalize_cycle) > 6)
            $fatal(1, "phase1 drain did not sustain two tokens/cycle: %0d",
                   last_emit_cycle - finalize_cycle);
        if (accumulator_overflow)
            $fatal(1, "unexpected phase1 overflow");

        phase = 2;
        seen = '0;
        emitted_in_phase = 0;
        for (int token = 0; token < TOKENS; token = token + 1) begin
            expected0[token] = 12'sd2000;
            expected1[token] = -12'sd3;
        end
        expected_emit = '1;
        expected_emit[2] = 1'b0;
        start_group(32'h1234_0002);
        send_update(2, 100, 0);
        start_finalize(32'h1234_0002, 2000, -3);
        wait_done();
        if (seen != expected_emit)
            $fatal(1, "phase2 overflow quarantine seen=%b", seen);
        if (!accumulator_overflow)
            $fatal(1, "phase2 final-add overflow was not reported");
        if (count_updates != 6 || count_product_writes != 6 ||
            count_final_reads != 16 || count_final_emits != 15)
            $fatal(1,
                "counter mismatch update=%0d write=%0d read=%0d emit=%0d",
                count_updates, count_product_writes, count_final_reads,
                count_final_emits);
        if (count_final_stall_cycles == 0)
            $fatal(1, "final backpressure was not counted");

        $display("PASS: implicit-bias finalizer exact pow2=%0d reads=%0d emits=%0d stalls=%0d",
                 TOKENS, count_final_reads, count_final_emits,
                 count_final_stall_cycles);
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
