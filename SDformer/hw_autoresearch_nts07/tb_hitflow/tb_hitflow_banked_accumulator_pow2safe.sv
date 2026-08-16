`timescale 1ns/1ps
`default_nettype none

// Power-of-two leaf TB. ACC_DUT can select the mainline or historical patch DUT.
// Self-checking product + bias-final for all tokens.
module tb_hitflow_banked_accumulator_pow2safe;
`ifndef TB_TOKENS
`define TB_TOKENS 32
`endif
`ifndef ACC_DUT
`define ACC_DUT hitflow_banked_accumulator_pow2safe
`endif
    localparam int TOKENS = `TB_TOKENS;
    localparam int BANKS = 2;
    localparam int PRODUCT_W = 17;
    localparam int ACC_W = 24;
    localparam int OUT_TILE = 2;
    localparam int TAG_W = 16;
    localparam int TOKEN_ID_W = $clog2(TOKENS);

    logic clk_core = 1'b0;
    logic rst_core;
    logic group_start_valid;
    logic group_start_ready;
    logic [TAG_W-1:0] group_start_tag;
    logic [BANKS-1:0] update_valid;
    logic [BANKS-1:0] update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] update_token_ids;
    logic [TAG_W-1:0] update_tag;
    logic update_is_bias;
    logic [(OUT_TILE*PRODUCT_W)-1:0] update_values;
    logic [(OUT_TILE*ACC_W)-1:0] update_bias_values;
    logic [BANKS-1:0] final_valid;
    logic [BANKS-1:0] final_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [TAG_W-1:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic group_finish_valid;
    logic group_finish_ready;
    logic [TAG_W-1:0] group_finish_tag;
    logic protocol_error;
    logic accumulator_overflow;
    logic [31:0] count_updates;
    logic [31:0] count_writes;
    logic [31:0] count_bias_commits;
    logic [31:0] count_bank_stall_cycles;
    logic [31:0] count_final_stall_cycles;

    logic [TOKENS-1:0] final_seen;
    logic signed [ACC_W-1:0] expected_lane0 [0:TOKENS-1];
    logic signed [ACC_W-1:0] expected_lane1 [0:TOKENS-1];
    integer errors;

    initial begin
        forever #1 clk_core = ~clk_core;
    end

    `ACC_DUT #(
        .TOKENS(TOKENS), .BANKS(BANKS), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_ID_W)
    ) dut (.*);

    task automatic check(input logic condition, input string message);
        if (!condition) begin
            $error("%s", message);
            errors = errors + 1;
        end
    endtask

    // Both banks share the same OUT_TILE product vector (matches GPT leaf TB).
    task automatic send_pair(
        input logic [TOKEN_ID_W-1:0] t0,
        input logic [TOKEN_ID_W-1:0] t1,
        input logic is_bias,
        input logic signed [PRODUCT_W-1:0] lane0,
        input logic signed [PRODUCT_W-1:0] lane1
    );
        integer guard;
        begin
            update_token_ids[0 +: TOKEN_ID_W] = t0;
            update_token_ids[TOKEN_ID_W +: TOKEN_ID_W] = t1;
            update_is_bias = is_bias;
            update_values[0 +: PRODUCT_W] = lane0;
            update_values[PRODUCT_W +: PRODUCT_W] = lane1;
            update_bias_values[0 +: ACC_W] = ACC_W'(lane0);
            update_bias_values[ACC_W +: ACC_W] = ACC_W'(lane1);
            guard = 0;
            while ((update_ready & 2'b11) != 2'b11) begin
                @(negedge clk_core);
                guard = guard + 1;
                if (guard > 10000)
                    $fatal(1, "timeout wait ready t0=%0d t1=%0d", t0, t1);
            end
            update_valid = 2'b11;
            @(posedge clk_core);
            #0.1 update_valid = '0;
        end
    endtask

    always @(posedge clk_core) begin
        integer bank;
        integer tid;
        if (!rst_core) begin
            for (bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    tid = final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W];
                    check(final_tag == 16'hB032, "final tag错误");
                    check(!final_seen[tid], "同一token final重复");
                    check($signed(final_values[(bank*OUT_TILE*ACC_W) +: ACC_W])
                          == expected_lane0[tid], "final lane0错误");
                    check($signed(final_values[(bank*OUT_TILE*ACC_W)+ACC_W +: ACC_W])
                          == expected_lane1[tid], "final lane1错误");
                    final_seen[tid] <= 1'b1;
                end
            end
        end
    end

    initial begin
        integer t;
        integer guard;
        errors = 0;
        rst_core = 1'b1;
        group_start_valid = 1'b0;
        group_start_tag = '0;
        update_valid = '0;
        update_token_ids = '0;
        update_tag = 16'hB032;
        update_is_bias = 1'b0;
        update_values = '0;
        update_bias_values = '0;
        final_ready = {BANKS{1'b1}};
        group_finish_valid = 1'b0;
        final_seen = '0;

        // Base: bias only. Tokens 0/1 also get product (10, -4).
        for (t = 0; t < TOKENS; t = t + 1) begin
            expected_lane0[t] = 24'sd100;
            expected_lane1[t] = -24'sd10;
        end
        expected_lane0[0] = 24'sd110;
        expected_lane1[0] = -24'sd14;
        expected_lane0[1] = 24'sd110;
        expected_lane1[1] = -24'sd14;

        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        $display("POW2SAFE leaf T=%0d: group start + product on 0/1", TOKENS);
        group_start_tag = 16'hB032;
        group_start_valid = 1'b1;
        do @(posedge clk_core); while (!group_start_ready);
        #0.1 group_start_valid = 1'b0;

        send_pair(TOKEN_ID_W'(0), TOKEN_ID_W'(1), 1'b0, 17'sd10, -17'sd4);

        $display("POW2SAFE leaf: bias all %0d tokens (paired banks)", TOKENS);
        for (t = 0; t < TOKENS; t = t + 2) begin
            send_pair(TOKEN_ID_W'(t), TOKEN_ID_W'(t + 1), 1'b1, 17'sd100, -17'sd10);
        end

        guard = 0;
        while (final_seen != {TOKENS{1'b1}}) begin
            @(posedge clk_core);
            guard = guard + 1;
            if (guard > 100000)
                $fatal(1, "timeout waiting finals seen=%h", final_seen);
        end

        check(group_finish_ready, "全部bias后应可 finish");
        check(count_bias_commits == 32'(TOKENS), "bias commit 计数");
        check(!protocol_error, "不应 protocol_error");
        check(!accumulator_overflow, "不应 overflow");
        group_finish_valid = 1'b1;
        @(posedge clk_core);
        #0.1 group_finish_valid = 1'b0;
        check(group_start_ready, "结束后可重启");

        if (errors == 0) begin
            $display("PASS: pow2safe banked accumulator TOKENS=%0d", TOKENS);
            $finish;
        end else begin
            $fatal(1, "FAIL: %0d errors", errors);
        end
    end
endmodule

`default_nettype wire
