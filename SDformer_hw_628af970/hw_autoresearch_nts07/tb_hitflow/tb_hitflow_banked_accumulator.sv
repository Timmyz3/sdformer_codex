`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_banked_accumulator;
    localparam int TOKENS = 6;
    localparam int BANKS = 2;
    localparam int PRODUCT_W = 17;
    localparam int ACC_W = 24;
    localparam int OUT_TILE = 2;
    localparam int TAG_W = 16;
    localparam int TOKEN_ID_W = $clog2(TOKENS);

    logic clk_core = 1'b0;
    logic rst_core;
    logic flush;
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
    logic [31:0] updates_before_flush;
    logic [31:0] writes_before_flush;
    logic [31:0] bias_before_flush;
    logic [31:0] bank_stalls_before_flush;
    logic [31:0] final_stalls_before_flush;

    initial begin
        forever #1 clk_core = ~clk_core;
    end

    hitflow_banked_accumulator #(
        .TOKENS(TOKENS), .BANKS(BANKS), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_ID_W)
    ) dut (.flush(flush), .*);

    task automatic check(input logic condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    task automatic send_update(
        input logic [BANKS-1:0] bank_mask,
        input logic [TOKEN_ID_W-1:0] bank0_token,
        input logic [TOKEN_ID_W-1:0] bank1_token,
        input logic is_bias,
        input logic signed [PRODUCT_W-1:0] lane0,
        input logic signed [PRODUCT_W-1:0] lane1
    );
        begin
            update_token_ids[0 +: TOKEN_ID_W] = bank0_token;
            update_token_ids[TOKEN_ID_W +: TOKEN_ID_W] = bank1_token;
            update_is_bias = is_bias;
            update_values[0 +: PRODUCT_W] = lane0;
            update_values[PRODUCT_W +: PRODUCT_W] = lane1;
            update_bias_values[0 +: ACC_W] = ACC_W'(lane0);
            update_bias_values[ACC_W +: ACC_W] = ACC_W'(lane1);
            while ((update_ready & bank_mask) != bank_mask) @(negedge clk_core);
            update_valid = bank_mask;
            @(posedge clk_core);
            #0.1 update_valid = '0;
        end
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    check(final_tag == 16'h2468, "final tag错误");
                    check(!final_seen[
                        final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]
                    ], "同一token final重复");
                    check($signed(final_values[
                              (bank*OUT_TILE*ACC_W) +: ACC_W
                          ]) == expected_lane0[
                              final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]
                          ], "final lane0错误");
                    check($signed(final_values[
                              (bank*OUT_TILE*ACC_W)+ACC_W +: ACC_W
                          ]) == expected_lane1[
                              final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]
                          ], "final lane1错误");
                    final_seen[
                        final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]
                    ] <= 1'b1;
                end
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        flush = 1'b0;
        group_start_valid = 1'b0;
        group_start_tag = '0;
        update_valid = '0;
        update_token_ids = '0;
        update_tag = 16'h2468;
        update_is_bias = 1'b0;
        update_values = '0;
        update_bias_values = '0;
        final_ready = '0;
        group_finish_valid = 1'b0;
        final_seen = '0;
        for (int token = 0; token < TOKENS; token = token + 1) begin
            expected_lane0[token] = 24'sd100;
            expected_lane1[token] = -24'sd10;
        end
        expected_lane0[0] = 24'sd113;
        expected_lane1[0] = -24'sd9;
        expected_lane0[1] = 24'sd110;
        expected_lane1[1] = -24'sd14;
        expected_lane0[4] = 24'sd98;
        expected_lane1[4] = -24'sd9;

        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        $display("阶段0A：普通update读出/写回中flush");
        group_start_tag = 16'h2468;
        group_start_valid = 1'b1;
        do @(posedge clk_core); while (!group_start_ready);
        #0.1 group_start_valid = 1'b0;

        send_update(2'b01, 3'd0, 3'd1, 1'b0, 17'sd13, 17'sd1);
        check(!update_ready[0], "普通update接受后bank应处于写回阶段");
        updates_before_flush = count_updates;
        writes_before_flush = count_writes;
        bias_before_flush = count_bias_commits;
        bank_stalls_before_flush = count_bank_stall_cycles;
        final_stalls_before_flush = count_final_stall_cycles;
        check(updates_before_flush == 1 && writes_before_flush == 0,
              "flush前普通update计数错误");
        flush = 1'b1;
        group_start_valid = 1'b1;
        update_valid = '1;
        group_finish_valid = 1'b1;
        #0.1;
        check(!group_start_ready && update_ready == '0 &&
              final_valid == '0 && !group_finish_ready,
              "flush拍必须屏蔽所有ready/final_valid");
        check(!protocol_error, "flush拍输入事务不得触发protocol_error");
        @(posedge clk_core);
        #0.1;
        check(count_updates == updates_before_flush &&
              count_writes == writes_before_flush &&
              count_bias_commits == bias_before_flush &&
              count_bank_stall_cycles == bank_stalls_before_flush &&
              count_final_stall_cycles == final_stalls_before_flush,
              "普通写回flush不得清除或增加计数器");
        flush = 1'b0;
        group_start_valid = 1'b0;
        update_valid = '0;
        group_finish_valid = 1'b0;
        #0.1;
        check(group_start_ready && final_valid == '0,
              "flush后必须idle且不得出现旧final");

        $display("阶段0B：bias final反压中flush");
        group_start_valid = 1'b1;
        do @(posedge clk_core); while (!group_start_ready);
        #0.1 group_start_valid = 1'b0;
        send_update(2'b01, 3'd0, 3'd1, 1'b0, 17'sd50, 17'sd0);
        send_update(2'b01, 3'd0, 3'd1, 1'b1, 17'sd100, 17'sd0);
        @(posedge clk_core);
        #0.1;
        check(final_valid[0] && !final_ready[0],
              "flush定向前bias final必须处于反压保持态");
        updates_before_flush = count_updates;
        writes_before_flush = count_writes;
        bias_before_flush = count_bias_commits;
        bank_stalls_before_flush = count_bank_stall_cycles;
        final_stalls_before_flush = count_final_stall_cycles;
        flush = 1'b1;
        group_start_valid = 1'b1;
        update_valid = '1;
        group_finish_valid = 1'b1;
        #0.1;
        check(!group_start_ready && update_ready == '0 &&
              final_valid == '0 && !group_finish_ready,
              "反压final的flush拍屏蔽失败");
        check(!protocol_error, "反压final flush拍不应产生protocol_error");
        @(posedge clk_core);
        #0.1;
        check(count_updates == updates_before_flush &&
              count_writes == writes_before_flush &&
              count_bias_commits == bias_before_flush &&
              count_bank_stall_cycles == bank_stalls_before_flush &&
              count_final_stall_cycles == final_stalls_before_flush,
              "final反压flush不得清除或增加计数器");
        flush = 1'b0;
        group_start_valid = 1'b0;
        update_valid = '0;
        group_finish_valid = 1'b0;
        #0.1;
        check(group_start_ready && final_valid == '0,
              "flush后旧bias final不得重现");
        @(posedge clk_core);
        #0.1;
        check(final_valid == '0, "flush后空闲周期不得泄漏旧final");

        $display("阶段1：相同tag重启并执行普通product累加");
        group_start_tag = 16'h2468;
        group_start_valid = 1'b1;
        do @(posedge clk_core); while (!group_start_ready);
        #0.1 group_start_valid = 1'b0;

        send_update(2'b11, 3'd0, 3'd1, 1'b0, 17'sd10, -17'sd4);
        update_valid = 2'b01;
        #0.1;
        check(!update_ready[0], "bank忙时必须反压下一更新");
        @(posedge clk_core);
        #0.1 update_valid = '0;
        send_update(2'b01, 3'd0, 3'd1, 1'b0, 17'sd3, 17'sd5);
        send_update(2'b01, 3'd4, 3'd1, 1'b0, -17'sd2, 17'sd1);

        $display("阶段2：bias提交即final输出，支持逐bank反压");
        final_ready = 2'b10;
        send_update(2'b11, 3'd0, 3'd1, 1'b1, 17'sd100, -17'sd10);
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(final_valid[0] &&
                  final_token_ids[0 +: TOKEN_ID_W] == 0 &&
                  $signed(final_values[0 +: ACC_W]) == 24'sd113,
                  "bank0 final反压期间不稳定");
        end
        final_ready = 2'b11;
        @(posedge clk_core);
        send_update(2'b11, 3'd2, 3'd3, 1'b1, 17'sd100, -17'sd10);
        send_update(2'b11, 3'd4, 3'd5, 1'b1, 17'sd100, -17'sd10);
        while (final_seen != {TOKENS{1'b1}}) @(posedge clk_core);
        #0.1;

        $display("阶段3：重复bias必须拒绝，完整bias后才允许结束");
        update_token_ids[0 +: TOKEN_ID_W] = 0;
        update_is_bias = 1'b1;
        update_valid = 2'b01;
        #0.1;
        check(protocol_error && !update_ready[0], "重复bias必须拒绝");
        update_valid = '0;

        check(group_finish_ready, "全部bias提交后group必须可结束");
        check(group_finish_tag == 16'h2468, "finish tag错误");
        check(count_updates == 13 && count_writes == 11 &&
              count_bias_commits == 6, "accumulator计数器错误");
        check(count_final_stall_cycles >= 2, "final反压周期未统计");
        check(count_bank_stall_cycles >= 1, "bank反压周期未统计");
        check(!accumulator_overflow, "正常向量不应溢出");
        group_finish_valid = 1'b1;
        @(posedge clk_core);
        #0.1 group_finish_valid = 1'b0;
        check(group_start_ready, "group结束后必须可重新启动");

        $display("PASS: HIT-Flow banked accumulator");
        $finish;
    end

endmodule

`default_nettype wire
