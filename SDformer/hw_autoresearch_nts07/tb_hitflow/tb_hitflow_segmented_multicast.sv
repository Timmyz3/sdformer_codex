`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_segmented_multicast;
    localparam int TOKENS = 10;
    localparam int SEGMENT_TOKENS = 4;
    localparam int BANKS = 2;
    localparam int PRODUCT_W = 17;
    localparam int OUT_TILE = 2;
    localparam int TAG_W = 16;
    localparam int TOKEN_ID_W = $clog2(TOKENS);

    logic clk_core = 1'b0;
    logic rst_core;
    logic product_valid;
    logic product_ready;
    logic [TAG_W-1:0] product_tag;
    logic [TOKENS-1:0] product_destination_bitmap;
    logic [(OUT_TILE*PRODUCT_W)-1:0] product_values;
    logic [BANKS-1:0] update_valid;
    logic [BANKS-1:0] update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] update_token_ids;
    logic [TAG_W-1:0] update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] update_values;
    logic product_done_valid;
    logic product_done_ready;
    logic [TAG_W-1:0] product_done_tag;
    logic protocol_error;
    logic [31:0] count_products;
    logic [31:0] count_destinations;
    logic [31:0] count_issue_cycles;
    logic [31:0] count_segment_advances;
    logic [31:0] count_bank_stall_cycles;

    logic [TOKENS-1:0] seen;
    logic [TOKENS-1:0] monitor_fire_mask;
    logic [31:0] monitor_fire_count;
    logic [31:0] accepted;

    initial begin
        forever #1 clk_core = ~clk_core;
    end

    hitflow_segmented_multicast #(
        .TOKENS(TOKENS), .SEGMENT_TOKENS(SEGMENT_TOKENS),
        .BANKS(BANKS), .PRODUCT_W(PRODUCT_W), .OUT_TILE(OUT_TILE),
        .TAG_W(TAG_W), .TOKEN_ID_W(TOKEN_ID_W)
    ) dut (.*);

    task automatic check(input logic condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    always_comb begin
        monitor_fire_mask = '0;
        monitor_fire_count = '0;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            if (update_valid[bank] && update_ready[bank]) begin
                monitor_fire_mask[
                    update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]
                ] = 1'b1;
                monitor_fire_count = monitor_fire_count + 1'b1;
            end
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (update_valid[bank] && update_ready[bank]) begin
                    check(update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] <
                          TOKEN_ID_W'(TOKENS),
                          "update token越界");
                    check(!seen[update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]],
                          "同一目的token重复提交");
                end
            end
            seen <= seen | monitor_fire_mask;
            accepted <= accepted + monitor_fire_count;
        end
    end

    initial begin
        rst_core = 1'b1;
        product_valid = 1'b0;
        product_tag = '0;
        product_destination_bitmap = '0;
        product_values = '0;
        update_ready = '0;
        product_done_ready = 1'b0;
        seen = '0;
        accepted = 0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        $display("阶段1：跨segment、同bank串行和独立反压");
        product_tag = 16'h1357;
        product_destination_bitmap = 10'b1010011111;
        product_values[0 +: PRODUCT_W] = -17'sd321;
        product_values[PRODUCT_W +: PRODUCT_W] = 17'sd456;
        product_valid = 1'b1;
        do @(posedge clk_core); while (!product_ready);
        #0.1 product_valid = 1'b0;

        update_ready = 2'b01;
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(update_tag == 16'h1357 &&
                  $signed(update_values[0 +: PRODUCT_W]) == -17'sd321 &&
                  $signed(update_values[PRODUCT_W +: PRODUCT_W]) == 17'sd456,
                  "反压时共享product数据错误");
        end
        update_ready = 2'b11;
        do @(posedge clk_core); while (!product_done_valid);
        #0.1;
        check(product_done_tag == 16'h1357, "done tag错误");
        check(seen == 10'b1010011111, "目的bitmap提交不完整");
        check(accepted == 7, "目的提交数量错误");
        check(count_products == 1 && count_destinations == 7,
              "产品或目的计数器错误");
        check(count_segment_advances == 2, "segment推进次数错误");
        check(count_bank_stall_cycles >= 1, "bank反压周期未统计");

        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(product_done_valid && product_done_tag == 16'h1357,
                  "done反压期间不稳定");
        end
        product_done_ready = 1'b1;
        @(posedge clk_core);
        #0.1 product_done_ready = 1'b0;

        $display("阶段2：空bitmap必须拒绝");
        product_destination_bitmap = '0;
        product_valid = 1'b1;
        #0.1;
        check(protocol_error && !product_ready, "空bitmap product必须拒绝");
        product_valid = 1'b0;

        check(count_issue_cycles >= 4, "issue周期计数异常");
        $display("PASS: HIT-Flow segmented multicast");
        $finish;
    end

endmodule

`default_nettype wire
