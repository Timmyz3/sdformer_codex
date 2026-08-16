`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_event_lifetime_router;

    localparam int DATA_W = 32;
    localparam int TAG_W = 16;
    localparam int COUNTER_W = 32;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [1:0] in_route;
    logic [1:0] in_pair_slot;
    logic [DATA_W-1:0] in_data;
    logic [TAG_W-1:0] in_tag;
    logic single_valid;
    logic single_ready;
    logic [DATA_W-1:0] single_data;
    logic [TAG_W-1:0] single_tag;
    logic fanout_q_valid;
    logic fanout_q_ready;
    logic [DATA_W-1:0] fanout_q_data;
    logic [TAG_W-1:0] fanout_q_tag;
    logic fanout_k_valid;
    logic fanout_k_ready;
    logic [DATA_W-1:0] fanout_k_data;
    logic [TAG_W-1:0] fanout_k_tag;
    logic pair_valid;
    logic pair_ready;
    logic [(4*DATA_W)-1:0] pair_data;
    logic [TAG_W-1:0] pair_tag;
    logic pair_tag_mismatch;
    logic pair_duplicate_slot;
    logic route_unsupported;
    logic [COUNTER_W-1:0] count_accepted;
    logic [COUNTER_W-1:0] count_single_forwarded;
    logic [COUNTER_W-1:0] count_fanout_q;
    logic [COUNTER_W-1:0] count_fanout_k;
    logic [COUNTER_W-1:0] count_pair_issued;

    always #1 clk_core = ~clk_core;

    hitflow_event_lifetime_router #(
        .DATA_W(DATA_W),
        .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W)
    ) dut (
        .*
    );

    task automatic send_event(
        input logic [1:0] route,
        input logic [1:0] slot,
        input logic [TAG_W-1:0] tag,
        input logic [DATA_W-1:0] data
    );
        integer wait_cycles;
        begin
            in_route = route;
            in_pair_slot = slot;
            in_tag = tag;
            in_data = data;
            in_valid = 1'b1;
            wait_cycles = 0;
            do begin
                @(posedge clk_core);
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 100) begin
                    $fatal(1, "send_event等待in_ready超时 route=%0d slot=%0d tag=%0h", route, slot, tag);
                end
            end while (!in_ready);
            #0.1;
            in_valid = 1'b0;
        end
    endtask

    task automatic check(input logic condition, input string message);
        begin
            if (!condition) begin
                $fatal(1, "%s", message);
            end
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_route = 2'd0;
        in_pair_slot = 2'd0;
        in_data = '0;
        in_tag = '0;
        single_ready = 1'b0;
        fanout_q_ready = 1'b0;
        fanout_k_ready = 1'b0;
        pair_ready = 1'b0;

        repeat (3) @(posedge clk_core);
        #0.1;
        rst_core = 1'b0;

        $display("阶段1：single反压");
        send_event(2'd0, 2'd0, 16'h0011, 32'h1122_3344);
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(single_valid, "single输出在反压下必须保持valid");
            check(single_data == 32'h1122_3344, "single输出数据错误");
            check(single_tag == 16'h0011, "single输出tag错误");
        end
        single_ready = 1'b1;
        @(posedge clk_core);
        #0.1;
        single_ready = 1'b0;

        $display("阶段2：fanout独立反压");
        send_event(2'd1, 2'd0, 16'h0022, 32'ha5a5_5a5a);
        fanout_q_ready = 1'b1;
        @(posedge clk_core);
        #0.1;
        check(!fanout_q_valid, "Q消费者握手后不应重复发送");
        check(fanout_k_valid, "K消费者尚未握手时必须保留");
        check(fanout_k_data == 32'ha5a5_5a5a, "fanout K数据错误");
        fanout_q_ready = 1'b0;
        fanout_k_ready = 1'b1;
        in_route = 2'd1;
        in_tag = 16'h0023;
        in_data = 32'h5a5a_a5a5;
        in_valid = 1'b1;
        #0.1;
        check(in_ready, "fanout最后一个消费者完成时必须可同拍接收下一事件");
        @(posedge clk_core);
        #0.1;
        in_valid = 1'b0;
        fanout_k_ready = 1'b0;
        check(fanout_q_valid && fanout_k_valid, "fanout替换事件必须在下一拍同时送往Q/K");
        check(fanout_q_data == 32'h5a5a_a5a5, "fanout同拍替换数据错误");
        fanout_q_ready = 1'b1;
        fanout_k_ready = 1'b1;
        @(posedge clk_core);
        #0.1;
        fanout_q_ready = 1'b0;
        fanout_k_ready = 1'b0;

        $display("阶段3：pair tag错配和乱序组装");
        send_event(2'd2, 2'd2, 16'h0033, 32'h0000_00c0);
        in_route = 2'd2;
        in_pair_slot = 2'd0;
        in_tag = 16'h0044;
        in_data = 32'hffff_ffff;
        in_valid = 1'b1;
        #0.1;
        check(pair_tag_mismatch, "部分pair遇到不同tag必须报告错配");
        check(!in_ready, "tag错配输入不得被接收");
        in_valid = 1'b0;

        in_route = 2'd2;
        in_pair_slot = 2'd2;
        in_tag = 16'h0033;
        in_data = 32'hffff_ffff;
        in_valid = 1'b1;
        #0.1;
        check(pair_duplicate_slot, "重复pair slot必须报告错误");
        check(!in_ready, "重复pair slot不得被接收");
        in_valid = 1'b0;

        in_route = 2'd3;
        in_valid = 1'b1;
        #0.1;
        check(route_unsupported, "未支持route必须报告错误");
        check(!in_ready, "未支持route不得被接收");
        in_valid = 1'b0;

        send_event(2'd2, 2'd0, 16'h0033, 32'h0000_00a0);
        send_event(2'd2, 2'd3, 16'h0033, 32'h0000_00d0);
        send_event(2'd2, 2'd1, 16'h0033, 32'h0000_00b0);
        #0.1;
        check(pair_valid, "四个slot到齐后pair必须valid");
        check(pair_tag == 16'h0033, "pair tag错误");
        check(pair_data == {32'h0000_00d0, 32'h0000_00c0, 32'h0000_00b0, 32'h0000_00a0}, "pair拼接顺序错误");
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(pair_valid, "pair在反压下必须保持valid");
        end
        pair_ready = 1'b1;
        in_route = 2'd2;
        in_pair_slot = 2'd1;
        in_tag = 16'h0055;
        in_data = 32'h0000_01b0;
        in_valid = 1'b1;
        #0.1;
        check(in_ready, "pair退休时必须可同拍建立下一tag");
        check(!pair_tag_mismatch, "pair同拍替换不得误报tag错配");
        @(posedge clk_core);
        #0.1;
        in_valid = 1'b0;
        pair_ready = 1'b0;
        check(!pair_valid, "新pair只有一个slot时不得提前valid");
        send_event(2'd2, 2'd3, 16'h0055, 32'h0000_01d0);
        send_event(2'd2, 2'd0, 16'h0055, 32'h0000_01a0);
        send_event(2'd2, 2'd2, 16'h0055, 32'h0000_01c0);
        #0.1;
        check(pair_valid, "同拍替换后的第二个pair必须组装完成");
        check(pair_data == {32'h0000_01d0, 32'h0000_01c0, 32'h0000_01b0, 32'h0000_01a0}, "第二个pair拼接错误");
        pair_ready = 1'b1;
        @(posedge clk_core);
        #0.1;
        pair_ready = 1'b0;

        $display("阶段4：计数器守恒");
        repeat (2) @(posedge clk_core);
        #0.1;
        check(count_accepted == 32'd11, "accepted计数错误");
        check(count_single_forwarded == 32'd1, "single计数错误");
        check(count_fanout_q == 32'd2, "fanout Q计数错误");
        check(count_fanout_k == 32'd2, "fanout K计数错误");
        check(count_pair_issued == 32'd2, "pair计数错误");

        $display("PASS: HIT-Flow event lifetime router");
        $finish;
    end

endmodule

`default_nettype wire
