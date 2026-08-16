`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_nmf_g1_builder;
    localparam int TOKENS = 4;
    localparam int LANES = 4;
    localparam int GATE_W = 9;
    localparam int SLOTS = 2;
    localparam int TAG_W = 16;

    logic clk_core = 1'b0;
    logic rst_core;
    logic group_valid;
    logic group_ready;
    logic [TAG_W-1:0] group_tag;
    logic token_valid;
    logic token_ready;
    logic [$clog2(TOKENS)-1:0] token_id;
    logic [GATE_W-1:0] token_gate_code;
    logic [LANES-1:0] token_k_bits;
    logic token_last;
    logic term_valid;
    logic term_ready;
    logic [TAG_W-1:0] term_tag;
    logic [GATE_W-1:0] term_gate_code;
    logic [$clog2(LANES)-1:0] term_lane;
    logic [TOKENS-1:0] term_destination_bitmap;
    logic fallback_valid;
    logic fallback_ready;
    logic [TAG_W-1:0] fallback_tag;
    logic [$clog2(TOKENS)-1:0] fallback_token_id;
    logic [GATE_W-1:0] fallback_gate_code;
    logic [LANES-1:0] fallback_k_bits;
    logic group_done_valid;
    logic group_done_ready;
    logic [TAG_W-1:0] group_done_tag;
    logic overflow_seen;
    logic protocol_error;
    logic [31:0] count_tokens;
    logic [31:0] count_active_lanes;
    logic [31:0] count_terms;
    logic [31:0] count_fallback_tokens;

    integer seen_terms;
    integer seen_fallback;
    logic held_term;
    logic [GATE_W-1:0] held_gate;
    logic [$clog2(LANES)-1:0] held_lane;
    logic [TOKENS-1:0] held_bitmap;

    always #1 clk_core = ~clk_core;

    hitflow_nmf_g1_builder #(
        .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
        .SLOTS(SLOTS), .TAG_W(TAG_W)
    ) dut (.*);

    task automatic check(input logic condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    task automatic start_group(input logic [TAG_W-1:0] tag);
        begin
            group_tag = tag;
            group_valid = 1'b1;
            do @(posedge clk_core); while (!group_ready);
            #0.1 group_valid = 1'b0;
        end
    endtask

    task automatic send_token(
        input logic [$clog2(TOKENS)-1:0] id,
        input logic [GATE_W-1:0] gate,
        input logic [LANES-1:0] k,
        input logic last
    );
        begin
            token_id = id;
            token_gate_code = gate;
            token_k_bits = k;
            token_last = last;
            token_valid = 1'b1;
            do @(posedge clk_core); while (!token_ready);
            #0.1 token_valid = 1'b0;
        end
    endtask

    task automatic check_term;
        begin
            if (term_gate_code == 9'd64 && term_lane == 0) begin
                check(term_destination_bitmap == 4'b0011, "gate64 lane0 bitmap错误");
            end else if (term_gate_code == 9'd64 && term_lane == 1) begin
                check(term_destination_bitmap == 4'b0001, "gate64 lane1 bitmap错误");
            end else if (term_gate_code == 9'd128 && term_lane == 2) begin
                check(term_destination_bitmap == 4'b0100, "gate128 lane2 bitmap错误");
            end else begin
                $fatal(1, "出现未知目录term gate=%0d lane=%0d bitmap=%b",
                       term_gate_code, term_lane, term_destination_bitmap);
            end
            check(term_tag == 16'h1234, "目录term tag错误");
            seen_terms = seen_terms + 1;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = '0;
        token_valid = 1'b0;
        token_id = '0;
        token_gate_code = '0;
        token_k_bits = '0;
        token_last = 1'b0;
        term_ready = 1'b0;
        fallback_ready = 1'b0;
        group_done_ready = 1'b0;
        seen_terms = 0;
        seen_fallback = 0;
        held_term = 1'b0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        $display("阶段1：同gate目录合并、第二gate分配和第三gate无损fallback");
        start_group(16'h1234);
        send_token(0, 9'd64, 4'b0011, 1'b0);
        send_token(1, 9'd64, 4'b0001, 1'b0);
        send_token(2, 9'd128, 4'b0100, 1'b0);
        send_token(3, 9'd192, 4'b1000, 1'b1);

        while (!group_done_valid) begin
            @(posedge clk_core);
            #0.1;
            if (term_valid && !held_term) begin
                held_term = 1'b1;
                held_gate = term_gate_code;
                held_lane = term_lane;
                held_bitmap = term_destination_bitmap;
                term_ready = 1'b0;
            end else if (term_valid && held_term) begin
                check(term_gate_code == held_gate && term_lane == held_lane &&
                      term_destination_bitmap == held_bitmap,
                      "term反压期间输出不稳定");
                check_term();
                term_ready = 1'b1;
                held_term = 1'b0;
            end else begin
                term_ready = 1'b1;
            end
            if (fallback_valid) begin
                check(fallback_tag == 16'h1234, "fallback tag错误");
                check(fallback_token_id == 3, "fallback token错误");
                check(fallback_gate_code == 9'd192, "fallback gate错误");
                check(fallback_k_bits == 4'b1000, "fallback K错误");
                fallback_ready = 1'b1;
                seen_fallback = seen_fallback + 1;
            end
        end
        term_ready = 1'b0;
        fallback_ready = 1'b0;
        check(seen_terms == 3, "目录term数量错误");
        check(seen_fallback == 1, "fallback数量错误");
        check(overflow_seen, "overflow标志未置位");
        check(count_tokens == 4, "token计数错误");
        check(count_active_lanes == 5, "活动lane计数错误");
        check(count_terms == 3, "term计数错误");
        check(count_fallback_tokens == 1, "fallback计数错误");
        check(group_done_tag == 16'h1234, "完成tag错误");
        group_done_ready = 1'b1;
        @(posedge clk_core);
        #0.1 group_done_ready = 1'b0;

        $display("阶段2：gate=0与K-zero均不生成投影事务");
        start_group(16'h5678);
        send_token(0, 9'd0, 4'b1111, 1'b0);
        send_token(1, 9'd32, 4'b0000, 1'b0);
        send_token(2, 9'd0, 4'b0000, 1'b0);
        send_token(3, 9'd16, 4'b0000, 1'b1);
        term_ready = 1'b1;
        fallback_ready = 1'b1;
        while (!group_done_valid) begin
            @(posedge clk_core);
            #0.1;
            check(!term_valid, "零乘积group不应输出目录term");
            check(!fallback_valid, "零乘积group不应输出fallback");
        end
        check(!overflow_seen, "空group不应overflow");
        check(count_terms == 0 && count_fallback_tokens == 0, "空group事务计数错误");
        group_done_ready = 1'b1;
        @(posedge clk_core);
        #0.1 group_done_ready = 1'b0;

        $display("阶段3：连续overflow通过单项弹性fallback反压且不丢失");
        start_group(16'haaaa);
        fallback_ready = 1'b1;
        term_ready = 1'b1;
        send_token(0, 9'd1, 4'b0001, 1'b0);
        send_token(1, 9'd2, 4'b0010, 1'b0);
        send_token(2, 9'd3, 4'b0100, 1'b0);
        send_token(3, 9'd4, 4'b1000, 1'b1);
        while (!group_done_valid) begin
            @(posedge clk_core);
            #0.1;
        end
        check(overflow_seen, "连续overflow必须置位标志");
        check(count_tokens == 4, "连续overflow token计数错误");
        check(count_terms == 2, "连续overflow目录term计数错误");
        check(count_fallback_tokens == 2, "连续overflow fallback计数错误");
        group_done_ready = 1'b1;
        @(posedge clk_core);
        #0.1 group_done_ready = 1'b0;
        fallback_ready = 1'b0;
        term_ready = 1'b0;

        $display("阶段4：乱序token必须拒绝");
        start_group(16'h9abc);
        token_id = 1;
        token_gate_code = 9'd1;
        token_k_bits = 4'b0001;
        token_last = 1'b0;
        token_valid = 1'b1;
        #0.1;
        check(protocol_error, "乱序token必须报协议错误");
        check(!token_ready, "乱序token不得被接收");
        token_valid = 1'b0;

        $display("PASS: HIT-Flow NMF G1 builder");
        $finish;
    end

endmodule

`default_nettype wire
