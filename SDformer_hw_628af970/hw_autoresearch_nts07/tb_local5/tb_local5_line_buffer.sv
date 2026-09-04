`timescale 1ns/1ps
`default_nettype none

module tb_local5_line_buffer;
    localparam int HEAD_DIM = 32;
    localparam int ROW_TOKENS = 4;

    logic clk_core, rst_core;
    logic row_push_valid, row_push_ready;
    logic [15:0] row_push_tag;
    logic [HEAD_DIM-1:0] row_push_q [0:ROW_TOKENS-1];
    logic [HEAD_DIM-1:0] row_push_k [0:ROW_TOKENS-1];
    logic [ROW_TOKENS-1:0] row_push_valid_mask;
    logic rd_valid, rd_ready;
    logic [1:0] rd_row_sel;
    logic [$clog2(ROW_TOKENS)-1:0] rd_x;
    logic rd_rsp_valid, rd_rsp_ready;
    logic [HEAD_DIM-1:0] rd_q_bits, rd_k_bits;
    logic rd_token_valid;
    logic [15:0] curr_row_tag;
    logic [1:0] rows_filled;
    logic protocol_error;

    local5_line_buffer_3row #(
        .HEAD_DIM(HEAD_DIM), .ROW_TOKENS(ROW_TOKENS)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    int errors;

    initial begin
        clk_core = 0;
        rst_core = 1;
        row_push_valid = 0;
        rd_valid = 0;
        rd_rsp_ready = 1;
        errors = 0;
        for (int i = 0; i < ROW_TOKENS; i++) begin
            row_push_q[i] = '0;
            row_push_k[i] = '0;
        end
        row_push_valid_mask = '0;
        repeat (3) @(posedge clk_core);
        rst_core = 0;

        // push 3 rows
        for (int r = 0; r < 3; r++) begin
            @(posedge clk_core);
            row_push_valid = 1;
            row_push_tag = 16'(r + 10);
            row_push_valid_mask = 4'b1111;
            for (int i = 0; i < ROW_TOKENS; i++) begin
                row_push_q[i] = 32'(r * 16 + i + 1);
                row_push_k[i] = 32'(r * 16 + i + 100);
            end
            while (!(row_push_valid && row_push_ready)) @(posedge clk_core);
            @(posedge clk_core);
            row_push_valid = 0;
        end

        if (rows_filled !== 2'd3) begin
            $error("filled=%0d", rows_filled);
            errors++;
        end

        // read prev row x=1 → first pushed row after 3 rotates:
        // after push0: next=r0
        // after push1: curr=r0 next=r1
        // after push2: prev=r0 curr=r1 next=r2
        @(posedge clk_core);
        rd_valid = 1;
        rd_row_sel = 2'd0; // prev
        rd_x = 1;
        while (!(rd_valid && rd_ready)) @(posedge clk_core);
        @(posedge clk_core);
        rd_valid = 0;
        while (!rd_rsp_valid) @(posedge clk_core);
        if (rd_k_bits !== 32'(100 + 1)) begin
            $error("prev k got %0d", rd_k_bits);
            errors++;
        end
        if (!rd_token_valid) begin
            $error("token invalid");
            errors++;
        end
        @(posedge clk_core);

        // curr x=2 → r1
        rd_valid = 1;
        rd_row_sel = 2'd1;
        rd_x = 2;
        while (!(rd_valid && rd_ready)) @(posedge clk_core);
        @(posedge clk_core);
        rd_valid = 0;
        while (!rd_rsp_valid) @(posedge clk_core);
        if (rd_q_bits !== 32'(16 + 2 + 1)) begin
            $error("curr q got %0d exp %0d", rd_q_bits, 16+2+1);
            errors++;
        end
        @(posedge clk_core);

        if (protocol_error) errors++;
        if (errors) $fatal(1, "FAIL %0d", errors);
        $display("PASS tb_local5_line_buffer");
        $finish;
    end

    initial begin
        #50000; $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
