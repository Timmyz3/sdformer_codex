`timescale 1ns/1ps
`default_nettype none

module tb_qfit_sync_1r1w_bank;
    logic clk_core;
    logic rst_core;
    logic wr_en;
    logic [2:0] wr_addr;
    logic [15:0] wr_data;
    logic rd_en;
    logic [2:0] rd_addr;
    logic rd_valid;
    logic [15:0] rd_data;

    qfit_sync_1r1w_bank #(
        .DATA_W(16),
        .DEPTH(8)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        wr_en = 1'b0;
        wr_addr = '0;
        wr_data = '0;
        rd_en = 1'b0;
        rd_addr = '0;
        repeat (3) @(negedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        wr_en = 1'b1;
        wr_addr = 3'd3;
        wr_data = 16'h1234;
        @(negedge clk_core);
        wr_en = 1'b0;

        rd_en = 1'b1;
        rd_addr = 3'd3;
        wr_en = 1'b1;
        wr_addr = 3'd3;
        wr_data = 16'habcd;
        @(negedge clk_core);
        if (!rd_valid || rd_data != 16'h1234)
            $fatal(
                1,
                "read-first collision failed valid=%0b data=%h",
                rd_valid,
                rd_data
            );

        wr_en = 1'b0;
        rd_en = 1'b1;
        rd_addr = 3'd3;
        @(negedge clk_core);
        if (!rd_valid || rd_data != 16'habcd)
            $fatal(1, "post-write read failed data=%h", rd_data);

        rd_en = 1'b0;
        @(negedge clk_core);
        if (rd_valid)
            $fatal(1, "rd_valid latency contract failed");
        $display("PASS qfit_sync_1r1w_bank read-first");
        $finish;
    end
endmodule

`default_nettype wire
