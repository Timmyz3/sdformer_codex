`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_product_bitmap_join;
    localparam int TOKENS = 8;
    localparam int OUT_TILE = 2;
    localparam int PRODUCT_W = 17;
    logic clk_core;
    logic rst_core;
    logic product_valid;
    logic product_ready;
    logic [15:0] product_tag;
    logic [5:0] product_input_channel;
    logic [3:0] product_output_tile;
    logic [12:0] product_issue_seq;
    logic [33:0] product_values;
    logic bitmap_valid;
    logic bitmap_ready;
    logic [15:0] bitmap_tag;
    logic [12:0] bitmap_issue_seq;
    logic [7:0] bitmap_destinations;
    logic joined_valid;
    logic joined_ready;
    logic [15:0] joined_tag;
    logic [5:0] joined_input_channel;
    logic [3:0] joined_output_tile;
    logic [12:0] joined_issue_seq;
    logic [7:0] joined_destinations;
    logic [33:0] joined_values;
    logic protocol_error;
    logic [31:0] count_joined_terms;
    logic [31:0] count_product_wait_cycles;
    logic [31:0] count_bitmap_wait_cycles;
    logic [31:0] count_output_stall_cycles;

    gatestack_product_bitmap_join #(
        .TOKENS(TOKENS),
        .OUT_TILE(OUT_TILE),
        .PRODUCT_W(PRODUCT_W),
        .INPUT_CH_W(6),
        .OUTPUT_TILE_W(4),
        .TAG_W(16)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic send_product(input int index);
        begin
            @(negedge clk_core);
            product_tag = 16'h5000 + 16'(index);
            product_input_channel = 6'(index + 2);
            product_output_tile = 4'(index + 1);
            product_issue_seq = 13'(index);
            product_values = {17'(index + 20), 17'(index + 10)};
            product_valid = 1'b1;
            do @(posedge clk_core); while (!product_ready);
            @(negedge clk_core);
            product_valid = 1'b0;
        end
    endtask

    task automatic send_bitmap(input int index);
        begin
            @(negedge clk_core);
            bitmap_tag = 16'h5000 + 16'(index);
            bitmap_issue_seq = 13'(index);
            bitmap_destinations = 8'(8'h05 << index);
            bitmap_valid = 1'b1;
            do @(posedge clk_core); while (!bitmap_ready);
            @(negedge clk_core);
            bitmap_valid = 1'b0;
        end
    endtask

    task automatic check_join(input int index);
        begin
            wait (joined_valid);
            if (joined_tag != 16'h5000 + 16'(index) ||
                joined_input_channel != 6'(index + 2) ||
                joined_output_tile != 4'(index + 1) ||
                joined_issue_seq != 13'(index) ||
                joined_destinations != 8'(8'h05 << index) ||
                joined_values != {17'(index + 20), 17'(index + 10)}) begin
                $fatal(1, "joined packet mismatch index=%0d", index);
            end
            @(negedge clk_core);
            joined_ready = 1'b0;
            @(posedge clk_core);
            @(negedge clk_core);
            joined_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            joined_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        product_valid = 1'b0;
        product_tag = '0;
        product_input_channel = '0;
        product_output_tile = '0;
        product_issue_seq = '0;
        product_values = '0;
        bitmap_valid = 1'b0;
        bitmap_tag = '0;
        bitmap_issue_seq = '0;
        bitmap_destinations = '0;
        joined_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        fork
            send_bitmap(0);
            begin
                repeat (3) @(posedge clk_core);
                send_product(0);
            end
            check_join(0);
        join
        fork
            send_product(1);
            begin
                repeat (3) @(posedge clk_core);
                send_bitmap(1);
            end
            check_join(1);
        join

        // A mismatched sequence is dropped and flagged instead of misrouting.
        @(negedge clk_core);
        product_tag = 16'h6000;
        product_issue_seq = 13'd7;
        product_valid = 1'b1;
        bitmap_tag = 16'h6000;
        bitmap_issue_seq = 13'd8;
        bitmap_destinations = 8'h01;
        bitmap_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        product_valid = 1'b0;
        bitmap_valid = 1'b0;
        repeat (2) @(posedge clk_core);

        if (!protocol_error || joined_valid || count_joined_terms != 2 ||
            count_product_wait_cycles == 0 || count_bitmap_wait_cycles == 0 ||
            count_output_stall_cycles == 0) begin
            $fatal(1, "join counters/error mismatch");
        end
        $display("PASS: product-bitmap join joined=%0d product_wait=%0d bitmap_wait=%0d output_stall=%0d",
                 count_joined_terms, count_product_wait_cycles,
                 count_bitmap_wait_cycles, count_output_stall_cycles);
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk_core);
        $fatal(1, "product-bitmap join TB timeout");
    end
endmodule

`default_nettype wire
