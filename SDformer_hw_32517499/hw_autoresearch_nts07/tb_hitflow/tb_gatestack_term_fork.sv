`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_term_fork;
    logic clk_core, rst_core, term_valid, term_ready;
    logic [31:0] term_tag;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [9:0] term_input_channel;
    logic [7:0] term_output_tile;
    logic [7:0] term_destination_count;
    logic [12:0] term_issue_seq;
    logic term_head_last;
    logic product_term_valid, product_term_ready;
    logic [31:0] product_term_tag;
    logic [8:0] product_term_gate_code;
    logic [9:0] product_term_input_channel;
    logic [7:0] product_term_output_tile;
    logic [12:0] product_term_issue_seq;
    logic bitmap_term_valid, bitmap_term_ready;
    logic [31:0] bitmap_term_tag;
    logic [8:0] bitmap_term_gate_code;
    logic [4:0] bitmap_term_lane_id;
    logic [7:0] bitmap_term_destination_count;
    logic [12:0] bitmap_term_issue_seq;
    logic bitmap_term_head_last;
    logic [31:0] count_terms, count_product_wait_cycles;
    logic [31:0] count_bitmap_wait_cycles;

    gatestack_term_fork dut (.*);
    always #5 clk_core <= ~clk_core;

    initial begin
        clk_core=0; rst_core=1; term_valid=0; term_tag=0;
        term_gate_code=0; term_lane_id=0; term_input_channel=0;
        term_output_tile=0; term_destination_count=0; term_issue_seq=0;
        term_head_last=0; product_term_ready=0; bitmap_term_ready=0;
        repeat (5) @(posedge clk_core); rst_core=0;
        @(negedge clk_core);
        term_tag=32'hf001; term_gate_code=9'd31; term_lane_id=5'd7;
        term_input_channel=10'd71; term_output_tile=8'd4;
        term_destination_count=8'd9; term_issue_seq=13'd6;
        term_head_last=1; term_valid=1;
        @(posedge clk_core); @(negedge clk_core); term_valid=0;
        if (!product_term_valid || !bitmap_term_valid || term_ready)
            $fatal(1,"fork did not capture both branches");
        @(posedge clk_core); @(negedge clk_core);
        bitmap_term_ready=1;
        @(posedge clk_core); @(negedge clk_core); bitmap_term_ready=0;
        repeat (2) @(posedge clk_core);
        if (!product_term_valid || bitmap_term_valid ||
            product_term_tag!=32'hf001 || product_term_gate_code!=31 ||
            product_term_input_channel!=71 || product_term_output_tile!=4 ||
            product_term_issue_seq!=6 || bitmap_term_tag!=32'hf001 ||
            bitmap_term_gate_code!=31 || bitmap_term_lane_id!=7 ||
            bitmap_term_destination_count!=9 || bitmap_term_issue_seq!=6 ||
            !bitmap_term_head_last) $fatal(1,"fork payload/retirement mismatch");
        @(negedge clk_core); product_term_ready=1;
        @(posedge clk_core); @(negedge clk_core); product_term_ready=0;
        repeat (2) @(posedge clk_core);
        if (!term_ready || count_terms!=1 || count_product_wait_cycles==0 ||
            count_bitmap_wait_cycles==0) $fatal(1,"fork counters mismatch");
        $display("PASS: term fork terms=%0d product_wait=%0d bitmap_wait=%0d",
                 count_terms,count_product_wait_cycles,count_bitmap_wait_cycles);
        $finish;
    end
    initial begin repeat (1000) @(posedge clk_core); $fatal(1,"fork timeout"); end
endmodule

`default_nettype wire
