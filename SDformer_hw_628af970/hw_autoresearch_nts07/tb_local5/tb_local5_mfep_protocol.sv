`timescale 1ns/1ps
`default_nettype none

module tb_local5_mfep_protocol;
    logic clk_core;
    logic rst_core;
    logic dest_valid;
    logic dest_ready;
    logic [15:0] dest_tag;
    logic [7:0] dest_id;
    logic edge_valid;
    logic edge_ready;
    logic [2:0] edge_dir;
    logic [8:0] edge_gate_q17;
    logic [31:0] edge_k_bits;
    logic edge_last;
    logic term_valid;
    logic term_ready;
    logic [15:0] term_tag;
    logic [7:0] term_dest_id;
    logic [4:0] term_lane;
    logic [8:0] term_gate_q17;
    logic [2:0] term_multiplicity;
    logic term_last;
    logic dest_done_valid;
    logic dest_done_ready;
    logic [15:0] dest_done_tag;
    logic protocol_error;
    logic [15:0] count_edges;
    logic [15:0] count_terms;
    logic [15:0] count_naive_products;

    local5_mfep_term_builder dut (.*);
    always #5 clk_core = ~clk_core;

    task automatic open_dest(input int tag_value);
        @(negedge clk_core);
        dest_valid = 1'b1;
        dest_tag = 16'(tag_value);
        dest_id = 8'(tag_value);
        while (!dest_ready) @(negedge clk_core);
        @(negedge clk_core);
        dest_valid = 1'b0;
    endtask

    task automatic send_edge(
        input int dir_value,
        input bit last_value
    );
        @(negedge clk_core);
        edge_valid = 1'b1;
        edge_dir = 3'(dir_value);
        edge_gate_q17 = 9'd64;
        edge_k_bits = 32'h1;
        edge_last = last_value;
        while (!edge_ready) @(negedge clk_core);
        @(negedge clk_core);
        edge_valid = 1'b0;
    endtask

    task automatic expect_protocol_abort(input int tag_value);
        wait (dest_done_valid);
        if (!protocol_error || dest_done_tag !== 16'(tag_value) || term_valid)
            $fatal(1, "expected atomic protocol abort tag=%0d", tag_value);
        @(negedge clk_core);
        dest_done_ready = 1'b1;
        @(negedge clk_core);
        dest_done_ready = 1'b0;
        wait (dest_ready);
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        dest_valid = 1'b0;
        edge_valid = 1'b0;
        term_ready = 1'b1;
        dest_done_ready = 1'b0;
        dest_tag = '0;
        dest_id = '0;
        edge_dir = '0;
        edge_gate_q17 = '0;
        edge_k_bits = '0;
        edge_last = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        open_dest(1);
        send_edge(2, 1'b0);
        send_edge(2, 1'b1);
        expect_protocol_abort(1);

        open_dest(2);
        send_edge(7, 1'b1);
        expect_protocol_abort(2);

        $display("PASS tb_local5_mfep_protocol");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
