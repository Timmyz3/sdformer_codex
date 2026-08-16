`timescale 1ns/1ps
`default_nettype none

module tb_local5_mfep_sparse_last;
    logic clk_core, rst_core;
    logic dest_valid, dest_ready;
    logic [15:0] dest_tag;
    logic [7:0] dest_id;
    logic edge_valid, edge_ready;
    logic [2:0] edge_dir;
    logic [8:0] edge_gate_q17;
    logic [31:0] edge_k_bits;
    logic edge_last;
    logic term_valid, term_ready;
    logic [15:0] term_tag;
    logic [7:0] term_dest_id;
    logic [4:0] term_lane;
    logic [8:0] term_gate_q17;
    logic [2:0] term_multiplicity;
    logic term_last;
    logic dest_done_valid, dest_done_ready;
    logic [15:0] dest_done_tag;
    logic protocol_error;
    logic [15:0] count_edges, count_terms, count_naive_products;

    local5_mfep_term_builder dut (.*);
    always #5 clk_core = ~clk_core;

    initial begin
        clk_core = 0;
        rst_core = 1;
        dest_valid = 0;
        edge_valid = 0;
        term_ready = 0;
        dest_done_ready = 1;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        @(posedge clk_core);
        dest_valid = 1;
        dest_tag = 16'h55aa;
        dest_id = 8'd3;
        while (!dest_ready) @(posedge clk_core);
        @(posedge clk_core);
        dest_valid = 0;

        edge_valid = 1;
        edge_dir = 0;
        edge_gate_q17 = 9'd17;
        edge_k_bits = 32'h0000_0001;
        edge_last = 1;
        while (!edge_ready) @(posedge clk_core);
        @(posedge clk_core);
        edge_valid = 0;

        while (!term_valid) @(posedge clk_core);
        if (term_lane != 0 || term_gate_q17 != 17
            || term_multiplicity != 1 || !term_last) begin
            $fatal(
                1,
                "bad sparse tail lane=%0d gate=%0d mult=%0d last=%0b",
                term_lane, term_gate_q17, term_multiplicity, term_last
            );
        end
        repeat (3) begin
            @(posedge clk_core);
            if (!term_valid || !term_last || term_lane != 0)
                $fatal(1, "term changed under backpressure");
        end
        term_ready = 1;
        @(posedge clk_core);
        term_ready = 0;
        while (!dest_done_valid) @(posedge clk_core);
        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (count_terms != 1)
            $fatal(1, "term count %0d", count_terms);

        $display("PASS tb_local5_mfep_sparse_last");
        $finish;
    end
endmodule

`default_nettype wire
