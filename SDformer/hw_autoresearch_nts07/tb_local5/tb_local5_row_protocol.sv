`timescale 1ns/1ps
`default_nettype none

module tb_local5_row_protocol;
    logic clk_core, rst_core;
    logic anchor_valid, anchor_ready;
    logic [15:0] anchor_tag;
    logic [7:0] anchor_dest_id;
    logic [31:0] anchor_q_bits, anchor_k_bits;
    logic [4:0] anchor_valid_mask;
    logic probe_valid, probe_ready;
    logic [2:0] probe_dir;
    logic [31:0] probe_k_bits;
    logic probe_last;
    logic edge_valid, edge_ready;
    logic [15:0] edge_tag;
    logic [7:0] edge_dest_id;
    logic [2:0] edge_dir;
    logic [31:0] edge_k_bits;
    logic [8:0] edge_gate_q17;
    logic signed [15:0] edge_score_q7;
    logic edge_last;
    logic row_done_valid, row_done_ready;
    logic [15:0] row_done_tag;
    logic [2:0] row_done_degree;
    logic protocol_error;
    logic [15:0] perf_probe_count, perf_edge_emit_count;

    local5_row_context_engine dut (.*);
    always #5 clk_core = ~clk_core;

    task automatic send_anchor(input logic [15:0] tag);
        while (!anchor_ready) @(posedge clk_core);
        anchor_valid = 1;
        anchor_tag = tag;
        anchor_dest_id = tag[7:0];
        anchor_q_bits = 32'h1234_5678;
        anchor_k_bits = 32'h89ab_cdef;
        anchor_valid_mask = 5'b0_0111;
        @(posedge clk_core);
        anchor_valid = 0;
    endtask

    task automatic send_probe(
        input logic [2:0] direction,
        input logic last
    );
        while (!probe_ready) @(posedge clk_core);
        probe_valid = 1;
        probe_dir = direction;
        probe_k_bits = 32'h1111_0000 ^ direction;
        probe_last = last;
        @(posedge clk_core);
        probe_valid = 0;
    endtask

    task automatic expect_aborted(input logic [15:0] tag);
        while (!row_done_valid) begin
            @(posedge clk_core);
            if (edge_valid)
                $fatal(1, "malformed row emitted an edge");
        end
        if (!protocol_error || row_done_tag != tag)
            $fatal(1, "malformed row did not report protocol error");
        @(posedge clk_core);
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        anchor_valid = 0;
        probe_valid = 0;
        edge_ready = 1;
        row_done_ready = 1;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        // Early probe_last: only one of two required neighbors arrived.
        send_anchor(16'h1001);
        send_probe(3'd1, 1'b1);
        expect_aborted(16'h1001);

        // Duplicate direction must not satisfy the expected probe count.
        send_anchor(16'h1002);
        send_probe(3'd1, 1'b0);
        send_probe(3'd1, 1'b1);
        expect_aborted(16'h1002);

        if (perf_edge_emit_count != 0)
            $fatal(1, "aborted rows emitted %0d edges", perf_edge_emit_count);
        $display("PASS tb_local5_row_protocol");
        $finish;
    end
endmodule

`default_nettype wire
