`timescale 1ns/1ps
`default_nettype none

module tb_local5_mfep_t450_counter;
    localparam int DESTS = 450;

    logic clk_core, rst_core;
    logic dest_valid, dest_ready;
    logic [15:0] dest_tag;
    logic [8:0] dest_id;
    logic edge_valid, edge_ready;
    logic [2:0] edge_dir;
    logic [8:0] edge_gate_q17;
    logic [31:0] edge_k_bits;
    logic edge_last;
    logic term_valid, term_ready;
    logic [15:0] term_tag;
    logic [8:0] term_dest_id;
    logic [4:0] term_lane;
    logic [8:0] term_gate_q17;
    logic [2:0] term_multiplicity;
    logic term_last;
    logic dest_done_valid, dest_done_ready;
    logic [15:0] dest_done_tag;
    logic protocol_error;
    logic [31:0] count_edges, count_terms, count_naive_products;
    int observed_terms;

    local5_mfep_term_builder #(
        .DEST_W(9),
        .COUNTER_W(32)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            observed_terms <= 0;
        end else if (term_valid && term_ready) begin
            observed_terms <= observed_terms + 1;
            if (term_multiplicity != 3'd1)
                $fatal(1, "multiplicity mismatch");
        end
    end

    initial begin
        clk_core = 0;
        rst_core = 1;
        dest_valid = 0;
        edge_valid = 0;
        term_ready = 1;
        dest_done_ready = 1;
        repeat (5) @(posedge clk_core);
        rst_core = 0;

        for (int dest = 0; dest < DESTS; dest++) begin
            @(negedge clk_core);
            dest_valid = 1;
            dest_tag = 16'(dest);
            dest_id = 9'(dest);
            while (!(dest_valid && dest_ready)) @(posedge clk_core);
            @(negedge clk_core);
            dest_valid = 0;

            for (int e = 0; e < 5; e++) begin
                @(negedge clk_core);
                edge_valid = 1;
                edge_dir = 3'(e);
                edge_gate_q17 = 9'(e + 1);
                edge_k_bits = 32'hffff_ffff;
                edge_last = (e == 4);
                while (!(edge_valid && edge_ready)) @(posedge clk_core);
                @(negedge clk_core);
                edge_valid = 0;
            end

            while (!dest_done_valid) @(posedge clk_core);
            if (dest_done_tag != 16'(dest))
                $fatal(1, "done tag mismatch");
            @(posedge clk_core);
        end

        if (protocol_error)
            $fatal(1, "protocol_error");
        if (count_edges != 32'd2250)
            $fatal(1, "edge count got=%0d", count_edges);
        if (count_terms != 32'd72000 || observed_terms != 72000)
            $fatal(1, "term count got=%0d observed=%0d",
                   count_terms, observed_terms);
        if (count_naive_products != 32'd72000)
            $fatal(1, "naive count got=%0d", count_naive_products);

        $display("PASS tb_local5_mfep_t450_counter edges=%0d terms=%0d naive=%0d",
                 count_edges, count_terms, count_naive_products);
        $finish;
    end

    initial begin
        #10000000;
        $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
