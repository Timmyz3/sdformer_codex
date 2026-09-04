`timescale 1ns/1ps
`default_nettype none

// Symmetry audit for the baseline SHA frozen by the r3 matched-island
// contract.  The candidate repair must not hide an equivalent acceptance
// window on the denominator side.
module tb_m102_r3_baseline_same_cycle_counterexample;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic lookup_valid = 1'b0;
    logic lookup_ready;
    logic [3:0] lookup_source = '0;
    logic [2:0] lookup_block = '0;
    logic [1:0] lookup_beat = '0;
    logic [31:0] lookup_tag = '0;
    logic [255:0] bank_words = {8{32'h807f_01ff}};
    logic [79:0] bank_row_addresses;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic output_escape;
    logic [1151:0] output_values;
    logic output_accept;
    logic protocol_error;
    logic busy;

    m102_bit_sparse_weight_stream dut (.*);
    always #1.5 clk_core = ~clk_core;

    initial begin
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Produce one complete fixed8 vector and hold its M82 result.
        output_ready = 1'b0;
        for (int beat = 0; beat < 3; beat++) begin
            @(negedge clk_core);
            lookup_valid = 1'b1;
            lookup_source = 4'd2;
            lookup_block = 3'd3;
            lookup_beat = beat[1:0];
            lookup_tag = 32'hba5e_3001;
            do @(posedge clk_core); while (!lookup_ready);
        end
        @(negedge clk_core);
        lookup_valid = 1'b0;
        #0.1;
        if (!output_valid || output_accept || protocol_error
                || !dut.m82_output_valid)
            $fatal(1, "baseline counterexample setup failed");

        // Orphan beat1 is illegal while idle.  Release ready in the same
        // combinational window, exactly matching the repaired candidate test.
        lookup_valid = 1'b1;
        lookup_beat = 2'd1;
        lookup_tag = 32'hba5e_3002;
        output_ready = 1'b1;
        #0.1;
        if (dut.request_semantically_valid || lookup_ready
                || protocol_error || !output_valid || !output_accept
                || !dut.m82_output_valid)
            $fatal(1, "expected baseline same-cycle window was absent semantic=%0d lookup_ready=%0d fault=%0d output_valid=%0d output_accept=%0d m82_valid=%0d",
                   dut.request_semantically_valid, lookup_ready,
                   protocol_error, output_valid, output_accept,
                   dut.m82_output_valid);
        $display("COUNTEREXAMPLE M102_R3_BASELINE_PREEDGE semantic_valid=0 protocol_error=0 output_valid=1 output_accept=1 m82_valid=1");

        @(posedge clk_core);
        #0.1;
        if (!dut.request_fault_q || !protocol_error || dut.m82_output_valid)
            $fatal(1, "baseline counterexample did not retire old output while registering fault");
        $display("COUNTEREXAMPLE_CONFIRMED M102_R3 baseline_old_output_retired_on_invalid_edge=1");
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "M102 r3 baseline counterexample watchdog");
    end
endmodule

`default_nettype wire
