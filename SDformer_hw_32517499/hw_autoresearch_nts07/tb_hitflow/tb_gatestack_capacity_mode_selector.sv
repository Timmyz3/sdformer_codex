`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_capacity_mode_selector;
    logic clk_core;
    logic rst_core;
    logic request_valid;
    logic request_ready;
    logic [31:0] request_tag;
    logic [3:0] request_active_classes;
    logic [7:0] request_class_terms;
    logic [12:0] request_active_lanes;
    logic response_valid;
    logic response_ready;
    logic [31:0] response_tag;
    logic response_is_csr;
    logic [1:0] response_reason;
    logic [15:0] response_csr_bits;
    logic [31:0] count_requests;
    logic [31:0] count_csr;
    logic [31:0] count_raw_class_overflow;
    logic [31:0] count_raw_capacity_overflow;
    logic [31:0] prng_q;
    int expected_csr;
    int expected_class;
    int expected_capacity;

    gatestack_capacity_mode_selector dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic run_case(
        input int classes,
        input int terms,
        input int active,
        input logic [31:0] tag
    );
        int bits;
        logic expect_csr;
        logic [1:0] expect_reason;
        int stalls;
        begin
            bits = 128 + ((terms + 1) / 2) * 64 + active * 8;
            if (classes > 4) begin
                expect_csr = 1'b0;
                expect_reason = 2'd1;
                expected_class = expected_class + 1;
            end else if (bits > 6642) begin
                expect_csr = 1'b0;
                expect_reason = 2'd2;
                expected_capacity = expected_capacity + 1;
            end else begin
                expect_csr = 1'b1;
                expect_reason = 2'd0;
                expected_csr = expected_csr + 1;
            end
            while (!request_ready) @(posedge clk_core);
            @(negedge clk_core);
            request_valid = 1'b1;
            request_tag = tag;
            request_active_classes = 4'(classes);
            request_class_terms = 8'(terms);
            request_active_lanes = 13'(active);
            @(posedge clk_core);
            @(negedge clk_core);
            request_valid = 1'b0;
            response_ready = 1'b0;
            stalls = 1 + (tag & 3);
            repeat (stalls) @(posedge clk_core);
            if (!response_valid || response_tag != tag ||
                response_is_csr != expect_csr || response_reason != expect_reason ||
                32'(response_csr_bits) != bits)
                $fatal(1, "capacity selector mismatch tag=%h bits=%0d", tag, bits);
            @(negedge clk_core);
            response_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            response_ready = 1'b0;
        end
    endtask

    initial begin
        int classes;
        int terms;
        int active;
        clk_core = 1'b0;
        rst_core = 1'b1;
        request_valid = 1'b0;
        request_tag = '0;
        request_active_classes = '0;
        request_class_terms = '0;
        request_active_lanes = '0;
        response_ready = 1'b0;
        prng_q = 32'hca9a_c17e;
        expected_csr = 0;
        expected_class = 0;
        expected_capacity = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        run_case(4, 6, 790, 32'h5000);  // 6640: largest byte-aligned CSR
        run_case(4, 6, 791, 32'h5001);  // 6648: first next-byte RAW
        run_case(5, 1, 1, 32'h5002);    // class overflow has priority
        run_case(0, 0, 0, 32'h5003);

        for (int trial = 0; trial < 500; trial = trial + 1) begin
            prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
            classes = 32'(prng_q[2:0]);
            prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
            terms = 32'(prng_q[7:0]) % 129;
            prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
            active = 32'(prng_q[12:0]) % 1644;
            run_case(classes, terms, active, 32'h5100 + trial);
        end

        if (count_requests != 504 || count_csr != expected_csr ||
            count_raw_class_overflow != expected_class ||
            count_raw_capacity_overflow != expected_capacity)
            $fatal(1, "capacity counters mismatch");
        $display("PASS: capacity selector req=%0d csr=%0d raw_class=%0d raw_capacity=%0d",
                 count_requests, count_csr, count_raw_class_overflow,
                 count_raw_capacity_overflow);
        $finish;
    end
endmodule

`default_nettype wire
