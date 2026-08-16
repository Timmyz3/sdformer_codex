`timescale 1ns/1ps
`default_nettype none

module tb_qfit_direct_1rw_reference_timing;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #5 clk_core = ~clk_core;

    logic run_start;
    logic run_accumulate;
    logic update_valid;
    logic update_ready;
    logic [1:0] update_addr;
    logic [31:0] update_delta;
    logic flush_valid;
    logic flush_ready;
    logic flush_done;
    logic read_valid;
    logic read_ready;
    logic [1:0] read_addr;
    logic read_data_valid;
    logic [31:0] read_data;
    logic protocol_error;
    logic [31:0] perf_updates;
    logic [31:0] perf_reads;
    logic [31:0] perf_writes;
    integer cycle;
    integer accept [0:5];

    qfit_direct_1rw_acc_bank #(
        .DEPTH(4), .OUT_DIM(1), .ACC_W(32), .MEMORY_IMPL(0)
    ) dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .run_start(run_start), .run_accumulate(run_accumulate),
        .update_valid(update_valid), .update_ready(update_ready),
        .update_addr(update_addr), .update_delta(update_delta),
        .flush_valid(flush_valid), .flush_ready(flush_ready),
        .flush_done(flush_done),
        .read_valid(read_valid), .read_ready(read_ready),
        .read_addr(read_addr), .read_data_valid(read_data_valid),
        .read_data(read_data), .protocol_error(protocol_error),
        .perf_updates(perf_updates), .perf_sram_reads(perf_reads),
        .perf_sram_writes(perf_writes)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) cycle <= 0;
        else cycle <= cycle + 1;
    end

    task automatic start_run(input logic accumulate);
        begin
            @(negedge clk_core);
            run_accumulate = accumulate;
            run_start = 1'b1;
            @(negedge clk_core);
            run_start = 1'b0;
        end
    endtask

    task automatic send_update(
        input logic [1:0] address,
        input logic [31:0] delta,
        input integer slot
    );
        begin
            update_addr = address;
            update_delta = delta;
            update_valid = 1'b1;
            while (!update_ready) @(negedge clk_core);
            @(posedge clk_core);
            accept[slot] = cycle;
            @(negedge clk_core);
            update_valid = 1'b0;
        end
    endtask

    initial begin
        run_start = 1'b0;
        run_accumulate = 1'b0;
        update_valid = 1'b0;
        update_addr = '0;
        update_delta = '0;
        flush_valid = 1'b0;
        read_valid = 1'b0;
        read_addr = '0;
        repeat (3) @(posedge clk_core);
        rst_core = 1'b0;

        start_run(1'b0);
        send_update(2'd0, 32'd1, 0);
        send_update(2'd0, 32'd2, 1);
        send_update(2'd0, 32'd3, 2);
        send_update(2'd1, 32'd4, 3);

        if (accept[1] - accept[0] != 1)
            $fatal(1, "first-touch did not sustain one-cycle issue");
        if (accept[2] - accept[1] != 2)
            $fatal(1, "RMW did not insert exactly one issue stall");
        if (accept[3] - accept[2] != 2)
            $fatal(1, "second RMW timing mismatch");
        if (perf_updates != 4 || perf_reads != 2 || perf_writes != 4)
            $fatal(1, "first run counters mismatch u=%0d r=%0d w=%0d",
                   perf_updates, perf_reads, perf_writes);

        start_run(1'b1);
        send_update(2'd1, 32'd5, 4);
        send_update(2'd2, 32'd6, 5);
        if (accept[5] - accept[4] != 2)
            $fatal(1, "preserved address did not execute as RMW");
        if (perf_updates != 2 || perf_reads != 1 || perf_writes != 2)
            $fatal(1, "accumulate run counters mismatch u=%0d r=%0d w=%0d",
                   perf_updates, perf_reads, perf_writes);
        if (protocol_error) $fatal(1, "protocol_error asserted");
        $display(
            "PASS DIRECT_1RW_REFERENCE_TIMING accepts=%0d,%0d,%0d,%0d,%0d,%0d",
            accept[0], accept[1], accept[2], accept[3], accept[4], accept[5]
        );
        $finish;
    end
endmodule

`default_nettype wire
