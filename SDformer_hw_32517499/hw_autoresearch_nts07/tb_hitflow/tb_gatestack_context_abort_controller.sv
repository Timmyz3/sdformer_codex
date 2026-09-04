`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_context_abort_controller;
    logic clk_core, rst_core;
    logic group_accept_pulse;
    logic [15:0] group_accept_tag;
    logic normal_done_fire, normal_done_error, fabric_error;
    logic fabric_reset_pulse, abort_done_valid, abort_done_ready;
    logic [15:0] abort_done_tag;
    logic abort_done_error, admission_blocked, group_active;
    logic protocol_error;
    logic [31:0] count_context_resets, count_error_aborts;
    logic [31:0] count_timeout_aborts;

    gatestack_context_abort_controller #(
        .TAG_W(16), .TIMEOUT_CYCLES(8)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic accept_group(input logic [15:0] tag_value);
        begin
            @(negedge clk_core);
            group_accept_tag = tag_value;
            group_accept_pulse = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            group_accept_pulse = 1'b0;
        end
    endtask

    task automatic accept_abort(input logic [15:0] tag_value);
        begin
            while (!abort_done_valid) @(posedge clk_core);
            if (abort_done_tag != tag_value || !abort_done_error ||
                !admission_blocked)
                $fatal(1, "abort response mismatch");
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            abort_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            abort_done_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_accept_pulse = 1'b0;
        group_accept_tag = '0;
        normal_done_fire = 1'b0;
        normal_done_error = 1'b0;
        fabric_error = 1'b0;
        abort_done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        // Normal retirement does not flush the context.
        accept_group(16'h8100);
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        normal_done_fire = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        normal_done_fire = 1'b0;
        if (group_active || fabric_reset_pulse || abort_done_valid)
            $fatal(1, "normal retirement flushed context");

        // Fabric error returns one stable synthetic error completion.
        accept_group(16'h8101);
        @(negedge clk_core);
        fabric_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        fabric_error = 1'b0;
        if (!fabric_reset_pulse)
            $fatal(1, "fabric error did not pulse reset");
        accept_abort(16'h8101);

        // Watchdog provides bounded retirement without any downstream progress.
        accept_group(16'h8102);
        accept_abort(16'h8102);

        // A normally returned error is forwarded by the fabric and still
        // triggers one cleanup reset without a duplicate synthetic response.
        accept_group(16'h8103);
        @(negedge clk_core);
        normal_done_error = 1'b1;
        normal_done_fire = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        normal_done_error = 1'b0;
        normal_done_fire = 1'b0;
        if (!fabric_reset_pulse || abort_done_valid)
            $fatal(1, "normal error cleanup mismatch");

        repeat (3) @(posedge clk_core);
        if (!protocol_error || count_context_resets != 3 ||
            count_error_aborts != 2 || count_timeout_aborts != 1)
            $fatal(1, "abort counters mismatch");
        $display("PASS: context resets=%0d error=%0d timeout=%0d",
                 count_context_resets, count_error_aborts,
                 count_timeout_aborts);
        $finish;
    end

    initial begin
        repeat (1000) @(posedge clk_core);
        $fatal(1, "context abort controller timeout");
    end
endmodule

`default_nettype wire
