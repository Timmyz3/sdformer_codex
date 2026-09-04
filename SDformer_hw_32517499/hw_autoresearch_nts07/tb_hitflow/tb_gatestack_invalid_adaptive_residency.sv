`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_invalid_adaptive_residency;
    logic clk_core, rst_core;
    logic group_valid;
    logic group_ready, protocol_error;

    gatestack_single_context_execution_top #(
        .CSR_FORMAT_FADC24(2),
        .ENABLE_RESIDENCY(1)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .group_valid(group_valid),
        .group_tag(32'd0),
        .group_head_count(6'd1),
        .group_first_output_tile(8'd0),
        .group_output_tile_count(8'd1),
        .group_done_ready(1'b1),
        .group_ready(group_ready),
        .protocol_error(protocol_error)
    );

    always #5 clk_core <= ~clk_core;
    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        if (protocol_error || !group_ready)
            $fatal(1, "Adaptive plus IPD-only residency admission failed");
        $display("PASS: Adaptive plus IPD-only residency configuration admitted");
        $finish;
    end
endmodule

`default_nettype wire
