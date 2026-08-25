`timescale 1ns/1ps
`default_nettype none

// Expected-fail target: proves the vendor adapter cannot silently fall back
// to registers or truncate an unsupported storage geometry.
module tb_qfit_tsmc28_unsupported_geometry;
    logic clk_core;
    logic [511:0] read_data;

    qfit_sync_1rw_acc_bank #(
        .DEPTH(64), .DATA_W(512), .ADDR_W(6)
    ) must_fail (
        .clk_core(clk_core), .enable(1'b0), .write_enable(1'b0),
        .address('0), .write_data('0), .read_data(read_data)
    );

    initial begin
        clk_core = 1'b0;
        #20 $fatal(1, "unsupported TSMC28 geometry failed to terminate");
    end
    always #1 clk_core = ~clk_core;
endmodule

`default_nettype wire
