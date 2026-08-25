`timescale 1ns/1ps
`default_nettype none

// Explicit synchronous 1RW boundary for one Acc32 bank.  The behavioral
// array is used for RTL/VCS and pre-macro synthesis only; a paper PPA run must
// replace this module with the characterized 128x512b SRAM macro view.
module qfit_sync_1rw_acc_bank #(
    parameter int DEPTH = 128,
    parameter int DATA_W = 512,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk_core,
    input  logic                  enable,
    input  logic                  write_enable,
    input  logic [ADDR_W-1:0]     address,
    input  logic [DATA_W-1:0]     write_data,
    output logic [DATA_W-1:0]     read_data
);
    logic [DATA_W-1:0] memory_q [0:DEPTH-1];

    initial begin
        if (DEPTH < 1 || DATA_W < 1)
            $error("SRAM bank geometry must be positive");
    end

    always_ff @(posedge clk_core) begin
        if (enable) begin
            if (write_enable)
                memory_q[address] <= write_data;
            else
                read_data <= memory_q[address];
        end
    end
endmodule

`default_nettype wire
