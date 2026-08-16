`timescale 1ns/1ps
`default_nettype none

// Functional-only models for the two open SRAM macros used by the relation
// bank wrapper. OpenROAD reads timing and geometry from Liberty/LEF instead.
module fakeram45_256x16 (
    input  logic        clk,
    input  logic        ce_in,
    input  logic        we_in,
    input  logic [15:0] w_mask_in,
    input  logic [7:0]  addr_in,
    input  logic [15:0] wd_in,
    output logic [15:0] rd_out
);
    logic [15:0] memory [0:255];
    always_ff @(posedge clk) begin
        if (ce_in && we_in)
            memory[addr_in] <= (memory[addr_in] & ~w_mask_in)
                             | (wd_in & w_mask_in);
        if (ce_in && !we_in)
            rd_out <= memory[addr_in];
    end
endmodule

module fakeram45_256x32 (
    input  logic        clk,
    input  logic        ce_in,
    input  logic        we_in,
    input  logic [31:0] w_mask_in,
    input  logic [7:0]  addr_in,
    input  logic [31:0] wd_in,
    output logic [31:0] rd_out
);
    logic [31:0] memory [0:255];
    always_ff @(posedge clk) begin
        if (ce_in && we_in)
            memory[addr_in] <= (memory[addr_in] & ~w_mask_in)
                             | (wd_in & w_mask_in);
        if (ce_in && !we_in)
            rd_out <= memory[addr_in];
    end
endmodule

`default_nettype wire
