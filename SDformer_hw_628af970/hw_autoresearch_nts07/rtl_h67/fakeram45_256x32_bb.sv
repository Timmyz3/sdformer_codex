`timescale 1ns/1ps
`default_nettype none

(* blackbox *)
module fakeram45_256x32 (
    input  logic        clk,
    input  logic        ce_in,
    input  logic        we_in,
    input  logic [31:0] w_mask_in,
    input  logic [7:0]  addr_in,
    input  logic [31:0] wd_in,
    output logic [31:0] rd_out
);
endmodule

`default_nettype wire
