`timescale 1ns/1ps
`default_nettype none

// Pre-macro synthesis/formal shell only.  Functional simulation must compile
// qfit_sync_1rw_acc_bank.sv instead.  The unresolved memory boundary prevents
// DC from converting six 128x512-bit SRAMs into hundreds of thousands of
// flip-flops while retaining every address/data/control pin for logic QoR and
// Formality.  It has zero intrinsic area/timing and is never paper-PPA ready.
(* black_box *)
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
endmodule

`default_nettype wire
