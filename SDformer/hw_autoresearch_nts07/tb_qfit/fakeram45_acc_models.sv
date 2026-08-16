`timescale 1ns/1ps
`default_nettype none

// Functional-only model. OpenROAD obtains timing, power, and geometry from
// the matching Liberty and LEF views.
module fakeram45_128x256 (
    input  logic         clk,
    input  logic         ce_in,
    input  logic         we_in,
    input  logic [255:0] w_mask_in,
    input  logic [6:0]   addr_in,
    input  logic [255:0] wd_in,
    output logic [255:0] rd_out
);
    logic [255:0] memory [0:127];

    always_ff @(posedge clk) begin
        if (ce_in && we_in)
            memory[addr_in] <= (memory[addr_in] & ~w_mask_in)
                             | (wd_in & w_mask_in);
        if (ce_in && !we_in)
            rd_out <= memory[addr_in];
    end
endmodule

`default_nettype wire
