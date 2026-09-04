`timescale 1ns/1ps
`default_nettype none

// Single-write append bank used by one context and one token parity.
module gatestack_ppdi_token_bank #(
    parameter int TOKEN_ID_W = 8,
    parameter int DEPTH = 21,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk_core,
    input  logic                  write_enable,
    input  logic [ADDR_W-1:0]     write_address,
    input  logic [TOKEN_ID_W-1:0] write_data,
    input  logic [ADDR_W-1:0]     read_address,
    output logic [TOKEN_ID_W-1:0] read_data
);
    logic [TOKEN_ID_W-1:0] token_q [0:DEPTH-1];

    assign read_data = token_q[read_address];

    always_ff @(posedge clk_core) begin
        if (write_enable)
            token_q[write_address] <= write_data;
    end
endmodule

`default_nettype wire
