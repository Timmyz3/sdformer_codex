`timescale 1ns/1ps
`default_nettype none

// One-cycle synchronous read, one write per cycle, explicit read-first collision.
module qfit_sync_1r1w_bank #(
    parameter int DATA_W = 32,
    parameter int DEPTH = 45,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk_core,
    input  logic                  rst_core,
    input  logic                  wr_en,
    input  logic [ADDR_W-1:0]     wr_addr,
    input  logic [DATA_W-1:0]     wr_data,
    input  logic                  rd_en,
    input  logic [ADDR_W-1:0]     rd_addr,
    output logic                  rd_valid,
    output logic [DATA_W-1:0]     rd_data
);
    logic [DATA_W-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            rd_valid <= 1'b0;
            rd_data <= '0;
        end else begin
            rd_valid <= rd_en;
            if (rd_en)
                rd_data <= mem[rd_addr];
            if (wr_en)
                mem[wr_addr] <= wr_data;
        end
    end
endmodule

`default_nettype wire
