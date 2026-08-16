`timescale 1ns/1ps
`default_nettype none

// One-cycle synchronous single-port SRAM contract: one read or one write.
module qfit_sync_1rw_bank #(
    parameter int DATA_W = 32,
    parameter int DEPTH = 45,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk_core,
    input  logic                  rst_core,
    input  logic                  mem_en,
    input  logic                  mem_write,
    input  logic [ADDR_W-1:0]     mem_addr,
    input  logic [DATA_W-1:0]     mem_write_data,
    output logic                  mem_read_valid,
    output logic [DATA_W-1:0]     mem_read_data
);
    logic [DATA_W-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            mem_read_valid <= 1'b0;
            mem_read_data <= '0;
        end else begin
            mem_read_valid <= mem_en && !mem_write;
            if (mem_en) begin
                if (mem_write)
                    mem[mem_addr] <= mem_write_data;
                else
                    mem_read_data <= mem[mem_addr];
            end
        end
    end
endmodule

`default_nettype wire
