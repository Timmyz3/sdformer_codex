`timescale 1ns/1ps
`default_nettype none

// One-command-per-cycle synchronous 1RW memory contract. Reads and writes
// share the same address port and can never occur in the same cycle.
module qfit_single_port_acc_memory #(
    parameter int DEPTH = 90,
    parameter int VEC_W = 64,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int MEMORY_IMPL = 0
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   command_valid,
    input  logic                   command_write,
    input  logic [ADDR_W-1:0]      command_addr,
    input  logic [VEC_W-1:0]       command_write_data,
    output logic                   read_data_valid,
    output logic [VEC_W-1:0]       read_data
);
    generate
        if (MEMORY_IMPL == 0) begin : g_inferred
            logic [VEC_W-1:0] memory [0:DEPTH-1];

            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    read_data_valid <= 1'b0;
                    read_data <= '0;
                end else begin
                    read_data_valid <= command_valid && !command_write;
                    if (command_valid && command_write)
                        memory[command_addr] <= command_write_data;
                    else if (command_valid)
                        read_data <= memory[command_addr];
                end
            end
        end else begin : g_fakeram45
            qfit_fakeram45_acc_memory_90x1024 #(
                .DEPTH(DEPTH), .VEC_W(VEC_W), .ADDR_W(ADDR_W)
            ) u_macro_memory (.*);
        end
    endgenerate
endmodule

`default_nettype wire
