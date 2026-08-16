`timescale 1ns/1ps
`default_nettype none

// 仅用于功能仿真的开放SRAM代理模型；接口和同步读时序与PnR宏合同一致。
module fakeram45_256x32 (
    input  logic        clk,
    input  logic        ce_in,
    input  logic        we_in,
    input  logic [31:0] w_mask_in,
    input  logic [7:0]  addr_in,
    input  logic [31:0] wd_in,
    output logic [31:0] rd_out
);
    logic [31:0] mem [0:255];
    integer bit_index;

    always_ff @(posedge clk) begin
        if (ce_in) begin
            if (we_in) begin
                for (bit_index = 0; bit_index < 32; bit_index = bit_index + 1) begin
                    if (w_mask_in[bit_index])
                        mem[addr_in][bit_index] <= wd_in[bit_index];
                end
            end else begin
                rd_out <= mem[addr_in];
            end
        end
    end
endmodule

`default_nettype wire
