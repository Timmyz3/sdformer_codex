`include "utils.vh"

module top #(
    parameter integer LANES = 8,
    parameter integer DATA_W = 8,
    parameter integer MEM_W = 12,
    parameter integer ACC_W = 24,
    parameter integer THRESHOLD = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     in_valid,
    input  wire                     in_last,
    output wire                     in_ready,
    input  wire [LANES*DATA_W-1:0]  in_data,
    output wire                     out_valid,
    output wire                     out_last,
    input  wire                     out_ready,
    output wire [LANES-1:0]         out_data
) ;
    wire [LANES*DATA_W-1:0] attn_data;
    wire [LANES*DATA_W-1:0] mixed_data;
    wire fire_en;
    wire [LANES-1:0] sram_data;

    controller u_controller (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .out_ready(out_ready),
        .in_ready(in_ready),
        .out_valid(out_valid),
        .fire_en(fire_en)
    );

    attention_unit #(
        .LANES(LANES),
        .DATA_W(DATA_W)
    ) u_attention (
        .token_in(in_data),
        .token_out(attn_data)
    );

    token_mixer #(
        .LANES(LANES),
        .DATA_W(DATA_W)
    ) u_mixer (
        .token_in(attn_data),
        .token_out(mixed_data)
    );

    spike_unit #(
        .LANES(LANES),
        .DATA_W(DATA_W),
        .MEM_W(MEM_W),
        .THRESHOLD(THRESHOLD)
    ) u_spike (
        .clk(clk),
        .rst_n(rst_n),
        .fire_en(fire_en),
        .token_in(mixed_data),
        .spike_out(out_data)
    );

    assign out_last = in_last;
endmodule

