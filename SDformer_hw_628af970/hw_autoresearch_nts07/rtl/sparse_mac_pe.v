`include "nts07_pkg.vh"

// Bit-serial sparse MAC: 1-bit spike input × INT8 weight, zero-skip.
module sparse_mac_pe #(
    parameter integer WGT_W = 8,
    parameter integer ACC_W = 24
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,
    input  wire                     spike_in,
    input  wire signed [WGT_W-1:0]  weight,
    input  wire                     acc_clear,
    output reg  signed [ACC_W-1:0]  acc_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 0;
        end else if (acc_clear) begin
            acc_out <= 0;
        end else if (enable && spike_in) begin
            acc_out <= acc_out + {{(ACC_W-WGT_W){weight[WGT_W-1]}}, weight};
        end
    end
endmodule


module sparse_mac_lane #(
    parameter integer LANES = 8,
    parameter integer WGT_W = 8,
    parameter integer ACC_W = 24
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     fire,
    input  wire [LANES-1:0]         spike_vec,
    input  wire signed [LANES*WGT_W-1:0] weight_vec,
    input  wire                     acc_clear,
    output wire signed [ACC_W-1:0]  acc_lane [0:LANES-1]
);
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_pe
            sparse_mac_pe #(.WGT_W(WGT_W), .ACC_W(ACC_W)) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                .enable(fire),
                .spike_in(spike_vec[i]),
                .weight(weight_vec[i*WGT_W +: WGT_W]),
                .acc_clear(acc_clear),
                .acc_out(acc_lane[i])
            );
        end
    endgenerate
endmodule