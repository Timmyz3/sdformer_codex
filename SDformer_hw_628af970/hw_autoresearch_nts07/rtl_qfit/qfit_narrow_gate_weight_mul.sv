`timescale 1ns/1ps
`default_nettype none

module qfit_narrow_gate_weight_mul #(
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int PRODUCT_W = GATE_W + W_W
) (
    input  logic                         enable,
    input  logic [GATE_W-1:0]            gate,
    input  logic signed [W_W-1:0]        weight,
    output logic signed [PRODUCT_W-1:0]  product
);
    logic signed [GATE_W:0] gate_operand;
    logic signed [W_W-1:0] weight_operand;

    always_comb begin
        gate_operand = enable ? $signed({1'b0, gate}) : '0;
        weight_operand = enable ? weight : '0;
        product = gate_operand * weight_operand;
    end
endmodule

`default_nettype wire
