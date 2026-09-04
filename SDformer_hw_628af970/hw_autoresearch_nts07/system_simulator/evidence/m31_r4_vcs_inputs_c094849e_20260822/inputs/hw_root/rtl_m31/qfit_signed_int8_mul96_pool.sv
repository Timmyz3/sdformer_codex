`timescale 1ns/1ps
`default_nettype none

// Named multiplier leaf retained as a synthesis hierarchy boundary.  The
// arithmetic is still technology-mapped inside this leaf; the boundary lets
// DC prove that exactly 96 nonempty multiplier cones remain under one pool.
module qfit_signed_int8_mul_leaf #(
    parameter int IN_W = 8
) (
    input  logic signed [IN_W-1:0] operand_a,
    input  logic signed [IN_W-1:0] operand_b,
    output wire signed [(2*IN_W)-1:0] product
);
    assign product = $signed(operand_a) * $signed(operand_b);
endmodule

// The only signed INT8 multiplier pool admitted by the M31 unified temporal
// engine.  Operand selection happens above this pool; DC audits the pool and
// every named leaf to reject fused or duplicated T10/T2 arithmetic resources.
module qfit_signed_int8_mul96_pool #(
    parameter int MULTIPLIERS = 96,
    parameter int IN_W = 8
) (
    input  logic signed [IN_W-1:0] operand_a [0:MULTIPLIERS-1],
    input  logic signed [IN_W-1:0] operand_b [0:MULTIPLIERS-1],
    output wire signed [(2*IN_W)-1:0] product [0:MULTIPLIERS-1]
);
`ifndef SYNTHESIS
    initial begin
        if (MULTIPLIERS != 96 || IN_W != 8)
            $fatal(1, "M31 multiplier pool resource contract drift");
    end
`endif

    for (genvar multiplier = 0; multiplier < MULTIPLIERS; multiplier++) begin
        qfit_signed_int8_mul_leaf #(.IN_W(IN_W)) u_mul (
            .operand_a(operand_a[multiplier]),
            .operand_b(operand_b[multiplier]),
            .product(product[multiplier])
        );
    end
endmodule

`default_nettype wire
