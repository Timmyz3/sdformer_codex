`include "unibin_h60_pkg.vh"

module gated_k_unit #(
    parameter integer DATA_W = `UBIN_DATA_W,
    parameter integer GATE_W = `UBIN_GATE_W,
    parameter integer OUT_W = DATA_W + GATE_W
)(
    input  wire                     k_event,
    input  wire signed [DATA_W-1:0] k_value,
    input  wire [GATE_W-1:0]        gate,
    output wire signed [OUT_W-1:0]  gated_out
);
    assign gated_out = k_event ? ($signed(k_value) * $signed({1'b0, gate})) : {OUT_W{1'b0}};
endmodule
