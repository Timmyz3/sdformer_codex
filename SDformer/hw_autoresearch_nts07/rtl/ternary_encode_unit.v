`include "nts07_pkg.vh"

// Legacy alias: Q/K ternary path wraps unified ATLIF encoder with ternary_en=1.
module ternary_encode_unit #(
    parameter integer DATA_W = 16,
    parameter integer THRESH_W = 16
)(
    input  wire signed [DATA_W-1:0]     activation,
    input  wire signed [THRESH_W-1:0]   pos_thresh,
    input  wire signed [THRESH_W-1:0]   neg_thresh,
    output wire [1:0]                   ternary_out
);
    wire binary_unused;
    atlif_unified_encode_unit #(
        .DATA_W(DATA_W), .THRESH_W(THRESH_W)
    ) u_unified (
        .ternary_en(1'b1),
        .activation(activation),
        .pos_thresh(pos_thresh),
        .neg_thresh(neg_thresh),
        .spike_out(ternary_out),
        .binary_out(binary_unused)
    );
endmodule