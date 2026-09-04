`include "unibin_h60_pkg.vh"

module binary_atlif_unit #(
    parameter integer DATA_W = 16,
    parameter integer THRESH_W = 16
)(
    input  wire signed [DATA_W-1:0]   membrane,
    input  wire signed [THRESH_W-1:0] threshold,
    output wire                       event_out
);
    assign event_out = (membrane >= threshold);
endmodule
