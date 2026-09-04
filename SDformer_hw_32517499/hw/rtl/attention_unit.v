module attention_unit #(
    parameter integer LANES = 8,
    parameter integer DATA_W = 8
) (
    input  wire [LANES*DATA_W-1:0] token_in,
    output wire [LANES*DATA_W-1:0] token_out
);
    assign token_out = token_in;
endmodule

