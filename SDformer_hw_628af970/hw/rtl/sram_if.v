module sram_if #(
    parameter integer DATA_W = 8,
    parameter integer LANES = 8
) (
    input  wire                    clk,
    input  wire                    we,
    input  wire [LANES-1:0]        data_in,
    output reg  [LANES-1:0]        data_out
);
    reg [LANES-1:0] buffer_a;

    always @(posedge clk) begin
        if (we) begin
            buffer_a <= data_in;
        end
        data_out <= buffer_a;
    end
endmodule

