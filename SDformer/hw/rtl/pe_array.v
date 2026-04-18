module pe_array #(
    parameter integer LANES = 8,
    parameter integer DATA_W = 8,
    parameter integer ACC_W = 24
) (
    input  wire [LANES*DATA_W-1:0] a_data,
    input  wire [LANES*DATA_W-1:0] b_data,
    output reg  [ACC_W-1:0]        acc_out
);
    integer i;
    reg signed [DATA_W-1:0] a_lane;
    reg signed [DATA_W-1:0] b_lane;
    reg signed [ACC_W-1:0]  sum;

    always @* begin
        sum = '0;
        for (i = 0; i < LANES; i = i + 1) begin
            a_lane = a_data[i*DATA_W +: DATA_W];
            b_lane = b_data[i*DATA_W +: DATA_W];
            sum = sum + a_lane * b_lane;
        end
        acc_out = sum;
    end
endmodule

