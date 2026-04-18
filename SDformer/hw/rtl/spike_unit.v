module spike_unit #(
    parameter integer LANES = 8,
    parameter integer DATA_W = 8,
    parameter integer MEM_W = 12,
    parameter integer THRESHOLD = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     fire_en,
    input  wire [LANES*DATA_W-1:0]  token_in,
    output reg  [LANES-1:0]         spike_out
);
    integer i;
    reg signed [MEM_W-1:0] membrane [0:LANES-1];
    reg signed [DATA_W-1:0] lane_value;
    reg signed [MEM_W-1:0] next_mem;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < LANES; i = i + 1) begin
                membrane[i] <= '0;
                spike_out[i] <= 1'b0;
            end
        end else if (fire_en) begin
            for (i = 0; i < LANES; i = i + 1) begin
                lane_value = token_in[i*DATA_W +: DATA_W];
                next_mem = membrane[i] + lane_value;
                if (next_mem >= THRESHOLD) begin
                    membrane[i] <= '0;
                    spike_out[i] <= 1'b1;
                end else begin
                    membrane[i] <= next_mem;
                    spike_out[i] <= 1'b0;
                end
            end
        end
    end
endmodule

