module controller (
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,
    input  wire out_ready,
    output wire in_ready,
    output reg  out_valid,
    output reg  fire_en
);
    assign in_ready = out_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            fire_en <= 1'b0;
        end else begin
            fire_en <= in_valid & out_ready;
            out_valid <= in_valid;
        end
    end
endmodule

