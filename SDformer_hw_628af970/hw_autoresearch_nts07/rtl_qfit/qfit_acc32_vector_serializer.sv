`timescale 1ns/1ps
`default_nettype none

// Common post-vector boundary for B0v and B2v. One complete Acc32 row is
// accepted atomically and serialized without rereading the backing SRAM.
module qfit_acc32_vector_serializer #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int OUT_DIM = 32,
    parameter int ACC_W = 32,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [PLANE_W-1:0]           in_plane,
    input  logic [Y_W-1:0]               in_y,
    input  logic [X_W-1:0]               in_x,
    input  logic [OUT_DIM*ACC_W-1:0]     in_data,
    input  logic                         in_last,
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [PLANE_W-1:0]           out_plane,
    output logic [Y_W-1:0]               out_y,
    output logic [X_W-1:0]               out_x,
    output logic [OUT_W-1:0]             out_index,
    output logic signed [ACC_W-1:0]      out_data,
    output logic                         out_last
);
    logic full_q;
    logic [PLANE_W-1:0] plane_q;
    logic [Y_W-1:0] y_q;
    logic [X_W-1:0] x_q;
    logic [OUT_DIM*ACC_W-1:0] data_q;
    logic last_q;
    logic [OUT_W-1:0] out_q;
    logic input_fire;
    logic output_fire;

    assign in_ready = !full_q;
    assign input_fire = in_valid && in_ready;
    assign out_valid = full_q;
    assign output_fire = out_valid && out_ready;
    assign out_plane = plane_q;
    assign out_y = y_q;
    assign out_x = x_q;
    assign out_index = out_q;
    assign out_data = data_q[out_q*ACC_W +: ACC_W];
    assign out_last = last_q && 32'(out_q) + 1 == OUT_DIM;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            full_q <= 1'b0;
            plane_q <= '0;
            y_q <= '0;
            x_q <= '0;
            data_q <= '0;
            last_q <= 1'b0;
            out_q <= '0;
        end else begin
            if (input_fire) begin
                full_q <= 1'b1;
                plane_q <= in_plane;
                y_q <= in_y;
                x_q <= in_x;
                data_q <= in_data;
                last_q <= in_last;
                out_q <= '0;
            end else if (output_fire) begin
                if (32'(out_q) + 1 == OUT_DIM) begin
                    full_q <= 1'b0;
                    out_q <= '0;
                end else begin
                    out_q <= out_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
