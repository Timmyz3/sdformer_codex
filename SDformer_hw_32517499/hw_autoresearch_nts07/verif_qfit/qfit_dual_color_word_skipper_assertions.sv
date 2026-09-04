`timescale 1ns/1ps
`default_nettype none

module qfit_dual_color_word_skipper_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [SOURCE_ID_W-1:0] out_source_id,
    input logic [PLANE_W-1:0] out_source_plane,
    input logic [Y_W-1:0] out_source_y,
    input logic [X_W-1:0] out_source_x,
    input logic out_last,
    input logic [31:0] perf_bank_conflicts
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready
        |=> out_valid && $stable({
            out_source_id, out_source_plane, out_source_y, out_source_x, out_last
        })
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid |-> 32'(out_source_plane) < TIME_PLANES
                  && 32'(out_source_y) < HEIGHT
                  && 32'(out_source_x) < WIDTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        perf_bank_conflicts == 0
    );
endmodule

bind qfit_dual_color_word_skipper_index
    qfit_dual_color_word_skipper_assertions #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .Y_W(Y_W), .X_W(X_W), .PLANE_W(PLANE_W),
        .SOURCE_ID_W(SOURCE_ID_W)
    ) u_qfit_dual_color_word_skipper_assertions (
        .clk_core(clk_core), .rst_core(rst_core),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_source_id(out_source_id),
        .out_source_plane(out_source_plane), .out_source_y(out_source_y),
        .out_source_x(out_source_x), .out_last(out_last),
        .perf_bank_conflicts(perf_bank_conflicts)
    );

`default_nettype wire
