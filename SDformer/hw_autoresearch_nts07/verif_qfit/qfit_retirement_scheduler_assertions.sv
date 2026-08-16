`timescale 1ns/1ps
`default_nettype none

module qfit_retirement_scheduler_assertions #(
    parameter int MODE = 0,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input logic clk_core,
    input logic rst_core,
    input logic retire_valid,
    input logic retire_ready,
    input logic [SOURCE_ID_W-1:0] retire_source_id,
    input logic [Y_W-1:0] retire_y,
    input logic [X_W-1:0] retire_x,
    input logic [2:0] perf_max_pending
);
    localparam int TOTAL = HEIGHT * WIDTH * TIME_PLANES;

    property p_retire_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            retire_valid && !retire_ready
            |=> retire_valid
                && $stable(retire_source_id)
                && $stable(retire_y)
                && $stable(retire_x);
    endproperty

    property p_retire_id_in_range;
        @(posedge clk_core) disable iff (rst_core)
            retire_valid |-> 32'(retire_source_id) < 32'(TOTAL);
    endproperty

    property p_pending_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
            perf_max_pending <= 3'd2;
    endproperty

    property p_coordinate_in_range;
        @(posedge clk_core) disable iff (rst_core)
            retire_valid
            |-> 32'(retire_y) < 32'(HEIGHT)
                && 32'(retire_x) < 32'(WIDTH);
    endproperty

    assert property (p_retire_stable_under_backpressure);
    assert property (p_retire_id_in_range);
    assert property (p_pending_is_bounded);
    assert property (p_coordinate_in_range);
endmodule

bind qfit_retirement_scheduler
    qfit_retirement_scheduler_assertions #(
        .MODE(MODE),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .Y_W(Y_W),
        .X_W(X_W),
        .SOURCE_ID_W(SOURCE_ID_W)
    ) u_qfit_retirement_scheduler_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .retire_valid(retire_valid),
        .retire_ready(retire_ready),
        .retire_source_id(retire_source_id),
        .retire_y(retire_y),
        .retire_x(retire_x),
        .perf_max_pending(perf_max_pending)
    );

`default_nettype wire
