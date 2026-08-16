`timescale 1ns/1ps
`default_nettype none

module qfit_retirement_scheduler_openproxy #(
    parameter int MODE = 0,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int DILATION = 1,
    parameter int STRIPE_RING_ROWS = 4,
    parameter int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH)
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   plane_start,
    input  logic [PLANE_W-1:0]     plane_id,
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [Y_W-1:0]         in_y,
    input  logic [X_W-1:0]         in_x,
    input  logic [4:0]             in_candidate_valid,
    output logic                   retire_valid,
    input  logic                   retire_ready,
    output logic [SOURCE_ID_W-1:0] retire_source_id,
    output logic [Y_W-1:0]         retire_y,
    output logic [X_W-1:0]         retire_x,
    output logic                   plane_idle,
    output logic [31:0]            perf_producer_stalls,
    output logic [2:0]             perf_max_pending
);
    qfit_retirement_scheduler #(
        .MODE(MODE),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .DILATION(DILATION),
        .FILTER_FCSR_EVENTS(MODE == 0),
        .STRIPE_EARLY_FILL(MODE == 2),
        .STRIPE_RING_ROWS(STRIPE_RING_ROWS)
    ) u_scheduler (.*);
endmodule

module qfit_fcsr_scheduler_openproxy (
    input  logic clk_core, rst_core, plane_start, in_valid, retire_ready,
    input  logic [0:0] plane_id,
    input  logic [3:0] in_y, in_x,
    input  logic [4:0] in_candidate_valid,
    output logic in_ready, retire_valid, plane_idle,
    output logic [8:0] retire_source_id,
    output logic [3:0] retire_y, retire_x,
    output logic [31:0] perf_producer_stalls,
    output logic [2:0] perf_max_pending
);
    qfit_retirement_scheduler_openproxy #(.MODE(0)) u_proxy (.*);
endmodule

module qfit_dynamic_scheduler_openproxy (
    input  logic clk_core, rst_core, plane_start, in_valid, retire_ready,
    input  logic [0:0] plane_id,
    input  logic [3:0] in_y, in_x,
    input  logic [4:0] in_candidate_valid,
    output logic in_ready, retire_valid, plane_idle,
    output logic [8:0] retire_source_id,
    output logic [3:0] retire_y, retire_x,
    output logic [31:0] perf_producer_stalls,
    output logic [2:0] perf_max_pending
);
    qfit_retirement_scheduler_openproxy #(.MODE(1)) u_proxy (.*);
endmodule

module qfit_stripe_scheduler_openproxy (
    input  logic clk_core, rst_core, plane_start, in_valid, retire_ready,
    input  logic [0:0] plane_id,
    input  logic [3:0] in_y, in_x,
    input  logic [4:0] in_candidate_valid,
    output logic in_ready, retire_valid, plane_idle,
    output logic [8:0] retire_source_id,
    output logic [3:0] retire_y, retire_x,
    output logic [31:0] perf_producer_stalls,
    output logic [2:0] perf_max_pending
);
    qfit_retirement_scheduler_openproxy #(.MODE(2)) u_proxy (.*);
endmodule

module qfit_banked_dynamic_scheduler_openproxy (
    input  logic clk_core, rst_core, plane_start, in_valid, retire_ready,
    input  logic [0:0] plane_id,
    input  logic [3:0] in_y, in_x,
    input  logic [4:0] in_candidate_valid,
    output logic in_ready, retire_valid, plane_idle,
    output logic [8:0] retire_source_id,
    output logic [3:0] retire_y, retire_x,
    output logic [31:0] perf_producer_stalls,
    output logic [2:0] perf_max_pending
);
    qfit_banked_dynamic_retirement_scheduler u_scheduler (.*);
endmodule

module qfit_crossr2_compiled_scheduler_openproxy (
    input  logic clk_core, rst_core, plane_start, in_valid, retire_ready,
    input  logic [0:0] plane_id,
    input  logic [3:0] in_y, in_x,
    input  logic [4:0] in_candidate_valid,
    output logic in_ready, retire_valid, plane_idle,
    output logic [8:0] retire_source_id,
    output logic [3:0] retire_y, retire_x,
    output logic [31:0] perf_producer_stalls,
    output logic [2:0] perf_max_pending
);
    qfit_retirement_scheduler_openproxy #(
        .MODE(0),
        .DILATION(2)
    ) u_proxy (.*);
endmodule

module qfit_crossr2_banked_dynamic_scheduler_openproxy (
    input  logic clk_core, rst_core, plane_start, in_valid, retire_ready,
    input  logic [0:0] plane_id,
    input  logic [3:0] in_y, in_x,
    input  logic [4:0] in_candidate_valid,
    output logic in_ready, retire_valid, plane_idle,
    output logic [8:0] retire_source_id,
    output logic [3:0] retire_y, retire_x,
    output logic [31:0] perf_producer_stalls,
    output logic [2:0] perf_max_pending
);
    qfit_banked_dynamic_retirement_scheduler #(
        .DILATION(2)
    ) u_scheduler (.*);
endmodule

`default_nettype wire
