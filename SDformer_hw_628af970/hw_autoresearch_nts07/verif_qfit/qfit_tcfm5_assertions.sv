`timescale 1ns/1ps
`default_nettype none

module qfit_tcfm5_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
) (
    input logic clk_core,
    input logic rst_core,
    input logic term_valid,
    input logic term_ready,
    input logic [PLANE_W-1:0] term_source_plane,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic [LANE_W-1:0] term_lane,
    input logic [GATE_W-1:0] term_gate,
    input logic [4:0] term_destination_mask,
    input logic term_window_last,
    input logic window_close,
    input logic window_close_ready,
    input logic run_busy,
    input logic run_done,
    input logic weight_context_release,
    input logic weight_context_release_ready,
    input logic weight_ready,
    input logic protocol_error,
    input logic term_fire,
    input logic term_commit,
    input logic bank_update_any,
    input logic all_banks_idle,
    input logic [31:0] perf_product_terms,
    input logic [31:0] perf_destination_updates
);
    property p_invalid_term_has_no_side_effect;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && !term_commit
            |=> protocol_error
                && $stable(perf_product_terms)
                && $stable(perf_destination_updates);
    endproperty

    property p_invalid_term_has_no_bank_update;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && !term_commit |-> !bank_update_any;
    endproperty

    property p_close_enters_drain_or_done;
        @(posedge clk_core) disable iff (rst_core)
            window_close && window_close_ready |=> run_busy || run_done;
    endproperty

    property p_invalid_close_is_reported;
        @(posedge clk_core) disable iff (rst_core)
            window_close && !window_close_ready |=> protocol_error;
    endproperty

    property p_done_is_not_busy;
        @(posedge clk_core) disable iff (rst_core)
            run_done |-> !run_busy && all_banks_idle;
    endproperty

    property p_term_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && !term_ready
            |=> term_valid
                && $stable(term_source_plane)
                && $stable(term_source_y)
                && $stable(term_source_x)
                && $stable(term_lane)
                && $stable(term_gate)
                && $stable(term_destination_mask)
                && $stable(term_window_last);
    endproperty

    property p_release_requires_done;
        @(posedge clk_core) disable iff (rst_core)
            weight_context_release_ready |-> run_done;
    endproperty

    property p_release_reopens_weight_load;
        @(posedge clk_core) disable iff (rst_core)
            weight_context_release && weight_context_release_ready
            |=> weight_ready && !run_done;
    endproperty

    property p_illegal_release_is_reported;
        @(posedge clk_core) disable iff (rst_core)
            weight_context_release && !weight_context_release_ready
            |=> protocol_error;
    endproperty

    assert property (p_invalid_term_has_no_side_effect);
    assert property (p_invalid_term_has_no_bank_update);
    assert property (p_term_stable_under_backpressure);
    assert property (p_close_enters_drain_or_done);
    assert property (p_invalid_close_is_reported);
    assert property (p_done_is_not_busy);
    assert property (p_release_requires_done);
    assert property (p_release_reopens_weight_load);
    assert property (p_illegal_release_is_reported);
endmodule

bind qfit_tcfm5_projection_top qfit_tcfm5_assertions #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .PLANE_W(PLANE_W),
        .LANE_W(LANE_W)
    ) u_qfit_tcfm5_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_plane(term_source_plane),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_window_last(term_window_last),
        .window_close(window_close),
        .window_close_ready(window_close_ready),
        .run_busy(run_busy),
        .run_done(run_done),
        .weight_context_release(weight_context_release),
        .weight_context_release_ready(weight_context_release_ready),
        .weight_ready(weight_ready),
        .protocol_error(protocol_error),
        .term_fire(term_fire),
        .term_commit(term_commit),
        .bank_update_any(bank_update_any),
        .all_banks_idle(all_banks_idle),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates)
    );

`default_nettype wire
