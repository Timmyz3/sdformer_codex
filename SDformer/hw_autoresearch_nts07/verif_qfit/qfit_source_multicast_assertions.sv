`timescale 1ns/1ps
`default_nettype none

module qfit_source_multicast_assertions #(
    parameter int GATE_W = 9,
    parameter int SOURCE_ID_W = 9,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int LANE_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic term_valid,
    input logic term_ready,
    input logic [SOURCE_ID_W-1:0] term_source_id,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic [LANE_W-1:0] term_lane,
    input logic [GATE_W-1:0] term_gate,
    input logic [4:0] term_destination_mask,
    input logic term_last
);
    property p_term_is_nonzero;
        @(posedge clk_core) disable iff (rst_core)
            term_valid
            |-> term_gate != '0
                && term_destination_mask != '0;
    endproperty

    property p_term_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && !term_ready
            |=> term_valid
                && $stable(term_source_id)
                && $stable(term_source_y)
                && $stable(term_source_x)
                && $stable(term_lane)
                && $stable(term_gate)
                && $stable(term_destination_mask)
                && $stable(term_last);
    endproperty

    assert property (p_term_is_nonzero);
    assert property (p_term_stable_under_backpressure);
endmodule

bind qfit_source_multicast_term_builder
    qfit_source_multicast_assertions #(
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .LANE_W(LANE_W)
    ) u_qfit_source_multicast_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_last)
    );

`default_nettype wire
