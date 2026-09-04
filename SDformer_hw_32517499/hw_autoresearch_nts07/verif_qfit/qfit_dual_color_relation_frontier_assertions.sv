`timescale 1ns/1ps
`default_nettype none

module qfit_dual_color_relation_frontier_assertions #(
    parameter int SOURCE_ID_W = 9,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int K_W = 32,
    parameter int GATE_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [SOURCE_ID_W-1:0] descriptor_source_id,
    input logic [PLANE_W-1:0] descriptor_plane,
    input logic [Y_W-1:0] descriptor_y,
    input logic [X_W-1:0] descriptor_x,
    input logic [K_W-1:0] descriptor_k,
    input logic [5*GATE_W-1:0] descriptor_incoming_gates,
    input logic [4:0] descriptor_valid_mask,
    input logic descriptor_last,
    input logic protocol_error
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_valid && !descriptor_ready
        |=> $stable({
            descriptor_source_id,
            descriptor_plane,
            descriptor_y,
            descriptor_x,
            descriptor_k,
            descriptor_incoming_gates,
            descriptor_valid_mask,
            descriptor_last
        })
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_valid |-> descriptor_k != 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_valid |-> descriptor_valid_mask != 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error
    );
endmodule

bind qfit_dual_color_relation_frontier
    qfit_dual_color_relation_frontier_assertions #(
        .SOURCE_ID_W(SOURCE_ID_W),
        .PLANE_W(PLANE_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .K_W(K_W),
        .GATE_W(GATE_W)
    ) u_qfit_dual_color_relation_frontier_assertions (.*);

`default_nettype wire
