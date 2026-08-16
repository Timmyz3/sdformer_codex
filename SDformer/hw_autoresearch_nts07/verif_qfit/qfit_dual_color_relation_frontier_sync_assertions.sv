`timescale 1ns/1ps
`default_nettype none

module qfit_relation_frontier_sync_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int K_W = 32,
    parameter int GATE_W = 9,
    parameter int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH)
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
    input logic protocol_error,
    input logic read_pending_q,
    input logic k_read_data_valid
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_valid && !descriptor_ready
        |=> descriptor_valid && $stable({
            descriptor_source_id, descriptor_plane, descriptor_y,
            descriptor_x, descriptor_k, descriptor_incoming_gates,
            descriptor_valid_mask, descriptor_last
        })
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_valid |-> descriptor_k != 0 && descriptor_valid_mask != 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        k_read_data_valid |-> read_pending_q
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error
    );
endmodule

bind qfit_dual_color_relation_frontier_sync
    qfit_relation_frontier_sync_assertions #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .K_W(K_W), .GATE_W(GATE_W), .SOURCE_ID_W(SOURCE_ID_W),
        .PLANE_W(PLANE_W), .Y_W(Y_W), .X_W(X_W)
    ) u_qfit_relation_frontier_sync_assertions (.*);

module qfit_sync_relation_bank_assertions #(
    parameter int DEPTH = 450,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input logic clk_core,
    input logic rst_core,
    input logic write_valid,
    input logic [ADDR_W-1:0] write_addr,
    input logic read_valid,
    input logic [ADDR_W-1:0] read_addr
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        write_valid |-> 32'(write_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        read_valid |-> 32'(read_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        !(write_valid && read_valid)
    );
endmodule

bind qfit_sync_relation_bank qfit_sync_relation_bank_assertions #(
    .DEPTH(DEPTH), .ADDR_W(ADDR_W)
) u_qfit_sync_relation_bank_assertions (
    .clk_core(clk_core), .rst_core(rst_core),
    .write_valid(write_valid), .write_addr(write_addr),
    .read_valid(read_valid), .read_addr(read_addr)
);

`default_nettype wire
