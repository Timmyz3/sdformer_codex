`timescale 1ns/1ps
`default_nettype none

module qfit_single_port_acc_memory_assertions #(
    parameter int DEPTH = 90,
    parameter int VEC_W = 64,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_write,
    input logic [ADDR_W-1:0] command_addr,
    input logic [VEC_W-1:0] command_write_data
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        command_valid |-> 32'(command_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        command_valid && command_write |-> !$isunknown(command_write_data)
    );
endmodule

bind qfit_single_port_acc_memory
    qfit_single_port_acc_memory_assertions #(
        .DEPTH(DEPTH), .VEC_W(VEC_W), .ADDR_W(ADDR_W)
    ) u_qfit_single_port_acc_memory_assertions (.*);

`default_nettype wire
