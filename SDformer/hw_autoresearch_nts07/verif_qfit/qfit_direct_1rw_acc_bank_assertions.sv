`timescale 1ns/1ps
`default_nettype none

module qfit_direct_1rw_acc_bank_assertions #(
    parameter int DEPTH = 90,
    parameter int VEC_W = 64,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input logic clk_core,
    input logic rst_core,
    input logic run_start,
    input logic update_valid,
    input logic update_ready,
    input logic [ADDR_W-1:0] update_addr,
    input logic [VEC_W-1:0] update_delta,
    input logic flush_valid,
    input logic flush_ready,
    input logic read_valid,
    input logic read_ready,
    input logic [ADDR_W-1:0] read_addr,
    input logic protocol_error,
    input logic memory_command_valid,
    input logic [ADDR_W-1:0] memory_command_addr
);
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        update_valid && !update_ready
        |=> update_valid && $stable({update_addr, update_delta})
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        read_valid && !read_ready |=> read_valid && $stable(read_addr)
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        memory_command_valid |-> 32'(memory_command_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        update_valid |-> 32'(update_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        read_valid |-> 32'(read_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        !protocol_error
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        !(flush_valid && flush_ready && update_valid && update_ready)
    );
endmodule

bind qfit_direct_1rw_acc_bank
    qfit_direct_1rw_acc_bank_assertions #(
        .DEPTH(DEPTH), .VEC_W(VEC_W), .ADDR_W(ADDR_W)
    ) u_qfit_direct_1rw_acc_bank_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .run_start(run_start),
        .update_valid(update_valid), .update_ready(update_ready),
        .update_addr(update_addr), .update_delta(update_delta),
        .flush_valid(flush_valid), .flush_ready(flush_ready),
        .read_valid(read_valid), .read_ready(read_ready), .read_addr(read_addr),
        .protocol_error(protocol_error),
        .memory_command_valid(memory_command_valid),
        .memory_command_addr(memory_command_addr)
    );

`default_nettype wire
