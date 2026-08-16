`timescale 1ns/1ps
`default_nettype none

module qfit_gasr2c_acc_bank_assertions #(
    parameter int DEPTH = 90,
    parameter int VEC_W = 64,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input logic clk_core,
    input logic rst_core,
    input logic run_start,
    input logic prepare_valid,
    input logic prepare_ready,
    input logic [ADDR_W-1:0] prepare_addr,
    input logic activate_valid,
    input logic activate_ready,
    input logic [ADDR_W-1:0] activate_addr,
    input logic update_valid,
    input logic update_ready,
    input logic [ADDR_W-1:0] update_addr,
    input logic [VEC_W-1:0] update_delta,
    input logic read_valid,
    input logic read_ready,
    input logic [ADDR_W-1:0] read_addr,
    input logic protocol_error,
    input logic memory_command_valid,
    input logic [ADDR_W-1:0] memory_command_addr,
    input logic active_valid_q,
    input logic active_sel_q,
    input logic prepared_valid_q,
    input logic prepared_sel_q,
    input logic [ADDR_W-1:0] prepared_addr_q,
    input logic slot_valid_q [0:1],
    input logic [ADDR_W-1:0] slot_addr_q [0:1]
);
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        prepare_valid && !prepare_ready
        |=> prepare_valid && $stable(prepare_addr)
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        activate_valid && !activate_ready
        |=> activate_valid && $stable(activate_addr)
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
        prepare_valid |-> 32'(prepare_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        activate_valid |-> 32'(activate_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        update_valid |-> 32'(update_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        read_valid |-> 32'(read_addr) < DEPTH
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        active_valid_q |-> slot_valid_q[active_sel_q]
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        prepared_valid_q |-> (
            slot_valid_q[prepared_sel_q]
            && slot_addr_q[prepared_sel_q] == prepared_addr_q
        )
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        !(slot_valid_q[0] && slot_valid_q[1]
          && slot_addr_q[0] == slot_addr_q[1])
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        update_valid && update_ready |-> (
            active_valid_q && slot_addr_q[active_sel_q] == update_addr
        )
    );
    assert property (@(posedge clk_core) disable iff (rst_core || run_start)
        !protocol_error
    );
endmodule

bind qfit_gasr2c_acc_bank
    qfit_gasr2c_acc_bank_assertions #(
        .DEPTH(DEPTH), .VEC_W(VEC_W), .ADDR_W(ADDR_W)
    ) u_qfit_gasr2c_acc_bank_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .run_start(run_start),
        .prepare_valid(prepare_valid), .prepare_ready(prepare_ready),
        .prepare_addr(prepare_addr), .activate_valid(activate_valid),
        .activate_ready(activate_ready), .activate_addr(activate_addr),
        .update_valid(update_valid), .update_ready(update_ready),
        .update_addr(update_addr), .update_delta(update_delta),
        .read_valid(read_valid), .read_ready(read_ready), .read_addr(read_addr),
        .protocol_error(protocol_error),
        .memory_command_valid(memory_command_valid),
        .memory_command_addr(memory_command_addr),
        .active_valid_q(active_valid_q), .active_sel_q(active_sel_q),
        .prepared_valid_q(prepared_valid_q), .prepared_sel_q(prepared_sel_q),
        .prepared_addr_q(prepared_addr_q), .slot_valid_q(slot_valid_q),
        .slot_addr_q(slot_addr_q)
    );

`default_nettype wire
