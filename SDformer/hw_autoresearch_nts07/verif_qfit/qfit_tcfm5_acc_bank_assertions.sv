`timescale 1ns/1ps
`default_nettype none

module qfit_tcfm5_acc_bank_assertions #(
    parameter int DEPTH = 90,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int VEC_W = 128
) (
    input logic clk_core,
    input logic rst_core,
    input logic clear_valid,
    input logic [ADDR_W-1:0] clear_addr,
    input logic update_valid,
    input logic [ADDR_W-1:0] update_addr,
    input logic read_valid,
    input logic [ADDR_W-1:0] read_addr,
    input logic read_data_valid,
    input logic update_pipe_valid_q,
    input logic [ADDR_W-1:0] update_pipe_addr_q,
    input logic [VEC_W-1:0] write_value,
    input logic collision_forward_valid_q,
    input logic [VEC_W-1:0] collision_forward_data_q,
    input logic [VEC_W-1:0] update_base
);
    property p_ports_are_phase_exclusive;
        @(posedge clk_core) disable iff (rst_core)
            clear_valid |-> !update_valid && !read_valid;
    endproperty

    property p_single_read_port_is_not_double_booked;
        @(posedge clk_core) disable iff (rst_core)
            !(update_valid && read_valid);
    endproperty

    property p_clear_does_not_drop_pending_writeback;
        @(posedge clk_core) disable iff (rst_core)
            clear_valid |-> !update_pipe_valid_q;
    endproperty

    property p_clear_address_is_in_range;
        @(posedge clk_core) disable iff (rst_core)
            clear_valid |-> 32'(clear_addr) < DEPTH;
    endproperty

    property p_update_address_is_in_range;
        @(posedge clk_core) disable iff (rst_core)
            update_valid |-> 32'(update_addr) < DEPTH;
    endproperty

    property p_read_address_is_in_range;
        @(posedge clk_core) disable iff (rst_core)
            read_valid |-> 32'(read_addr) < DEPTH;
    endproperty

    property p_read_has_one_cycle_response;
        @(posedge clk_core) disable iff (rst_core)
            read_valid |=> read_data_valid;
    endproperty

    property p_same_cycle_raw_sets_forwarding;
        @(posedge clk_core) disable iff (rst_core)
            update_valid
            && update_pipe_valid_q
            && update_addr == update_pipe_addr_q
            |=> collision_forward_valid_q
                && collision_forward_data_q == $past(write_value);
    endproperty

    property p_forwarding_selects_exact_base;
        @(posedge clk_core) disable iff (rst_core)
            collision_forward_valid_q
            |-> update_base == collision_forward_data_q;
    endproperty

    assert property (p_ports_are_phase_exclusive);
    assert property (p_single_read_port_is_not_double_booked);
    assert property (p_clear_does_not_drop_pending_writeback);
    assert property (p_clear_address_is_in_range);
    assert property (p_update_address_is_in_range);
    assert property (p_read_address_is_in_range);
    assert property (p_read_has_one_cycle_response);
    assert property (p_same_cycle_raw_sets_forwarding);
    assert property (p_forwarding_selects_exact_base);
endmodule

bind qfit_tcfm5_acc_bank qfit_tcfm5_acc_bank_assertions #(
        .DEPTH(DEPTH),
        .ADDR_W(ADDR_W),
        .VEC_W(VEC_W)
    ) u_qfit_tcfm5_acc_bank_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .clear_valid(clear_valid),
        .clear_addr(clear_addr),
        .update_valid(update_valid),
        .update_addr(update_addr),
        .read_valid(read_valid),
        .read_addr(read_addr),
        .read_data_valid(read_data_valid),
        .update_pipe_valid_q(update_pipe_valid_q),
        .update_pipe_addr_q(update_pipe_addr_q),
        .write_value(write_value),
        .collision_forward_valid_q(collision_forward_valid_q),
        .collision_forward_data_q(collision_forward_data_q),
        .update_base(update_base)
    );

`default_nettype wire
