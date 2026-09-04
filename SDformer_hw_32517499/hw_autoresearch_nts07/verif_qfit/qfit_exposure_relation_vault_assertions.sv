`timescale 1ns/1ps
`default_nettype none

module qfit_exposure_relation_vault_assertions #(
    parameter int SOURCE_ID_W = 9,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int PTR_W = 10,
    parameter int DEPTH = 512,
    parameter int REPLAY_FIFO_DEPTH = 4,
    parameter int REPLAY_FIFO_COUNT_W = $clog2(REPLAY_FIFO_DEPTH + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic live_valid,
    input logic live_ready,
    input logic [SOURCE_ID_W-1:0] live_source_id,
    input logic [Y_W-1:0] live_y,
    input logic [X_W-1:0] live_x,
    input logic [HEAD_DIM-1:0] live_k,
    input logic [5*GATE_W-1:0] live_gates,
    input logic [4:0] live_valid_mask,
    input logic live_last,
    input logic replay_valid,
    input logic replay_ready,
    input logic [SOURCE_ID_W-1:0] replay_source_id,
    input logic [Y_W-1:0] replay_y,
    input logic [X_W-1:0] replay_x,
    input logic [HEAD_DIM-1:0] replay_k,
    input logic [5*GATE_W-1:0] replay_gates,
    input logic [4:0] replay_valid_mask,
    input logic replay_last,
    input logic replay_active_q,
    input logic mem_write_valid,
    input logic mem_read_valid,
    input logic mem_read_data_valid,
    input logic [PTR_W-1:0] committed_ptr_q,
    input logic [REPLAY_FIFO_COUNT_W-1:0] replay_fifo_count_q,
    input logic [REPLAY_FIFO_COUNT_W-1:0] read_tag_count_q,
    input logic replay_pop,
    input logic protocol_error
);
    property p_live_stable_when_blocked;
        @(posedge clk_core) disable iff (rst_core)
            live_valid && !live_ready
            |=> live_valid
                && $stable(live_source_id)
                && $stable(live_y)
                && $stable(live_x)
                && $stable(live_k)
                && $stable(live_gates)
                && $stable(live_valid_mask)
                && $stable(live_last);
    endproperty

    property p_replay_stable_when_blocked;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid && !replay_ready
            |=> replay_valid
                && $stable(replay_source_id)
                && $stable(replay_y)
                && $stable(replay_x)
                && $stable(replay_k)
                && $stable(replay_gates)
                && $stable(replay_valid_mask)
                && $stable(replay_last);
    endproperty

    property p_single_port_phase_exclusion;
        @(posedge clk_core) disable iff (rst_core)
            !(mem_write_valid && mem_read_valid);
    endproperty

    property p_fifo_reservation_bounded;
        @(posedge clk_core) disable iff (rst_core)
            32'(replay_fifo_count_q) + 32'(read_tag_count_q)
                <= REPLAY_FIFO_DEPTH;
    endproperty

    property p_response_has_tag;
        @(posedge clk_core) disable iff (rst_core)
            mem_read_data_valid |-> read_tag_count_q != 0;
    endproperty

    property p_replay_requires_active_command;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid |-> replay_active_q;
    endproperty

    property p_last_pop_finishes_stream;
        @(posedge clk_core) disable iff (rst_core)
            replay_pop && replay_last |=> !replay_active_q;
    endproperty

    property p_committed_pointer_bounded;
        @(posedge clk_core) disable iff (rst_core)
            32'(committed_ptr_q) <= DEPTH;
    endproperty

    property p_committed_pointer_monotonic_in_window;
        @(posedge clk_core) disable iff (rst_core)
            !window_start && !$past(window_start)
            |-> committed_ptr_q >= $past(committed_ptr_q);
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
            $past(protocol_error) |-> protocol_error;
    endproperty

    assert property (p_live_stable_when_blocked);
    assert property (p_replay_stable_when_blocked);
    assert property (p_single_port_phase_exclusion);
    assert property (p_fifo_reservation_bounded);
    assert property (p_response_has_tag);
    assert property (p_replay_requires_active_command);
    assert property (p_last_pop_finishes_stream);
    assert property (p_committed_pointer_bounded);
    assert property (p_committed_pointer_monotonic_in_window);
    assert property (p_protocol_error_sticky);
endmodule

bind qfit_exposure_relation_vault
    qfit_exposure_relation_vault_assertions #(
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .PTR_W(PTR_W),
        .DEPTH(DEPTH),
        .REPLAY_FIFO_DEPTH(REPLAY_FIFO_DEPTH),
        .REPLAY_FIFO_COUNT_W(REPLAY_FIFO_COUNT_W)
    ) u_qfit_exposure_relation_vault_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .live_valid(live_valid),
        .live_ready(live_ready),
        .live_source_id(live_source_id),
        .live_y(live_y),
        .live_x(live_x),
        .live_k(live_k),
        .live_gates(live_gates),
        .live_valid_mask(live_valid_mask),
        .live_last(live_last),
        .replay_valid(replay_valid),
        .replay_ready(replay_ready),
        .replay_source_id(replay_source_id),
        .replay_y(replay_y),
        .replay_x(replay_x),
        .replay_k(replay_k),
        .replay_gates(replay_gates),
        .replay_valid_mask(replay_valid_mask),
        .replay_last(replay_last),
        .replay_active_q(replay_active_q),
        .mem_write_valid(mem_write_valid),
        .mem_read_valid(mem_read_valid),
        .mem_read_data_valid(mem_read_data_valid),
        .committed_ptr_q(committed_ptr_q),
        .replay_fifo_count_q(replay_fifo_count_q),
        .read_tag_count_q(read_tag_count_q),
        .replay_pop(replay_pop),
        .protocol_error(protocol_error)
    );

`default_nettype wire
