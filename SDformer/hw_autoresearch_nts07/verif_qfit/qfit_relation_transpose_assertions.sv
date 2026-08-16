`timescale 1ns/1ps
`default_nettype none

module qfit_relation_transpose_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int K_W = 32,
    parameter int GATE_W = 9,
    parameter int SCHED_MODE = 0,
    parameter bit SKIP_ZERO_K = 1'b0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input logic clk_core,
    input logic rst_core,
    input logic plane_start_fire,
    input logic plane_idle,
    input logic write_fire,
    input logic [Y_W-1:0] in_y,
    input logic [X_W-1:0] in_x,
    input logic [K_W-1:0] in_k_self,
    input logic [4:0] scheduler_candidate_valid,
    input logic [2:0] fcsr_active_events,
    input logic [$clog2(HEIGHT*WIDTH+1)-1:0] accepted_tokens_q,
    input logic plane_input_complete_q,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [SOURCE_ID_W-1:0] descriptor_source_id,
    input logic [Y_W-1:0] descriptor_y,
    input logic [X_W-1:0] descriptor_x,
    input logic [K_W-1:0] descriptor_k,
    input logic [5*GATE_W-1:0] descriptor_incoming_gates,
    input logic [4:0] descriptor_valid_mask,
    input logic read_inflight_q,
    input logic retire_valid,
    input logic retire_ready,
    input logic retire_k_active,
    input logic read_issue,
    input logic read_response,
    input logic [1:0] fifo_count_q,
    input logic fifo_pop,
    input logic k_rd_valid,
    input logic self_rd_valid,
    input logic n_rd_valid,
    input logic s_rd_valid,
    input logic e_rd_valid,
    input logic w_rd_valid
);
    property p_descriptor_stable;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_valid && !descriptor_ready
            |=> descriptor_valid
                && $stable(descriptor_source_id)
                && $stable(descriptor_y)
                && $stable(descriptor_x)
                && $stable(descriptor_k)
                && $stable(descriptor_incoming_gates)
                && $stable(descriptor_valid_mask);
    endproperty

    property p_boundary_mask;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_valid
            |-> (descriptor_y != 0 || !descriptor_valid_mask[2])
                && (
                    descriptor_y != HEIGHT - 1
                    || !descriptor_valid_mask[1]
                )
                && (
                    descriptor_x != WIDTH - 1
                    || !descriptor_valid_mask[3]
                )
                && (
                    descriptor_x != 0
                    || !descriptor_valid_mask[4]
                );
    endproperty

    property p_plane_start_only_when_idle;
        @(posedge clk_core) disable iff (rst_core)
            plane_start_fire |-> plane_idle;
    endproperty

    property p_input_is_raster_ordered;
        @(posedge clk_core) disable iff (rst_core)
            write_fire
            |-> 32'(in_y) * WIDTH + 32'(in_x)
                == 32'(accepted_tokens_q);
    endproperty

    property p_input_count_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
            32'(accepted_tokens_q) <= HEIGHT * WIDTH;
    endproperty

    property p_complete_has_exact_count;
        @(posedge clk_core) disable iff (rst_core)
            plane_input_complete_q
            |-> 32'(accepted_tokens_q) == HEIGHT * WIDTH;
    endproperty

    property p_fifo_never_overflows;
        @(posedge clk_core) disable iff (rst_core)
            fifo_count_q <= 2;
    endproperty

    property p_response_has_space;
        @(posedge clk_core) disable iff (rst_core)
            read_response |-> fifo_count_q < 2 || fifo_pop;
    endproperty

    property p_six_bank_response_is_atomic;
        @(posedge clk_core) disable iff (rst_core)
            (
                k_rd_valid
                || self_rd_valid
                || n_rd_valid
                || s_rd_valid
                || e_rd_valid
                || w_rd_valid
            )
            |-> read_inflight_q
                && k_rd_valid
                && self_rd_valid
                && n_rd_valid
                && s_rd_valid
                && e_rd_valid
                && w_rd_valid;
    endproperty

    property p_zero_k_retirement_skips_read;
        @(posedge clk_core) disable iff (rst_core)
            SKIP_ZERO_K && retire_valid && retire_ready && !retire_k_active
            |-> !read_issue;
    endproperty

    property p_active_retirement_issues_read;
        @(posedge clk_core) disable iff (rst_core)
            SKIP_ZERO_K && retire_valid && retire_ready && retire_k_active
            |-> read_issue;
    endproperty

    property p_fcsr_active_event_filter_is_exact;
        @(posedge clk_core) disable iff (rst_core)
            SKIP_ZERO_K && SCHED_MODE == 0 && write_fire
            |-> scheduler_candidate_valid[2:0] == fcsr_active_events;
    endproperty

    property p_fcsr_final_self_uses_current_k;
        @(posedge clk_core) disable iff (rst_core)
            SKIP_ZERO_K && SCHED_MODE == 0 && write_fire
                && 32'(in_y) == HEIGHT - 1
                && 32'(in_x) == WIDTH - 1
            |-> scheduler_candidate_valid[2] == (in_k_self != '0);
    endproperty

    assert property (p_descriptor_stable);
    assert property (p_boundary_mask);
    assert property (p_plane_start_only_when_idle);
    assert property (p_input_is_raster_ordered);
    assert property (p_input_count_is_bounded);
    assert property (p_complete_has_exact_count);
    assert property (p_fifo_never_overflows);
    assert property (p_response_has_space);
    assert property (p_six_bank_response_is_atomic);
    assert property (p_zero_k_retirement_skips_read);
    assert property (p_active_retirement_issues_read);
    assert property (p_fcsr_active_event_filter_is_exact);
    assert property (p_fcsr_final_self_uses_current_k);
endmodule

bind qfit_relation_transpose_leaf
    qfit_relation_transpose_assertions #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(K_W),
        .GATE_W(GATE_W),
        .SCHED_MODE(SCHED_MODE),
        .SKIP_ZERO_K(SKIP_ZERO_K),
        .Y_W(Y_W),
        .X_W(X_W),
        .SOURCE_ID_W(SOURCE_ID_W)
    ) u_qfit_relation_transpose_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .plane_start_fire(plane_start_fire),
        .plane_idle(plane_idle),
        .write_fire(write_fire),
        .in_y(in_y),
        .in_x(in_x),
        .in_k_self(in_k_self),
        .scheduler_candidate_valid(scheduler_candidate_valid),
        .fcsr_active_events(fcsr_active_events),
        .accepted_tokens_q(accepted_tokens_q),
        .plane_input_complete_q(plane_input_complete_q),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_incoming_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .read_inflight_q(read_inflight_q),
        .retire_valid(retire_valid),
        .retire_ready(retire_ready),
        .retire_k_active(retire_k_active),
        .read_issue(read_issue),
        .read_response(read_response),
        .fifo_count_q(fifo_count_q),
        .fifo_pop(fifo_pop),
        .k_rd_valid(k_rd_valid),
        .self_rd_valid(self_rd_valid),
        .n_rd_valid(n_rd_valid),
        .s_rd_valid(s_rd_valid),
        .e_rd_valid(e_rd_valid),
        .w_rd_valid(w_rd_valid)
    );

`default_nettype wire
