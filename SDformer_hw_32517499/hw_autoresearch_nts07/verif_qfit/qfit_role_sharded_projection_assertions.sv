`timescale 1ns/1ps
`default_nettype none

module qfit_role_sharded_projection_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int ROLE_DEPTH = TIME_PLANES * HEIGHT * WIDTH,
    parameter int ROLE_ADDR_W =
        (ROLE_DEPTH <= 1) ? 1 : $clog2(ROLE_DEPTH),
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
    input logic term_fire,
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
    input logic protocol_error,
    input logic term_contract_valid,
    input logic [31:0] term_update_count,
    input logic [31:0] perf_product_terms,
    input logic [31:0] perf_destination_updates,
    input logic role_valid_0,
    input logic role_valid_1,
    input logic role_valid_2,
    input logic role_valid_3,
    input logic role_valid_4,
    input logic role_update_valid_0,
    input logic role_update_valid_1,
    input logic role_update_valid_2,
    input logic role_update_valid_3,
    input logic role_update_valid_4,
    input logic [ROLE_ADDR_W-1:0] role_addr_0,
    input logic [ROLE_ADDR_W-1:0] role_addr_1,
    input logic [ROLE_ADDR_W-1:0] role_addr_2,
    input logic [ROLE_ADDR_W-1:0] role_addr_3,
    input logic [ROLE_ADDR_W-1:0] role_addr_4,
    input logic read_fire,
    input logic read_data_valid,
    input logic draining,
    input logic all_roles_idle
);
    property p_term_contract;
        @(posedge clk_core) disable iff (rst_core)
            term_valid
            |-> 32'(term_source_plane) < TIME_PLANES
                && 32'(term_source_y) < HEIGHT
                && 32'(term_source_x) < WIDTH
                && 32'(term_lane) < HEAD_DIM
                && term_gate != '0
                && term_destination_mask != '0;
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

    property p_no_bubble_after_normal_term;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && term_contract_valid && !term_window_last
            |=> term_ready;
    endproperty

    property p_full_mask_updates_five_legal_roles;
        @(posedge clk_core) disable iff (rst_core)
            term_fire
            && term_contract_valid
            && term_destination_mask == 5'b11111
            && role_valid_0
            && role_valid_1
            && role_valid_2
            && role_valid_3
            && role_valid_4
            |-> role_update_valid_0
                && role_update_valid_1
                && role_update_valid_2
                && role_update_valid_3
                && role_update_valid_4;
    endproperty

    property p_product_counter_exact;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && term_contract_valid
            |=> perf_product_terms == $past(perf_product_terms) + 1'b1;
    endproperty

    property p_update_counter_exact;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && term_contract_valid
            |=> perf_destination_updates
                == $past(perf_destination_updates)
                + $past(term_update_count);
    endproperty

    property p_close_enters_drain_or_done;
        @(posedge clk_core) disable iff (rst_core)
            window_close && window_close_ready |=> run_busy || run_done;
    endproperty

    property p_invalid_close_is_reported;
        @(posedge clk_core) disable iff (rst_core)
            window_close && !window_close_ready |=> protocol_error;
    endproperty

    property p_drain_waits_for_idle;
        @(posedge clk_core) disable iff (rst_core)
            draining && !all_roles_idle |=> !run_done;
    endproperty

    property p_idle_drain_completes;
        @(posedge clk_core) disable iff (rst_core)
            draining && all_roles_idle |=> run_done;
    endproperty

    property p_read_has_synchronous_response;
        @(posedge clk_core) disable iff (rst_core)
            read_fire |=> read_data_valid;
    endproperty

    property p_done_is_not_busy;
        @(posedge clk_core) disable iff (rst_core)
            run_done |-> !run_busy;
    endproperty

    property p_role_0_address;
        @(posedge clk_core) disable iff (rst_core)
            role_update_valid_0 |-> 32'(role_addr_0) < ROLE_DEPTH;
    endproperty

    property p_role_1_address;
        @(posedge clk_core) disable iff (rst_core)
            role_update_valid_1 |-> 32'(role_addr_1) < ROLE_DEPTH;
    endproperty

    property p_role_2_address;
        @(posedge clk_core) disable iff (rst_core)
            role_update_valid_2 |-> 32'(role_addr_2) < ROLE_DEPTH;
    endproperty

    property p_role_3_address;
        @(posedge clk_core) disable iff (rst_core)
            role_update_valid_3 |-> 32'(role_addr_3) < ROLE_DEPTH;
    endproperty

    property p_role_4_address;
        @(posedge clk_core) disable iff (rst_core)
            role_update_valid_4 |-> 32'(role_addr_4) < ROLE_DEPTH;
    endproperty

    assert property (p_term_contract);
    assert property (p_term_stable_under_backpressure);
    assert property (p_no_bubble_after_normal_term);
    assert property (p_full_mask_updates_five_legal_roles);
    assert property (p_product_counter_exact);
    assert property (p_update_counter_exact);
    assert property (p_close_enters_drain_or_done);
    assert property (p_invalid_close_is_reported);
    assert property (p_drain_waits_for_idle);
    assert property (p_idle_drain_completes);
    assert property (p_read_has_synchronous_response);
    assert property (p_done_is_not_busy);
    assert property (p_role_0_address);
    assert property (p_role_1_address);
    assert property (p_role_2_address);
    assert property (p_role_3_address);
    assert property (p_role_4_address);
endmodule

bind qfit_role_sharded_projection_top
    qfit_role_sharded_projection_assertions #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .ROLE_DEPTH(ROLE_DEPTH),
        .ROLE_ADDR_W(ROLE_ADDR_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .PLANE_W(PLANE_W),
        .LANE_W(LANE_W)
    ) u_qfit_role_sharded_projection_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_fire(term_fire),
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
        .protocol_error(protocol_error),
        .term_contract_valid(term_contract_valid),
        .term_update_count(term_update_count),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates),
        .role_valid_0(role_valid[0]),
        .role_valid_1(role_valid[1]),
        .role_valid_2(role_valid[2]),
        .role_valid_3(role_valid[3]),
        .role_valid_4(role_valid[4]),
        .role_update_valid_0(role_update_valid[0]),
        .role_update_valid_1(role_update_valid[1]),
        .role_update_valid_2(role_update_valid[2]),
        .role_update_valid_3(role_update_valid[3]),
        .role_update_valid_4(role_update_valid[4]),
        .role_addr_0(role_addr[0]),
        .role_addr_1(role_addr[1]),
        .role_addr_2(role_addr[2]),
        .role_addr_3(role_addr[3]),
        .role_addr_4(role_addr[4]),
        .read_fire(read_fire),
        .read_data_valid(read_data_valid),
        .draining(state_q == 3'd3),
        .all_roles_idle(all_roles_idle)
    );

`default_nettype wire
