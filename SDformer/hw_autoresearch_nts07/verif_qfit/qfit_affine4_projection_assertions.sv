`timescale 1ns/1ps
`default_nettype none

module qfit_affine4_projection_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int ACC_VEC_W = 128,
    parameter int BANK_DEPTH = 120,
    parameter int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH),
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
    input logic term_contract_ok,
    input logic term_fire,
    input logic term_update_fire,
    input logic term_conflict,
    input logic [2:0] term_valid_update_count,
    input logic [3:0] primary_update_vec,
    input logic [3:0] bank_update_valid_vec,
    input logic [ACC_VEC_W-1:0] product_vector,
    input logic replay_valid_q,
    input logic [1:0] replay_bank_q,
    input logic [BANK_ADDR_W-1:0] replay_addr_q,
    input logic [ACC_VEC_W-1:0] replay_product_q,
    input logic [1:0] north_bank,
    input logic [BANK_ADDR_W-1:0] north_addr,
    input logic all_banks_idle,
    input logic [31:0] perf_product_terms,
    input logic [31:0] perf_destination_updates,
    input logic [31:0] perf_replay_updates
);
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

    property p_legal_term_metadata;
        @(posedge clk_core) disable iff (rst_core)
            term_update_fire
            |-> 32'(term_source_plane) < TIME_PLANES
                && 32'(term_source_y) < HEIGHT
                && 32'(term_source_x) < WIDTH
                && 32'(term_lane) < HEAD_DIM
                && term_gate != '0
                && term_destination_mask != '0
                && term_contract_ok;
    endproperty

    property p_conflict_is_captured_exactly;
        @(posedge clk_core) disable iff (rst_core)
            term_update_fire && term_conflict
            |=> replay_valid_q
                && replay_bank_q == $past(north_bank)
                && replay_addr_q == $past(north_addr)
                && replay_product_q == $past(product_vector);
    endproperty

    property p_nonconflict_does_not_allocate_replay;
        @(posedge clk_core) disable iff (rst_core)
            term_update_fire && !term_conflict
            |=> !replay_valid_q;
    endproperty

    property p_replay_owns_exactly_one_bank;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid_q
            |-> !term_ready && $onehot(bank_update_valid_vec);
    endproperty

    property p_replay_retires_in_one_cycle;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid_q |=> !replay_valid_q;
    endproperty

    property p_primary_cardinality_matches_roles;
        @(posedge clk_core) disable iff (rst_core)
            term_update_fire
            |-> $countones(primary_update_vec)
                == term_valid_update_count
                    - {2'b00, term_conflict};
    endproperty

    property p_conflict_free_term_has_no_added_bubble;
        @(posedge clk_core) disable iff (rst_core)
            term_update_fire && !term_conflict && !term_window_last
            |=> term_ready;
    endproperty

    property p_zero_mask_is_rejected_without_work;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && term_destination_mask == '0
            |=> protocol_error
                && $stable(perf_product_terms)
                && $stable(perf_destination_updates)
                && $stable(perf_replay_updates);
    endproperty

    property p_close_enters_busy_drain_or_done;
        @(posedge clk_core) disable iff (rst_core)
            window_close && window_close_ready |=> run_busy || run_done;
    endproperty

    property p_invalid_close_is_reported;
        @(posedge clk_core) disable iff (rst_core)
            window_close && !window_close_ready |=> protocol_error;
    endproperty

    property p_done_has_no_pending_update;
        @(posedge clk_core) disable iff (rst_core)
            run_done |-> !run_busy && !replay_valid_q && all_banks_idle;
    endproperty

    assert property (p_term_stable_under_backpressure);
    assert property (p_legal_term_metadata);
    assert property (p_conflict_is_captured_exactly);
    assert property (p_nonconflict_does_not_allocate_replay);
    assert property (p_replay_owns_exactly_one_bank);
    assert property (p_replay_retires_in_one_cycle);
    assert property (p_primary_cardinality_matches_roles);
    assert property (p_conflict_free_term_has_no_added_bubble);
    assert property (p_zero_mask_is_rejected_without_work);
    assert property (p_close_enters_busy_drain_or_done);
    assert property (p_invalid_close_is_reported);
    assert property (p_done_has_no_pending_update);
endmodule

bind qfit_affine4_projection_top
    qfit_affine4_projection_assertions #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .ACC_VEC_W(ACC_VEC_W),
        .BANK_DEPTH(BANK_DEPTH),
        .BANK_ADDR_W(BANK_ADDR_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .PLANE_W(PLANE_W),
        .LANE_W(LANE_W)
    ) u_qfit_affine4_projection_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid),
        .term_ready(term_ready),
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
        .term_contract_ok(term_contract_ok),
        .term_fire(term_fire),
        .term_update_fire(term_update_fire),
        .term_conflict(term_conflict),
        .term_valid_update_count(term_valid_update_count),
        .primary_update_vec(primary_update_vec),
        .bank_update_valid_vec(bank_update_valid_vec),
        .product_vector(product_vector),
        .replay_valid_q(replay_valid_q),
        .replay_bank_q(replay_bank_q),
        .replay_addr_q(replay_addr_q),
        .replay_product_q(replay_product_q),
        .north_bank(role_bank[2]),
        .north_addr(role_addr[2]),
        .all_banks_idle(all_banks_idle),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates),
        .perf_replay_updates(perf_replay_updates)
    );

`default_nettype wire
