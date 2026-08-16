`timescale 1ns/1ps
`default_nettype none

module qfit_local5_projection_tile_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES)
) (
    input logic clk_core,
    input logic rst_core,
    input logic term_valid,
    input logic [SOURCE_ID_W-1:0] term_source_id,
    input logic [PLANE_W-1:0] term_source_plane,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic projection_start,
    input logic projection_start_ready,
    input logic projection_start_fire,
    input logic projection_close,
    input logic projection_close_ready,
    input logic stream_idle,
    input logic run_active_q,
    input logic plane_open_q,
    input logic [31:0] planes_completed,
    input logic [31:0] run_descriptors,
    input logic run_protocol_error_q,
    input logic weight_valid,
    input logic backend_weight_ready,
    input logic weight_request_legal,
    input logic weight_fire,
    input logic weights_loaded_q,
    input logic [31:0] weight_count,
    input logic weight_protocol_error_q,
    input logic weight_context_release,
    input logic weight_context_release_ready,
    input logic weight_context_release_fire,
    input logic plane_start,
    input logic plane_request_legal,
    input logic in_valid,
    input logic input_request_legal,
    input logic descriptor_fire,
    input logic descriptor_attempt,
    input logic descriptor_contract_valid,
    input logic term_issue_enable,
    input logic backend_term_ready,
    input logic term_descriptor_last,
    input logic term_run_last
);
    property p_term_identity_is_consistent;
        @(posedge clk_core) disable iff (rst_core)
            term_valid
            |-> 32'(term_source_id)
                == 32'(term_source_plane) * (HEIGHT * WIDTH)
                    + 32'(term_source_y) * WIDTH
                    + 32'(term_source_x);
    endproperty

    property p_illegal_close_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            projection_close && !projection_close_ready
            |=> run_protocol_error_q;
    endproperty

    property p_illegal_start_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            projection_start && !projection_start_ready
            |=> run_protocol_error_q;
    endproperty

    property p_start_fire_matches_handshake;
        @(posedge clk_core) disable iff (rst_core)
            projection_start_fire
            == (projection_start && projection_start_ready);
    endproperty

    property p_close_ready_requires_complete_run;
        @(posedge clk_core) disable iff (rst_core)
            projection_close_ready
            |-> stream_idle
                && run_active_q
                && !plane_open_q
                && planes_completed == TIME_PLANES
                && run_descriptors == TIME_PLANES * HEIGHT * WIDTH;
    endproperty

    property p_illegal_weight_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            weight_valid && backend_weight_ready && !weight_request_legal
            |=> weight_protocol_error_q;
    endproperty

    property p_weight_fire_requires_legal_request;
        @(posedge clk_core) disable iff (rst_core)
            weight_fire |-> weight_request_legal && backend_weight_ready;
    endproperty

    property p_weights_loaded_requires_full_ledger;
        @(posedge clk_core) disable iff (rst_core)
            weights_loaded_q
            |-> weight_count == HEAD_DIM * OUT_DIM;
    endproperty

    property p_weight_release_matches_handshake;
        @(posedge clk_core) disable iff (rst_core)
            weight_context_release_fire
            == (weight_context_release && weight_context_release_ready);
    endproperty

    property p_weight_release_clears_ledger;
        @(posedge clk_core) disable iff (rst_core)
            weight_context_release_fire
            |=> !weights_loaded_q && weight_count == 0;
    endproperty

    property p_illegal_weight_release_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            weight_context_release && !weight_context_release_ready
            |=> weight_protocol_error_q;
    endproperty

    property p_illegal_plane_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            plane_start && !plane_request_legal |=> run_protocol_error_q;
    endproperty

    property p_illegal_input_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            in_valid && !input_request_legal |=> run_protocol_error_q;
    endproperty

    property p_illegal_descriptor_is_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_attempt && !descriptor_contract_valid
            |=> run_protocol_error_q;
    endproperty

    property p_illegal_descriptor_is_atomically_rejected;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_attempt && !descriptor_contract_valid
            |-> !descriptor_fire;
    endproperty

    property p_run_last_is_final_descriptor_last;
        @(posedge clk_core) disable iff (rst_core)
            term_run_last
            |-> term_descriptor_last
                && run_descriptors == TIME_PLANES * HEIGHT * WIDTH;
    endproperty

    property p_blocked_run_last_is_held;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && backend_term_ready
                && !term_issue_enable && term_run_last
            |=> term_valid && term_descriptor_last && term_run_last;
    endproperty

    assert property (p_term_identity_is_consistent);
    assert property (p_illegal_close_is_sticky_error);
    assert property (p_illegal_start_is_sticky_error);
    assert property (p_start_fire_matches_handshake);
    assert property (p_close_ready_requires_complete_run);
    assert property (p_illegal_weight_is_sticky_error);
    assert property (p_weight_fire_requires_legal_request);
    assert property (p_weights_loaded_requires_full_ledger);
    assert property (p_weight_release_matches_handshake);
    assert property (p_weight_release_clears_ledger);
    assert property (p_illegal_weight_release_is_sticky_error);
    assert property (p_illegal_plane_is_sticky_error);
    assert property (p_illegal_input_is_sticky_error);
    assert property (p_illegal_descriptor_is_sticky_error);
    assert property (p_illegal_descriptor_is_atomically_rejected);
    assert property (p_run_last_is_final_descriptor_last);
    assert property (p_blocked_run_last_is_held);
endmodule

bind qfit_local5_projection_tile
    qfit_local5_projection_tile_assertions #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .PLANE_W(PLANE_W)
    ) u_qfit_local5_projection_tile_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid),
        .term_source_id(term_source_id),
        .term_source_plane(term_source_plane),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .projection_start(projection_start),
        .projection_start_ready(projection_start_ready),
        .projection_start_fire(projection_start_fire),
        .projection_close(projection_close),
        .projection_close_ready(projection_close_ready),
        .stream_idle(stream_idle),
        .run_active_q(run_active_q),
        .plane_open_q(plane_open_q),
        .planes_completed(32'(planes_completed_q)),
        .run_descriptors(32'(run_descriptors_q)),
        .run_protocol_error_q(run_protocol_error_q),
        .weight_valid(weight_valid),
        .backend_weight_ready(backend_weight_ready),
        .weight_request_legal(weight_request_legal),
        .weight_fire(weight_fire),
        .weights_loaded_q(weights_loaded_q),
        .weight_count(32'(weight_count_q)),
        .weight_protocol_error_q(weight_protocol_error_q),
        .weight_context_release(weight_context_release),
        .weight_context_release_ready(weight_context_release_ready),
        .weight_context_release_fire(weight_context_release_fire),
        .plane_start(plane_start),
        .plane_request_legal(plane_request_legal),
        .in_valid(in_valid),
        .input_request_legal(input_request_legal),
        .descriptor_fire(descriptor_fire),
        .descriptor_attempt(descriptor_attempt),
        .descriptor_contract_valid(descriptor_contract_valid),
        .term_issue_enable(term_issue_enable),
        .backend_term_ready(backend_term_ready),
        .term_descriptor_last(term_descriptor_last),
        .term_run_last(term_run_last)
    );

`default_nettype wire
