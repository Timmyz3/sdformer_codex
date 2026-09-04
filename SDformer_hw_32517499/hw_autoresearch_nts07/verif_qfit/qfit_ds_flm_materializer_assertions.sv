`timescale 1ns/1ps
`default_nettype none

module qfit_ds_flm_materializer_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int SOURCE_ID_W = 9,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int LANE_W = 5,
    parameter int TERM_COUNT_W = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [HEAD_DIM-1:0] descriptor_k,
    input logic [5*GATE_W-1:0] descriptor_incoming_gates,
    input logic [4:0] descriptor_valid_mask,
    input logic busy,
    input logic locked_mode,
    input logic [HEAD_DIM-1:0] captured_k,
    input logic [2:0] unique_count,
    input logic [2:0] gate_index,
    input logic [TERM_COUNT_W-1:0] terms_remaining,
    input logic term_valid,
    input logic term_ready,
    input logic [SOURCE_ID_W-1:0] term_source_id,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic [LANE_W-1:0] term_lane,
    input logic [GATE_W-1:0] term_gate,
    input logic [4:0] term_destination_mask,
    input logic term_last,
    input logic [31:0] perf_terms,
    input logic [31:0] perf_destination_updates
);
    function automatic logic has_nonzero_gate(
        input logic [5*GATE_W-1:0] gates,
        input logic [4:0] valid_mask
    );
        logic found;
        found = 1'b0;
        for (integer role = 0; role < 5; role = role + 1) begin
            if (
                valid_mask[role]
                && gates[role*GATE_W +: GATE_W] != '0
            )
                found = 1'b1;
        end
        has_nonzero_gate = found;
    endfunction

    property p_term_is_canonical_nonzero;
        @(posedge clk_core) disable iff (rst_core)
            term_valid
            |-> term_gate != '0
                && term_destination_mask != '0
                && 32'(term_lane) < HEAD_DIM;
    endproperty

    property p_term_lane_was_captured_active;
        @(posedge clk_core) disable iff (rst_core)
            term_valid |-> captured_k[term_lane];
    endproperty

    property p_scan_indices_in_range;
        @(posedge clk_core) disable iff (rst_core)
            term_valid
            |-> unique_count inside {[3'd1:3'd5]}
                && gate_index < unique_count
                && terms_remaining != '0;
    endproperty

    property p_last_matches_remaining_count;
        @(posedge clk_core) disable iff (rst_core)
            term_valid
            |-> term_last == (terms_remaining == TERM_COUNT_W'(1));
    endproperty

    property p_term_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && !term_ready
            |=> term_valid
                && $stable(term_source_id)
                && $stable(term_source_y)
                && $stable(term_source_x)
                && $stable(term_lane)
                && $stable(term_gate)
                && $stable(term_destination_mask)
                && $stable(term_last);
    endproperty

    property p_last_requires_valid;
        @(posedge clk_core) disable iff (rst_core)
            term_last |-> term_valid;
    endproperty

    property p_busy_owns_descriptor_port;
        @(posedge clk_core) disable iff (rst_core)
            busy |-> !descriptor_ready;
    endproperty

    property p_descriptor_ready_matches_idle;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_ready == !busy;
    endproperty

    property p_locked_mode_stable_while_busy;
        @(posedge clk_core) disable iff (rst_core)
            busy && !(term_valid && term_ready && term_last)
            |=> $stable(locked_mode);
    endproperty

    property p_nonlast_keeps_context;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && term_ready && !term_last
            |=> busy && term_valid && !descriptor_ready;
    endproperty

    property p_last_releases_context;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && term_ready && term_last
            |=> !busy && descriptor_ready && !term_valid;
    endproperty

    property p_zero_k_retires_without_term;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_valid && descriptor_ready
                && descriptor_k == '0
            |=> descriptor_ready && !term_valid;
    endproperty

    property p_zero_gate_retires_without_term;
        @(posedge clk_core) disable iff (rst_core)
            descriptor_valid && descriptor_ready
                && !has_nonzero_gate(
                    descriptor_incoming_gates,
                    descriptor_valid_mask
                )
            |=> descriptor_ready && !term_valid;
    endproperty

    property p_term_counter_tracks_handshake;
        @(posedge clk_core) disable iff (rst_core)
            $past(!rst_core)
            |-> perf_terms
                == $past(perf_terms)
                    + 32'($past(term_valid && term_ready));
    endproperty

    property p_update_counter_tracks_handshake;
        @(posedge clk_core) disable iff (rst_core)
            $past(!rst_core)
            |-> perf_destination_updates
                == $past(perf_destination_updates)
                    + (
                        $past(term_valid && term_ready)
                        ? 32'($countones(
                            $past(term_destination_mask)
                        ))
                        : 32'd0
                    );
    endproperty

    assert property (p_term_is_canonical_nonzero);
    assert property (p_term_lane_was_captured_active);
    assert property (p_scan_indices_in_range);
    assert property (p_last_matches_remaining_count);
    assert property (p_term_stable_under_backpressure);
    assert property (p_last_requires_valid);
    assert property (p_busy_owns_descriptor_port);
    assert property (p_descriptor_ready_matches_idle);
    assert property (p_locked_mode_stable_while_busy);
    assert property (p_nonlast_keeps_context);
    assert property (p_last_releases_context);
    assert property (p_zero_k_retires_without_term);
    assert property (p_zero_gate_retires_without_term);
    assert property (p_term_counter_tracks_handshake);
    assert property (p_update_counter_tracks_handshake);
endmodule

bind qfit_ds_flm_materializer
    qfit_ds_flm_materializer_assertions #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .LANE_W(LANE_W),
        .TERM_COUNT_W(TERM_COUNT_W)
    ) u_qfit_ds_flm_materializer_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_incoming_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .busy(state_q != 1'b0),
        .locked_mode(mode_q),
        .captured_k(active_k_q),
        .unique_count(unique_count_q),
        .gate_index(gate_index_q),
        .terms_remaining(terms_remaining_q),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_last),
        .perf_terms(perf_terms),
        .perf_destination_updates(perf_destination_updates)
    );

`default_nettype wire
