`timescale 1ns/1ps
`default_nettype none

// Exact per-tile choice between a current-activation Local replay and a
// signed temporal Motion update.  The upstream descriptor builder must emit
// current one-bits for Local or 0->1 / 1->0 transitions for Motion.  This
// block deliberately does not predict transition density from aggregate
// sparsity; both counts must arrive with the tile command.
module qfit_dual_line_tile_selector #(
    parameter int TAG_W = 24,
    parameter int COUNT_W = 16,
    parameter int PERF_W = 32
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     request_valid,
    output logic                     request_ready,
    input  logic [TAG_W-1:0]         request_tag,
    input  logic [COUNT_W-1:0]       request_valid_bits,
    input  logic [COUNT_W-1:0]       request_current_nonzero,
    input  logic [COUNT_W-1:0]       request_positive_transitions,
    input  logic [COUNT_W-1:0]       request_negative_transitions,
    input  logic                     request_prior_state_valid,
    input  logic                     request_sequence_boundary,
    input  logic                     request_force_refresh,

    output logic                     decision_valid,
    input  logic                     decision_ready,
    output logic [TAG_W-1:0]         decision_tag,
    output logic                     decision_use_motion,
    output logic                     decision_seed_previous,
    output logic [COUNT_W:0]         decision_work_count,
    output logic [COUNT_W:0]         decision_local_work_count,
    output logic [COUNT_W:0]         decision_transition_work_count,
    output logic                     decision_force_local,
    output logic                     decision_counts_legal,

    output logic                     protocol_error,
    output logic [PERF_W-1:0]        perf_decisions,
    output logic [PERF_W-1:0]        perf_local_decisions,
    output logic [PERF_W-1:0]        perf_motion_decisions,
    output logic [PERF_W-1:0]        perf_local_work,
    output logic [PERF_W-1:0]        perf_transition_work,
    output logic [PERF_W-1:0]        perf_selected_work
);
    logic [COUNT_W:0] transition_work;
    logic counts_legal;
    logic force_local;
    logic choose_motion;
    logic request_fire;
    logic decision_fire;

    assign transition_work = {1'b0, request_positive_transitions}
                           + {1'b0, request_negative_transitions};
    assign counts_legal = ({1'b0, request_current_nonzero}
                           <= {1'b0, request_valid_bits})
                       && (transition_work <= {1'b0, request_valid_bits});
    assign force_local = !request_prior_state_valid
                      || request_sequence_boundary
                      || request_force_refresh
                      || !counts_legal;
    assign choose_motion = !force_local
                        && transition_work
                           < {1'b0, request_current_nonzero};

    assign request_ready = !decision_valid || decision_ready;
    assign request_fire = request_valid && request_ready;
    assign decision_fire = decision_valid && decision_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            decision_valid <= 1'b0;
            decision_tag <= '0;
            decision_use_motion <= 1'b0;
            decision_seed_previous <= 1'b0;
            decision_work_count <= '0;
            decision_local_work_count <= '0;
            decision_transition_work_count <= '0;
            decision_force_local <= 1'b1;
            decision_counts_legal <= 1'b1;
            protocol_error <= 1'b0;
            perf_decisions <= '0;
            perf_local_decisions <= '0;
            perf_motion_decisions <= '0;
            perf_local_work <= '0;
            perf_transition_work <= '0;
            perf_selected_work <= '0;
        end else begin
            if (decision_fire)
                decision_valid <= 1'b0;

            if (request_fire) begin
                decision_valid <= 1'b1;
                decision_tag <= request_tag;
                decision_use_motion <= choose_motion;
                decision_seed_previous <= choose_motion;
                decision_local_work_count <= {1'b0, request_current_nonzero};
                decision_transition_work_count <= transition_work;
                decision_work_count <= choose_motion
                    ? transition_work : {1'b0, request_current_nonzero};
                decision_force_local <= force_local;
                decision_counts_legal <= counts_legal;
                if (!counts_legal)
                    protocol_error <= 1'b1;
            end

            if (decision_fire) begin
                perf_decisions <= perf_decisions + 1'b1;
                perf_local_work <= perf_local_work
                    + PERF_W'(decision_local_work_count);
                perf_transition_work <= perf_transition_work
                    + PERF_W'(decision_transition_work_count);
                perf_selected_work <= perf_selected_work
                    + PERF_W'(decision_work_count);
                if (decision_use_motion)
                    perf_motion_decisions <= perf_motion_decisions + 1'b1;
                else
                    perf_local_decisions <= perf_local_decisions + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
