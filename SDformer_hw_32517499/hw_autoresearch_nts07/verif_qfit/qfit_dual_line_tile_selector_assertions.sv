`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_tile_selector_assertions #(
    parameter int TAG_W = 24,
    parameter int COUNT_W = 16,
    parameter int PERF_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic decision_valid,
    input logic decision_ready,
    input logic [TAG_W-1:0] decision_tag,
    input logic decision_use_motion,
    input logic decision_seed_previous,
    input logic [COUNT_W:0] decision_work_count,
    input logic [COUNT_W:0] decision_local_work_count,
    input logic [COUNT_W:0] decision_transition_work_count,
    input logic decision_force_local,
    input logic decision_counts_legal,
    input logic protocol_error,
    input logic [PERF_W-1:0] perf_decisions,
    input logic [PERF_W-1:0] perf_local_decisions,
    input logic [PERF_W-1:0] perf_motion_decisions
);
    property p_decision_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            decision_valid && !decision_ready
            |=> decision_valid
                && $stable(decision_tag)
                && $stable(decision_use_motion)
                && $stable(decision_seed_previous)
                && $stable(decision_work_count)
                && $stable(decision_local_work_count)
                && $stable(decision_transition_work_count)
                && $stable(decision_force_local)
                && $stable(decision_counts_legal);
    endproperty

    property p_motion_is_strictly_less_work;
        @(posedge clk_core) disable iff (rst_core)
            decision_valid && decision_use_motion
            |-> decision_counts_legal && !decision_force_local
                && decision_transition_work_count < decision_local_work_count
                && decision_work_count == decision_transition_work_count
                && decision_seed_previous;
    endproperty

    property p_local_is_forced_or_no_worse;
        @(posedge clk_core) disable iff (rst_core)
            decision_valid && !decision_use_motion
            |-> decision_force_local
                || decision_local_work_count <= decision_transition_work_count;
    endproperty

    property p_local_work_count_is_selected;
        @(posedge clk_core) disable iff (rst_core)
            decision_valid && !decision_use_motion
            |-> decision_work_count == decision_local_work_count
                && !decision_seed_previous;
    endproperty

    property p_illegal_count_fails_safe;
        @(posedge clk_core) disable iff (rst_core)
            decision_valid && !decision_counts_legal
            |-> decision_force_local && !decision_use_motion;
    endproperty

    property p_protocol_error_is_sticky;
        @(posedge clk_core) disable iff (rst_core)
            $past(protocol_error) |-> protocol_error;
    endproperty

    property p_decision_counters_balance;
        @(posedge clk_core) disable iff (rst_core)
            perf_decisions == perf_local_decisions + perf_motion_decisions;
    endproperty

    assert property (p_decision_stable_under_backpressure);
    assert property (p_motion_is_strictly_less_work);
    assert property (p_local_is_forced_or_no_worse);
    assert property (p_local_work_count_is_selected);
    assert property (p_illegal_count_fails_safe);
    assert property (p_protocol_error_is_sticky);
    assert property (p_decision_counters_balance);

    cover property (@(posedge clk_core) disable iff (rst_core)
        decision_valid && decision_ready && decision_use_motion);
    cover property (@(posedge clk_core) disable iff (rst_core)
        decision_valid && decision_ready && decision_force_local);
    cover property (@(posedge clk_core) disable iff (rst_core)
        decision_valid && decision_ready
        && decision_local_work_count == decision_transition_work_count);
endmodule

bind qfit_dual_line_tile_selector
    qfit_dual_line_tile_selector_assertions #(
        .TAG_W(TAG_W), .COUNT_W(COUNT_W), .PERF_W(PERF_W)
    ) u_qfit_dual_line_tile_selector_assertions (
        .clk_core(clk_core), .rst_core(rst_core),
        .decision_valid(decision_valid), .decision_ready(decision_ready),
        .decision_tag(decision_tag),
        .decision_use_motion(decision_use_motion),
        .decision_seed_previous(decision_seed_previous),
        .decision_work_count(decision_work_count),
        .decision_local_work_count(decision_local_work_count),
        .decision_transition_work_count(decision_transition_work_count),
        .decision_force_local(decision_force_local),
        .decision_counts_legal(decision_counts_legal),
        .protocol_error(protocol_error),
        .perf_decisions(perf_decisions),
        .perf_local_decisions(perf_local_decisions),
        .perf_motion_decisions(perf_motion_decisions)
    );

`default_nettype wire
