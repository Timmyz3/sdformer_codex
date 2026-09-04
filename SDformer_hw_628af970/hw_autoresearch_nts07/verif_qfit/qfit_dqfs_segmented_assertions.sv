`timescale 1ns/1ps
`default_nettype none

module qfit_dqfs_segmented_assertions #(
    parameter int GATE_W = 9,
    parameter int EPOCH_W = 4,
    parameter int TILE_W = 4,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int DEST_MASK_W = 5,
    parameter int COUNT_W = 5,
    parameter int LANE_W = 2
) (
    input logic clk_core,
    input logic rst_core,
    input logic txn_done,
    input logic group_valid,
    input logic group_ready,
    input logic [LANE_W-1:0] group_lane,
    input logic [GATE_W-1:0] group_gate,
    input logic [EPOCH_W-1:0] group_epoch,
    input logic [TILE_W-1:0] group_output_tile,
    input logic [COUNT_W-1:0] group_member_count,
    input logic member_valid,
    input logic member_ready,
    input logic [PLANE_W-1:0] member_source_plane,
    input logic [Y_W-1:0] member_source_y,
    input logic [X_W-1:0] member_source_x,
    input logic [DEST_MASK_W-1:0] member_destination_mask,
    input logic member_group_last,
    input logic member_row_last,
    input logic member_window_last,
    input logic [31:0] perf_accepted_terms,
    input logic [31:0] perf_emitted_members,
    input logic seal_request,
    input logic in_fire,
    input logic read_pending_q,
    input logic member_valid_q
);
    property p_group_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
            group_valid && !group_ready
            |=> group_valid
                && $stable({
                    group_lane,
                    group_gate,
                    group_epoch,
                    group_output_tile,
                    group_member_count
                });
    endproperty

    property p_member_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
            member_valid && !member_ready
            |=> member_valid
                && $stable({
                    member_source_plane,
                    member_source_y,
                    member_source_x,
                    member_destination_mask,
                    member_group_last,
                    member_row_last,
                    member_window_last
                });
    endproperty

    property p_nonzero_group;
        @(posedge clk_core) disable iff (rst_core)
            group_valid |-> group_member_count != '0;
    endproperty

    property p_emitted_not_ahead;
        @(posedge clk_core) disable iff (rst_core)
            perf_emitted_members <= perf_accepted_terms;
    endproperty

    property p_done_conserves_terms;
        @(posedge clk_core) disable iff (rst_core)
            txn_done |-> perf_emitted_members == perf_accepted_terms;
    endproperty

    property p_last_hierarchy;
        @(posedge clk_core) disable iff (rst_core)
            member_valid && (member_row_last || member_window_last)
            |-> member_group_last;
    endproperty

    property p_seal_does_not_accept_trigger;
        @(posedge clk_core) disable iff (rst_core)
            seal_request |-> !in_fire;
    endproperty

    property p_single_read_or_held_member;
        @(posedge clk_core) disable iff (rst_core)
            !(read_pending_q && member_valid_q);
    endproperty

    assert property (p_group_stable_under_stall);
    assert property (p_member_stable_under_stall);
    assert property (p_nonzero_group);
    assert property (p_emitted_not_ahead);
    assert property (p_done_conserves_terms);
    assert property (p_last_hierarchy);
    assert property (p_seal_does_not_accept_trigger);
    assert property (p_single_read_or_held_member);
endmodule

bind qfit_dqfs_segmented_leaf qfit_dqfs_segmented_assertions #(
    .GATE_W(GATE_W),
    .EPOCH_W(EPOCH_W),
    .TILE_W(TILE_W),
    .PLANE_W(PLANE_W),
    .Y_W(Y_W),
    .X_W(X_W),
    .DEST_MASK_W(DEST_MASK_W),
    .COUNT_W(COUNT_W),
    .LANE_W(LANE_W)
) u_qfit_dqfs_segmented_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .txn_done(txn_done),
    .group_valid(group_valid),
    .group_ready(group_ready),
    .group_lane(group_lane),
    .group_gate(group_gate),
    .group_epoch(group_epoch),
    .group_output_tile(group_output_tile),
    .group_member_count(group_member_count),
    .member_valid(member_valid),
    .member_ready(member_ready),
    .member_source_plane(member_source_plane),
    .member_source_y(member_source_y),
    .member_source_x(member_source_x),
    .member_destination_mask(member_destination_mask),
    .member_group_last(member_group_last),
    .member_row_last(member_row_last),
    .member_window_last(member_window_last),
    .perf_accepted_terms(perf_accepted_terms),
    .perf_emitted_members(perf_emitted_members),
    .seal_request(seal_request),
    .in_fire(in_fire),
    .read_pending_q(read_pending_q),
    .member_valid_q(member_valid_q)
);

`default_nettype wire
