`timescale 1ns/1ps
`default_nettype none

module qfit_temporal_destination_commit_engine_assertions #(
    parameter int CTX_W = 2,
    parameter int LANE_TILE_W = 5,
    parameter int EPOCH_W = 16,
    parameter int DOMAIN_W = 8,
    parameter int STEP_W = 4,
    parameter int LEN_W = 4,
    parameter int TAG_W = 32,
    parameter int LANES = 16,
    parameter int ACC_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic [DOMAIN_W-1:0] active_domain,
    input logic commit_valid,
    input logic commit_ready,
    input logic [CTX_W-1:0] commit_context,
    input logic [LANE_TILE_W-1:0] commit_lane_tile,
    input logic [EPOCH_W-1:0] commit_epoch,
    input logic [DOMAIN_W-1:0] commit_domain,
    input logic [STEP_W-1:0] commit_temporal_step,
    input logic [LEN_W-1:0] commit_temporal_length,
    input logic commit_temporal_first,
    input logic commit_temporal_last,
    input logic commit_use_motion,
    input logic [TAG_W-1:0] commit_tag,
    input logic [(LANES*ACC_W)-1:0] commit_acc,
    input logic [(LANES*ACC_W)-1:0] commit_prior_acc,
    input logic protocol_error,
    input logic abort_valid,
    input logic abort_ready,
    input logic abort_error,
    input logic [CTX_W-1:0] abort_context,
    input logic [LANE_TILE_W-1:0] abort_lane_tile,
    input logic [EPOCH_W-1:0] abort_epoch,
    input logic [DOMAIN_W-1:0] abort_domain,
    input logic [TAG_W-1:0] abort_tag,
    input logic commit_fire,
    input logic abort_fire,
    input logic output_valid,
    input logic output_ready,
    input logic [CTX_W-1:0] output_context,
    input logic [LANE_TILE_W-1:0] output_lane_tile,
    input logic [EPOCH_W-1:0] output_epoch,
    input logic [DOMAIN_W-1:0] output_domain,
    input logic [STEP_W-1:0] output_temporal_step,
    input logic [LEN_W-1:0] output_temporal_length,
    input logic output_temporal_first,
    input logic output_temporal_last,
    input logic output_used_motion,
    input logic [TAG_W-1:0] output_tag,
    input logic [(LANES*ACC_W)-1:0] output_current_acc
);
    integer accepted_seen = 0;
    integer local_seen = 0;
    integer motion_seen = 0;
    integer rejected_seen = 0;
    integer abort_seen = 0;
    integer abort_rejected_seen = 0;
    integer output_stall_seen = 0;

    property p_output_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            output_valid && !output_ready |=> output_valid &&
            $stable(output_context) && $stable(output_lane_tile) &&
            $stable(output_epoch) && $stable(output_temporal_step) &&
            $stable(output_domain) &&
            $stable(output_temporal_length) &&
            $stable(output_temporal_first) &&
            $stable(output_temporal_last) && $stable(output_used_motion) &&
            $stable(output_tag) && $stable(output_current_acc);
    endproperty

    property p_domain_changes_only_behind_reset_fence;
        @(posedge clk_core)
            $changed(active_domain) |-> rst_core;
    endproperty

    property p_reset_quiesces_protocol;
        @(posedge clk_core)
            rst_core |-> !commit_ready && !protocol_error &&
                        !abort_ready && !abort_error;
    endproperty

    property p_protocol_error_rejects_input;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |-> commit_valid && !commit_ready;
    endproperty

    property p_accepted_input_has_no_error;
        @(posedge clk_core) disable iff (rst_core)
            commit_fire |-> commit_valid && commit_ready && !protocol_error;
    endproperty

    property p_abort_error_rejects_input;
        @(posedge clk_core) disable iff (rst_core)
            abort_error |-> abort_valid && !abort_ready;
    endproperty

    property p_accepted_metadata_exact;
        @(posedge clk_core) disable iff (rst_core)
            commit_fire |=> output_valid &&
                output_context == $past(commit_context) &&
                output_lane_tile == $past(commit_lane_tile) &&
                output_epoch == $past(commit_epoch) &&
                output_domain == $past(commit_domain) &&
                output_temporal_step == $past(commit_temporal_step) &&
                output_temporal_length == $past(commit_temporal_length) &&
                output_temporal_first == $past(commit_temporal_first) &&
                output_temporal_last == $past(commit_temporal_last) &&
                output_used_motion == $past(commit_use_motion) &&
                output_tag == $past(commit_tag);
    endproperty

    property p_stalled_same_sequence_abort_is_rejected;
        @(posedge clk_core) disable iff (rst_core)
            abort_valid && output_valid && !output_ready &&
            output_context == abort_context &&
            output_lane_tile == abort_lane_tile && output_epoch == abort_epoch &&
            output_domain == abort_domain && output_tag == abort_tag
            |-> abort_error && !abort_ready;
    endproperty

    property p_motion_never_starts_sequence;
        @(posedge clk_core) disable iff (rst_core)
            output_valid && output_temporal_first |->
                !output_used_motion && output_temporal_step == '0;
    endproperty

    assert property (p_output_stable_under_backpressure);
    assert property (p_reset_quiesces_protocol);
    assert property (p_domain_changes_only_behind_reset_fence);
    assert property (p_protocol_error_rejects_input);
    assert property (p_accepted_input_has_no_error);
    assert property (p_abort_error_rejects_input);
    assert property (p_accepted_metadata_exact);
    assert property (p_stalled_same_sequence_abort_is_rejected);
    assert property (p_motion_never_starts_sequence);

    for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : g_exact_value
        property p_local_overwrites_exactly;
            @(posedge clk_core) disable iff (rst_core)
                commit_fire && !commit_use_motion |=>
                    $signed(output_current_acc[(lane*ACC_W) +: ACC_W]) ==
                    $past($signed(commit_acc[(lane*ACC_W) +: ACC_W]));
        endproperty

        property p_motion_adds_resident_state_exactly;
            @(posedge clk_core) disable iff (rst_core)
                commit_fire && commit_use_motion |=>
                    $signed(output_current_acc[(lane*ACC_W) +: ACC_W]) ==
                    ($past($signed(commit_prior_acc[(lane*ACC_W) +: ACC_W])) +
                     $past($signed(commit_acc[(lane*ACC_W) +: ACC_W])));
        endproperty

        assert property (p_local_overwrites_exactly);
        assert property (p_motion_adds_resident_state_exactly);
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (commit_fire) begin
                accepted_seen <= accepted_seen + 1;
                if (commit_use_motion)
                    motion_seen <= motion_seen + 1;
                else
                    local_seen <= local_seen + 1;
            end
            if (protocol_error)
                rejected_seen <= rejected_seen + 1;
            if (abort_fire)
                abort_seen <= abort_seen + 1;
            if (abort_error)
                abort_rejected_seen <= abort_rejected_seen + 1;
            if (output_valid && !output_ready)
                output_stall_seen <= output_stall_seen + 1;
        end
    end

    final begin
        $display("M8_SVA_COVERAGE accepted=%0d local=%0d motion=%0d rejected=%0d abort=%0d abort_rejected=%0d output_stall=%0d",
                 accepted_seen, local_seen, motion_seen, rejected_seen,
                 abort_seen, abort_rejected_seen, output_stall_seen);
        if (accepted_seen <= 0 || local_seen <= 0 || motion_seen <= 0 ||
            rejected_seen <= 0 || abort_seen <= 0 || abort_rejected_seen <= 0 ||
            output_stall_seen <= 0)
            $error("M8 bound-SVA runtime coverage is incomplete");
    end
endmodule

bind qfit_temporal_destination_commit_engine
qfit_temporal_destination_commit_engine_assertions #(
    .CTX_W(CTX_W), .LANE_TILE_W(LANE_TILE_W), .EPOCH_W(EPOCH_W),
    .DOMAIN_W(DOMAIN_W),
    .STEP_W(STEP_W), .LEN_W(LEN_W), .TAG_W(TAG_W),
    .LANES(LANES), .ACC_W(ACC_W)
) u_qfit_temporal_destination_commit_engine_assertions (
    .clk_core, .rst_core, .active_domain,
    .commit_valid, .commit_ready, .commit_context, .commit_lane_tile,
    .commit_epoch, .commit_domain, .commit_temporal_step, .commit_temporal_length,
    .commit_temporal_first, .commit_temporal_last, .commit_use_motion,
    .commit_tag, .commit_acc, .commit_prior_acc, .protocol_error,
    .abort_valid, .abort_ready, .abort_error, .abort_context,
    .abort_lane_tile, .abort_epoch, .abort_domain, .abort_tag,
    .commit_fire, .abort_fire,
    .output_valid, .output_ready, .output_context, .output_lane_tile,
    .output_epoch, .output_domain, .output_temporal_step, .output_temporal_length,
    .output_temporal_first, .output_temporal_last,
    .output_used_motion, .output_tag, .output_current_acc
);

`default_nettype wire
