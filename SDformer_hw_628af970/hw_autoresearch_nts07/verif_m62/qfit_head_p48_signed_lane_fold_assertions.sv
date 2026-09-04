`timescale 1ns/1ps
`default_nettype none

module qfit_head_p48_signed_lane_fold_assertions #(
    parameter int PIXELS = 48,
    parameter int OUTPUTS = 2,
    parameter int SOURCE_SLOTS = 8,
    parameter int W_W = 8,
    parameter int ACC_W = 13,
    parameter int TAG_W = 48,
    parameter int LANES = PIXELS * OUTPUTS
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic command_accept,
    input logic [TAG_W-1:0] command_tag,
    input logic [LANES*ACC_W-1:0] command_seed_acc,
    input logic command_zero_event_group,
    input logic event_valid,
    input logic event_ready,
    input logic event_accept,
    input logic event_last,
    input logic [SOURCE_SLOTS-1:0] event_source_valid,
    input logic [SOURCE_SLOTS*PIXELS-1:0] event_positive_mask,
    input logic [SOURCE_SLOTS*PIXELS-1:0] event_negative_mask,
    input logic [SOURCE_SLOTS*OUTPUTS*W_W-1:0] event_weight,
    input logic output_valid,
    input logic output_ready,
    input logic output_accept,
    input logic [TAG_W-1:0] output_tag,
    input logic [LANES*ACC_W-1:0] output_acc,
    input logic [15:0] output_source_issues,
    input logic [15:0] output_signed_events,
    input logic protocol_error,
    input logic busy
);
    function automatic logic event_geometry_legal;
        logic any_source;
        begin
            event_geometry_legal = 1'b1;
            any_source = 1'b0;
            for (int slot = 0; slot < SOURCE_SLOTS; slot++) begin
                logic any_mask;
                any_mask = |event_positive_mask[slot*PIXELS +: PIXELS]
                    || |event_negative_mask[slot*PIXELS +: PIXELS];
                if (event_source_valid[slot]) begin
                    any_source = 1'b1;
                    if (!any_mask) event_geometry_legal = 1'b0;
                    for (int output_index = 0; output_index < OUTPUTS;
                            output_index++)
                        if (event_weight[
                            (slot*OUTPUTS+output_index)*W_W +: W_W]
                            == {1'b1, {(W_W-1){1'b0}}})
                            event_geometry_legal = 1'b0;
                end else if (any_mask) begin
                    event_geometry_legal = 1'b0;
                end
                if (|(event_positive_mask[slot*PIXELS +: PIXELS]
                      & event_negative_mask[slot*PIXELS +: PIXELS]))
                    event_geometry_legal = 1'b0;
            end
            if (!any_source) event_geometry_legal = 1'b0;
        end
    endfunction

    ap_command_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core) command_accept == (command_valid && command_ready));
    ap_event_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept == (event_valid && event_ready));
    ap_output_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_acc,
                                     output_source_issues,
                                     output_signed_events}));
    ap_accepted_event_geometry: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept |-> event_geometry_legal());
    ap_last_event_completes: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept && event_last
        |=> output_valid || protocol_error);
    ap_zero_group_completes: assert property (@(posedge clk_core)
        disable iff (rst_core) command_accept && command_zero_event_group
        |=> output_valid && output_tag == $past(command_tag)
            && output_acc == $past(command_seed_acc));
    ap_nonzero_output_has_work: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_source_issues != 0
        |-> output_signed_events != 0);
    ap_fault_is_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error |=> protocol_error);
    ap_fault_blocks_accepts: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error
        |-> !command_accept && !event_accept && !output_accept);

    cp_zero_group: cover property (@(posedge clk_core)
        disable iff (rst_core) command_accept && command_zero_event_group);
    cp_full_eight_source_event: cover property (@(posedge clk_core)
        disable iff (rst_core) event_accept && &event_source_valid);
    cp_positive_and_negative: cover property (@(posedge clk_core)
        disable iff (rst_core) event_accept && |event_positive_mask
            && |event_negative_mask);
    cp_output_stall: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready);
    cp_protocol_fault: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);
endmodule

`default_nettype wire
