`timescale 1ns/1ps
`default_nettype none

// Prediction-head specialization of the shared 96-lane event datapath.
// The physical lanes are transposed from 96 output channels to
// 48 pixels x 2 output channels.  Each accepted cycle consumes as many as
// eight source channels, with independent positive/negative masks per pixel.
module qfit_head_p48_signed_lane_fold #(
    parameter int PIXELS = 48,
    parameter int OUTPUTS = 2,
    parameter int SOURCE_SLOTS = 8,
    parameter int W_W = 8,
    parameter int ACC_W = 13,
    parameter int TAG_W = 48,
    parameter int LANES = PIXELS * OUTPUTS,
    parameter int SOURCE_COUNT_W = 16,
    parameter int SIGNED_EVENT_COUNT_W = 16
) (
    input  logic                                     clk_core,
    input  logic                                     rst_core,

    input  logic                                     command_valid,
    output logic                                     command_ready,
    input  logic [TAG_W-1:0]                         command_tag,
    input  logic [LANES*ACC_W-1:0]                   command_seed_acc,
    input  logic                                     command_zero_event_group,
    output logic                                     command_accept,

    input  logic                                     event_valid,
    output logic                                     event_ready,
    input  logic                                     event_last,
    input  logic [SOURCE_SLOTS-1:0]                  event_source_valid,
    input  logic [SOURCE_SLOTS*PIXELS-1:0]           event_positive_mask,
    input  logic [SOURCE_SLOTS*PIXELS-1:0]           event_negative_mask,
    input  logic [SOURCE_SLOTS*OUTPUTS*W_W-1:0]      event_weight,
    output logic                                     event_accept,

    output logic                                     output_valid,
    input  logic                                     output_ready,
    output logic [TAG_W-1:0]                         output_tag,
    output logic [LANES*ACC_W-1:0]                   output_acc,
    output logic [SOURCE_COUNT_W-1:0]                output_source_issues,
    output logic [SIGNED_EVENT_COUNT_W-1:0]          output_signed_events,
    output logic                                     output_accept,

    output logic                                     protocol_error,
    output logic                                     busy
);
    logic active_q;
    logic faulted_q;
    logic [TAG_W-1:0] active_tag_q;
    logic signed [ACC_W-1:0] accumulator_q [0:LANES-1];
    logic [SOURCE_COUNT_W-1:0] source_issues_q;
    logic [SIGNED_EVENT_COUNT_W-1:0] signed_events_q;

    logic output_valid_q;
    logic [TAG_W-1:0] output_tag_q;
    logic signed [ACC_W-1:0] output_acc_q [0:LANES-1];
    logic [SOURCE_COUNT_W-1:0] output_source_issues_q;
    logic [SIGNED_EVENT_COUNT_W-1:0] output_signed_events_q;

    logic signed [W_W:0] lane_term [0:LANES-1][0:SOURCE_SLOTS-1];
    logic signed [W_W+3:0] lane_cycle_sum [0:LANES-1];
    logic signed [ACC_W:0] lane_accumulator_wide [0:LANES-1];
    logic signed [ACC_W-1:0] lane_accumulator_next [0:LANES-1];
    logic mask_overlap;
    logic invalid_slot_mask;
    logic reserved_negative_weight;
    logic event_has_source;
    logic accumulator_overflow;
    logic [SOURCE_COUNT_W-1:0] event_source_count;
    logic [SIGNED_EVENT_COUNT_W-1:0] event_signed_count;

`ifndef SYNTHESIS
    initial begin
        if (PIXELS != 48 || OUTPUTS != 2 || SOURCE_SLOTS != 8
                || W_W != 8 || ACC_W != 13 || LANES != 96)
            $fatal(1, "M62 frozen P48x2 S8 signed13 geometry drift");
    end
`endif

    assign output_valid = !faulted_q && output_valid_q;
    assign output_accept = output_valid && output_ready;
    assign output_tag = output_valid_q ? output_tag_q : '0;
    assign output_source_issues = output_valid_q
        ? output_source_issues_q : '0;
    assign output_signed_events = output_valid_q
        ? output_signed_events_q : '0;
    always_comb begin
        output_acc = '0;
        if (output_valid_q)
            for (int lane = 0; lane < LANES; lane++)
                output_acc[lane*ACC_W +: ACC_W] = output_acc_q[lane];
    end

    // A new zero-event command may replace an output only when that output is
    // accepted in the same cycle.  This is the sole one-cycle group path.
    assign command_ready = !faulted_q && !active_q
        && (!output_valid_q || output_ready);
    assign command_accept = command_valid && command_ready;
    assign event_ready = !faulted_q && active_q && !output_valid_q;
    assign event_accept = event_valid && event_ready;
    assign protocol_error = faulted_q;
    assign busy = active_q || output_valid_q;

    always_comb begin : build_eight_source_signed_reduction
        mask_overlap = 1'b0;
        invalid_slot_mask = 1'b0;
        reserved_negative_weight = 1'b0;
        event_has_source = |event_source_valid;
        event_source_count = '0;
        event_signed_count = '0;
        accumulator_overflow = 1'b0;
        for (int slot = 0; slot < SOURCE_SLOTS; slot++) begin
            if (!event_source_valid[slot]
                    && (|event_positive_mask[slot*PIXELS +: PIXELS]
                        || |event_negative_mask[slot*PIXELS +: PIXELS]))
                invalid_slot_mask = 1'b1;
            if (event_source_valid[slot])
                event_source_count = event_source_count + 1'b1;
            for (int output_index = 0; output_index < OUTPUTS;
                    output_index++)
                if (event_source_valid[slot]
                        && event_weight[
                            (slot*OUTPUTS + output_index)*W_W +: W_W]
                            == {1'b1, {(W_W-1){1'b0}}})
                    reserved_negative_weight = 1'b1;
            for (int pixel = 0; pixel < PIXELS; pixel++) begin
                logic positive;
                logic negative;
                positive = event_positive_mask[slot*PIXELS + pixel];
                negative = event_negative_mask[slot*PIXELS + pixel];
                if (event_source_valid[slot] && positive && negative)
                    mask_overlap = 1'b1;
                if (event_source_valid[slot] && (positive ^ negative))
                    event_signed_count = event_signed_count + 1'b1;
            end
        end
        for (int pixel = 0; pixel < PIXELS; pixel++) begin
            for (int output_index = 0; output_index < OUTPUTS;
                    output_index++) begin
                int lane;
                lane = pixel * OUTPUTS + output_index;
                lane_cycle_sum[lane] = '0;
                for (int slot = 0; slot < SOURCE_SLOTS; slot++) begin
                    logic positive;
                    logic negative;
                    logic signed [W_W-1:0] weight;
                    positive = event_positive_mask[slot*PIXELS + pixel];
                    negative = event_negative_mask[slot*PIXELS + pixel];
                    weight = event_weight[
                        (slot*OUTPUTS + output_index)*W_W +: W_W];
                    if (event_source_valid[slot] && positive && !negative)
                        lane_term[lane][slot]
                            = {{1{weight[W_W-1]}}, weight};
                    else if (event_source_valid[slot] && negative && !positive)
                        lane_term[lane][slot]
                            = -{{1{weight[W_W-1]}}, weight};
                    else
                        lane_term[lane][slot] = '0;
                    lane_cycle_sum[lane] = lane_cycle_sum[lane]
                        + {{3{lane_term[lane][slot][W_W]}},
                           lane_term[lane][slot]};
                end
                lane_accumulator_wide[lane]
                    = {{1{accumulator_q[lane][ACC_W-1]}},
                        accumulator_q[lane]}
                      + {{(ACC_W-(W_W+3)){
                            lane_cycle_sum[lane][W_W+3]}},
                         lane_cycle_sum[lane]};
                lane_accumulator_next[lane]
                    = lane_accumulator_wide[lane][ACC_W-1:0];
                if (lane_accumulator_wide[lane][ACC_W:ACC_W-1]
                        != 2'b00
                        && lane_accumulator_wide[lane][ACC_W:ACC_W-1]
                        != 2'b11)
                    accumulator_overflow = 1'b1;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            faulted_q <= 1'b0;
            active_tag_q <= '0;
            source_issues_q <= '0;
            signed_events_q <= '0;
            output_valid_q <= 1'b0;
            output_tag_q <= '0;
            output_source_issues_q <= '0;
            output_signed_events_q <= '0;
            for (int lane = 0; lane < LANES; lane++) begin
                accumulator_q[lane] <= '0;
                output_acc_q[lane] <= '0;
            end
        end else begin
            if (output_accept)
                output_valid_q <= 1'b0;

            if (command_accept) begin
                active_tag_q <= command_tag;
                source_issues_q <= '0;
                signed_events_q <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    accumulator_q[lane]
                        <= command_seed_acc[lane*ACC_W +: ACC_W];
                if (command_zero_event_group) begin
                    active_q <= 1'b0;
                    output_valid_q <= 1'b1;
                    output_tag_q <= command_tag;
                    output_source_issues_q <= '0;
                    output_signed_events_q <= '0;
                    for (int lane = 0; lane < LANES; lane++)
                        output_acc_q[lane]
                            <= command_seed_acc[lane*ACC_W +: ACC_W];
                end else begin
                    active_q <= 1'b1;
                end
            end

            if (event_accept) begin
                if (!event_has_source || event_signed_count == 0
                        || mask_overlap || invalid_slot_mask
                        || reserved_negative_weight || accumulator_overflow) begin
                    faulted_q <= 1'b1;
                end else begin
                    source_issues_q <= source_issues_q + event_source_count;
                    signed_events_q <= signed_events_q + event_signed_count;
                    for (int lane = 0; lane < LANES; lane++)
                        accumulator_q[lane] <= lane_accumulator_next[lane];
                    if (event_last) begin
                        active_q <= 1'b0;
                        output_valid_q <= 1'b1;
                        output_tag_q <= active_tag_q;
                        output_source_issues_q
                            <= source_issues_q + event_source_count;
                        output_signed_events_q
                            <= signed_events_q + event_signed_count;
                        for (int lane = 0; lane < LANES; lane++)
                            output_acc_q[lane]
                                <= lane_accumulator_next[lane];
                    end
                end
            end

            if (event_valid && !active_q && !command_accept)
                faulted_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
