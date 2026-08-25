`timescale 1ns/1ps
`default_nettype none

// M8.1 Local/Motion destination-state commit slice. Local inputs are absolute
// Acc32 refreshes; Motion inputs are signed deltas applied in place. A serial
// generation, exact temporal step/length, sequence tag, and per-entry abort
// make duplicate, skipped, stale, and stranded updates fail closed.
module qfit_temporal_destination_commit_engine #(
    parameter int CONTEXTS = 4,
    parameter int LANE_TILES = 32,
    parameter int LANES = 16,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int EPOCH_W = 16,
    parameter int DOMAIN_W = 8,
    parameter int STEP_W = 4,
    parameter int LEN_W = 4,
    parameter int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int LANE_TILE_W = (LANE_TILES <= 1) ? 1 : $clog2(LANE_TILES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    // Must change on every system reset fence. Producers stamp queued work
    // with this domain so pre-reset traffic cannot be accepted post-reset.
    input  logic [DOMAIN_W-1:0]          active_domain,

    input  logic                         commit_valid,
    output logic                         commit_ready,
    input  logic [CTX_W-1:0]             commit_context,
    input  logic [LANE_TILE_W-1:0]       commit_lane_tile,
    input  logic [EPOCH_W-1:0]           commit_epoch,
    input  logic [DOMAIN_W-1:0]          commit_domain,
    input  logic [STEP_W-1:0]            commit_temporal_step,
    input  logic [LEN_W-1:0]             commit_temporal_length,
    input  logic                         commit_temporal_first,
    input  logic                         commit_temporal_last,
    input  logic                         commit_use_motion,
    // The tag is a sequence identity and must remain constant for all steps.
    input  logic [TAG_W-1:0]             commit_tag,
    input  logic [(LANES*ACC_W)-1:0]     commit_acc,

    // A matching abort invalidates one stranded resident sequence but keeps
    // its generation watermark, so the same/stale generation cannot restart.
    input  logic                         abort_valid,
    output logic                         abort_ready,
    input  logic [CTX_W-1:0]             abort_context,
    input  logic [LANE_TILE_W-1:0]       abort_lane_tile,
    input  logic [EPOCH_W-1:0]           abort_epoch,
    input  logic [DOMAIN_W-1:0]          abort_domain,
    input  logic [TAG_W-1:0]             abort_tag,
    output logic                         abort_error,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [CTX_W-1:0]             output_context,
    output logic [LANE_TILE_W-1:0]       output_lane_tile,
    output logic [EPOCH_W-1:0]           output_epoch,
    output logic [DOMAIN_W-1:0]          output_domain,
    output logic [STEP_W-1:0]            output_temporal_step,
    output logic [LEN_W-1:0]             output_temporal_length,
    output logic                         output_temporal_first,
    output logic                         output_temporal_last,
    output logic                         output_used_motion,
    output logic [TAG_W-1:0]             output_tag,
    output logic [(LANES*ACC_W)-1:0]     output_current_acc,

    output logic                         protocol_error
);
    logic signed [ACC_W-1:0] state_q [0:CONTEXTS-1][0:LANE_TILES-1][0:LANES-1];
    logic [EPOCH_W-1:0] state_epoch_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [DOMAIN_W-1:0] state_domain_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [STEP_W-1:0] next_step_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [LEN_W-1:0] sequence_length_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [TAG_W-1:0] sequence_tag_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic epoch_initialized_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic state_valid_q [0:CONTEXTS-1][0:LANE_TILES-1];
    logic sequence_open_q [0:CONTEXTS-1][0:LANE_TILES-1];

    logic output_valid_q;
    logic output_available;
    logic commit_index_valid;
    logic abort_index_valid;
    logic length_admitted;
    logic epoch_fresh;
    logic [EPOCH_W-1:0] epoch_delta;
    logic first_admitted;
    logic continuation_admitted;
    logic protocol_admitted;
    logic abort_admitted;
    logic abort_output_conflict;
    logic commit_fire;
    logic abort_fire;
    logic signed [ACC_W-1:0] current_value [0:LANES-1];
    // Kept as a packed internal signal so bound SVA can prove Motion addition.
    logic [(LANES*ACC_W)-1:0] commit_prior_acc;

    assign output_available = ~output_valid_q | output_ready;
    assign commit_index_valid = ($unsigned(commit_context) < CONTEXTS) &&
                                ($unsigned(commit_lane_tile) < LANE_TILES);
    assign abort_index_valid = ($unsigned(abort_context) < CONTEXTS) &&
                               ($unsigned(abort_lane_tile) < LANE_TILES);
    assign length_admitted = (commit_temporal_length == LEN_W'(2)) ||
                             (commit_temporal_length == LEN_W'(10));

    always_comb begin
        epoch_delta = '0;
        epoch_fresh = 1'b0;
        first_admitted = 1'b0;
        continuation_admitted = 1'b0;
        abort_admitted = 1'b0;
        if (commit_index_valid) begin
            epoch_delta = commit_epoch -
                state_epoch_q[commit_context][commit_lane_tile];
            // RFC1982-style serial freshness: accept forward deltas smaller
            // than half the generation space, including FFFF->0000 wrap.
            epoch_fresh = !epoch_initialized_q[commit_context][commit_lane_tile] ||
                ((state_domain_q[commit_context][commit_lane_tile] == commit_domain) &&
                 (epoch_delta != '0) && !epoch_delta[EPOCH_W-1]);
            first_admitted = commit_temporal_first &&
                !commit_temporal_last && !commit_use_motion &&
                (commit_domain == active_domain) &&
                (commit_temporal_step == '0) && length_admitted &&
                !sequence_open_q[commit_context][commit_lane_tile] &&
                epoch_fresh;
            continuation_admitted = !commit_temporal_first &&
                state_valid_q[commit_context][commit_lane_tile] &&
                sequence_open_q[commit_context][commit_lane_tile] &&
                (commit_domain == active_domain) &&
                (state_domain_q[commit_context][commit_lane_tile] == commit_domain) &&
                (state_epoch_q[commit_context][commit_lane_tile] == commit_epoch) &&
                (sequence_tag_q[commit_context][commit_lane_tile] == commit_tag) &&
                (sequence_length_q[commit_context][commit_lane_tile] ==
                    commit_temporal_length) &&
                (next_step_q[commit_context][commit_lane_tile] ==
                    commit_temporal_step) &&
                (commit_temporal_last ==
                    (commit_temporal_step == (commit_temporal_length - 1'b1)));
        end
        if (abort_index_valid) begin
            abort_admitted = (abort_domain == active_domain) &&
                epoch_initialized_q[abort_context][abort_lane_tile] &&
                state_valid_q[abort_context][abort_lane_tile] &&
                sequence_open_q[abort_context][abort_lane_tile] &&
                (state_domain_q[abort_context][abort_lane_tile] == abort_domain) &&
                (state_epoch_q[abort_context][abort_lane_tile] == abort_epoch) &&
                (sequence_tag_q[abort_context][abort_lane_tile] == abort_tag) &&
                !abort_output_conflict;
        end
    end

    // A committed output is irrevocable. Abort terminates only future steps,
    // so the same sequence cannot be aborted while its output is stalled.
    assign abort_output_conflict = output_valid_q && !output_ready &&
        (output_context == abort_context) &&
        (output_lane_tile == abort_lane_tile) &&
        (output_epoch == abort_epoch) && (output_domain == abort_domain) &&
        (output_tag == abort_tag);

    assign protocol_admitted = first_admitted | continuation_admitted;
    assign commit_ready = !rst_core && !abort_valid && output_available &&
                          protocol_admitted;
    assign commit_fire = commit_valid && commit_ready;
    // An illegal request remains observable even while the output is stalled.
    assign protocol_error = !rst_core && commit_valid && !protocol_admitted;
    assign abort_ready = !rst_core && abort_admitted;
    assign abort_error = !rst_core && abort_valid && !abort_admitted;
    assign abort_fire = abort_valid && abort_ready;
    assign output_valid = output_valid_q;

    for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : g_current
        logic signed [ACC_W-1:0] input_value;
        always_comb begin
            input_value = commit_acc[(lane*ACC_W) +: ACC_W];
            commit_prior_acc[(lane*ACC_W) +: ACC_W] = '0;
            current_value[lane] = input_value;
            if (commit_index_valid) begin
                commit_prior_acc[(lane*ACC_W) +: ACC_W] =
                    state_q[commit_context][commit_lane_tile][lane];
                if (commit_use_motion)
                    current_value[lane] =
                        state_q[commit_context][commit_lane_tile][lane] + input_value;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            output_valid_q <= 1'b0;
            output_context <= '0;
            output_lane_tile <= '0;
            output_epoch <= '0;
            output_domain <= '0;
            output_temporal_step <= '0;
            output_temporal_length <= '0;
            output_temporal_first <= 1'b0;
            output_temporal_last <= 1'b0;
            output_used_motion <= 1'b0;
            output_tag <= '0;
            output_current_acc <= '0;
            for (int ctx_i = 0; ctx_i < CONTEXTS; ctx_i = ctx_i + 1) begin
                for (int tile_i = 0; tile_i < LANE_TILES; tile_i = tile_i + 1) begin
                    state_epoch_q[ctx_i][tile_i] <= '0;
                    state_domain_q[ctx_i][tile_i] <= '0;
                    next_step_q[ctx_i][tile_i] <= '0;
                    sequence_length_q[ctx_i][tile_i] <= '0;
                    sequence_tag_q[ctx_i][tile_i] <= '0;
                    epoch_initialized_q[ctx_i][tile_i] <= 1'b0;
                    state_valid_q[ctx_i][tile_i] <= 1'b0;
                    sequence_open_q[ctx_i][tile_i] <= 1'b0;
                end
            end
        end else begin
            if (output_valid_q && output_ready)
                output_valid_q <= 1'b0;
            if (abort_fire) begin
                state_valid_q[abort_context][abort_lane_tile] <= 1'b0;
                sequence_open_q[abort_context][abort_lane_tile] <= 1'b0;
            end
            if (commit_fire) begin
                output_valid_q <= 1'b1;
                output_context <= commit_context;
                output_lane_tile <= commit_lane_tile;
                output_epoch <= commit_epoch;
                output_domain <= commit_domain;
                output_temporal_step <= commit_temporal_step;
                output_temporal_length <= commit_temporal_length;
                output_temporal_first <= commit_temporal_first;
                output_temporal_last <= commit_temporal_last;
                output_used_motion <= commit_use_motion;
                output_tag <= commit_tag;
                state_valid_q[commit_context][commit_lane_tile] <= 1'b1;
                sequence_open_q[commit_context][commit_lane_tile] <=
                    !commit_temporal_last;
                next_step_q[commit_context][commit_lane_tile] <=
                    commit_temporal_step + 1'b1;
                if (commit_temporal_first) begin
                    state_epoch_q[commit_context][commit_lane_tile] <= commit_epoch;
                    state_domain_q[commit_context][commit_lane_tile] <= commit_domain;
                    sequence_length_q[commit_context][commit_lane_tile] <=
                        commit_temporal_length;
                    sequence_tag_q[commit_context][commit_lane_tile] <= commit_tag;
                    epoch_initialized_q[commit_context][commit_lane_tile] <= 1'b1;
                end
                for (int lane = 0; lane < LANES; lane = lane + 1) begin
                    state_q[commit_context][commit_lane_tile][lane] <= current_value[lane];
                    output_current_acc[(lane*ACC_W) +: ACC_W] <= current_value[lane];
                end
            end
        end
    end
endmodule

`default_nettype wire
