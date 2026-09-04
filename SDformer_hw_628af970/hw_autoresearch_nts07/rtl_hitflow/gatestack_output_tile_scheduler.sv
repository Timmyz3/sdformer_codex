`timescale 1ns/1ps
`default_nettype none

// Sequences output tiles outside input heads. Descriptor/event resources remain
// owned by the context until the final output-tile visit of each head.
module gatestack_output_tile_scheduler #(
    parameter int CONTEXTS             = 2,
    parameter int HEADS                = 24,
    parameter int LANES                = 32,
    parameter int TAG_W                = 32,
    parameter int INPUT_CH_W           = 10,
    parameter int OUTPUT_TILE_W         = 8,
    parameter int OUTPUT_TILE_COUNT_W   = 8,
    parameter int HEAD_COUNT_W          = 6,
    parameter int COUNTER_W             = 32,
    parameter int CONTEXT_ID_W          = (CONTEXTS <= 1) ?
                                           1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W             = (HEADS <= 1) ? 1 : $clog2(HEADS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         group_valid,
    output logic                         group_ready,
    input  logic [CONTEXT_ID_W-1:0]      group_context_id,
    input  logic [TAG_W-1:0]             group_tag,
    input  logic [HEAD_COUNT_W-1:0]      group_head_count,
    input  logic [OUTPUT_TILE_W-1:0]     group_first_output_tile,
    input  logic [OUTPUT_TILE_COUNT_W-1:0]
                                            group_output_tile_count,

    output logic                         tile_start_valid,
    input  logic                         tile_start_ready,
    output logic [TAG_W-1:0]             tile_start_tag,
    output logic [OUTPUT_TILE_W-1:0]     tile_start_output_tile,
    output logic [HEAD_COUNT_W-1:0]      tile_start_head_count,

    output logic                         head_issue_valid,
    input  logic                         head_issue_ready,
    output logic [CONTEXT_ID_W-1:0]      head_issue_context_id,
    output logic [TAG_W-1:0]             head_issue_tag,
    output logic [HEAD_ID_W-1:0]         head_issue_head_id,
    output logic [HEAD_COUNT_W-1:0]      head_issue_head_index,
    output logic [INPUT_CH_W-1:0]        head_issue_input_channel_base,
    output logic [OUTPUT_TILE_W-1:0]     head_issue_output_tile,
    output logic                         head_issue_last_head,
    output logic                         head_issue_last_output_tile,

    input  logic                         head_done_valid,
    output logic                         head_done_ready,
    input  logic [TAG_W-1:0]             head_done_tag,
    input  logic [HEAD_ID_W-1:0]         head_done_head_id,
    input  logic                         head_done_error,

    input  logic                         tile_done_valid,
    output logic                         tile_done_ready,
    input  logic [TAG_W-1:0]             tile_done_tag,
    input  logic                         tile_done_error,

    output logic                         group_done_valid,
    input  logic                         group_done_ready,
    output logic [TAG_W-1:0]             group_done_tag,
    output logic                         group_done_error,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_groups,
    output logic [COUNTER_W-1:0]         count_tile_starts,
    output logic [COUNTER_W-1:0]         count_head_issues,
    output logic [COUNTER_W-1:0]         count_group_errors
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_TILE_START,
        ST_HEAD_ISSUE,
        ST_HEAD_WAIT,
        ST_TILE_WAIT,
        ST_GROUP_DONE
    } state_t;

    state_t state_q;
    logic [CONTEXT_ID_W-1:0] context_q;
    logic [TAG_W-1:0] tag_q;
    logic [TAG_W-1:0] tile_tag_q;
    logic [HEAD_COUNT_W-1:0] head_count_q;
    logic [OUTPUT_TILE_COUNT_W-1:0] tile_count_q;
    logic [OUTPUT_TILE_COUNT_W-1:0] tile_ordinal_q;
    logic [OUTPUT_TILE_W-1:0] output_tile_q;
    logic [HEAD_COUNT_W-1:0] head_index_q;
    logic group_error_q;

    logic group_context_legal;
    logic group_counts_legal;
    logic group_tile_range_legal;
    logic group_channel_range_legal;
    logic group_legal;
    logic group_fire;
    logic tile_start_fire;
    logic head_issue_fire;
    logic head_done_fire;
    logic tile_done_fire;
    logic group_done_fire;
    logic head_done_mismatch;
    logic tile_done_mismatch;
    logic last_head;
    logic last_output_tile;
    logic [31:0] final_tile_extended;
    logic [31:0] final_channel_base_extended;

    assign group_context_legal = 32'(group_context_id) < 32'(CONTEXTS);
    assign group_counts_legal = group_head_count != '0 &&
                                32'(group_head_count) <= 32'(HEADS) &&
                                group_output_tile_count != '0;
    assign final_tile_extended = 32'(group_first_output_tile) +
                                 32'(group_output_tile_count) - 32'd1;
    assign group_tile_range_legal = group_output_tile_count != '0 &&
                                    final_tile_extended <
                                    (32'(1) << OUTPUT_TILE_W);
    assign final_channel_base_extended =
        (32'(group_head_count) - 32'd1) * 32'(LANES);
    assign group_channel_range_legal = group_head_count != '0 &&
                                       final_channel_base_extended <
                                       (32'(1) << INPUT_CH_W);
    assign group_legal = group_context_legal && group_counts_legal &&
                         group_tile_range_legal &&
                         group_channel_range_legal;
    assign group_ready = state_q == ST_IDLE && group_legal;
    assign group_fire = group_valid && group_ready;

    assign last_head = 32'(head_index_q) + 1 == 32'(head_count_q);
    assign last_output_tile = 32'(tile_ordinal_q) + 1 ==
                              32'(tile_count_q);

    assign tile_start_valid = state_q == ST_TILE_START;
    assign tile_start_tag = tile_tag_q;
    assign tile_start_output_tile = output_tile_q;
    assign tile_start_head_count = head_count_q;
    assign tile_start_fire = tile_start_valid && tile_start_ready;

    assign head_issue_valid = state_q == ST_HEAD_ISSUE;
    assign head_issue_context_id = context_q;
    assign head_issue_tag = tile_tag_q;
    assign head_issue_head_id = HEAD_ID_W'(head_index_q);
    assign head_issue_head_index = head_index_q;
    assign head_issue_input_channel_base =
        INPUT_CH_W'(32'(head_index_q) * 32'(LANES));
    assign head_issue_output_tile = output_tile_q;
    assign head_issue_last_head = last_head;
    assign head_issue_last_output_tile = last_output_tile;
    assign head_issue_fire = head_issue_valid && head_issue_ready;

    assign head_done_ready = state_q == ST_HEAD_WAIT;
    assign head_done_fire = head_done_valid && head_done_ready;
    assign head_done_mismatch = head_done_tag != tile_tag_q ||
                                head_done_head_id !=
                                    HEAD_ID_W'(head_index_q) ||
                                head_done_error;

    assign tile_done_ready = state_q == ST_TILE_WAIT;
    assign tile_done_fire = tile_done_valid && tile_done_ready;
    assign tile_done_mismatch = tile_done_tag != tile_tag_q ||
                                tile_done_error;

    assign group_done_valid = state_q == ST_GROUP_DONE;
    assign group_done_tag = tag_q;
    assign group_done_error = group_error_q;
    assign group_done_fire = group_done_valid && group_done_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            context_q <= '0;
            tag_q <= '0;
            tile_tag_q <= '0;
            head_count_q <= '0;
            tile_count_q <= '0;
            tile_ordinal_q <= '0;
            output_tile_q <= '0;
            head_index_q <= '0;
            group_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_groups <= '0;
            count_tile_starts <= '0;
            count_head_issues <= '0;
            count_group_errors <= '0;
        end else begin
            if (group_valid && state_q == ST_IDLE && !group_legal)
                protocol_error <= 1'b1;

            if (group_fire) begin
                context_q <= group_context_id;
                tag_q <= group_tag;
                tile_tag_q <= group_tag;
                head_count_q <= group_head_count;
                tile_count_q <= group_output_tile_count;
                tile_ordinal_q <= '0;
                output_tile_q <= group_first_output_tile;
                head_index_q <= '0;
                group_error_q <= 1'b0;
                count_groups <= count_groups + 1'b1;
                state_q <= ST_TILE_START;
            end

            if (tile_start_fire) begin
                count_tile_starts <= count_tile_starts + 1'b1;
                state_q <= ST_HEAD_ISSUE;
            end

            if (head_issue_fire) begin
                count_head_issues <= count_head_issues + 1'b1;
                state_q <= ST_HEAD_WAIT;
            end

            if (head_done_fire) begin
                if (head_done_mismatch) begin
                    group_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    state_q <= ST_GROUP_DONE;
                end else begin
                    if (last_head) begin
                        state_q <= ST_TILE_WAIT;
                    end else begin
                        head_index_q <= head_index_q + 1'b1;
                        state_q <= ST_HEAD_ISSUE;
                    end
                end
            end

            if (tile_done_fire) begin
                if (tile_done_mismatch) begin
                    group_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    state_q <= ST_GROUP_DONE;
                end else begin
                    if (last_output_tile) begin
                        state_q <= ST_GROUP_DONE;
                    end else begin
                        tile_ordinal_q <= tile_ordinal_q + 1'b1;
                        output_tile_q <= output_tile_q + 1'b1;
                        tile_tag_q <= tile_tag_q + 1'b1;
                        head_index_q <= '0;
                        state_q <= ST_TILE_START;
                    end
                end
            end

            if (group_done_fire) begin
                if (group_error_q)
                    count_group_errors <= count_group_errors + 1'b1;
                state_q <= ST_IDLE;
            end
        end
    end
endmodule

`default_nettype wire
