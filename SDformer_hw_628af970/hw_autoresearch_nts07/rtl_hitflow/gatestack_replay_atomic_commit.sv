`timescale 1ns/1ps
`default_nettype none

// Commits one immutable replay plan only when projection, lifecycle and the
// optional slot replay port can acquire the session in the same cycle.
module gatestack_replay_atomic_commit #(
    parameter int CONTEXTS        = 2,
    parameter int HEADS           = 24,
    parameter int RESIDENT_TERMS  = 80,
    parameter int ENABLE_RESIDENCY = 1,
    parameter int TAG_W           = 32,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int HEAD_COUNT_W    = 6,
    parameter int EVENT_COUNT_W   = 13,
    parameter int WORD_INDEX_W    = 7,
    parameter int FORMAT_W        = 2,
    parameter int COUNTER_W       = 32,
    parameter int CONTEXT_ID_W    = (CONTEXTS <= 1) ?
                                     1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W       = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int ROUTE_W         = 2
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         plan_valid,
    output logic                         plan_ready,
    input  logic [CONTEXT_ID_W-1:0]      plan_context_id,
    input  logic [HEAD_ID_W-1:0]         plan_head_id,
    input  logic [TAG_W-1:0]             plan_payload_tag,
    input  logic [TAG_W-1:0]             plan_execution_tag,
    input  logic [ROUTE_W-1:0]           plan_route,
    input  logic [FORMAT_W-1:0]          plan_format,
    input  logic [HEAD_COUNT_W-1:0]      plan_head_index,
    input  logic [INPUT_CH_W-1:0]        plan_input_channel_base,
    input  logic [OUTPUT_TILE_W-1:0]     plan_output_tile,
    input  logic                         plan_last_head,
    input  logic                         plan_last_output_tile,
    input  logic                         plan_cache_owned,
    input  logic                         plan_slot_replay_required,
    input  logic [WORD_INDEX_W-1:0]      plan_replay_start_word,
    input  logic [7:0]                   plan_resident_term_count,
    input  logic [EVENT_COUNT_W-1:0]     plan_resident_event_count,

    output logic                         projection_commit_pulse,
    input  logic                         projection_reserve_ready,
    output logic [CONTEXT_ID_W-1:0]      projection_context_id,
    output logic [HEAD_ID_W-1:0]         projection_head_id,
    output logic [TAG_W-1:0]             projection_payload_tag,
    output logic [TAG_W-1:0]             projection_execution_tag,
    output logic [ROUTE_W-1:0]           projection_route,
    output logic [FORMAT_W-1:0]          projection_format,
    output logic [HEAD_COUNT_W-1:0]      projection_head_index,
    output logic [INPUT_CH_W-1:0]        projection_input_channel_base,
    output logic [OUTPUT_TILE_W-1:0]     projection_output_tile,
    output logic                         projection_last_head,
    output logic [7:0]                   projection_resident_term_count,
    output logic [EVENT_COUNT_W-1:0]     projection_resident_event_count,

    output logic                         slot_commit_pulse,
    input  logic                         slot_reserve_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_context_id,
    output logic [HEAD_ID_W-1:0]         slot_head_id,
    output logic [TAG_W-1:0]             slot_payload_tag,
    output logic [WORD_INDEX_W-1:0]      slot_replay_start_word,

    output logic                         lifecycle_commit_pulse,
    input  logic                         lifecycle_reserve_ready,
    output logic [CONTEXT_ID_W-1:0]      lifecycle_context_id,
    output logic [HEAD_ID_W-1:0]         lifecycle_head_id,
    output logic [TAG_W-1:0]             lifecycle_payload_tag,
    output logic [TAG_W-1:0]             lifecycle_execution_tag,
    output logic                         lifecycle_cache_owned,
    output logic                         lifecycle_last_output_tile,

    output logic                         reject_valid,
    input  logic                         reject_ready,
    output logic [TAG_W-1:0]             reject_execution_tag,
    output logic                         commit_pulse,
    output logic [TAG_W-1:0]             commit_execution_tag,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_commits,
    output logic [COUNTER_W-1:0]         count_rejects
);
    localparam logic [ROUTE_W-1:0] ROUTE_RESIDENT = ROUTE_W'(0);
    localparam logic [ROUTE_W-1:0] ROUTE_IPD = ROUTE_W'(1);
    localparam logic [ROUTE_W-1:0] ROUTE_RAW = ROUTE_W'(2);
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);

    logic context_legal;
    logic head_legal;
    logic route_legal;
    logic format_legal;
    logic route_format_legal;
    logic resident_contract_legal;
    logic replay_contract_legal;
    logic ownership_legal;
    logic offset_legal;
    logic plan_legal;
    logic slot_ready_or_unused;
    logic all_resources_ready;
    logic commit_fire;
    logic reject_fire;

    assign context_legal = 32'(plan_context_id) < 32'(CONTEXTS);
    assign head_legal = 32'(plan_head_id) < 32'(HEADS) &&
                        32'(plan_head_index) < 32'(HEADS);
    assign route_legal = plan_route == ROUTE_RESIDENT ||
                         plan_route == ROUTE_IPD ||
                         plan_route == ROUTE_RAW;
    assign format_legal = plan_format == FORMAT_RAW ||
                          plan_format == FORMAT_IPD32W ||
                          plan_format == FORMAT_FADC24;
    assign route_format_legal =
        (plan_route == ROUTE_RESIDENT && plan_format == FORMAT_IPD32W) ||
        (plan_route == ROUTE_IPD &&
         (plan_format == FORMAT_IPD32W || plan_format == FORMAT_FADC24)) ||
        (plan_route == ROUTE_RAW && plan_format == FORMAT_RAW);
    assign resident_contract_legal =
        plan_route != ROUTE_RESIDENT ||
        (32'(plan_resident_term_count) <= 32'(RESIDENT_TERMS) &&
         ((plan_resident_term_count == '0) ==
          (plan_resident_event_count == '0)) &&
         plan_cache_owned);
    assign replay_contract_legal =
        (plan_route == ROUTE_RESIDENT &&
         (plan_slot_replay_required ==
          (plan_resident_event_count != '0))) ||
        ((plan_route == ROUTE_IPD || plan_route == ROUTE_RAW) &&
         plan_slot_replay_required && !plan_cache_owned);
    assign ownership_legal =
        plan_cache_owned == (plan_route == ROUTE_RESIDENT);
    assign offset_legal = plan_route == ROUTE_RESIDENT ?
        plan_replay_start_word == WORD_INDEX_W'(
            2 + ((32'(plan_resident_term_count) + 1) >> 1)) :
        plan_replay_start_word == '0;
    assign plan_legal = context_legal && head_legal && route_legal &&
                        format_legal && route_format_legal &&
                        resident_contract_legal && replay_contract_legal &&
                        ownership_legal && offset_legal;

    assign slot_ready_or_unused = !plan_slot_replay_required ||
                                  slot_reserve_ready;
    assign all_resources_ready = projection_reserve_ready &&
                                 lifecycle_reserve_ready &&
                                 slot_ready_or_unused;
    assign plan_ready = plan_legal ? all_resources_ready : reject_ready;
    assign commit_fire = plan_valid && plan_legal && all_resources_ready;
    assign reject_fire = plan_valid && !plan_legal && reject_ready;

    assign projection_commit_pulse = commit_fire;
    assign slot_commit_pulse = commit_fire && plan_slot_replay_required;
    assign lifecycle_commit_pulse = commit_fire;
    assign reject_valid = plan_valid && !plan_legal;

    assign projection_context_id = plan_context_id;
    assign projection_head_id = plan_head_id;
    assign projection_payload_tag = plan_payload_tag;
    assign projection_execution_tag = plan_execution_tag;
    assign projection_route = plan_route;
    assign projection_format = plan_format;
    assign projection_head_index = plan_head_index;
    assign projection_input_channel_base = plan_input_channel_base;
    assign projection_output_tile = plan_output_tile;
    assign projection_last_head = plan_last_head;
    assign projection_resident_term_count = plan_resident_term_count;
    assign projection_resident_event_count = plan_resident_event_count;

    assign slot_context_id = plan_context_id;
    assign slot_head_id = plan_head_id;
    assign slot_payload_tag = plan_payload_tag;
    assign slot_replay_start_word = plan_replay_start_word;

    assign lifecycle_context_id = plan_context_id;
    assign lifecycle_head_id = plan_head_id;
    assign lifecycle_payload_tag = plan_payload_tag;
    assign lifecycle_execution_tag = plan_execution_tag;
    // Final-tile release is issued for every residency-managed IPD payload.
    // The descriptor cache accepts an absent line as an idempotent no-op, which
    // also covers a cold final tile or a capacity-bypassed fill.
    assign lifecycle_cache_owned = ENABLE_RESIDENCY != 0 &&
                                   plan_format == FORMAT_IPD32W;
    assign lifecycle_last_output_tile = plan_last_output_tile;
    assign reject_execution_tag = plan_execution_tag;
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            commit_pulse <= 1'b0;
            commit_execution_tag <= '0;
            protocol_error <= 1'b0;
            count_commits <= '0;
            count_rejects <= '0;
        end else begin
            commit_pulse <= commit_fire;
            if (commit_fire) begin
                commit_execution_tag <= plan_execution_tag;
                count_commits <= count_commits + 1'b1;
            end
            if (reject_fire) begin
                protocol_error <= 1'b1;
                count_rejects <= count_rejects + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
