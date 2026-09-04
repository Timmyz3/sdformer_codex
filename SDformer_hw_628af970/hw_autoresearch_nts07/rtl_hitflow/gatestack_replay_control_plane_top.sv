`timescale 1ns/1ps
`default_nettype none

// Single replay transaction owner: metadata PLAN, atomic resource COMMIT and
// dual-tag last-use lifecycle. Datapath streams remain outside this boundary.
module gatestack_replay_control_plane_top #(
    parameter int CONTEXTS        = 2,
    parameter int HEADS           = 24,
    parameter int HEAD_BITS       = 6642,
    parameter int SLOT_CAPACITY_BITS = ((HEAD_BITS + 63) / 64) * 64,
    parameter int RESIDENT_TERMS  = 80,
    parameter int ENABLE_RESIDENCY = 1,
    parameter int CSR_FORMAT_FADC24 = 0,
    parameter int TAG_W           = 32,
    parameter int SIZE_W          = 16,
    parameter int FORMAT_W        = 2,
    parameter int EVENT_COUNT_W   = 13,
    parameter int WORD_INDEX_W    = 7,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int HEAD_COUNT_W    = 6,
    parameter int COUNTER_W       = 32,
    parameter int CONTEXT_ID_W    = (CONTEXTS <= 1) ?
                                     1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W       = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int ROUTE_W         = 2
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         head_request_valid,
    output logic                         head_request_ready,
    input  logic [CONTEXT_ID_W-1:0]      head_request_context_id,
    input  logic [HEAD_ID_W-1:0]         head_request_head_id,
    input  logic [TAG_W-1:0]             head_request_execution_tag,
    input  logic [HEAD_COUNT_W-1:0]      head_request_head_index,
    input  logic [INPUT_CH_W-1:0]        head_request_input_channel_base,
    input  logic [OUTPUT_TILE_W-1:0]     head_request_output_tile,
    input  logic                         head_request_last_head,
    input  logic                         head_request_last_output_tile,

    output logic                         slot_inspect_valid,
    input  logic                         slot_inspect_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_inspect_context_id,
    output logic [HEAD_ID_W-1:0]         slot_inspect_head_id,
    input  logic                         slot_meta_valid,
    output logic                         slot_meta_ready,
    input  logic                         slot_meta_exists,
    input  logic [TAG_W-1:0]             slot_meta_tag,
    input  logic                         slot_meta_mode_is_csr,
    input  logic [FORMAT_W-1:0]          slot_meta_format,
    input  logic [SIZE_W-1:0]            slot_meta_payload_bits,
    input  logic [SIZE_W-1:0]            slot_meta_word_count,

    output logic                         cache_lookup_valid,
    input  logic                         cache_lookup_ready,
    output logic [CONTEXT_ID_W-1:0]      cache_lookup_context_id,
    output logic [HEAD_ID_W-1:0]         cache_lookup_head_id,
    output logic [TAG_W-1:0]             cache_lookup_expected_tag,
    input  logic                         cache_meta_valid,
    output logic                         cache_meta_ready,
    input  logic                         cache_meta_hit,
    input  logic [TAG_W-1:0]             cache_meta_tag,
    input  logic [7:0]                   cache_meta_term_count,

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
    output logic [CONTEXT_ID_W-1:0]      slot_replay_context_id,
    output logic [HEAD_ID_W-1:0]         slot_replay_head_id,
    output logic [TAG_W-1:0]             slot_replay_payload_tag,
    output logic [WORD_INDEX_W-1:0]      slot_replay_start_word,

    input  logic                         decoder_done_valid,
    output logic                         decoder_done_ready,
    input  logic [TAG_W-1:0]             decoder_done_payload_tag,
    input  logic                         decoder_done_error,
    input  logic                         backend_done_valid,
    output logic                         backend_done_ready,
    input  logic [TAG_W-1:0]             backend_done_execution_tag,
    input  logic                         backend_done_error,

    output logic                         slot_release_valid,
    input  logic                         slot_release_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_release_context_id,
    output logic [HEAD_ID_W-1:0]         slot_release_head_id,
    output logic                         cache_release_valid,
    input  logic                         cache_release_ready,
    output logic [CONTEXT_ID_W-1:0]      cache_release_context_id,
    output logic [HEAD_ID_W-1:0]         cache_release_head_id,
    output logic [TAG_W-1:0]             cache_release_payload_tag,

    output logic                         head_complete_valid,
    input  logic                         head_complete_ready,
    output logic [CONTEXT_ID_W-1:0]      head_complete_context_id,
    output logic [HEAD_ID_W-1:0]         head_complete_head_id,
    output logic [HEAD_COUNT_W-1:0]      head_complete_head_index,
    output logic                         head_complete_last_head,
    output logic [TAG_W-1:0]             head_complete_payload_tag,
    output logic [TAG_W-1:0]             head_complete_execution_tag,
    output logic                         head_complete_error,

    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_requests,
    output logic [COUNTER_W-1:0]         count_commits,
    output logic [COUNTER_W-1:0]         count_rejects,
    output logic [COUNTER_W-1:0]         count_sessions
);
    logic plan_valid, plan_ready;
    logic builder_request_ready;
    logic [CONTEXT_ID_W-1:0] plan_context_id;
    logic [HEAD_ID_W-1:0] plan_head_id;
    logic [TAG_W-1:0] plan_payload_tag, plan_execution_tag;
    logic [ROUTE_W-1:0] plan_route;
    logic [FORMAT_W-1:0] plan_format;
    logic [HEAD_COUNT_W-1:0] plan_head_index;
    logic [INPUT_CH_W-1:0] plan_input_channel_base;
    logic [OUTPUT_TILE_W-1:0] plan_output_tile;
    logic plan_last_head, plan_last_output_tile;
    logic plan_cache_owned, plan_slot_replay_required;
    logic [WORD_INDEX_W-1:0] plan_replay_start_word;
    logic [7:0] plan_resident_term_count;
    logic [EVENT_COUNT_W-1:0] plan_resident_event_count;
    logic builder_reject_valid, builder_reject_ready;
    logic [TAG_W-1:0] builder_reject_payload_tag;
    logic [TAG_W-1:0] builder_reject_execution_tag;
    logic builder_protocol_error;
    logic [COUNTER_W-1:0] builder_count_requests;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] count_resident_plans;
    logic [COUNTER_W-1:0] count_ipd_plans;
    logic [COUNTER_W-1:0] count_fadc_plans;
    logic [COUNTER_W-1:0] count_raw_plans;
    logic [COUNTER_W-1:0] builder_count_rejects;
    /* verilator lint_on UNUSEDSIGNAL */

    logic lifecycle_commit_pulse;
    logic lifecycle_reserve_ready;
    logic [CONTEXT_ID_W-1:0] lifecycle_context_id;
    logic [HEAD_ID_W-1:0] lifecycle_head_id;
    logic [TAG_W-1:0] lifecycle_payload_tag;
    logic [TAG_W-1:0] lifecycle_execution_tag;
    logic lifecycle_cache_owned, lifecycle_last_output_tile;
    logic atomic_reject_valid, atomic_reject_ready;
    logic [TAG_W-1:0] atomic_reject_execution_tag;
    /* verilator lint_off UNUSEDSIGNAL */
    logic atomic_commit_done_pulse;
    logic [TAG_W-1:0] atomic_commit_execution_tag;
    /* verilator lint_on UNUSEDSIGNAL */
    logic atomic_protocol_error;
    logic [COUNTER_W-1:0] atomic_count_commits;
    logic [COUNTER_W-1:0] atomic_count_rejects;

    logic lifecycle_done_valid, lifecycle_done_ready;
    logic [TAG_W-1:0] lifecycle_done_payload_tag;
    logic [TAG_W-1:0] lifecycle_done_execution_tag;
    logic lifecycle_done_error, lifecycle_protocol_error;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] count_final_tile_releases;
    logic [COUNTER_W-1:0] count_cache_releases;
    logic [COUNTER_W-1:0] count_session_errors;
    /* verilator lint_on UNUSEDSIGNAL */

    logic [CONTEXT_ID_W-1:0] request_context_q;
    logic [HEAD_ID_W-1:0] request_head_q;
    logic [HEAD_COUNT_W-1:0] request_head_index_q;
    logic request_last_head_q;
    logic request_outstanding_q;
    logic request_fire;
    logic completion_fire;
    logic builder_reject_selected, atomic_reject_selected;

    assign head_request_ready = builder_request_ready &&
                                !request_outstanding_q;
    assign request_fire = head_request_valid && head_request_ready;
    assign completion_fire = head_complete_valid && head_complete_ready;
    assign count_requests = builder_count_requests;
    assign count_commits = atomic_count_commits;
    assign count_rejects = builder_count_rejects + atomic_count_rejects;

    gatestack_replay_plan_builder #(
        .CONTEXTS(CONTEXTS), .HEADS(HEADS), .HEAD_BITS(HEAD_BITS),
        .SLOT_CAPACITY_BITS(SLOT_CAPACITY_BITS),
        .RESIDENT_TERMS(RESIDENT_TERMS),
        .ENABLE_RESIDENCY(ENABLE_RESIDENCY),
        .CSR_FORMAT_FADC24(CSR_FORMAT_FADC24),
        .TAG_W(TAG_W), .SIZE_W(SIZE_W), .FORMAT_W(FORMAT_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .WORD_INDEX_W(WORD_INDEX_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .COUNTER_W(COUNTER_W),
        .CONTEXT_ID_W(CONTEXT_ID_W), .HEAD_ID_W(HEAD_ID_W),
        .ROUTE_W(ROUTE_W)
    ) u_plan_builder (
        .clk_core, .rst_core,
        .request_valid(head_request_valid && !request_outstanding_q),
        .request_ready(builder_request_ready),
        .request_context_id(head_request_context_id),
        .request_head_id(head_request_head_id),
        .request_execution_tag(head_request_execution_tag),
        .request_head_index(head_request_head_index),
        .request_input_channel_base(head_request_input_channel_base),
        .request_output_tile(head_request_output_tile),
        .request_last_head(head_request_last_head),
        .request_last_output_tile(head_request_last_output_tile),
        .slot_inspect_valid, .slot_inspect_ready,
        .slot_inspect_context_id, .slot_inspect_head_id,
        .slot_meta_valid, .slot_meta_ready, .slot_meta_exists,
        .slot_meta_tag, .slot_meta_mode_is_csr, .slot_meta_format,
        .slot_meta_payload_bits, .slot_meta_word_count,
        .cache_lookup_valid, .cache_lookup_ready,
        .cache_lookup_context_id, .cache_lookup_head_id,
        .cache_lookup_expected_tag, .cache_meta_valid,
        .cache_meta_ready, .cache_meta_hit, .cache_meta_tag,
        .cache_meta_term_count, .plan_valid, .plan_ready,
        .plan_context_id, .plan_head_id, .plan_payload_tag,
        .plan_execution_tag, .plan_route, .plan_format, .plan_head_index,
        .plan_input_channel_base, .plan_output_tile, .plan_last_head,
        .plan_last_output_tile, .plan_cache_owned,
        .plan_slot_replay_required, .plan_replay_start_word,
        .plan_resident_term_count, .plan_resident_event_count,
        .reject_valid(builder_reject_valid),
        .reject_ready(builder_reject_ready),
        .reject_payload_tag(builder_reject_payload_tag),
        .reject_execution_tag(builder_reject_execution_tag),
        .protocol_error(builder_protocol_error),
        .count_requests(builder_count_requests),
        .count_resident_plans, .count_ipd_plans, .count_fadc_plans,
        .count_raw_plans,
        .count_rejects(builder_count_rejects)
    );

    gatestack_replay_atomic_commit #(
        .CONTEXTS(CONTEXTS), .HEADS(HEADS),
        .RESIDENT_TERMS(RESIDENT_TERMS), .TAG_W(TAG_W),
        .ENABLE_RESIDENCY(ENABLE_RESIDENCY),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .EVENT_COUNT_W(EVENT_COUNT_W),
        .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
        .FORMAT_W(FORMAT_W),
        .CONTEXT_ID_W(CONTEXT_ID_W), .HEAD_ID_W(HEAD_ID_W),
        .ROUTE_W(ROUTE_W)
    ) u_atomic_commit (
        .clk_core, .rst_core, .plan_valid, .plan_ready,
        .plan_context_id, .plan_head_id, .plan_payload_tag,
        .plan_execution_tag, .plan_route, .plan_format, .plan_head_index,
        .plan_input_channel_base, .plan_output_tile, .plan_last_head,
        .plan_last_output_tile, .plan_cache_owned,
        .plan_slot_replay_required, .plan_replay_start_word,
        .plan_resident_term_count, .plan_resident_event_count,
        .projection_commit_pulse, .projection_reserve_ready,
        .projection_context_id, .projection_head_id,
        .projection_payload_tag, .projection_execution_tag,
        .projection_route, .projection_format, .projection_head_index,
        .projection_input_channel_base, .projection_output_tile,
        .projection_last_head, .projection_resident_term_count,
        .projection_resident_event_count, .slot_commit_pulse,
        .slot_reserve_ready, .slot_context_id(slot_replay_context_id),
        .slot_head_id(slot_replay_head_id),
        .slot_payload_tag(slot_replay_payload_tag),
        .slot_replay_start_word, .lifecycle_commit_pulse,
        .lifecycle_reserve_ready, .lifecycle_context_id,
        .lifecycle_head_id, .lifecycle_payload_tag,
        .lifecycle_execution_tag, .lifecycle_cache_owned,
        .lifecycle_last_output_tile,
        .reject_valid(atomic_reject_valid),
        .reject_ready(atomic_reject_ready),
        .reject_execution_tag(atomic_reject_execution_tag),
        .commit_pulse(atomic_commit_done_pulse),
        .commit_execution_tag(atomic_commit_execution_tag),
        .protocol_error(atomic_protocol_error),
        .count_commits(atomic_count_commits),
        .count_rejects(atomic_count_rejects)
    );

    gatestack_dualtag_replay_lifecycle_manager #(
        .CONTEXTS(CONTEXTS), .HEADS(HEADS), .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W), .CONTEXT_ID_W(CONTEXT_ID_W),
        .HEAD_ID_W(HEAD_ID_W)
    ) u_lifecycle (
        .clk_core, .rst_core,
        .session_valid(lifecycle_commit_pulse),
        .session_ready(lifecycle_reserve_ready),
        .session_context_id(lifecycle_context_id),
        .session_head_id(lifecycle_head_id),
        .session_payload_tag(lifecycle_payload_tag),
        .session_execution_tag(lifecycle_execution_tag),
        .session_cache_owned(lifecycle_cache_owned),
        .session_last_output_tile(lifecycle_last_output_tile),
        .decoder_done_valid, .decoder_done_ready,
        .decoder_done_payload_tag, .decoder_done_error,
        .backend_done_valid, .backend_done_ready,
        .backend_done_execution_tag, .backend_done_error,
        .slot_release_valid, .slot_release_ready,
        .slot_release_context_id, .slot_release_head_id,
        .cache_release_valid, .cache_release_ready,
        .cache_release_context_id, .cache_release_head_id,
        .cache_release_payload_tag,
        .session_done_valid(lifecycle_done_valid),
        .session_done_ready(lifecycle_done_ready),
        .session_done_payload_tag(lifecycle_done_payload_tag),
        .session_done_execution_tag(lifecycle_done_execution_tag),
        .session_done_error(lifecycle_done_error),
        .protocol_error(lifecycle_protocol_error),
        .count_sessions, .count_final_tile_releases,
        .count_cache_releases, .count_session_errors
    );

    assign builder_reject_selected = builder_reject_valid;
    assign atomic_reject_selected = !builder_reject_selected &&
                                    atomic_reject_valid;
    assign head_complete_valid = lifecycle_done_valid ||
                                 builder_reject_selected ||
                                 atomic_reject_selected;
    assign lifecycle_done_ready = head_complete_ready &&
                                  lifecycle_done_valid;
    assign builder_reject_ready = head_complete_ready &&
                                  !lifecycle_done_valid;
    assign atomic_reject_ready = head_complete_ready &&
                                 !lifecycle_done_valid &&
                                 !builder_reject_valid;

    assign head_complete_context_id = request_context_q;
    assign head_complete_head_id = request_head_q;
    assign head_complete_head_index = request_head_index_q;
    assign head_complete_last_head = request_last_head_q;
    assign head_complete_payload_tag = lifecycle_done_valid ?
        lifecycle_done_payload_tag :
        (builder_reject_selected ? builder_reject_payload_tag :
         plan_payload_tag);
    assign head_complete_execution_tag = lifecycle_done_valid ?
        lifecycle_done_execution_tag :
        (builder_reject_selected ? builder_reject_execution_tag :
         atomic_reject_execution_tag);
    assign head_complete_error = lifecycle_done_valid ?
                                 lifecycle_done_error : 1'b1;
    assign protocol_error = builder_protocol_error || atomic_protocol_error ||
                            lifecycle_protocol_error;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            request_context_q <= '0;
            request_head_q <= '0;
            request_head_index_q <= '0;
            request_last_head_q <= 1'b0;
            request_outstanding_q <= 1'b0;
        end else begin
            if (request_fire) begin
                request_context_q <= head_request_context_id;
                request_head_q <= head_request_head_id;
                request_head_index_q <= head_request_head_index;
                request_last_head_q <= head_request_last_head;
                request_outstanding_q <= 1'b1;
            end
            if (completion_fire)
                request_outstanding_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
