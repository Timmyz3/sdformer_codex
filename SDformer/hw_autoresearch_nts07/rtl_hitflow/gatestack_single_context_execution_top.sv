`timescale 1ns/1ps
`default_nettype none

// Single-context execution slice. Upstream payload/cache fill and downstream
// weight/bias/final interfaces remain explicit architectural boundaries.
module gatestack_single_context_execution_top #(
    parameter int TOKENS = 162,
    parameter int HEADS = 24,
    parameter int HEAD_BITS = 6642,
    parameter int SLOT_CAPACITY_BITS = ((HEAD_BITS + 63) / 64) * 64,
    parameter int MAX_TERMS = 128,
    parameter int RESIDENT_TERMS = 80,
    parameter int ENABLE_RESIDENCY = 1,
    parameter int EXTERNAL_SLOT_SERVICE_ENABLE = 0,
    // 0: IPD32W, 1: FADC24, 2: header-steered adaptive IPD32W/FADC24.
    parameter int CSR_FORMAT_FADC24 = 0,
    parameter int EVENT_WAYS = 4,
    parameter int OUT_TILE = 32,
    parameter int BANKS = 2,
    parameter int SEGMENT_TOKENS = 18,
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int OUTPUT_TILE_COUNT_W = 8,
    parameter int HEAD_COUNT_W = 6,
    parameter int SIZE_W = 16,
    parameter int FORMAT_W = 2,
    parameter int ISSUE_SEQ_W = 13,
    parameter int EVENT_COUNT_W = 13,
    parameter int WORD_INDEX_W = 7,
    parameter int COUNTER_W = 32,
    parameter int TOKEN_ID_W = 8,
    parameter int HEAD_ID_W = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int RES_TERM_IDX_W = (RESIDENT_TERMS <= 1) ?
                                     1 : $clog2(RESIDENT_TERMS),
    parameter int ROUTE_W = 2,
    parameter int WEIGHT_W = 8,
    parameter int GATE_W = 9,
    parameter int PRODUCT_W = GATE_W + WEIGHT_W,
    parameter int ACC_W = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter int ABORT_TIMEOUT_CYCLES = 1000000
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      group_valid,
    output logic                                      group_ready,
    input  logic [TAG_W-1:0]                          group_tag,
    input  logic [HEAD_COUNT_W-1:0]                   group_head_count,
    input  logic [OUTPUT_TILE_W-1:0]                  group_first_output_tile,
    input  logic [OUTPUT_TILE_COUNT_W-1:0]            group_output_tile_count,
    output logic                                      group_done_valid,
    input  logic                                      group_done_ready,
    output logic [TAG_W-1:0]                          group_done_tag,
    output logic                                      group_done_error,

    input  logic                                      payload_commit_begin_valid,
    output logic                                      payload_commit_begin_ready,
    input  logic [HEAD_ID_W-1:0]                      payload_commit_head_id,
    input  logic [TAG_W-1:0]                          payload_commit_tag,
    input  logic                                      payload_commit_mode_is_csr,
    input  logic [SIZE_W-1:0]                         payload_commit_bits,
    input  logic                                      payload_commit_word_valid,
    output logic                                      payload_commit_word_ready,
    input  logic [63:0]                               payload_commit_word_data,
    input  logic                                      payload_commit_word_last,

    output logic                                      external_slot_inspect_valid,
    input  logic                                      external_slot_inspect_ready,
    output logic                                      external_slot_inspect_context_id,
    output logic [HEAD_ID_W-1:0]                      external_slot_inspect_head_id,
    input  logic                                      external_slot_inspect_meta_valid,
    output logic                                      external_slot_inspect_meta_ready,
    input  logic                                      external_slot_inspect_exists,
    input  logic [TAG_W-1:0]                          external_slot_inspect_tag,
    input  logic                                      external_slot_inspect_mode_is_csr,
    input  logic [FORMAT_W-1:0]                       external_slot_inspect_format,
    input  logic [SIZE_W-1:0]                         external_slot_inspect_payload_bits,
    input  logic [SIZE_W-1:0]                         external_slot_inspect_word_count,

    output logic                                      external_slot_replay_begin_valid,
    input  logic                                      external_slot_replay_begin_ready,
    output logic                                      external_slot_replay_context_id,
    output logic [HEAD_ID_W-1:0]                      external_slot_replay_head_id,
    output logic [TAG_W-1:0]                          external_slot_replay_payload_tag,
    output logic [WORD_INDEX_W-1:0]                   external_slot_replay_start_word,
    input  logic                                      external_slot_replay_word_valid,
    output logic                                      external_slot_replay_word_ready,
    input  logic [63:0]                               external_slot_replay_word_data,
    input  logic [WORD_INDEX_W-1:0]                   external_slot_replay_word_index,
    input  logic                                      external_slot_replay_word_last,
    input  logic [TAG_W-1:0]                          external_slot_replay_tag,
    input  logic                                      external_slot_replay_mode_is_csr,
    input  logic [FORMAT_W-1:0]                       external_slot_replay_format,
    input  logic [SIZE_W-1:0]                         external_slot_replay_payload_bits,

    output logic                                      external_slot_release_valid,
    input  logic                                      external_slot_release_ready,
    output logic                                      external_slot_release_context_id,
    output logic [HEAD_ID_W-1:0]                      external_slot_release_head_id,
    input  logic [HEADS-1:0]                          external_slot_valid_flat,
    input  logic                                      external_slot_protocol_error,
    input  logic [COUNTER_W-1:0]                      external_slot_count_replays,
    input  logic [COUNTER_W-1:0]                      external_slot_count_releases,
    output logic                                      external_slot_reset_pulse,

    input  logic                                      descriptor_fill_begin_valid,
    output logic                                      descriptor_fill_begin_ready,
    input  logic [HEAD_ID_W-1:0]                      descriptor_fill_head_id,
    input  logic [TAG_W-1:0]                          descriptor_fill_tag,
    input  logic [7:0]                                descriptor_fill_term_count,
    input  logic [FORMAT_W-1:0]                       descriptor_fill_format,
    output logic                                      descriptor_fill_begin_cacheable,
    input  logic                                      descriptor_fill_entry_valid,
    output logic                                      descriptor_fill_entry_ready,
    input  logic [8:0]                                descriptor_fill_gate_code,
    input  logic [4:0]                                descriptor_fill_lane_id,
    input  logic [7:0]                                descriptor_fill_destination_count,
    input  logic                                      descriptor_fill_entry_last,

    output logic                                      weight_req_valid,
    input  logic                                      weight_req_ready,
    output logic [TAG_W-1:0]                          weight_req_tag,
    output logic [INPUT_CH_W-1:0]                     weight_req_input_channel,
    output logic [OUTPUT_TILE_W-1:0]                  weight_req_output_tile,
    input  logic                                      weight_rsp_valid,
    output logic                                      weight_rsp_ready,
    input  logic [TAG_W-1:0]                          weight_rsp_tag,
    input  logic [INPUT_CH_W-1:0]                     weight_rsp_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]                  weight_rsp_output_tile,
    input  logic [(OUT_TILE*WEIGHT_W)-1:0]            weight_rsp_weights,
    output logic                                      bias_req_valid,
    input  logic                                      bias_req_ready,
    output logic [TAG_W-1:0]                          bias_req_tag,
    output logic [OUTPUT_TILE_W-1:0]                  bias_req_output_tile,
    output logic [TOKEN_ID_W-1:0]                     bias_req_token_id,
    input  logic                                      bias_rsp_valid,
    output logic                                      bias_rsp_ready,
    input  logic [TAG_W-1:0]                          bias_rsp_tag,
    input  logic [TOKEN_ID_W-1:0]                     bias_rsp_token_id,
    input  logic [(OUT_TILE*ACC_W)-1:0]               bias_rsp_values,
    output logic [BANKS-1:0]                          final_valid,
    input  logic [BANKS-1:0]                          final_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]             final_token_ids,
    output logic [TAG_W-1:0]                          final_tag,
    output logic [(BANKS*OUT_TILE*ACC_W)-1:0]         final_values,

    output logic                                      protocol_error,
    output logic [COUNTER_W-1:0]                      count_groups,
    output logic [COUNTER_W-1:0]                      count_tile_starts,
    output logic [COUNTER_W-1:0]                      count_head_issues,
    output logic [COUNTER_W-1:0]                      count_control_requests,
    output logic [COUNTER_W-1:0]                      count_control_commits,
    output logic [COUNTER_W-1:0]                      count_control_rejects,
    output logic [COUNTER_W-1:0]                      count_control_sessions,
    output logic [COUNTER_W-1:0]                      count_slot_replays,
    output logic [COUNTER_W-1:0]                      count_slot_releases,
    output logic [COUNTER_W-1:0]                      count_cache_hits,
    output logic [COUNTER_W-1:0]                      count_cache_releases,
    output logic [COUNTER_W-1:0]                      count_projection_heads,
    output logic [COUNTER_W-1:0]                      count_projection_terms,
    output logic [COUNTER_W-1:0]                      count_bias_commits,
    output logic [COUNTER_W-1:0]                      count_context_resets,
    output logic [COUNTER_W-1:0]                      count_error_aborts,
    output logic [COUNTER_W-1:0]                      count_timeout_aborts
);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    logic fabric_rst_core;
    logic fabric_reset_pulse, fabric_protocol_error;
    logic abort_done_valid, abort_done_ready, abort_done_error;
    logic [TAG_W-1:0] abort_done_tag;
    logic abort_admission_blocked, abort_group_active;
    logic abort_protocol_error;
    logic scheduler_group_ready;
    logic scheduler_group_done_valid, scheduler_group_done_ready;
    logic [TAG_W-1:0] scheduler_group_done_tag;
    logic scheduler_group_done_error;
    logic tile_start_valid, tile_start_ready;
    logic [TAG_W-1:0] tile_start_tag;
    logic [OUTPUT_TILE_W-1:0] tile_start_output_tile;
    logic [HEAD_COUNT_W-1:0] tile_start_head_count;
    logic head_issue_valid, head_issue_ready;
    logic head_issue_context_id;
    logic [TAG_W-1:0] head_issue_tag;
    logic [HEAD_ID_W-1:0] head_issue_head_id;
    logic [HEAD_COUNT_W-1:0] head_issue_head_index;
    logic [INPUT_CH_W-1:0] head_issue_input_channel_base;
    logic [OUTPUT_TILE_W-1:0] head_issue_output_tile;
    logic head_issue_last_head, head_issue_last_output_tile;
    logic scheduler_head_done_valid, scheduler_head_done_ready;
    logic [TAG_W-1:0] scheduler_head_done_tag;
    logic [HEAD_ID_W-1:0] scheduler_head_done_head_id;
    logic scheduler_head_done_error;
    logic tile_done_valid, tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic scheduler_protocol_error;
    logic [COUNTER_W-1:0] count_group_errors;

    // These cache/fill nets are intentionally optimized away when the
    // compile-time no-residency baseline is selected.
    /* verilator lint_off UNUSEDSIGNAL */
    logic slot_inspect_valid, slot_inspect_ready;
    logic [HEAD_ID_W-1:0] slot_inspect_head_id;
    logic slot_meta_valid, slot_meta_ready, slot_meta_exists;
    logic [TAG_W-1:0] slot_meta_tag;
    logic slot_meta_mode_is_csr;
    logic [FORMAT_W-1:0] slot_meta_format;
    logic [SIZE_W-1:0] slot_meta_payload_bits, slot_meta_word_count;
    logic cache_lookup_valid, cache_lookup_ready;
    logic [HEAD_ID_W-1:0] cache_lookup_head_id;
    logic [TAG_W-1:0] cache_lookup_expected_tag;
    logic cache_meta_valid, cache_meta_ready, cache_meta_hit;
    logic [TAG_W-1:0] cache_meta_tag;
    logic [7:0] cache_meta_term_count;
    logic projection_commit_pulse, projection_reserve_ready;
    logic [HEAD_ID_W-1:0] projection_head_id;
    logic [TAG_W-1:0] projection_payload_tag;
    logic [TAG_W-1:0] projection_execution_tag;
    logic [ROUTE_W-1:0] projection_route;
    logic [FORMAT_W-1:0] projection_format;
    logic [HEAD_COUNT_W-1:0] projection_head_index;
    logic [INPUT_CH_W-1:0] projection_input_channel_base;
    logic [OUTPUT_TILE_W-1:0] projection_output_tile;
    logic projection_last_head;
    logic [7:0] projection_resident_term_count;
    logic [EVENT_COUNT_W-1:0] projection_resident_event_count;
    logic slot_commit_pulse, slot_reserve_ready;
    logic [HEAD_ID_W-1:0] slot_replay_head_id;
    logic [TAG_W-1:0] slot_replay_payload_tag;
    logic [WORD_INDEX_W-1:0] slot_replay_start_word;
    logic control_decoder_done_valid, control_decoder_done_ready;
    logic [TAG_W-1:0] control_decoder_done_payload_tag;
    logic control_decoder_done_error;
    logic control_backend_done_valid, control_backend_done_ready;
    logic [TAG_W-1:0] control_backend_done_execution_tag;
    logic control_backend_done_error;
    logic slot_release_valid, slot_release_ready;
    logic [HEAD_ID_W-1:0] slot_release_head_id;
    logic cache_release_valid, cache_release_ready;
    logic [HEAD_ID_W-1:0] cache_release_head_id;
    logic [TAG_W-1:0] cache_release_payload_tag;
    logic head_complete_valid, head_complete_ready;
    logic [HEAD_ID_W-1:0] head_complete_head_id;
    logic [HEAD_COUNT_W-1:0] head_complete_head_index;
    logic head_complete_last_head;
    logic [TAG_W-1:0] head_complete_payload_tag;
    logic [TAG_W-1:0] head_complete_execution_tag;
    logic head_complete_error, control_protocol_error;

    logic slot_replay_word_valid, slot_replay_word_ready;
    logic [63:0] slot_replay_word_data;
    logic [WORD_INDEX_W-1:0] slot_replay_word_index;
    logic slot_replay_word_last;
    logic [TAG_W-1:0] slot_replay_tag;
    logic slot_replay_mode_is_csr;
    logic [FORMAT_W-1:0] slot_replay_format;
    logic [SIZE_W-1:0] slot_replay_payload_bits;
    logic slot_adapter_replay_begin_ready;
    logic slot_commit_session_active, slot_replay_session_active;
    logic [HEADS-1:0] slot_valid_flat;
    logic slot_protocol_error;
    logic [COUNTER_W-1:0] count_slot_commit_heads;
    logic [COUNTER_W-1:0] count_slot_invalid_headers;
    logic [COUNTER_W-1:0] count_slot_commit_stalls;
    logic [COUNTER_W-1:0] count_slot_replay_stalls;

    logic resident_word_valid, resident_word_ready;
    logic [63:0] resident_word_data;
    logic [WORD_INDEX_W-1:0] resident_word_index;
    logic resident_word_last;
    logic ipd_word_valid, ipd_word_ready;
    logic [63:0] ipd_word_data;
    logic [WORD_INDEX_W-1:0] ipd_word_index;
    logic ipd_word_last;
    logic raw_word_valid, raw_word_ready;
    logic [63:0] raw_word_data;
    logic [WORD_INDEX_W-1:0] raw_word_index;
    logic raw_word_last, replay_router_protocol_error;
    logic replay_router_active, replay_router_start_ready;

    logic descriptor_entry_valid, descriptor_entry_ready;
    logic [8:0] descriptor_gate_code;
    logic [4:0] descriptor_lane_id;
    logic [7:0] descriptor_destination_count;
    logic [RES_TERM_IDX_W-1:0] descriptor_term_index;
    logic descriptor_entry_last, cache_protocol_error;
    logic [(HEADS)-1:0] cache_valid_flat;
    logic [COUNTER_W-1:0] count_cached_heads, count_bypass_heads;
    logic [COUNTER_W-1:0] count_cache_misses;
    logic [COUNTER_W-1:0] count_cache_release_noops;
    logic [COUNTER_W-1:0] count_cache_release_tag_mismatches;
    logic cache_fill_begin_valid, cache_fill_begin_ready;
    logic [HEAD_ID_W-1:0] cache_fill_begin_head_id;
    logic [TAG_W-1:0] cache_fill_begin_tag;
    logic [7:0] cache_fill_begin_term_count;
    logic cache_fill_begin_cacheable;
    logic cache_fill_entry_valid, cache_fill_entry_ready;
    logic [8:0] cache_fill_entry_gate_code;
    logic [4:0] cache_fill_entry_lane_id;
    logic [7:0] cache_fill_entry_destination_count;
    logic cache_fill_entry_last;
    logic auto_fill_begin_valid, auto_fill_begin_ready;
    logic [HEAD_ID_W-1:0] auto_fill_begin_head_id;
    logic [TAG_W-1:0] auto_fill_begin_tag;
    logic [7:0] auto_fill_begin_term_count;
    logic auto_fill_entry_valid, auto_fill_entry_ready;
    logic [8:0] auto_fill_entry_gate_code;
    logic [4:0] auto_fill_entry_lane_id;
    logic [7:0] auto_fill_entry_destination_count;
    logic auto_fill_entry_last, auto_fill_adapter_active;
    logic auto_fill_protocol_error;
    logic [COUNTER_W-1:0] count_auto_cacheable_fills;
    logic [COUNTER_W-1:0] count_auto_bypass_fills;
    logic external_fill_cache_begin_valid;
    logic external_fill_cache_begin_ready;
    logic [HEAD_ID_W-1:0] external_fill_cache_begin_head_id;
    logic [TAG_W-1:0] external_fill_cache_begin_tag;
    logic [7:0] external_fill_cache_begin_term_count;
    logic external_fill_cache_entry_valid;
    logic external_fill_cache_entry_ready;
    logic [8:0] external_fill_cache_entry_gate_code;
    logic [4:0] external_fill_cache_entry_lane_id;
    logic [7:0] external_fill_cache_entry_destination_count;
    logic external_fill_cache_entry_last;
    logic external_fill_adapter_active, external_fill_protocol_error;
    logic [COUNTER_W-1:0] count_external_cacheable_fills;
    logic [COUNTER_W-1:0] count_external_bypass_fills;
    /* verilator lint_on UNUSEDSIGNAL */

    logic projection_head_start_ready;
    logic projection_decoder_done_valid, projection_decoder_done_ready;
    logic [TAG_W-1:0] projection_decoder_done_payload_tag;
    logic projection_decoder_done_error;
    logic projection_head_done_valid, projection_head_done_ready;
    logic [TAG_W-1:0] projection_head_done_tag;
    logic [HEAD_COUNT_W-1:0] projection_head_done_index;
    logic projection_head_done_last, projection_head_done_error;
    logic projection_protocol_error, accumulator_overflow;
    logic [COUNTER_W-1:0] count_projection_completed_terms;
    /* verilator lint_off UNUSEDSIGNAL */
    logic projection_ipd_fill_begin_valid;
    logic projection_ipd_fill_begin_ready;
    logic [TAG_W-1:0] projection_ipd_fill_begin_tag;
    logic [7:0] projection_ipd_fill_begin_term_count;
    logic projection_ipd_fill_entry_valid;
    logic projection_ipd_fill_entry_ready;
    logic [8:0] projection_ipd_fill_gate_code;
    logic [4:0] projection_ipd_fill_lane_id;
    logic [7:0] projection_ipd_fill_destination_count;
    logic projection_ipd_fill_entry_last;
    logic projection_ipd_fill_cache_allowed;
    /* verilator lint_on UNUSEDSIGNAL */
    logic guard_start_ready, guard_protocol_error, guard_active;
    logic [HEAD_COUNT_W-1:0] unused_head_complete_index;
    logic slot_inspect_context_id, cache_lookup_context_id;
    logic projection_context_id, slot_replay_context_id;
    logic slot_release_context_id, cache_release_context_id;
    logic head_complete_context_id;

    gatestack_output_tile_scheduler #(
        .CONTEXTS(1), .HEADS(HEADS), .LANES(32), .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .OUTPUT_TILE_COUNT_W(OUTPUT_TILE_COUNT_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .COUNTER_W(COUNTER_W),
        .CONTEXT_ID_W(1), .HEAD_ID_W(HEAD_ID_W)
    ) u_scheduler (
        .clk_core, .rst_core(fabric_rst_core),
        .group_valid(group_valid && !abort_admission_blocked),
        .group_ready(scheduler_group_ready),
        .group_context_id(1'b0), .group_tag, .group_head_count,
        .group_first_output_tile, .group_output_tile_count,
        .tile_start_valid, .tile_start_ready, .tile_start_tag,
        .tile_start_output_tile, .tile_start_head_count,
        .head_issue_valid, .head_issue_ready,
        .head_issue_context_id, .head_issue_tag, .head_issue_head_id,
        .head_issue_head_index, .head_issue_input_channel_base,
        .head_issue_output_tile, .head_issue_last_head,
        .head_issue_last_output_tile,
        .head_done_valid(scheduler_head_done_valid),
        .head_done_ready(scheduler_head_done_ready),
        .head_done_tag(scheduler_head_done_tag),
        .head_done_head_id(scheduler_head_done_head_id),
        .head_done_error(scheduler_head_done_error),
        .tile_done_valid, .tile_done_ready, .tile_done_tag,
        .tile_done_error(projection_protocol_error),
        .group_done_valid(scheduler_group_done_valid),
        .group_done_ready(scheduler_group_done_ready),
        .group_done_tag(scheduler_group_done_tag),
        .group_done_error(scheduler_group_done_error),
        .protocol_error(scheduler_protocol_error),
        .count_groups, .count_tile_starts, .count_head_issues,
        .count_group_errors
    );

    generate
        if (EXTERNAL_SLOT_SERVICE_ENABLE == 0) begin : g_internal_slot_service
            gatestack_head_slot_sram_adapter #(
                .CONTEXTS(1), .HEADS(HEADS), .HEAD_BITS(HEAD_BITS),
                .SLOT_CAPACITY_BITS(SLOT_CAPACITY_BITS),
                .TAG_W(TAG_W), .SIZE_W(SIZE_W), .COUNTER_W(COUNTER_W),
                .FORMAT_W(FORMAT_W),
                .CONTEXT_ID_W(1), .HEAD_ID_W(HEAD_ID_W),
                .WORD_INDEX_W(WORD_INDEX_W)
            ) u_head_slot (
                .clk_core, .rst_core(fabric_rst_core),
                .commit_begin_valid(payload_commit_begin_valid),
                .commit_begin_ready(payload_commit_begin_ready),
                .commit_context_id(1'b0), .commit_head_id(payload_commit_head_id),
                .commit_tag(payload_commit_tag),
                .commit_mode_is_csr(payload_commit_mode_is_csr),
                .commit_payload_bits(payload_commit_bits),
                .commit_word_valid(payload_commit_word_valid),
                .commit_word_ready(payload_commit_word_ready),
                .commit_word_data(payload_commit_word_data),
                .commit_word_last(payload_commit_word_last),
                .inspect_valid(slot_inspect_valid),
                .inspect_ready(slot_inspect_ready),
                .inspect_context_id(1'b0),
                .inspect_head_id(slot_inspect_head_id),
                .inspect_meta_valid(slot_meta_valid),
                .inspect_meta_ready(slot_meta_ready),
                .inspect_exists(slot_meta_exists),
                .inspect_tag(slot_meta_tag),
                .inspect_mode_is_csr(slot_meta_mode_is_csr),
                .inspect_format(slot_meta_format),
                .inspect_payload_bits(slot_meta_payload_bits),
                .inspect_word_count(slot_meta_word_count),
                .replay_begin_valid(slot_commit_pulse),
                .replay_begin_ready(slot_adapter_replay_begin_ready),
                .replay_context_id(1'b0),
                .replay_head_id(slot_replay_head_id),
                .replay_start_word(slot_replay_start_word),
                .replay_word_valid(slot_replay_word_valid),
                .replay_word_ready(slot_replay_word_ready),
                .replay_word_data(slot_replay_word_data),
                .replay_word_index(slot_replay_word_index),
                .replay_word_last(slot_replay_word_last),
                .replay_tag(slot_replay_tag),
                .replay_mode_is_csr(slot_replay_mode_is_csr),
                .replay_format(slot_replay_format),
                .replay_payload_bits(slot_replay_payload_bits),
                .release_valid(slot_release_valid),
                .release_ready(slot_release_ready),
                .release_context_id(1'b0),
                .release_head_id(slot_release_head_id),
                .commit_session_active(slot_commit_session_active),
                .replay_session_active(slot_replay_session_active),
                .slot_valid_flat(slot_valid_flat),
                .protocol_error(slot_protocol_error),
                .count_commit_heads(count_slot_commit_heads),
                .count_replay_heads(count_slot_replays),
                .count_release_heads(count_slot_releases),
                .count_invalid_headers(count_slot_invalid_headers),
                .count_commit_stall_cycles(count_slot_commit_stalls),
                .count_replay_stall_cycles(count_slot_replay_stalls)
            );

            assign external_slot_inspect_valid = 1'b0;
            assign external_slot_inspect_context_id = 1'b0;
            assign external_slot_inspect_head_id = '0;
            assign external_slot_inspect_meta_ready = 1'b0;
            assign external_slot_replay_begin_valid = 1'b0;
            assign external_slot_replay_context_id = 1'b0;
            assign external_slot_replay_head_id = '0;
            assign external_slot_replay_payload_tag = '0;
            assign external_slot_replay_start_word = '0;
            assign external_slot_replay_word_ready = 1'b0;
            assign external_slot_release_valid = 1'b0;
            assign external_slot_release_context_id = 1'b0;
            assign external_slot_release_head_id = '0;
            assign external_slot_reset_pulse = 1'b0;
        end else begin : g_external_slot_service
            assign payload_commit_begin_ready = 1'b0;
            assign payload_commit_word_ready = 1'b0;

            assign external_slot_inspect_valid = slot_inspect_valid;
            assign external_slot_inspect_context_id = slot_inspect_context_id;
            assign external_slot_inspect_head_id = slot_inspect_head_id;
            assign slot_inspect_ready = external_slot_inspect_ready;
            assign slot_meta_valid = external_slot_inspect_meta_valid;
            assign external_slot_inspect_meta_ready = slot_meta_ready;
            assign slot_meta_exists = external_slot_inspect_exists;
            assign slot_meta_tag = external_slot_inspect_tag;
            assign slot_meta_mode_is_csr =
                external_slot_inspect_mode_is_csr;
            assign slot_meta_format = external_slot_inspect_format;
            assign slot_meta_payload_bits =
                external_slot_inspect_payload_bits;
            assign slot_meta_word_count = external_slot_inspect_word_count;

            assign external_slot_replay_begin_valid = slot_commit_pulse;
            assign external_slot_replay_context_id = slot_replay_context_id;
            assign external_slot_replay_head_id = slot_replay_head_id;
            assign external_slot_replay_payload_tag = slot_replay_payload_tag;
            assign external_slot_replay_start_word = slot_replay_start_word;
            assign slot_adapter_replay_begin_ready =
                external_slot_replay_begin_ready;
            assign slot_replay_word_valid = external_slot_replay_word_valid;
            assign external_slot_replay_word_ready = slot_replay_word_ready;
            assign slot_replay_word_data = external_slot_replay_word_data;
            assign slot_replay_word_index = external_slot_replay_word_index;
            assign slot_replay_word_last = external_slot_replay_word_last;
            assign slot_replay_tag = external_slot_replay_tag;
            assign slot_replay_mode_is_csr =
                external_slot_replay_mode_is_csr;
            assign slot_replay_format = external_slot_replay_format;
            assign slot_replay_payload_bits =
                external_slot_replay_payload_bits;

            assign external_slot_release_valid = slot_release_valid;
            assign external_slot_release_context_id = slot_release_context_id;
            assign external_slot_release_head_id = slot_release_head_id;
            assign slot_release_ready = external_slot_release_ready;

            assign slot_valid_flat = external_slot_valid_flat;
            assign slot_protocol_error = external_slot_protocol_error;
            assign count_slot_replays = external_slot_count_replays;
            assign count_slot_releases = external_slot_count_releases;
            assign external_slot_reset_pulse = fabric_reset_pulse;
            assign slot_commit_session_active = 1'b0;
            assign slot_replay_session_active = 1'b0;
            assign count_slot_commit_heads = '0;
            assign count_slot_invalid_headers = '0;
            assign count_slot_commit_stalls = '0;
            assign count_slot_replay_stalls = '0;
        end
    endgenerate

    gatestack_ipd_cache_fill_adapter #(
        .TAG_W(TAG_W), .HEAD_ID_W(HEAD_ID_W), .COUNTER_W(COUNTER_W)
    ) u_external_fill_adapter (
        .clk_core, .rst_core(fabric_rst_core),
        .begin_valid(descriptor_fill_begin_valid),
        .begin_ready(descriptor_fill_begin_ready),
        .begin_head_id(descriptor_fill_head_id),
        .begin_tag(descriptor_fill_tag),
        .begin_term_count(descriptor_fill_term_count),
        .begin_cache_allowed(descriptor_fill_format == FORMAT_IPD32W),
        .entry_valid(descriptor_fill_entry_valid),
        .entry_ready(descriptor_fill_entry_ready),
        .entry_gate_code(descriptor_fill_gate_code),
        .entry_lane_id(descriptor_fill_lane_id),
        .entry_destination_count(descriptor_fill_destination_count),
        .entry_last(descriptor_fill_entry_last),
        .cache_begin_valid(external_fill_cache_begin_valid),
        .cache_begin_ready(external_fill_cache_begin_ready),
        .cache_begin_head_id(external_fill_cache_begin_head_id),
        .cache_begin_tag(external_fill_cache_begin_tag),
        .cache_begin_term_count(external_fill_cache_begin_term_count),
        .cache_begin_cacheable(cache_fill_begin_cacheable),
        .cache_entry_valid(external_fill_cache_entry_valid),
        .cache_entry_ready(external_fill_cache_entry_ready),
        .cache_entry_gate_code(external_fill_cache_entry_gate_code),
        .cache_entry_lane_id(external_fill_cache_entry_lane_id),
        .cache_entry_destination_count(
            external_fill_cache_entry_destination_count),
        .cache_entry_last(external_fill_cache_entry_last),
        .session_active(external_fill_adapter_active),
        .protocol_error(external_fill_protocol_error),
        .count_cacheable_fills(count_external_cacheable_fills),
        .count_bypass_fills(count_external_bypass_fills)
    );

    assign descriptor_fill_begin_cacheable =
        descriptor_fill_format == FORMAT_IPD32W &&
        cache_fill_begin_cacheable;

    generate
        if (ENABLE_RESIDENCY != 0) begin : g_descriptor_residency
            gatestack_descriptor_residency_cache #(
                .CONTEXTS(1), .HEADS(HEADS),
                .CACHE_TERMS(RESIDENT_TERMS), .TAG_W(TAG_W),
                .COUNTER_W(COUNTER_W), .CONTEXT_ID_W(1),
                .HEAD_ID_W(HEAD_ID_W), .TERM_INDEX_W(RES_TERM_IDX_W)
            ) u_descriptor_cache (
                .clk_core, .rst_core(fabric_rst_core),
                .fill_begin_valid(cache_fill_begin_valid),
                .fill_begin_ready(cache_fill_begin_ready),
                .fill_context_id(1'b0),
                .fill_head_id(cache_fill_begin_head_id),
                .fill_tag(cache_fill_begin_tag),
                .fill_term_count(cache_fill_begin_term_count),
                .fill_begin_cacheable(cache_fill_begin_cacheable),
                .fill_entry_valid(cache_fill_entry_valid),
                .fill_entry_ready(cache_fill_entry_ready),
                .fill_gate_code(cache_fill_entry_gate_code),
                .fill_lane_id(cache_fill_entry_lane_id),
                .fill_destination_count(cache_fill_entry_destination_count),
                .fill_entry_last(cache_fill_entry_last),
                .lookup_valid(cache_lookup_valid),
                .lookup_ready(cache_lookup_ready),
                .lookup_context_id(1'b0),
                .lookup_head_id(cache_lookup_head_id),
                .lookup_expected_tag(cache_lookup_expected_tag),
                .lookup_meta_valid(cache_meta_valid),
                .lookup_meta_ready(cache_meta_ready),
                .lookup_hit(cache_meta_hit), .lookup_tag(cache_meta_tag),
                .lookup_term_count(cache_meta_term_count),
                .lookup_entry_valid(descriptor_entry_valid),
                .lookup_entry_ready(descriptor_entry_ready),
                .lookup_gate_code(descriptor_gate_code),
                .lookup_lane_id(descriptor_lane_id),
                .lookup_destination_count(descriptor_destination_count),
                .lookup_term_index(descriptor_term_index),
                .lookup_entry_last(descriptor_entry_last),
                .release_valid(cache_release_valid),
                .release_ready(cache_release_ready),
                .release_context_id(1'b0),
                .release_head_id(cache_release_head_id),
                .release_expected_tag(cache_release_payload_tag),
                .cache_valid_flat, .protocol_error(cache_protocol_error),
                .count_cached_heads, .count_bypass_heads,
                .count_lookup_hits(count_cache_hits),
                .count_lookup_misses(count_cache_misses),
                .count_releases(count_cache_releases),
                .count_release_noops(count_cache_release_noops),
                .count_release_tag_mismatches(
                    count_cache_release_tag_mismatches)
            );

            gatestack_ipd_cache_fill_adapter #(
                .TAG_W(TAG_W), .HEAD_ID_W(HEAD_ID_W),
                .COUNTER_W(COUNTER_W)
            ) u_ipd_fill_adapter (
                .clk_core, .rst_core(fabric_rst_core),
                .begin_valid(projection_ipd_fill_begin_valid),
                .begin_ready(projection_ipd_fill_begin_ready),
                .begin_head_id(projection_head_id),
                .begin_tag(projection_ipd_fill_begin_tag),
                .begin_term_count(projection_ipd_fill_begin_term_count),
                .begin_cache_allowed(projection_ipd_fill_cache_allowed),
                .entry_valid(projection_ipd_fill_entry_valid),
                .entry_ready(projection_ipd_fill_entry_ready),
                .entry_gate_code(projection_ipd_fill_gate_code),
                .entry_lane_id(projection_ipd_fill_lane_id),
                .entry_destination_count(projection_ipd_fill_destination_count),
                .entry_last(projection_ipd_fill_entry_last),
                .cache_begin_valid(auto_fill_begin_valid),
                .cache_begin_ready(auto_fill_begin_ready),
                .cache_begin_head_id(auto_fill_begin_head_id),
                .cache_begin_tag(auto_fill_begin_tag),
                .cache_begin_term_count(auto_fill_begin_term_count),
                .cache_begin_cacheable(cache_fill_begin_cacheable),
                .cache_entry_valid(auto_fill_entry_valid),
                .cache_entry_ready(auto_fill_entry_ready),
                .cache_entry_gate_code(auto_fill_entry_gate_code),
                .cache_entry_lane_id(auto_fill_entry_lane_id),
                .cache_entry_destination_count(
                    auto_fill_entry_destination_count
                ),
                .cache_entry_last(auto_fill_entry_last),
                .session_active(auto_fill_adapter_active),
                .protocol_error(auto_fill_protocol_error),
                .count_cacheable_fills(count_auto_cacheable_fills),
                .count_bypass_fills(count_auto_bypass_fills)
            );
        end else begin : g_no_descriptor_residency
            assign cache_fill_begin_ready = 1'b1;
            assign cache_fill_begin_cacheable = 1'b0;
            assign cache_fill_entry_ready = 1'b1;
            assign cache_lookup_ready = 1'b1;
            assign cache_meta_valid = 1'b0;
            assign cache_meta_hit = 1'b0;
            assign cache_meta_tag = '0;
            assign cache_meta_term_count = '0;
            assign descriptor_entry_valid = 1'b0;
            assign descriptor_gate_code = '0;
            assign descriptor_lane_id = '0;
            assign descriptor_destination_count = '0;
            assign descriptor_term_index = '0;
            assign descriptor_entry_last = 1'b0;
            assign cache_release_ready = 1'b1;
            assign cache_valid_flat = '0;
            assign cache_protocol_error = 1'b0;
            assign count_cached_heads = '0;
            assign count_bypass_heads = '0;
            assign count_cache_hits = '0;
            assign count_cache_misses = '0;
            assign count_cache_releases = '0;
            assign count_cache_release_noops = '0;
            assign count_cache_release_tag_mismatches = '0;
            assign projection_ipd_fill_begin_ready = 1'b1;
            assign projection_ipd_fill_entry_ready = 1'b1;
            assign auto_fill_begin_valid = 1'b0;
            assign auto_fill_begin_head_id = '0;
            assign auto_fill_begin_tag = '0;
            assign auto_fill_begin_term_count = '0;
            assign auto_fill_entry_valid = 1'b0;
            assign auto_fill_entry_gate_code = '0;
            assign auto_fill_entry_lane_id = '0;
            assign auto_fill_entry_destination_count = '0;
            assign auto_fill_entry_last = 1'b0;
            assign auto_fill_adapter_active = 1'b0;
            assign auto_fill_protocol_error = 1'b0;
            assign count_auto_cacheable_fills = '0;
            assign count_auto_bypass_fills = '0;
        end
    endgenerate

    gatestack_replay_control_plane_top #(
        .CONTEXTS(1), .HEADS(HEADS), .HEAD_BITS(HEAD_BITS),
        .SLOT_CAPACITY_BITS(SLOT_CAPACITY_BITS),
        .RESIDENT_TERMS(RESIDENT_TERMS),
        .ENABLE_RESIDENCY(ENABLE_RESIDENCY),
        .CSR_FORMAT_FADC24(CSR_FORMAT_FADC24),
        .TAG_W(TAG_W), .SIZE_W(SIZE_W),
        .FORMAT_W(FORMAT_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .WORD_INDEX_W(WORD_INDEX_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .COUNTER_W(COUNTER_W),
        .CONTEXT_ID_W(1), .HEAD_ID_W(HEAD_ID_W), .ROUTE_W(ROUTE_W)
    ) u_control (
        .clk_core, .rst_core(fabric_rst_core),
        .head_request_valid(head_issue_valid),
        .head_request_ready(head_issue_ready),
        .head_request_context_id(1'b0),
        .head_request_head_id(head_issue_head_id),
        .head_request_execution_tag(head_issue_tag),
        .head_request_head_index(head_issue_head_index),
        .head_request_input_channel_base(head_issue_input_channel_base),
        .head_request_output_tile(head_issue_output_tile),
        .head_request_last_head(head_issue_last_head),
        .head_request_last_output_tile(head_issue_last_output_tile),
        .slot_inspect_valid, .slot_inspect_ready,
        .slot_inspect_context_id, .slot_inspect_head_id,
        .slot_meta_valid, .slot_meta_ready, .slot_meta_exists, .slot_meta_tag,
        .slot_meta_mode_is_csr, .slot_meta_format, .slot_meta_payload_bits,
        .slot_meta_word_count, .cache_lookup_valid, .cache_lookup_ready,
        .cache_lookup_context_id, .cache_lookup_head_id,
        .cache_lookup_expected_tag, .cache_meta_valid, .cache_meta_ready,
        .cache_meta_hit, .cache_meta_tag, .cache_meta_term_count,
        .projection_commit_pulse, .projection_reserve_ready,
        .projection_context_id, .projection_head_id,
        .projection_payload_tag, .projection_execution_tag,
        .projection_route, .projection_format, .projection_head_index,
        .projection_input_channel_base, .projection_output_tile,
        .projection_last_head, .projection_resident_term_count,
        .projection_resident_event_count, .slot_commit_pulse,
        .slot_reserve_ready,
        .slot_replay_context_id, .slot_replay_head_id,
        .slot_replay_payload_tag, .slot_replay_start_word,
        .decoder_done_valid(control_decoder_done_valid),
        .decoder_done_ready(control_decoder_done_ready),
        .decoder_done_payload_tag(control_decoder_done_payload_tag),
        .decoder_done_error(control_decoder_done_error),
        .backend_done_valid(control_backend_done_valid),
        .backend_done_ready(control_backend_done_ready),
        .backend_done_execution_tag(control_backend_done_execution_tag),
        .backend_done_error(control_backend_done_error),
        .slot_release_valid, .slot_release_ready,
        .slot_release_context_id, .slot_release_head_id,
        .cache_release_valid, .cache_release_ready,
        .cache_release_context_id, .cache_release_head_id,
        .cache_release_payload_tag,
        .head_complete_valid, .head_complete_ready,
        .head_complete_context_id, .head_complete_head_id,
        .head_complete_head_index, .head_complete_last_head,
        .head_complete_payload_tag, .head_complete_execution_tag,
        .head_complete_error, .protocol_error(control_protocol_error),
        .count_requests(count_control_requests),
        .count_commits(count_control_commits),
        .count_rejects(count_control_rejects),
        .count_sessions(count_control_sessions)
    );

    gatestack_slot_replay_word_router #(
        .TAG_W(TAG_W), .WORD_INDEX_W(WORD_INDEX_W), .ROUTE_W(ROUTE_W)
    ) u_word_router (
        .clk_core, .rst_core(fabric_rst_core),
        .session_start_valid(slot_commit_pulse),
        .session_start_ready(replay_router_start_ready),
        .session_route(projection_route),
        .session_format(projection_format),
        .session_payload_tag(slot_replay_payload_tag),
        .input_valid(slot_replay_word_valid), .input_ready(slot_replay_word_ready),
        .input_data(slot_replay_word_data), .input_index(slot_replay_word_index),
        .input_last(slot_replay_word_last),
        .input_payload_tag(slot_replay_tag),
        .input_mode_is_csr(slot_replay_mode_is_csr),
        .input_format(slot_replay_format),
        .resident_valid(resident_word_valid), .resident_ready(resident_word_ready),
        .resident_data(resident_word_data), .resident_index(resident_word_index),
        .resident_last(resident_word_last), .ipd_valid(ipd_word_valid),
        .ipd_ready(ipd_word_ready), .ipd_data(ipd_word_data),
        .ipd_index(ipd_word_index), .ipd_last(ipd_word_last),
        .raw_valid(raw_word_valid), .raw_ready(raw_word_ready),
        .raw_data(raw_word_data), .raw_index(raw_word_index),
        .raw_last(raw_word_last), .session_active(replay_router_active),
        .protocol_error(replay_router_protocol_error)
    );

    gatestack_backend_done_guard #(
        .TAG_W(TAG_W), .HEAD_COUNT_W(HEAD_COUNT_W)
    ) u_done_guard (
        .clk_core, .rst_core(fabric_rst_core),
        .start_valid(projection_commit_pulse),
        .start_ready(guard_start_ready),
        .start_execution_tag(projection_execution_tag),
        .start_head_index(projection_head_index),
        .start_last_head(projection_last_head),
        .backend_done_valid(projection_head_done_valid),
        .backend_done_ready(projection_head_done_ready),
        .backend_done_execution_tag(projection_head_done_tag),
        .backend_done_head_index(projection_head_done_index),
        .backend_done_last_head(projection_head_done_last),
        .backend_done_error(projection_head_done_error),
        .checked_done_valid(control_backend_done_valid),
        .checked_done_ready(control_backend_done_ready),
        .checked_done_execution_tag(control_backend_done_execution_tag),
        .checked_done_error(control_backend_done_error),
        .session_active(guard_active), .protocol_error(guard_protocol_error)
    );

    gatestack_multihead_decoder_projection_top #(
        .TOKENS(TOKENS), .LANES(32), .MAX_TERMS(MAX_TERMS),
        .RESIDENT_TERMS(RESIDENT_TERMS),
        .CSR_FORMAT_FADC24(CSR_FORMAT_FADC24), .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS), .SEGMENT_TOKENS(SEGMENT_TOKENS),
        .GATE_W(GATE_W), .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .TAG_W(TAG_W), .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .WORD_INDEX_W(WORD_INDEX_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(5),
        .RES_TERM_IDX_W(RES_TERM_IDX_W), .ROUTE_W(ROUTE_W),
        .FORMAT_W(FORMAT_W)
    ) u_projection (
        .clk_core, .rst_core(fabric_rst_core),
        .tile_start_valid, .tile_start_ready,
        .tile_start_tag, .tile_start_output_tile, .tile_start_head_count,
        .head_start_valid(projection_commit_pulse),
        .head_start_ready(projection_head_start_ready),
        .head_start_tag(projection_execution_tag),
        .head_start_payload_tag(projection_payload_tag),
        .head_start_index(projection_head_index),
        .head_start_route_select(projection_route),
        .head_start_csr_format(projection_format),
        .head_start_input_channel_base(projection_input_channel_base),
        .head_start_last(projection_last_head),
        .resident_term_count(projection_resident_term_count),
        .resident_event_count(projection_resident_event_count),
        .resident_descriptor_valid(descriptor_entry_valid),
        .resident_descriptor_ready(descriptor_entry_ready),
        .resident_descriptor_gate_code(descriptor_gate_code),
        .resident_descriptor_lane_id(descriptor_lane_id),
        .resident_descriptor_destination_count(descriptor_destination_count),
        .resident_descriptor_term_index(descriptor_term_index),
        .resident_descriptor_last(descriptor_entry_last),
        .resident_word_valid, .resident_word_ready, .resident_word_data,
        .resident_word_index, .resident_word_last,
        .ipd_word_valid, .ipd_word_ready, .ipd_word_data,
        .ipd_word_index, .ipd_word_last, .raw_word_valid, .raw_word_ready,
        .raw_word_data, .raw_word_index, .raw_word_last,
        .ipd_fill_begin_valid(projection_ipd_fill_begin_valid),
        .ipd_fill_begin_ready(projection_ipd_fill_begin_ready),
        .ipd_fill_begin_tag(projection_ipd_fill_begin_tag),
        .ipd_fill_begin_term_count(projection_ipd_fill_begin_term_count),
        .ipd_fill_entry_valid(projection_ipd_fill_entry_valid),
        .ipd_fill_entry_ready(projection_ipd_fill_entry_ready),
        .ipd_fill_gate_code(projection_ipd_fill_gate_code),
        .ipd_fill_lane_id(projection_ipd_fill_lane_id),
        .ipd_fill_destination_count(projection_ipd_fill_destination_count),
        .ipd_fill_entry_last(projection_ipd_fill_entry_last),
        .ipd_fill_cache_allowed(projection_ipd_fill_cache_allowed),
        .weight_req_valid, .weight_req_ready, .weight_req_tag,
        .weight_req_input_channel, .weight_req_output_tile,
        .weight_rsp_valid, .weight_rsp_ready, .weight_rsp_tag,
        .weight_rsp_input_channel, .weight_rsp_output_tile,
        .weight_rsp_weights, .bias_req_valid, .bias_req_ready,
        .bias_req_tag, .bias_req_output_tile, .bias_req_token_id,
        .bias_rsp_valid, .bias_rsp_ready, .bias_rsp_tag,
        .bias_rsp_token_id, .bias_rsp_values, .final_valid, .final_ready,
        .final_token_ids, .final_tag, .final_values,
        .decoder_done_valid(projection_decoder_done_valid),
        .decoder_done_ready(projection_decoder_done_ready),
        .decoder_done_payload_tag(projection_decoder_done_payload_tag),
        .decoder_done_error(projection_decoder_done_error),
        .head_done_valid(projection_head_done_valid),
        .head_done_ready(projection_head_done_ready),
        .head_done_tag(projection_head_done_tag),
        .head_done_index(projection_head_done_index),
        .head_done_last(projection_head_done_last),
        .head_done_error(projection_head_done_error),
        .tile_done_valid, .tile_done_ready, .tile_done_tag,
        .protocol_error(projection_protocol_error), .accumulator_overflow,
        .count_heads(count_projection_heads),
        .count_terms(count_projection_terms),
        .count_completed_terms(count_projection_completed_terms),
        .count_bias_commits
    );

    assign projection_reserve_ready = projection_head_start_ready &&
                                      guard_start_ready;
    assign slot_reserve_ready = slot_adapter_replay_begin_ready &&
                                replay_router_start_ready;
    assign control_decoder_done_valid = projection_decoder_done_valid;
    assign projection_decoder_done_ready = control_decoder_done_ready;
    assign control_decoder_done_payload_tag =
        projection_decoder_done_payload_tag;
    assign control_decoder_done_error = projection_decoder_done_error;
    assign scheduler_head_done_valid = head_complete_valid;
    assign head_complete_ready = scheduler_head_done_ready;
    assign scheduler_head_done_tag = head_complete_execution_tag;
    assign scheduler_head_done_head_id = head_complete_head_id;
    assign scheduler_head_done_error = head_complete_error;
    assign unused_head_complete_index = head_complete_head_index;

    assign fabric_protocol_error = scheduler_protocol_error ||
        control_protocol_error || slot_protocol_error || cache_protocol_error ||
        auto_fill_protocol_error || external_fill_protocol_error ||
        replay_router_protocol_error || guard_protocol_error ||
        projection_protocol_error || accumulator_overflow ||
        (projection_commit_pulse &&
         (projection_output_tile != head_issue_output_tile ||
          projection_head_id != head_issue_head_id)) ||
        (head_complete_valid &&
         (head_complete_last_head != head_issue_last_head ||
          head_complete_payload_tag != slot_replay_payload_tag));
    assign protocol_error = fabric_protocol_error || abort_protocol_error;

    assign cache_fill_begin_valid = abort_group_active ?
                                    auto_fill_begin_valid :
                                    external_fill_cache_begin_valid;
    assign cache_fill_begin_head_id = abort_group_active ?
                                      auto_fill_begin_head_id :
                                      external_fill_cache_begin_head_id;
    assign cache_fill_begin_tag = abort_group_active ?
                                  auto_fill_begin_tag :
                                  external_fill_cache_begin_tag;
    assign cache_fill_begin_term_count = abort_group_active ?
                                         auto_fill_begin_term_count :
                                         external_fill_cache_begin_term_count;
    assign auto_fill_begin_ready = abort_group_active &&
                                   cache_fill_begin_ready;
    assign external_fill_cache_begin_ready = !abort_group_active &&
                                             cache_fill_begin_ready;
    assign cache_fill_entry_valid = abort_group_active ?
                                    auto_fill_entry_valid :
                                    external_fill_cache_entry_valid;
    assign cache_fill_entry_gate_code = abort_group_active ?
        auto_fill_entry_gate_code : external_fill_cache_entry_gate_code;
    assign cache_fill_entry_lane_id = abort_group_active ?
        auto_fill_entry_lane_id : external_fill_cache_entry_lane_id;
    assign cache_fill_entry_destination_count = abort_group_active ?
        auto_fill_entry_destination_count :
        external_fill_cache_entry_destination_count;
    assign cache_fill_entry_last = abort_group_active ?
                                   auto_fill_entry_last :
                                   external_fill_cache_entry_last;
    assign auto_fill_entry_ready = abort_group_active &&
                                   cache_fill_entry_ready;
    assign external_fill_cache_entry_ready = !abort_group_active &&
                                             cache_fill_entry_ready;

    assign fabric_rst_core = rst_core || fabric_reset_pulse;
    assign group_ready = scheduler_group_ready && !abort_admission_blocked;
    assign scheduler_group_done_ready = group_done_ready && !abort_done_valid;
    assign abort_done_ready = group_done_ready;
    assign group_done_valid = abort_done_valid || scheduler_group_done_valid;
    assign group_done_tag = abort_done_valid ? abort_done_tag :
                            scheduler_group_done_tag;
    assign group_done_error = abort_done_valid ? abort_done_error :
                              scheduler_group_done_error;

    gatestack_context_abort_controller #(
        .TAG_W(TAG_W), .TIMEOUT_CYCLES(ABORT_TIMEOUT_CYCLES),
        .COUNTER_W(COUNTER_W)
    ) u_abort_controller (
        .clk_core, .rst_core,
        .group_accept_pulse(group_valid && group_ready),
        .group_accept_tag(group_tag),
        .normal_done_fire(scheduler_group_done_valid &&
                          scheduler_group_done_ready),
        .normal_done_error(scheduler_group_done_error),
        .fabric_error(fabric_protocol_error),
        .fabric_reset_pulse, .abort_done_valid, .abort_done_ready,
        .abort_done_tag, .abort_done_error,
        .admission_blocked(abort_admission_blocked),
        .group_active(abort_group_active),
        .protocol_error(abort_protocol_error),
        .count_context_resets, .count_error_aborts,
        .count_timeout_aborts
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_status;
    assign unused_status = ^{count_group_errors, count_slot_commit_heads,
        count_slot_commit_stalls, count_slot_replay_stalls,
        slot_replay_payload_bits, cache_valid_flat, count_cached_heads,
        count_bypass_heads, count_cache_misses, count_cache_release_noops,
        count_cache_release_tag_mismatches,
        count_slot_invalid_headers,
        count_projection_completed_terms, replay_router_active, guard_active,
        unused_head_complete_index, head_issue_context_id,
        slot_commit_session_active, slot_replay_session_active, slot_valid_flat,
        slot_inspect_context_id, cache_lookup_context_id,
        projection_context_id, slot_replay_context_id,
        slot_release_context_id, cache_release_context_id,
        head_complete_context_id, abort_group_active,
        auto_fill_adapter_active, count_auto_cacheable_fills,
        count_auto_bypass_fills, external_fill_adapter_active,
        count_external_cacheable_fills, count_external_bypass_fills,
        payload_commit_begin_valid, payload_commit_head_id,
        payload_commit_tag, payload_commit_mode_is_csr, payload_commit_bits,
        payload_commit_word_valid, payload_commit_word_data,
        payload_commit_word_last,
        external_slot_inspect_ready, external_slot_inspect_meta_valid,
        external_slot_inspect_exists, external_slot_inspect_tag,
        external_slot_inspect_mode_is_csr, external_slot_inspect_format,
        external_slot_inspect_payload_bits, external_slot_inspect_word_count,
        external_slot_replay_begin_ready, external_slot_replay_word_valid,
        external_slot_replay_word_data, external_slot_replay_word_index,
        external_slot_replay_word_last, external_slot_replay_tag,
        external_slot_replay_mode_is_csr, external_slot_replay_format,
        external_slot_replay_payload_bits, external_slot_release_ready,
        external_slot_valid_flat, external_slot_protocol_error,
        external_slot_count_replays, external_slot_count_releases};
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
