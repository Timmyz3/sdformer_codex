`timescale 1ns/1ps
`default_nettype none

// Shared typed-slot integration from final-gate/K capture through projection.
// Builder payload SRAM is the execution slot service; no payload copy exists.
module gatestack_builder_projection_single_context_top #(
    parameter int BUILDER_C1_ENABLE = 0,
    parameter int TOKENS = 162,
    parameter int LANES = 32,
    parameter int GATE_W = 9,
    parameter int CLASS_SLOTS = 4,
    parameter int HEADS = 24,
    parameter int SLOT_WORDS = 104,
    parameter int WORD_W = 64,
    parameter int HEAD_BITS = TOKENS * (LANES + GATE_W),
    parameter int MAX_TERMS = 128,
    parameter int RESIDENT_TERMS = 80,
    parameter int ENABLE_RESIDENCY = 1,
    parameter int CSR_FORMAT_FADC24 = 0,
    parameter int EVENT_WAYS = 4,
    parameter int OUT_TILE = 32,
    parameter int BANKS = 2,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
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
    parameter int COUNTER_W = 32,
    parameter int TOKEN_ID_W = 8,
    parameter int BUILDER_TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int HEAD_ID_W = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int WORD_INDEX_W = (SLOT_WORDS <= 1) ? 1 : $clog2(SLOT_WORDS),
    parameter int RES_TERM_IDX_W = (RESIDENT_TERMS <= 1) ?
                                     1 : $clog2(RESIDENT_TERMS),
    parameter int ROUTE_W = 2,
    parameter int WEIGHT_W = 8,
    parameter int PRODUCT_W = GATE_W + WEIGHT_W,
    parameter int ACC_W = 32,
    parameter int DESTINATION_SCAN_MODE = 1,
    parameter int BITMAP_BYPASS_ENABLE = 1,
    parameter int EXPLICIT_BITMAP_BANK_ENABLE = 0,
    parameter int BUILD_TIMEOUT_CYCLES = 1000000,
    parameter int ABORT_TIMEOUT_CYCLES = 1000000
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic                                      head_begin_valid,
    output logic                                      head_begin_ready,
    input  logic [HEAD_ID_W-1:0]                      head_id,
    input  logic [TAG_W-1:0]                          head_tag,
    input  logic                                      token_valid,
    output logic                                      token_ready,
    input  logic [BUILDER_TOKEN_ID_W-1:0]             token_id,
    input  logic [GATE_W-1:0]                         token_gate_code,
    input  logic [LANES-1:0]                          token_k_bits,
    input  logic                                      token_last,

    input  logic                                      group_valid,
    output logic                                      group_ready,
    input  logic [TAG_W-1:0]                          group_tag,
    input  logic [OUTPUT_TILE_W-1:0]                  group_first_output_tile,
    input  logic [OUTPUT_TILE_COUNT_W-1:0]            group_output_tile_count,
    output logic                                      group_done_valid,
    input  logic                                      group_done_ready,
    output logic [TAG_W-1:0]                          group_done_tag,
    output logic                                      group_done_error,
    input  logic                                      batch_abort_valid,
    output logic                                      batch_abort_ready,

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

    output logic                                      builder_done_pulse,
    output logic [TAG_W-1:0]                          builder_done_tag,
    output logic [FORMAT_W-1:0]                       builder_done_format,
    output logic                                      builder_done_error,
    output logic [7:0]                                builder_done_word_count,
    output logic [2:0]                                builder_selected_reason,
    output logic [SIZE_W-1:0]                         builder_selected_payload_bits,
    output logic [HEAD_COUNT_W-1:0]                   batch_accepted_heads,
    output logic [HEAD_COUNT_W-1:0]                   batch_completed_heads,
    output logic [HEADS-1:0]                          slot_valid_flat,
    output logic                                      slot_reset_pulse,
    output logic                                      protocol_error,

    output logic [COUNTER_W-1:0]                      count_builder_heads,
    output logic [COUNTER_W-1:0]                      count_builder_raw_heads,
    output logic [COUNTER_W-1:0]                      count_builder_terms,
    output logic [COUNTER_W-1:0]                      count_builder_destinations,
    output logic [COUNTER_W-1:0]                      count_builder_scan_cycles,
    output logic [COUNTER_W-1:0]                      count_builder_output_stalls,
    output logic [COUNTER_W-1:0]                      count_builder_committed_heads,
    output logic [COUNTER_W-1:0]                      count_builder_aborted_heads,
    output logic [COUNTER_W-1:0]                      count_builder_committed_words,
    output logic [COUNTER_W-1:0]                      count_slot_commit_heads,
    output logic [COUNTER_W-1:0]                      count_slot_replay_heads,
    output logic [COUNTER_W-1:0]                      count_slot_release_heads,
    output logic [COUNTER_W-1:0]                      count_builder_capture_blocked_cycles,
    output logic [COUNTER_W-1:0]                      count_builder_overlap_cycles,
    output logic [COUNTER_W-1:0]                      count_builder_order_wait_cycles,
    output logic [COUNTER_W-1:0]                      count_payload_copy_words,
    output logic [COUNTER_W-1:0]                      count_groups,
    output logic [COUNTER_W-1:0]                      count_tile_starts,
    output logic [COUNTER_W-1:0]                      count_head_issues,
    output logic [COUNTER_W-1:0]                      count_control_requests,
    output logic [COUNTER_W-1:0]                      count_control_commits,
    output logic [COUNTER_W-1:0]                      count_control_rejects,
    output logic [COUNTER_W-1:0]                      count_control_sessions,
    output logic [COUNTER_W-1:0]                      count_projection_heads,
    output logic [COUNTER_W-1:0]                      count_projection_terms,
    output logic [COUNTER_W-1:0]                      count_bias_commits,
    output logic [COUNTER_W-1:0]                      count_context_resets,
    output logic [COUNTER_W-1:0]                      count_error_aborts,
    output logic [COUNTER_W-1:0]                      count_timeout_aborts
);
    localparam logic [HEADS-1:0] ALL_HEADS_VALID = {HEADS{1'b1}};

    logic builder_rst_core, execution_rst_core;
    logic execution_slot_reset_pulse, wrapper_abort_fire;
    logic manual_abort_fire, auto_abort_request, build_timeout;
    logic builder_head_begin_valid, builder_head_begin_ready;
    logic builder_done_valid;
    logic builder_workspace_error, builder_serializer_error;
    logic builder_slot_error;
    logic builder_inspect_valid, builder_inspect_ready;
    logic [HEAD_ID_W-1:0] builder_inspect_head_id;
    logic builder_inspect_meta_valid, builder_inspect_meta_ready;
    logic builder_inspect_exists;
    logic [TAG_W-1:0] builder_inspect_tag;
    logic builder_inspect_mode_is_csr;
    logic [FORMAT_W-1:0] builder_inspect_format;
    logic [SIZE_W-1:0] builder_inspect_payload_bits;
    logic [SIZE_W-1:0] builder_inspect_word_count;
    logic builder_replay_begin_valid, builder_replay_begin_ready;
    logic [HEAD_ID_W-1:0] builder_replay_head_id;
    logic [WORD_INDEX_W-1:0] builder_replay_start_word;
    logic builder_replay_word_valid, builder_replay_word_ready;
    logic [WORD_W-1:0] builder_replay_word_data;
    logic [WORD_INDEX_W-1:0] builder_replay_word_index;
    logic builder_replay_word_last;
    logic [TAG_W-1:0] builder_replay_tag;
    logic builder_replay_mode_is_csr;
    logic [FORMAT_W-1:0] builder_replay_format;
    logic [SIZE_W-1:0] builder_replay_payload_bits;
    logic builder_release_valid, builder_release_ready;
    logic [HEAD_ID_W-1:0] builder_release_head_id;

    logic execution_group_valid, execution_group_ready;
    logic execution_group_done_valid, execution_group_done_ready;
    logic [TAG_W-1:0] execution_group_done_tag;
    logic execution_group_done_error;
    logic execution_protocol_error;
    logic [COUNTER_W-1:0] count_execution_slot_replays;
    logic [COUNTER_W-1:0] count_execution_slot_releases;
    logic [COUNTER_W-1:0] unused_count_cache_hits;
    logic [COUNTER_W-1:0] unused_count_cache_releases;
    logic unused_payload_commit_begin_ready;
    logic unused_payload_commit_word_ready;
    logic unused_external_inspect_context_id;
    logic unused_external_replay_context_id;
    logic [TAG_W-1:0] unused_external_replay_payload_tag;
    logic unused_external_release_context_id;
    logic unused_descriptor_fill_begin_ready;
    logic unused_descriptor_fill_begin_cacheable;
    logic unused_descriptor_fill_entry_ready;
    logic [HEADS-1:0] accepted_bitmap_q;
    logic [HEAD_COUNT_W-1:0] accepted_count_q, completed_count_q;
    logic builder_error_q, batch_protocol_error_q;
    logic group_request_active_q, execution_launched_q;
    logic abort_done_valid_q;
    logic [TAG_W-1:0] abort_done_tag_q;
    logic [TAG_W-1:0] group_tag_q;
    logic [OUTPUT_TILE_W-1:0] group_first_output_tile_q;
    logic [OUTPUT_TILE_COUNT_W-1:0] group_output_tile_count_q;
    logic [COUNTER_W-1:0] build_watchdog_q;
    logic head_id_legal, head_duplicate, head_fire;
    logic batch_build_complete, group_fire, execution_group_fire;
    logic normal_group_done_fire, abort_group_done_fire;

    assign batch_abort_ready = group_request_active_q && !abort_done_valid_q;
    assign manual_abort_fire = batch_abort_valid && batch_abort_ready;
    assign build_timeout = group_request_active_q && !execution_launched_q &&
        build_watchdog_q >= COUNTER_W'(BUILD_TIMEOUT_CYCLES - 1);
    assign auto_abort_request = group_request_active_q &&
        !execution_launched_q && !abort_done_valid_q &&
        (builder_workspace_error || builder_serializer_error ||
         builder_slot_error || builder_error_q || batch_protocol_error_q ||
         build_timeout);
    assign wrapper_abort_fire = manual_abort_fire || auto_abort_request;
    assign execution_rst_core = rst_core || wrapper_abort_fire;
    assign slot_reset_pulse = execution_slot_reset_pulse || wrapper_abort_fire;
    assign builder_rst_core = rst_core || slot_reset_pulse;
    assign head_id_legal = HEAD_COUNT_W'(head_id) < HEAD_COUNT_W'(HEADS);
    assign head_duplicate = head_id_legal && accepted_bitmap_q[head_id];
    assign builder_head_begin_valid = head_begin_valid &&
        group_request_active_q && !execution_launched_q &&
        !head_duplicate && head_id_legal &&
        accepted_count_q < HEAD_COUNT_W'(HEADS);
    assign head_begin_ready = builder_head_begin_ready &&
        group_request_active_q && !execution_launched_q &&
        !head_duplicate &&
        head_id_legal && accepted_count_q < HEAD_COUNT_W'(HEADS);
    assign token_ready = builder_token_ready;
    assign head_fire = head_begin_valid && head_begin_ready;

    assign builder_done_pulse = builder_done_valid;
    assign batch_accepted_heads = accepted_count_q;
    assign batch_completed_heads = completed_count_q;
    assign batch_build_complete =
        accepted_count_q == HEAD_COUNT_W'(HEADS) &&
        completed_count_q == HEAD_COUNT_W'(HEADS) && !builder_error_q &&
        !batch_protocol_error_q && slot_valid_flat == ALL_HEADS_VALID;
    assign group_ready = !group_request_active_q && !abort_done_valid_q;
    assign group_fire = group_valid && group_ready;
    assign execution_group_valid = group_request_active_q &&
        batch_build_complete && !execution_launched_q &&
        execution_group_ready;
    assign execution_group_fire = execution_group_valid &&
                                  execution_group_ready;
    assign execution_group_done_ready = !abort_done_valid_q &&
                                        group_done_ready;
    assign group_done_valid = abort_done_valid_q ? 1'b1 :
                              execution_group_done_valid;
    assign group_done_tag = abort_done_valid_q ? abort_done_tag_q :
                            execution_group_done_tag;
    assign group_done_error = abort_done_valid_q ? 1'b1 :
                              execution_group_done_error;
    assign abort_group_done_fire = abort_done_valid_q && group_done_ready;
    assign normal_group_done_fire = !abort_done_valid_q &&
        execution_group_done_valid && group_done_ready;
    assign protocol_error = execution_protocol_error || builder_workspace_error ||
        builder_serializer_error || builder_slot_error || builder_error_q ||
        batch_protocol_error_q || abort_done_valid_q;
    assign count_payload_copy_words = '0;

    always_ff @(posedge clk_core) begin
        if (rst_core || slot_reset_pulse) begin
            accepted_bitmap_q <= '0;
            accepted_count_q <= '0;
            completed_count_q <= '0;
            builder_error_q <= 1'b0;
            batch_protocol_error_q <= 1'b0;
        end else begin
            if (head_fire) begin
                accepted_bitmap_q[head_id] <= 1'b1;
                accepted_count_q <= accepted_count_q + 1'b1;
            end
            if (head_begin_valid && (!head_id_legal || head_duplicate))
                batch_protocol_error_q <= 1'b1;
            if (builder_done_valid) begin
                if (completed_count_q < HEAD_COUNT_W'(HEADS))
                    completed_count_q <= completed_count_q + 1'b1;
                else
                    batch_protocol_error_q <= 1'b1;
                if (builder_done_error)
                    builder_error_q <= 1'b1;
            end
            if (normal_group_done_fire) begin
                accepted_bitmap_q <= '0;
                accepted_count_q <= '0;
                completed_count_q <= '0;
                builder_error_q <= 1'b0;
                batch_protocol_error_q <= 1'b0;
            end
        end
    end

    // A group request is accepted before its heads. This gives every build,
    // execution, timeout and abort path one tagged completion transaction.
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            group_request_active_q <= 1'b0;
            execution_launched_q <= 1'b0;
            abort_done_valid_q <= 1'b0;
            abort_done_tag_q <= '0;
            group_tag_q <= '0;
            group_first_output_tile_q <= '0;
            group_output_tile_count_q <= '0;
            build_watchdog_q <= '0;
        end else begin
            if (group_fire) begin
                group_request_active_q <= 1'b1;
                execution_launched_q <= 1'b0;
                group_tag_q <= group_tag;
                group_first_output_tile_q <= group_first_output_tile;
                group_output_tile_count_q <= group_output_tile_count;
                build_watchdog_q <= '0;
            end else if (group_request_active_q && !execution_launched_q &&
                         !wrapper_abort_fire) begin
                build_watchdog_q <= build_watchdog_q + 1'b1;
            end

            if (execution_group_fire)
                execution_launched_q <= 1'b1;

            if (wrapper_abort_fire) begin
                group_request_active_q <= 1'b0;
                execution_launched_q <= 1'b0;
                abort_done_valid_q <= 1'b1;
                abort_done_tag_q <= group_tag_q;
                build_watchdog_q <= '0;
            end else if (abort_group_done_fire) begin
                abort_done_valid_q <= 1'b0;
            end else if (normal_group_done_fire) begin
                group_request_active_q <= 1'b0;
                execution_launched_q <= 1'b0;
                build_watchdog_q <= '0;
            end
        end
    end

    logic builder_token_ready;
    generate
        if (BUILDER_C1_ENABLE == 0) begin : g_builder_c0
            gatestack_onchip_typed_builder_c0_top #(
                .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
                .CLASS_SLOTS(CLASS_SLOTS), .CONTEXTS(1), .HEADS(HEADS),
                .SLOT_WORDS(SLOT_WORDS), .WORD_W(WORD_W), .TAG_W(TAG_W),
                .FORMAT_W(FORMAT_W), .SIZE_W(SIZE_W),
                .COUNTER_W(COUNTER_W),
                .DESTINATION_SCAN_MODE(DESTINATION_SCAN_MODE),
                .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE),
                .EXPLICIT_BITMAP_BANK_ENABLE(EXPLICIT_BITMAP_BANK_ENABLE)
            ) u_builder (
                .clk_core, .rst_core(builder_rst_core),
                .head_begin_valid(builder_head_begin_valid),
                .head_begin_ready(builder_head_begin_ready),
                .head_context_id(1'b0), .head_id, .head_tag,
                .token_valid, .token_ready(builder_token_ready), .token_id,
                .token_gate_code, .token_k_bits, .token_last,
                .done_valid(builder_done_valid), .done_ready(1'b1),
                .done_tag(builder_done_tag), .done_format(builder_done_format),
                .done_error(builder_done_error),
                .done_word_count(builder_done_word_count),
                .selected_reason(builder_selected_reason),
                .selected_payload_bits(builder_selected_payload_bits),
                .inspect_valid(builder_inspect_valid),
                .inspect_ready(builder_inspect_ready),
                .inspect_context_id(1'b0),
                .inspect_head_id(builder_inspect_head_id),
                .inspect_meta_valid(builder_inspect_meta_valid),
                .inspect_meta_ready(builder_inspect_meta_ready),
                .inspect_exists(builder_inspect_exists),
                .inspect_tag(builder_inspect_tag),
                .inspect_mode_is_csr(builder_inspect_mode_is_csr),
                .inspect_format(builder_inspect_format),
                .inspect_payload_bits(builder_inspect_payload_bits),
                .inspect_word_count(builder_inspect_word_count),
                .replay_begin_valid(builder_replay_begin_valid),
                .replay_begin_ready(builder_replay_begin_ready),
                .replay_context_id(1'b0),
                .replay_head_id(builder_replay_head_id),
                .replay_start_word(builder_replay_start_word),
                .replay_word_valid(builder_replay_word_valid),
                .replay_word_ready(builder_replay_word_ready),
                .replay_word_data(builder_replay_word_data),
                .replay_word_index(builder_replay_word_index),
                .replay_word_last(builder_replay_word_last),
                .replay_tag(builder_replay_tag),
                .replay_mode_is_csr(builder_replay_mode_is_csr),
                .replay_format(builder_replay_format),
                .replay_payload_bits(builder_replay_payload_bits),
                .release_valid(builder_release_valid),
                .release_ready(builder_release_ready),
                .release_context_id(1'b0),
                .release_head_id(builder_release_head_id), .slot_valid_flat,
                .workspace_protocol_error(builder_workspace_error),
                .serializer_protocol_error(builder_serializer_error),
                .slot_protocol_error(builder_slot_error),
                .count_workspace_heads(count_builder_heads),
                .count_workspace_raw_fallback_heads(count_builder_raw_heads),
                .count_workspace_terms(count_builder_terms),
                .count_workspace_destinations(count_builder_destinations),
                .count_workspace_scan_cycles(count_builder_scan_cycles),
                .count_workspace_output_stall_cycles(count_builder_output_stalls),
                .count_builder_committed_heads,
                .count_builder_aborted_heads,
                .count_builder_committed_words,
                .count_slot_commit_heads, .count_slot_replay_heads,
                .count_slot_release_heads
            );
            assign count_builder_capture_blocked_cycles = '0;
            assign count_builder_overlap_cycles = '0;
            assign count_builder_order_wait_cycles = '0;
        end else begin : g_builder_c1
            logic [COUNTER_W-1:0] unused_done_sequence;
            gatestack_onchip_typed_builder_c1_top #(
                .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
                .CLASS_SLOTS(CLASS_SLOTS), .CONTEXTS(1), .HEADS(HEADS),
                .SLOT_WORDS(SLOT_WORDS), .WORD_W(WORD_W), .TAG_W(TAG_W),
                .FORMAT_W(FORMAT_W), .SIZE_W(SIZE_W),
                .COUNTER_W(COUNTER_W),
                .DESTINATION_SCAN_MODE(DESTINATION_SCAN_MODE),
                .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE),
                .EXPLICIT_BITMAP_BANK_ENABLE(EXPLICIT_BITMAP_BANK_ENABLE)
            ) u_builder (
                .clk_core, .rst_core(builder_rst_core),
                .head_begin_valid(builder_head_begin_valid),
                .head_begin_ready(builder_head_begin_ready),
                .head_context_id(1'b0), .head_id, .head_tag,
                .token_valid, .token_ready(builder_token_ready), .token_id,
                .token_gate_code, .token_k_bits, .token_last,
                .done_valid(builder_done_valid), .done_ready(1'b1),
                .done_tag(builder_done_tag), .done_format(builder_done_format),
                .done_error(builder_done_error),
                .done_word_count(builder_done_word_count),
                .selected_reason(builder_selected_reason),
                .selected_payload_bits(builder_selected_payload_bits),
                .done_sequence(unused_done_sequence),
                .inspect_valid(builder_inspect_valid),
                .inspect_ready(builder_inspect_ready),
                .inspect_context_id(1'b0),
                .inspect_head_id(builder_inspect_head_id),
                .inspect_meta_valid(builder_inspect_meta_valid),
                .inspect_meta_ready(builder_inspect_meta_ready),
                .inspect_exists(builder_inspect_exists),
                .inspect_tag(builder_inspect_tag),
                .inspect_mode_is_csr(builder_inspect_mode_is_csr),
                .inspect_format(builder_inspect_format),
                .inspect_payload_bits(builder_inspect_payload_bits),
                .inspect_word_count(builder_inspect_word_count),
                .replay_begin_valid(builder_replay_begin_valid),
                .replay_begin_ready(builder_replay_begin_ready),
                .replay_context_id(1'b0),
                .replay_head_id(builder_replay_head_id),
                .replay_start_word(builder_replay_start_word),
                .replay_word_valid(builder_replay_word_valid),
                .replay_word_ready(builder_replay_word_ready),
                .replay_word_data(builder_replay_word_data),
                .replay_word_index(builder_replay_word_index),
                .replay_word_last(builder_replay_word_last),
                .replay_tag(builder_replay_tag),
                .replay_mode_is_csr(builder_replay_mode_is_csr),
                .replay_format(builder_replay_format),
                .replay_payload_bits(builder_replay_payload_bits),
                .release_valid(builder_release_valid),
                .release_ready(builder_release_ready),
                .release_context_id(1'b0),
                .release_head_id(builder_release_head_id), .slot_valid_flat,
                .workspace_protocol_error(builder_workspace_error),
                .serializer_protocol_error(builder_serializer_error),
                .slot_protocol_error(builder_slot_error),
                .count_workspace_heads(count_builder_heads),
                .count_workspace_raw_fallback_heads(count_builder_raw_heads),
                .count_workspace_terms(count_builder_terms),
                .count_workspace_destinations(count_builder_destinations),
                .count_workspace_scan_cycles(count_builder_scan_cycles),
                .count_workspace_output_stall_cycles(count_builder_output_stalls),
                .count_builder_committed_heads,
                .count_builder_aborted_heads,
                .count_builder_committed_words,
                .count_slot_commit_heads, .count_slot_replay_heads,
                .count_slot_release_heads,
                .count_capture_blocked_cycles(
                    count_builder_capture_blocked_cycles),
                .count_capture_service_overlap_cycles(
                    count_builder_overlap_cycles),
                .count_order_wait_cycles(count_builder_order_wait_cycles)
            );
        end
    endgenerate

    gatestack_single_context_execution_top #(
        .TOKENS(TOKENS), .HEADS(HEADS), .HEAD_BITS(HEAD_BITS),
        .SLOT_CAPACITY_BITS(SLOT_WORDS * WORD_W), .MAX_TERMS(MAX_TERMS),
        .RESIDENT_TERMS(RESIDENT_TERMS),
        .ENABLE_RESIDENCY(ENABLE_RESIDENCY),
        .EXTERNAL_SLOT_SERVICE_ENABLE(1),
        .CSR_FORMAT_FADC24(CSR_FORMAT_FADC24), .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(SEGMENT_TOKENS), .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .OUTPUT_TILE_COUNT_W(OUTPUT_TILE_COUNT_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .SIZE_W(SIZE_W),
        .FORMAT_W(FORMAT_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .EVENT_COUNT_W(EVENT_COUNT_W), .WORD_INDEX_W(WORD_INDEX_W),
        .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W),
        .HEAD_ID_W(HEAD_ID_W), .RES_TERM_IDX_W(RES_TERM_IDX_W),
        .ROUTE_W(ROUTE_W), .WEIGHT_W(WEIGHT_W), .GATE_W(GATE_W),
        .PRODUCT_W(PRODUCT_W), .ACC_W(ACC_W),
        .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .ABORT_TIMEOUT_CYCLES(ABORT_TIMEOUT_CYCLES)
    ) u_execution (
        .clk_core, .rst_core(execution_rst_core),
        .group_valid(execution_group_valid),
        .group_ready(execution_group_ready), .group_tag(group_tag_q),
        .group_head_count(HEAD_COUNT_W'(HEADS)),
        .group_first_output_tile(group_first_output_tile_q),
        .group_output_tile_count(group_output_tile_count_q),
        .group_done_valid(execution_group_done_valid),
        .group_done_ready(execution_group_done_ready),
        .group_done_tag(execution_group_done_tag),
        .group_done_error(execution_group_done_error),
        .payload_commit_begin_valid(1'b0),
        .payload_commit_begin_ready(unused_payload_commit_begin_ready),
        .payload_commit_head_id('0),
        .payload_commit_tag('0), .payload_commit_mode_is_csr(1'b0),
        .payload_commit_bits('0), .payload_commit_word_valid(1'b0),
        .payload_commit_word_ready(unused_payload_commit_word_ready),
        .payload_commit_word_data('0),
        .payload_commit_word_last(1'b0),
        .external_slot_inspect_valid(builder_inspect_valid),
        .external_slot_inspect_ready(builder_inspect_ready),
        .external_slot_inspect_context_id(unused_external_inspect_context_id),
        .external_slot_inspect_head_id(builder_inspect_head_id),
        .external_slot_inspect_meta_valid(builder_inspect_meta_valid),
        .external_slot_inspect_meta_ready(builder_inspect_meta_ready),
        .external_slot_inspect_exists(builder_inspect_exists),
        .external_slot_inspect_tag(builder_inspect_tag),
        .external_slot_inspect_mode_is_csr(builder_inspect_mode_is_csr),
        .external_slot_inspect_format(builder_inspect_format),
        .external_slot_inspect_payload_bits(builder_inspect_payload_bits),
        .external_slot_inspect_word_count(builder_inspect_word_count),
        .external_slot_replay_begin_valid(builder_replay_begin_valid),
        .external_slot_replay_begin_ready(builder_replay_begin_ready),
        .external_slot_replay_context_id(unused_external_replay_context_id),
        .external_slot_replay_head_id(builder_replay_head_id),
        .external_slot_replay_payload_tag(
            unused_external_replay_payload_tag),
        .external_slot_replay_start_word(builder_replay_start_word),
        .external_slot_replay_word_valid(builder_replay_word_valid),
        .external_slot_replay_word_ready(builder_replay_word_ready),
        .external_slot_replay_word_data(builder_replay_word_data),
        .external_slot_replay_word_index(builder_replay_word_index),
        .external_slot_replay_word_last(builder_replay_word_last),
        .external_slot_replay_tag(builder_replay_tag),
        .external_slot_replay_mode_is_csr(builder_replay_mode_is_csr),
        .external_slot_replay_format(builder_replay_format),
        .external_slot_replay_payload_bits(builder_replay_payload_bits),
        .external_slot_release_valid(builder_release_valid),
        .external_slot_release_ready(builder_release_ready),
        .external_slot_release_context_id(unused_external_release_context_id),
        .external_slot_release_head_id(builder_release_head_id),
        .external_slot_valid_flat(slot_valid_flat),
        .external_slot_protocol_error(builder_slot_error),
        .external_slot_count_replays(count_slot_replay_heads),
        .external_slot_count_releases(count_slot_release_heads),
        .external_slot_reset_pulse(execution_slot_reset_pulse),
        .descriptor_fill_begin_valid(1'b0),
        .descriptor_fill_begin_ready(unused_descriptor_fill_begin_ready),
        .descriptor_fill_head_id('0),
        .descriptor_fill_tag('0), .descriptor_fill_term_count('0),
        .descriptor_fill_format('0),
        .descriptor_fill_begin_cacheable(
            unused_descriptor_fill_begin_cacheable),
        .descriptor_fill_entry_valid(1'b0),
        .descriptor_fill_entry_ready(unused_descriptor_fill_entry_ready),
        .descriptor_fill_gate_code('0),
        .descriptor_fill_lane_id('0),
        .descriptor_fill_destination_count('0),
        .descriptor_fill_entry_last(1'b0),
        .weight_req_valid, .weight_req_ready, .weight_req_tag,
        .weight_req_input_channel, .weight_req_output_tile,
        .weight_rsp_valid, .weight_rsp_ready, .weight_rsp_tag,
        .weight_rsp_input_channel, .weight_rsp_output_tile,
        .weight_rsp_weights, .bias_req_valid, .bias_req_ready,
        .bias_req_tag, .bias_req_output_tile, .bias_req_token_id,
        .bias_rsp_valid, .bias_rsp_ready, .bias_rsp_tag,
        .bias_rsp_token_id, .bias_rsp_values, .final_valid, .final_ready,
        .final_token_ids, .final_tag, .final_values,
        .protocol_error(execution_protocol_error), .count_groups,
        .count_tile_starts, .count_head_issues, .count_control_requests,
        .count_control_commits, .count_control_rejects,
        .count_control_sessions,
        .count_slot_replays(count_execution_slot_replays),
        .count_slot_releases(count_execution_slot_releases),
        .count_cache_hits(unused_count_cache_hits),
        .count_cache_releases(unused_count_cache_releases),
        .count_projection_heads, .count_projection_terms,
        .count_bias_commits, .count_context_resets, .count_error_aborts,
        .count_timeout_aborts
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_status;
    assign unused_status = ^{count_execution_slot_replays,
        count_execution_slot_releases, unused_count_cache_hits,
        unused_count_cache_releases, unused_payload_commit_begin_ready,
        unused_payload_commit_word_ready, unused_external_inspect_context_id,
        unused_external_replay_context_id,
        unused_external_replay_payload_tag,
        unused_external_release_context_id,
        unused_descriptor_fill_begin_ready,
        unused_descriptor_fill_begin_cacheable,
        unused_descriptor_fill_entry_ready};
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
