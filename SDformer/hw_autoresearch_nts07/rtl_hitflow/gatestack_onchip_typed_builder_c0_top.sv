`timescale 1ns/1ps
`default_nettype none

// Complete C0 path from final-gate/K tokens to typed atomic head slots.
module gatestack_onchip_typed_builder_c0_top #(
    parameter int TOKENS             = 162,
    parameter int LANES              = 32,
    parameter int GATE_W             = 9,
    parameter int CLASS_SLOTS        = 4,
    parameter int CONTEXTS           = 2,
    parameter int HEADS              = 24,
    parameter int SLOT_WORDS         = 104,
    parameter int WORD_W             = 64,
    parameter int TAG_W              = 32,
    parameter int FORMAT_W           = 2,
    parameter int SIZE_W             = 16,
    parameter int COUNTER_W          = 32,
    parameter int DESTINATION_SCAN_MODE = 1,
    parameter int BITMAP_BYPASS_ENABLE = 1,
    parameter int EXPLICIT_BITMAP_BANK_ENABLE = 0,
    parameter int CONTEXT_ID_W       = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W          = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int TOKEN_ID_W         = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int WORD_INDEX_W       = (SLOT_WORDS <= 1) ? 1 : $clog2(SLOT_WORDS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         head_begin_valid,
    output logic                         head_begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      head_context_id,
    input  logic [HEAD_ID_W-1:0]         head_id,
    input  logic [TAG_W-1:0]             head_tag,
    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [TOKEN_ID_W-1:0]        token_id,
    input  logic [GATE_W-1:0]            token_gate_code,
    input  logic [LANES-1:0]             token_k_bits,
    input  logic                         token_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_W-1:0]             done_tag,
    output logic [FORMAT_W-1:0]          done_format,
    output logic                         done_error,
    output logic [7:0]                   done_word_count,
    output logic [2:0]                   selected_reason,
    output logic [SIZE_W-1:0]            selected_payload_bits,

    input  logic                         inspect_valid,
    output logic                         inspect_ready,
    input  logic [CONTEXT_ID_W-1:0]      inspect_context_id,
    input  logic [HEAD_ID_W-1:0]         inspect_head_id,
    output logic                         inspect_meta_valid,
    input  logic                         inspect_meta_ready,
    output logic                         inspect_exists,
    output logic [TAG_W-1:0]             inspect_tag,
    output logic                         inspect_mode_is_csr,
    output logic [FORMAT_W-1:0]          inspect_format,
    output logic [SIZE_W-1:0]            inspect_payload_bits,
    output logic [SIZE_W-1:0]            inspect_word_count,

    input  logic                         replay_begin_valid,
    output logic                         replay_begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      replay_context_id,
    input  logic [HEAD_ID_W-1:0]         replay_head_id,
    input  logic [WORD_INDEX_W-1:0]      replay_start_word,
    output logic                         replay_word_valid,
    input  logic                         replay_word_ready,
    output logic [WORD_W-1:0]            replay_word_data,
    output logic [WORD_INDEX_W-1:0]      replay_word_index,
    output logic                         replay_word_last,
    output logic [TAG_W-1:0]             replay_tag,
    output logic                         replay_mode_is_csr,
    output logic [FORMAT_W-1:0]          replay_format,
    output logic [SIZE_W-1:0]            replay_payload_bits,

    input  logic                         release_valid,
    output logic                         release_ready,
    input  logic [CONTEXT_ID_W-1:0]      release_context_id,
    input  logic [HEAD_ID_W-1:0]         release_head_id,

    output logic [(CONTEXTS*HEADS)-1:0]  slot_valid_flat,
    output logic                         workspace_protocol_error,
    output logic                         serializer_protocol_error,
    output logic                         slot_protocol_error,
    output logic [COUNTER_W-1:0]         count_workspace_heads,
    output logic [COUNTER_W-1:0]         count_workspace_raw_fallback_heads,
    output logic [COUNTER_W-1:0]         count_workspace_terms,
    output logic [COUNTER_W-1:0]         count_workspace_destinations,
    output logic [COUNTER_W-1:0]         count_workspace_scan_cycles,
    output logic [COUNTER_W-1:0]         count_workspace_output_stall_cycles,
    output logic [COUNTER_W-1:0]         count_builder_committed_heads,
    output logic [COUNTER_W-1:0]         count_builder_aborted_heads,
    output logic [COUNTER_W-1:0]         count_builder_committed_words,
    output logic [COUNTER_W-1:0]         count_slot_commit_heads,
    output logic [COUNTER_W-1:0]         count_slot_replay_heads,
    output logic [COUNTER_W-1:0]         count_slot_release_heads
);

    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);

    logic workspace_metadata_valid;
    logic workspace_metadata_ready;
    logic [CONTEXT_ID_W-1:0] workspace_context;
    logic [HEAD_ID_W-1:0] workspace_head;
    logic [TAG_W-1:0] workspace_tag;
    logic [3:0] workspace_active_classes;
    logic [7:0] workspace_active_tokens;
    logic [7:0] workspace_term_count;
    logic [12:0] workspace_event_count;
    logic [7:0] workspace_bitmap_terms;
    logic [12:0] workspace_fadc_bytes;
    logic workspace_metadata_overflow;
    logic workspace_raw_capture_error;
    logic workspace_emit_start_valid;
    logic workspace_emit_start_ready;
    logic workspace_descriptor_valid;
    logic workspace_descriptor_ready;
    logic [GATE_W-1:0] workspace_descriptor_gate;
    logic [4:0] workspace_descriptor_lane;
    logic [7:0] workspace_descriptor_count;
    logic workspace_descriptor_last;
    logic workspace_destination_valid;
    logic workspace_destination_ready;
    logic [7:0] workspace_destination_token;
    logic workspace_destination_last;
    logic workspace_destination_bitmap_valid;
    logic workspace_destination_bitmap_ready;
    logic [TOKENS-1:0] workspace_destination_bitmap;
    logic workspace_raw_valid;
    logic workspace_raw_ready;
    logic [7:0] workspace_raw_token;
    logic [GATE_W-1:0] workspace_raw_gate;
    logic [LANES-1:0] workspace_raw_k;
    logic workspace_emit_done_valid;
    logic workspace_emit_done_ready;
    logic [TAG_W-1:0] workspace_emit_done_tag;
    logic workspace_emit_done_error;

    logic [FORMAT_W-1:0] policy_format;
    logic [2:0] policy_reason;
    logic [SIZE_W-1:0] policy_payload_bits;
    logic [7:0] policy_word_count;
    logic [SIZE_W-1:0] unused_policy_ipd_bytes;
    logic [SIZE_W-1:0] unused_policy_fadc_bytes;

    logic builder_begin_valid;
    logic builder_begin_ready;
    logic builder_done_valid;
    logic builder_done_ready;
    logic [TAG_W-1:0] builder_done_tag;
    logic [FORMAT_W-1:0] builder_done_format;
    logic builder_done_error;
    logic [7:0] builder_done_word_count;
    logic [COUNTER_W-1:0] unused_count_builder_heads;
    logic [COUNTER_W-1:0] unused_count_builder_input_stalls;
    logic [COUNTER_W-1:0] unused_count_builder_output_stalls;
    logic [COUNTER_W-1:0] unused_count_slot_invalid_headers;
    logic [COUNTER_W-1:0] unused_count_slot_commit_stalls;
    logic [COUNTER_W-1:0] unused_count_slot_replay_stalls;
    logic unused_commit_session_active;
    logic unused_replay_session_active;

    logic normal_active_q;
    logic abort_active_q;
    logic emit_started_q;
    logic [FORMAT_W-1:0] selected_format_q;
    logic [2:0] selected_reason_q;
    logic [SIZE_W-1:0] selected_payload_bits_q;

    assign builder_begin_valid =
        workspace_metadata_valid && !workspace_raw_capture_error &&
        !normal_active_q && !abort_active_q;
    assign workspace_metadata_ready = workspace_raw_capture_error ?
        (!normal_active_q && !abort_active_q) : builder_begin_ready;

    assign workspace_emit_start_valid =
        (normal_active_q || abort_active_q) && !emit_started_q;

    assign done_valid = abort_active_q ? workspace_emit_done_valid :
        (normal_active_q && workspace_emit_done_valid && builder_done_valid);
    assign done_tag = abort_active_q ? workspace_emit_done_tag : builder_done_tag;
    assign done_format = abort_active_q ? FORMAT_RAW : builder_done_format;
    assign done_error = abort_active_q ? 1'b1 :
        (workspace_emit_done_error || builder_done_error ||
         workspace_emit_done_tag != builder_done_tag ||
         builder_done_format != selected_format_q ||
         builder_done_word_count != policy_word_count);
    assign done_word_count = abort_active_q ? 8'd0 : builder_done_word_count;
    assign selected_reason = selected_reason_q;
    assign selected_payload_bits = selected_payload_bits_q;

    assign workspace_emit_done_ready = abort_active_q ? done_ready :
        (normal_active_q && builder_done_valid && done_ready);
    assign builder_done_ready =
        normal_active_q && workspace_emit_done_valid && done_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            normal_active_q <= 1'b0;
            abort_active_q <= 1'b0;
            emit_started_q <= 1'b0;
            selected_format_q <= FORMAT_RAW;
            selected_reason_q <= '0;
            selected_payload_bits_q <= '0;
        end else begin
            if (workspace_metadata_valid && workspace_metadata_ready &&
                !normal_active_q && !abort_active_q) begin
                selected_format_q <= workspace_raw_capture_error ?
                    FORMAT_RAW : policy_format;
                selected_reason_q <= policy_reason;
                selected_payload_bits_q <= workspace_raw_capture_error ?
                    SIZE_W'(TOKENS * (LANES + GATE_W)) : policy_payload_bits;
                if (workspace_raw_capture_error)
                    abort_active_q <= 1'b1;
                else
                    normal_active_q <= 1'b1;
            end
            if (workspace_emit_start_valid && workspace_emit_start_ready)
                emit_started_q <= 1'b1;
            if (done_valid && done_ready) begin
                normal_active_q <= 1'b0;
                abort_active_q <= 1'b0;
                emit_started_q <= 1'b0;
            end
        end
    end

    gatestack_canonical_head_workspace_c0 #(
        .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
        .CLASS_SLOTS(CLASS_SLOTS), .CONTEXTS(CONTEXTS), .HEADS(HEADS),
        .TAG_W(TAG_W), .FORMAT_W(FORMAT_W), .COUNTER_W(COUNTER_W),
        .DESTINATION_SCAN_MODE(DESTINATION_SCAN_MODE),
        .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE),
        .EXPLICIT_BITMAP_BANK_ENABLE(EXPLICIT_BITMAP_BANK_ENABLE)
    ) u_workspace (
        .clk_core, .rst_core,
        .head_begin_valid, .head_begin_ready, .head_context_id, .head_id,
        .head_tag, .token_valid, .token_ready, .token_id,
        .token_gate_code, .token_k_bits, .token_last,
        .metadata_valid(workspace_metadata_valid),
        .metadata_ready(workspace_metadata_ready),
        .metadata_context_id(workspace_context),
        .metadata_head_id(workspace_head), .metadata_tag(workspace_tag),
        .metadata_active_classes(workspace_active_classes),
        .metadata_active_tokens(workspace_active_tokens),
        .metadata_term_count(workspace_term_count),
        .metadata_event_count(workspace_event_count),
        .metadata_bitmap_term_count(workspace_bitmap_terms),
        .metadata_fadc_destination_bytes(workspace_fadc_bytes),
        .metadata_overflow(workspace_metadata_overflow),
        .raw_capture_error(workspace_raw_capture_error),
        .emit_start_valid(workspace_emit_start_valid),
        .emit_start_ready(workspace_emit_start_ready),
        .emit_start_format(selected_format_q),
        .descriptor_valid(workspace_descriptor_valid),
        .descriptor_ready(workspace_descriptor_ready),
        .descriptor_gate_code(workspace_descriptor_gate),
        .descriptor_lane_id(workspace_descriptor_lane),
        .descriptor_destination_count(workspace_descriptor_count),
        .descriptor_last(workspace_descriptor_last),
        .destination_valid(workspace_destination_valid),
        .destination_ready(workspace_destination_ready),
        .destination_token_id(workspace_destination_token),
        .destination_last_for_term(workspace_destination_last),
        .destination_bitmap_valid(workspace_destination_bitmap_valid),
        .destination_bitmap_ready(workspace_destination_bitmap_ready),
        .destination_bitmap(workspace_destination_bitmap),
        .raw_token_valid(workspace_raw_valid),
        .raw_token_ready(workspace_raw_ready),
        .raw_token_id(workspace_raw_token),
        .raw_gate_code(workspace_raw_gate), .raw_k_bits(workspace_raw_k),
        .emit_done_valid(workspace_emit_done_valid),
        .emit_done_ready(workspace_emit_done_ready),
        .emit_done_tag(workspace_emit_done_tag),
        .emit_done_error(workspace_emit_done_error),
        .protocol_error(workspace_protocol_error),
        .count_heads(count_workspace_heads),
        .count_raw_fallback_heads(count_workspace_raw_fallback_heads),
        .count_emitted_terms(count_workspace_terms),
        .count_emitted_destinations(count_workspace_destinations),
        .count_destination_scan_cycles(count_workspace_scan_cycles),
        .count_output_stall_cycles(count_workspace_output_stall_cycles)
    );

    gatestack_typed_format_policy #(
        .TOKENS(TOKENS), .HEAD_DIM(LANES), .GATE_W(GATE_W),
        .WORD_W(WORD_W), .SLOT_CAPACITY_BITS(SLOT_WORDS * WORD_W),
        .IPD_CLASS_SLOTS(CLASS_SLOTS), .FORMAT_W(FORMAT_W),
        .SIZE_W(SIZE_W)
    ) u_policy (
        .metadata_active_classes(workspace_active_classes),
        .metadata_term_count(workspace_term_count),
        .metadata_event_count(workspace_event_count),
        .metadata_fadc_destination_bytes(workspace_fadc_bytes),
        .metadata_overflow(workspace_metadata_overflow),
        .decision_format(policy_format), .decision_reason(policy_reason),
        .decision_payload_bits(policy_payload_bits),
        .decision_word_count(policy_word_count),
        .decision_ipd_payload_bytes(unused_policy_ipd_bytes),
        .decision_fadc_payload_bytes(unused_policy_fadc_bytes)
    );

    gatestack_typed_builder_commit_top #(
        .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
        .CONTEXTS(CONTEXTS), .HEADS(HEADS), .SLOT_WORDS(SLOT_WORDS),
        .WORD_W(WORD_W), .TAG_W(TAG_W), .FORMAT_W(FORMAT_W),
        .SIZE_W(SIZE_W), .COUNTER_W(COUNTER_W),
        .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE)
    ) u_builder_commit (
        .clk_core, .rst_core,
        .begin_valid(builder_begin_valid), .begin_ready(builder_begin_ready),
        .begin_context_id(workspace_context), .begin_head_id(workspace_head),
        .begin_tag(workspace_tag), .begin_format(policy_format),
        .begin_expected_payload_bits(policy_payload_bits),
        .begin_active_classes(workspace_active_classes),
        .begin_active_tokens(workspace_active_tokens),
        .begin_term_count(workspace_term_count),
        .begin_event_count(workspace_event_count),
        .begin_bitmap_term_count(workspace_bitmap_terms),
        .begin_fadc_destination_bytes(workspace_fadc_bytes),
        .descriptor_valid(workspace_descriptor_valid),
        .descriptor_ready(workspace_descriptor_ready),
        .descriptor_gate_code(workspace_descriptor_gate),
        .descriptor_lane_id(workspace_descriptor_lane),
        .descriptor_destination_count(workspace_descriptor_count),
        .descriptor_last(workspace_descriptor_last),
        .destination_valid(workspace_destination_valid),
        .destination_ready(workspace_destination_ready),
        .destination_token_id(workspace_destination_token),
        .destination_last_for_term(workspace_destination_last),
        .destination_bitmap_valid(workspace_destination_bitmap_valid),
        .destination_bitmap_ready(workspace_destination_bitmap_ready),
        .destination_bitmap(workspace_destination_bitmap),
        .raw_token_valid(workspace_raw_valid), .raw_token_ready(workspace_raw_ready),
        .raw_token_id(workspace_raw_token), .raw_gate_code(workspace_raw_gate),
        .raw_k_bits(workspace_raw_k),
        .builder_done_valid(builder_done_valid),
        .builder_done_ready(builder_done_ready), .builder_done_tag(builder_done_tag),
        .builder_done_format(builder_done_format),
        .builder_done_error(builder_done_error),
        .builder_done_word_count(builder_done_word_count),
        .inspect_valid, .inspect_ready, .inspect_context_id, .inspect_head_id,
        .inspect_meta_valid, .inspect_meta_ready, .inspect_exists, .inspect_tag,
        .inspect_mode_is_csr, .inspect_format, .inspect_payload_bits,
        .inspect_word_count, .replay_begin_valid, .replay_begin_ready,
        .replay_context_id, .replay_head_id, .replay_start_word,
        .replay_word_valid, .replay_word_ready, .replay_word_data,
        .replay_word_index, .replay_word_last, .replay_tag,
        .replay_mode_is_csr, .replay_format, .replay_payload_bits,
        .release_valid, .release_ready, .release_context_id, .release_head_id,
        .slot_valid_flat,
        .commit_session_active(unused_commit_session_active),
        .replay_session_active(unused_replay_session_active),
        .serializer_protocol_error, .slot_protocol_error,
        .count_builder_heads(unused_count_builder_heads),
        .count_builder_committed_heads, .count_builder_aborted_heads,
        .count_builder_committed_words,
        .count_builder_input_stall_cycles(unused_count_builder_input_stalls),
        .count_builder_output_stall_cycles(unused_count_builder_output_stalls),
        .count_slot_commit_heads, .count_slot_replay_heads,
        .count_slot_release_heads,
        .count_slot_invalid_headers(unused_count_slot_invalid_headers),
        .count_slot_commit_stall_cycles(unused_count_slot_commit_stalls),
        .count_slot_replay_stall_cycles(unused_count_slot_replay_stalls)
    );

endmodule

`default_nettype wire
