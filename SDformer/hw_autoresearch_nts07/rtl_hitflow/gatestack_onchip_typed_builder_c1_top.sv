`timescale 1ns/1ps
`default_nettype none

// C1 uses two canonical workspaces around one shared serializer/slot path.
// Sequence tags preserve in-order issue while capture overlaps prior service.
module gatestack_onchip_typed_builder_c1_top #(
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
    output logic [COUNTER_W-1:0]         done_sequence,

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
    output logic [COUNTER_W-1:0]         count_slot_release_heads,
    output logic [COUNTER_W-1:0]         count_capture_blocked_cycles,
    output logic [COUNTER_W-1:0]         count_capture_service_overlap_cycles,
    output logic [COUNTER_W-1:0]         count_order_wait_cycles
);

    localparam int WORKSPACES = 32'd2;
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);

    logic [WORKSPACES-1:0] ws_head_begin_valid;
    logic [WORKSPACES-1:0] ws_head_begin_ready;
    logic [WORKSPACES-1:0] ws_token_valid;
    logic [WORKSPACES-1:0] ws_token_ready;
    logic [WORKSPACES-1:0] ws_metadata_valid;
    logic [WORKSPACES-1:0] ws_metadata_ready;
    logic [CONTEXT_ID_W-1:0] ws_context [0:WORKSPACES-1];
    logic [HEAD_ID_W-1:0] ws_head [0:WORKSPACES-1];
    logic [TAG_W-1:0] ws_tag [0:WORKSPACES-1];
    logic [3:0] ws_active_classes [0:WORKSPACES-1];
    logic [7:0] ws_active_tokens [0:WORKSPACES-1];
    logic [7:0] ws_term_count [0:WORKSPACES-1];
    logic [12:0] ws_event_count [0:WORKSPACES-1];
    logic [7:0] ws_bitmap_terms [0:WORKSPACES-1];
    logic [12:0] ws_fadc_bytes [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_metadata_overflow;
    logic [WORKSPACES-1:0] ws_raw_capture_error;
    logic [WORKSPACES-1:0] ws_emit_start_valid;
    logic [WORKSPACES-1:0] ws_emit_start_ready;
    logic [FORMAT_W-1:0] ws_emit_start_format [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_descriptor_valid;
    logic [WORKSPACES-1:0] ws_descriptor_ready;
    logic [GATE_W-1:0] ws_descriptor_gate [0:WORKSPACES-1];
    logic [4:0] ws_descriptor_lane [0:WORKSPACES-1];
    logic [7:0] ws_descriptor_count [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_descriptor_last;
    logic [WORKSPACES-1:0] ws_destination_valid;
    logic [WORKSPACES-1:0] ws_destination_ready;
    logic [7:0] ws_destination_token [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_destination_last;
    logic [WORKSPACES-1:0] ws_destination_bitmap_valid;
    logic [WORKSPACES-1:0] ws_destination_bitmap_ready;
    logic [TOKENS-1:0] ws_destination_bitmap [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_raw_valid;
    logic [WORKSPACES-1:0] ws_raw_ready;
    logic [7:0] ws_raw_token [0:WORKSPACES-1];
    logic [GATE_W-1:0] ws_raw_gate [0:WORKSPACES-1];
    logic [LANES-1:0] ws_raw_k [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_emit_done_valid;
    logic [WORKSPACES-1:0] ws_emit_done_ready;
    logic [TAG_W-1:0] ws_emit_done_tag [0:WORKSPACES-1];
    logic [WORKSPACES-1:0] ws_emit_done_error;
    logic [WORKSPACES-1:0] ws_protocol_error;
    logic [COUNTER_W-1:0] ws_count_heads [0:WORKSPACES-1];
    logic [COUNTER_W-1:0] ws_count_raw [0:WORKSPACES-1];
    logic [COUNTER_W-1:0] ws_count_terms [0:WORKSPACES-1];
    logic [COUNTER_W-1:0] ws_count_destinations [0:WORKSPACES-1];
    logic [COUNTER_W-1:0] ws_count_scans [0:WORKSPACES-1];
    logic [COUNTER_W-1:0] ws_count_stalls [0:WORKSPACES-1];

    logic [FORMAT_W-1:0] policy_format [0:WORKSPACES-1];
    logic [2:0] policy_reason [0:WORKSPACES-1];
    logic [SIZE_W-1:0] policy_payload_bits [0:WORKSPACES-1];
    logic [7:0] policy_word_count [0:WORKSPACES-1];
    logic [SIZE_W-1:0] unused_policy_ipd_bytes [0:WORKSPACES-1];
    logic [SIZE_W-1:0] unused_policy_fadc_bytes [0:WORKSPACES-1];

    logic capture_active_q;
    logic capture_owner_q;
    logic allocate_owner;
    logic head_begin_fire;
    logic token_fire;
    logic [COUNTER_W-1:0] ws_sequence_q [0:WORKSPACES-1];
    logic [COUNTER_W-1:0] next_capture_sequence_q;
    logic [COUNTER_W-1:0] next_emit_sequence_q;
    logic oldest_valid;
    logic oldest_owner;
    logic metadata_fire;

    logic session_active_q;
    logic session_abort_q;
    logic emit_started_q;
    logic emit_owner_q;
    logic [FORMAT_W-1:0] selected_format_q;
    logic [2:0] selected_reason_q;
    logic [SIZE_W-1:0] selected_payload_bits_q;
    logic [7:0] selected_word_count_q;

    logic builder_begin_valid;
    logic builder_begin_ready;
    logic builder_descriptor_ready;
    logic builder_destination_ready;
    logic builder_destination_bitmap_ready;
    logic builder_raw_ready;
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

    assign allocate_owner = !ws_head_begin_ready[0] &&
                            ws_head_begin_ready[1];
    assign head_begin_ready = !capture_active_q &&
        (ws_head_begin_ready[0] || ws_head_begin_ready[1]);
    assign head_begin_fire = head_begin_valid && head_begin_ready;
    assign ws_head_begin_valid[0] = head_begin_valid && head_begin_ready &&
        !allocate_owner;
    assign ws_head_begin_valid[1] = head_begin_valid && head_begin_ready &&
        allocate_owner;
    assign ws_token_valid[0] = token_valid && capture_active_q &&
        !capture_owner_q;
    assign ws_token_valid[1] = token_valid && capture_active_q &&
        capture_owner_q;
    assign token_ready = capture_active_q ? ws_token_ready[capture_owner_q] :
                                           1'b0;
    assign token_fire = token_valid && token_ready;

    always_comb begin
        oldest_valid = 1'b0;
        oldest_owner = 1'b0;
        if (ws_metadata_valid[0] &&
            ws_sequence_q[0] == next_emit_sequence_q) begin
            oldest_valid = 1'b1;
            oldest_owner = 1'b0;
        end else if (ws_metadata_valid[1] &&
                     ws_sequence_q[1] == next_emit_sequence_q) begin
            oldest_valid = 1'b1;
            oldest_owner = 1'b1;
        end
    end

    assign builder_begin_valid = !session_active_q && oldest_valid &&
        !ws_raw_capture_error[oldest_owner];
    always_comb begin
        ws_metadata_ready = '0;
        if (!session_active_q && oldest_valid) begin
            if (ws_raw_capture_error[oldest_owner])
                ws_metadata_ready[oldest_owner] = 1'b1;
            else
                ws_metadata_ready[oldest_owner] = builder_begin_ready;
        end
    end
    assign metadata_fire = oldest_valid &&
        ws_metadata_valid[oldest_owner] && ws_metadata_ready[oldest_owner];

    always_comb begin
        ws_emit_start_valid = '0;
        ws_descriptor_ready = '0;
        ws_destination_ready = '0;
        ws_destination_bitmap_ready = '0;
        ws_raw_ready = '0;
        ws_emit_done_ready = '0;
        for (int workspace = 32'd0; workspace < WORKSPACES;
             workspace = workspace + 32'd1)
            ws_emit_start_format[workspace] = selected_format_q;
        if (session_active_q) begin
            ws_emit_start_valid[emit_owner_q] = !emit_started_q;
            ws_descriptor_ready[emit_owner_q] = builder_descriptor_ready;
            ws_destination_ready[emit_owner_q] = builder_destination_ready;
            ws_destination_bitmap_ready[emit_owner_q] =
                builder_destination_bitmap_ready;
            ws_raw_ready[emit_owner_q] = builder_raw_ready;
            if (session_abort_q)
                ws_emit_done_ready[emit_owner_q] = done_ready;
            else
                ws_emit_done_ready[emit_owner_q] =
                    builder_done_valid && done_ready;
        end
    end

    assign done_valid = session_abort_q ?
        (session_active_q && ws_emit_done_valid[emit_owner_q]) :
        (session_active_q && ws_emit_done_valid[emit_owner_q] &&
         builder_done_valid);
    assign done_tag = session_abort_q ? ws_emit_done_tag[emit_owner_q] :
                                       builder_done_tag;
    assign done_format = session_abort_q ? FORMAT_RAW : builder_done_format;
    assign done_error = session_abort_q ? 1'b1 :
        (ws_emit_done_error[emit_owner_q] || builder_done_error ||
         ws_emit_done_tag[emit_owner_q] != builder_done_tag ||
         builder_done_format != selected_format_q ||
         builder_done_word_count != selected_word_count_q);
    assign done_word_count = session_abort_q ? 8'd0 :
                                                builder_done_word_count;
    assign selected_reason = selected_reason_q;
    assign selected_payload_bits = selected_payload_bits_q;
    assign done_sequence = next_emit_sequence_q;
    assign builder_done_ready = session_active_q && !session_abort_q &&
        ws_emit_done_valid[emit_owner_q] && done_ready;

    assign workspace_protocol_error = |ws_protocol_error;
    assign count_workspace_heads = ws_count_heads[0] + ws_count_heads[1];
    assign count_workspace_raw_fallback_heads =
        ws_count_raw[0] + ws_count_raw[1];
    assign count_workspace_terms = ws_count_terms[0] + ws_count_terms[1];
    assign count_workspace_destinations =
        ws_count_destinations[0] + ws_count_destinations[1];
    assign count_workspace_scan_cycles =
        ws_count_scans[0] + ws_count_scans[1];
    assign count_workspace_output_stall_cycles =
        ws_count_stalls[0] + ws_count_stalls[1];

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            capture_active_q <= 1'b0;
            capture_owner_q <= 1'b0;
            ws_sequence_q[0] <= '0;
            ws_sequence_q[1] <= '0;
            next_capture_sequence_q <= '0;
            next_emit_sequence_q <= '0;
            session_active_q <= 1'b0;
            session_abort_q <= 1'b0;
            emit_started_q <= 1'b0;
            emit_owner_q <= 1'b0;
            selected_format_q <= FORMAT_RAW;
            selected_reason_q <= '0;
            selected_payload_bits_q <= '0;
            selected_word_count_q <= '0;
            count_capture_blocked_cycles <= '0;
            count_capture_service_overlap_cycles <= '0;
            count_order_wait_cycles <= '0;
        end else begin
            if (head_begin_valid && !head_begin_ready)
                count_capture_blocked_cycles <=
                    count_capture_blocked_cycles + 1'b1;
            if (capture_active_q && session_active_q)
                count_capture_service_overlap_cycles <=
                    count_capture_service_overlap_cycles + 1'b1;
            if (!session_active_q && |ws_metadata_valid && !oldest_valid)
                count_order_wait_cycles <= count_order_wait_cycles + 1'b1;

            if (head_begin_fire) begin
                capture_active_q <= 1'b1;
                capture_owner_q <= allocate_owner;
                ws_sequence_q[allocate_owner] <= next_capture_sequence_q;
                next_capture_sequence_q <= next_capture_sequence_q + 1'b1;
            end
            if (token_fire && token_last)
                capture_active_q <= 1'b0;

            if (metadata_fire) begin
                session_active_q <= 1'b1;
                session_abort_q <= ws_raw_capture_error[oldest_owner];
                emit_started_q <= 1'b0;
                emit_owner_q <= oldest_owner;
                selected_format_q <= ws_raw_capture_error[oldest_owner] ?
                    FORMAT_RAW : policy_format[oldest_owner];
                selected_reason_q <= policy_reason[oldest_owner];
                selected_payload_bits_q <=
                    ws_raw_capture_error[oldest_owner] ?
                    SIZE_W'(TOKENS * (LANES + GATE_W)) :
                    policy_payload_bits[oldest_owner];
                selected_word_count_q <= ws_raw_capture_error[oldest_owner] ?
                    8'd0 : policy_word_count[oldest_owner];
            end
            if (session_active_q && !emit_started_q &&
                ws_emit_start_valid[emit_owner_q] &&
                ws_emit_start_ready[emit_owner_q])
                emit_started_q <= 1'b1;
            if (done_valid && done_ready) begin
                session_active_q <= 1'b0;
                session_abort_q <= 1'b0;
                emit_started_q <= 1'b0;
                next_emit_sequence_q <= next_emit_sequence_q + 1'b1;
            end
        end
    end

    generate
        for (genvar workspace = 32'd0; workspace < WORKSPACES;
             workspace = workspace + 32'd1) begin : g_workspace
            gatestack_canonical_head_workspace_c0 #(
                .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
                .CLASS_SLOTS(CLASS_SLOTS), .CONTEXTS(CONTEXTS),
                .HEADS(HEADS), .TAG_W(TAG_W), .FORMAT_W(FORMAT_W),
                .COUNTER_W(COUNTER_W),
                .DESTINATION_SCAN_MODE(DESTINATION_SCAN_MODE),
                .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE),
                .EXPLICIT_BITMAP_BANK_ENABLE(EXPLICIT_BITMAP_BANK_ENABLE)
            ) u_workspace (
                .clk_core, .rst_core,
                .head_begin_valid(ws_head_begin_valid[workspace]),
                .head_begin_ready(ws_head_begin_ready[workspace]),
                .head_context_id, .head_id, .head_tag,
                .token_valid(ws_token_valid[workspace]),
                .token_ready(ws_token_ready[workspace]),
                .token_id, .token_gate_code, .token_k_bits, .token_last,
                .metadata_valid(ws_metadata_valid[workspace]),
                .metadata_ready(ws_metadata_ready[workspace]),
                .metadata_context_id(ws_context[workspace]),
                .metadata_head_id(ws_head[workspace]),
                .metadata_tag(ws_tag[workspace]),
                .metadata_active_classes(ws_active_classes[workspace]),
                .metadata_active_tokens(ws_active_tokens[workspace]),
                .metadata_term_count(ws_term_count[workspace]),
                .metadata_event_count(ws_event_count[workspace]),
                .metadata_bitmap_term_count(ws_bitmap_terms[workspace]),
                .metadata_fadc_destination_bytes(ws_fadc_bytes[workspace]),
                .metadata_overflow(ws_metadata_overflow[workspace]),
                .raw_capture_error(ws_raw_capture_error[workspace]),
                .emit_start_valid(ws_emit_start_valid[workspace]),
                .emit_start_ready(ws_emit_start_ready[workspace]),
                .emit_start_format(ws_emit_start_format[workspace]),
                .descriptor_valid(ws_descriptor_valid[workspace]),
                .descriptor_ready(ws_descriptor_ready[workspace]),
                .descriptor_gate_code(ws_descriptor_gate[workspace]),
                .descriptor_lane_id(ws_descriptor_lane[workspace]),
                .descriptor_destination_count(ws_descriptor_count[workspace]),
                .descriptor_last(ws_descriptor_last[workspace]),
                .destination_valid(ws_destination_valid[workspace]),
                .destination_ready(ws_destination_ready[workspace]),
                .destination_token_id(ws_destination_token[workspace]),
                .destination_last_for_term(ws_destination_last[workspace]),
                .destination_bitmap_valid(
                    ws_destination_bitmap_valid[workspace]),
                .destination_bitmap_ready(
                    ws_destination_bitmap_ready[workspace]),
                .destination_bitmap(ws_destination_bitmap[workspace]),
                .raw_token_valid(ws_raw_valid[workspace]),
                .raw_token_ready(ws_raw_ready[workspace]),
                .raw_token_id(ws_raw_token[workspace]),
                .raw_gate_code(ws_raw_gate[workspace]),
                .raw_k_bits(ws_raw_k[workspace]),
                .emit_done_valid(ws_emit_done_valid[workspace]),
                .emit_done_ready(ws_emit_done_ready[workspace]),
                .emit_done_tag(ws_emit_done_tag[workspace]),
                .emit_done_error(ws_emit_done_error[workspace]),
                .protocol_error(ws_protocol_error[workspace]),
                .count_heads(ws_count_heads[workspace]),
                .count_raw_fallback_heads(ws_count_raw[workspace]),
                .count_emitted_terms(ws_count_terms[workspace]),
                .count_emitted_destinations(
                    ws_count_destinations[workspace]),
                .count_destination_scan_cycles(ws_count_scans[workspace]),
                .count_output_stall_cycles(ws_count_stalls[workspace])
            );

            gatestack_typed_format_policy #(
                .TOKENS(TOKENS), .HEAD_DIM(LANES), .GATE_W(GATE_W),
                .WORD_W(WORD_W),
                .SLOT_CAPACITY_BITS(SLOT_WORDS * WORD_W),
                .IPD_CLASS_SLOTS(CLASS_SLOTS), .FORMAT_W(FORMAT_W),
                .SIZE_W(SIZE_W)
            ) u_policy (
                .metadata_active_classes(ws_active_classes[workspace]),
                .metadata_term_count(ws_term_count[workspace]),
                .metadata_event_count(ws_event_count[workspace]),
                .metadata_fadc_destination_bytes(ws_fadc_bytes[workspace]),
                .metadata_overflow(ws_metadata_overflow[workspace]),
                .decision_format(policy_format[workspace]),
                .decision_reason(policy_reason[workspace]),
                .decision_payload_bits(policy_payload_bits[workspace]),
                .decision_word_count(policy_word_count[workspace]),
                .decision_ipd_payload_bytes(
                    unused_policy_ipd_bytes[workspace]),
                .decision_fadc_payload_bytes(
                    unused_policy_fadc_bytes[workspace])
            );
        end
    endgenerate

    gatestack_typed_builder_commit_top #(
        .TOKENS(TOKENS), .LANES(LANES), .GATE_W(GATE_W),
        .CONTEXTS(CONTEXTS), .HEADS(HEADS), .SLOT_WORDS(SLOT_WORDS),
        .WORD_W(WORD_W), .TAG_W(TAG_W), .FORMAT_W(FORMAT_W),
        .SIZE_W(SIZE_W), .COUNTER_W(COUNTER_W),
        .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE)
    ) u_builder_commit (
        .clk_core, .rst_core,
        .begin_valid(builder_begin_valid), .begin_ready(builder_begin_ready),
        .begin_context_id(ws_context[oldest_owner]),
        .begin_head_id(ws_head[oldest_owner]),
        .begin_tag(ws_tag[oldest_owner]),
        .begin_format(policy_format[oldest_owner]),
        .begin_expected_payload_bits(policy_payload_bits[oldest_owner]),
        .begin_active_classes(ws_active_classes[oldest_owner]),
        .begin_active_tokens(ws_active_tokens[oldest_owner]),
        .begin_term_count(ws_term_count[oldest_owner]),
        .begin_event_count(ws_event_count[oldest_owner]),
        .begin_bitmap_term_count(ws_bitmap_terms[oldest_owner]),
        .begin_fadc_destination_bytes(ws_fadc_bytes[oldest_owner]),
        .descriptor_valid(session_active_q && !session_abort_q &&
                          ws_descriptor_valid[emit_owner_q]),
        .descriptor_ready(builder_descriptor_ready),
        .descriptor_gate_code(ws_descriptor_gate[emit_owner_q]),
        .descriptor_lane_id(ws_descriptor_lane[emit_owner_q]),
        .descriptor_destination_count(ws_descriptor_count[emit_owner_q]),
        .descriptor_last(ws_descriptor_last[emit_owner_q]),
        .destination_valid(session_active_q && !session_abort_q &&
                           ws_destination_valid[emit_owner_q]),
        .destination_ready(builder_destination_ready),
        .destination_token_id(ws_destination_token[emit_owner_q]),
        .destination_last_for_term(ws_destination_last[emit_owner_q]),
        .destination_bitmap_valid(session_active_q && !session_abort_q &&
                                  ws_destination_bitmap_valid[emit_owner_q]),
        .destination_bitmap_ready(builder_destination_bitmap_ready),
        .destination_bitmap(ws_destination_bitmap[emit_owner_q]),
        .raw_token_valid(session_active_q && !session_abort_q &&
                         ws_raw_valid[emit_owner_q]),
        .raw_token_ready(builder_raw_ready),
        .raw_token_id(ws_raw_token[emit_owner_q]),
        .raw_gate_code(ws_raw_gate[emit_owner_q]),
        .raw_k_bits(ws_raw_k[emit_owner_q]),
        .builder_done_valid, .builder_done_ready, .builder_done_tag,
        .builder_done_format, .builder_done_error, .builder_done_word_count,
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
