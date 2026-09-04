`timescale 1ns/1ps
`default_nettype none

// Bandwidth-matched vector-service island for the M88 cap11 numerator.
// One 256-bit slot serves either a packed PWP vector, a signed correction
// weight vector, or a real escape-fallback weight vector.  Matcher, active-
// source enumeration, DMA, memories, accumulation and phase scheduling remain
// explicit port cuts; this block must not be called a complete accelerator.
module m102_combined_candidate_service_top #(
    parameter int ROW_W = 10,
    parameter int TAG_W = 32,
    parameter int BUFFER_WORDS = 3680
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     phase_load_valid,
    output logic                     phase_load_ready,
    input  logic [591:0]             phase_metadata,
    output logic                     phase_loaded,
    output logic                     metadata_error,

    input  logic                     service_valid,
    output logic                     service_ready,
    // 0=PWP, 1=signed correction weight, 2=escape-fallback weight.
    input  logic [1:0]               service_kind,
    input  logic [3:0]               service_pattern,
    input  logic [3:0]               service_source,
    input  logic [2:0]               service_block,
    input  logic [2:0]               service_beat,
    input  logic                     service_negate,
    input  logic [TAG_W-1:0]         service_tag,
    input  logic [255:0]             bank_words,
    output logic [8*ROW_W-1:0]       bank_row_addresses,
    output logic                     bank_select_pwp,

    output logic                     output_valid,
    input  logic                     output_ready,
    output logic [TAG_W-1:0]         output_tag,
    output logic [1:0]               output_kind,
    output logic [3:0]               output_width,
    output logic                     output_escape,
    output logic [96*12-1:0]         output_values,
    output logic                     output_accept,
    output logic                     protocol_error,
    output logic                     busy
);
    localparam logic [1:0] KIND_PWP = 2'd0;
    localparam logic [1:0] KIND_CORRECTION = 2'd1;
    localparam logic [1:0] KIND_FALLBACK = 2'd2;

    logic [591:0] metadata_q;
    logic phase_loaded_q, phase_poison_q, request_fault_q;
    logic parse_active_q, parse_poison_q;
    logic [6:0] parse_index_q;
    logic [13:0] parse_cursor_q;
    logic [2:0] parse_code;
    logic [5:0] parse_used_words, parse_fetch_words;
    logic [13:0] parse_next_cursor, parse_fetch_end;
    logic [13:0] parse_pattern_base, parse_terminal_words;
    logic parse_entry_poison, parse_terminal_poison;

    logic transaction_active_q;
    logic [1:0] transaction_kind_q;
    logic [3:0] transaction_pattern_q, transaction_source_q;
    logic [2:0] transaction_block_q, expected_beat_q;
    logic transaction_negate_q;
    logic [TAG_W-1:0] transaction_tag_q;

    // A ready/valid producer may legally keep the just-accepted request
    // asserted until it observes the active edge.  After that edge the
    // transaction sequencer has already advanced, so the old request no
    // longer satisfies request_semantically_valid.  Remember its complete
    // identity and tolerate only that exact, already-consumed request.  A
    // changed request still enters fail-closed quarantine combinationally.
    logic accepted_grace_q;
    logic [1:0] accepted_grace_kind_q;
    logic [3:0] accepted_grace_pattern_q, accepted_grace_source_q;
    logic [2:0] accepted_grace_block_q, accepted_grace_beat_q;
    logic accepted_grace_negate_q;
    logic [TAG_W-1:0] accepted_grace_tag_q;

    logic [1:0] output_kind_q;
    logic output_negate_q;

    logic [2:0] selected_code;
    logic [13:0] prefix_words, logical_base_word, weight_base_word;
    logic [3:0] descriptor_width;
    logic [2:0] descriptor_beats;
    logic request_kind_valid, request_code_valid, request_identity_valid;
    logic request_semantically_valid, request_violation, request_last;
    logic accepted_grace_match;
    logic [2:0] base_bank;
    logic [ROW_W-1:0] base_row;
    logic [255:0] logical_words, masked_words;
    logic [3:0] keep_words;

    logic m82_beat_ready, m82_beat_accept;
    logic m82_output_valid, m82_output_accept;
    logic [TAG_W-1:0] m82_output_tag;
    logic [3:0] m82_output_width;
    logic m82_output_escape, m82_protocol_error, m82_collecting, m82_busy;
    logic [96*12-1:0] m82_output_values;

`ifndef SYNTHESIS
    initial begin
        if (ROW_W != 10 || TAG_W != 32 || BUFFER_WORDS != 3680)
            $fatal(1, "M102 combined frozen geometry drift");
    end
`endif

    function automatic logic [5:0] words_for_code(input logic [2:0] code);
        case (code)
            3'd0: words_for_code = 6'd24;
            3'd1: words_for_code = 6'd27;
            3'd2: words_for_code = 6'd30;
            3'd3: words_for_code = 6'd33;
            default: words_for_code = 6'd0;
        endcase
    endfunction

    always_comb begin : serial_metadata_audit
        parse_code = metadata_q[parse_index_q*3 +: 3];
        parse_used_words = '0;
        parse_fetch_words = '0;
        case (parse_code)
            3'd0: begin parse_used_words = 6'd24; parse_fetch_words = 6'd24; end
            3'd1: begin parse_used_words = 6'd27; parse_fetch_words = 6'd32; end
            3'd2: begin parse_used_words = 6'd30; parse_fetch_words = 6'd32; end
            3'd3: begin parse_used_words = 6'd33; parse_fetch_words = 6'd40; end
            3'd4: begin parse_used_words = 6'd0;  parse_fetch_words = 6'd0;  end
            default: begin parse_used_words = 6'd0; parse_fetch_words = 6'd0; end
        endcase
        parse_next_cursor = parse_cursor_q + parse_used_words;
        parse_fetch_end = parse_cursor_q + parse_fetch_words;
        parse_pattern_base = {1'b0, metadata_q[
            384 + parse_index_q[6:3]*13 +: 13]};
        parse_terminal_words = (parse_next_cursor + 14'd7) & 14'h3ff8;
        parse_entry_poison = parse_code > 3'd4
                           || parse_fetch_end > BUFFER_WORDS
                           || parse_next_cursor > BUFFER_WORDS
                           || (parse_index_q[2:0] == 0
                               && parse_pattern_base != parse_cursor_q);
        parse_terminal_poison = parse_terminal_words == 0
                              || parse_next_cursor > BUFFER_WORDS;
    end

    assign phase_loaded = phase_loaded_q;
    assign metadata_error = phase_poison_q;
    assign phase_load_ready = !protocol_error && !m82_busy && !parse_active_q
                            && !transaction_active_q && !service_valid;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            metadata_q <= '0;
            phase_loaded_q <= 1'b0;
            phase_poison_q <= 1'b0;
            request_fault_q <= 1'b0;
            parse_active_q <= 1'b0;
            parse_poison_q <= 1'b0;
            parse_index_q <= '0;
            parse_cursor_q <= '0;
            transaction_active_q <= 1'b0;
            transaction_kind_q <= '0;
            transaction_pattern_q <= '0;
            transaction_source_q <= '0;
            transaction_block_q <= '0;
            expected_beat_q <= '0;
            transaction_negate_q <= 1'b0;
            transaction_tag_q <= '0;
            accepted_grace_q <= 1'b0;
            accepted_grace_kind_q <= '0;
            accepted_grace_pattern_q <= '0;
            accepted_grace_source_q <= '0;
            accepted_grace_block_q <= '0;
            accepted_grace_beat_q <= '0;
            accepted_grace_negate_q <= 1'b0;
            accepted_grace_tag_q <= '0;
            output_kind_q <= '0;
            output_negate_q <= 1'b0;
        end else begin
            if (!service_valid || !accepted_grace_match)
                accepted_grace_q <= 1'b0;
            if (phase_load_valid && phase_load_ready) begin
                metadata_q <= phase_metadata;
                phase_loaded_q <= 1'b0;
                phase_poison_q <= 1'b0;
                parse_active_q <= 1'b1;
                parse_poison_q <= 1'b0;
                parse_index_q <= '0;
                parse_cursor_q <= '0;
                transaction_active_q <= 1'b0;
            end else begin
                if (parse_active_q) begin
                    parse_poison_q <= parse_poison_q || parse_entry_poison;
                    parse_cursor_q <= parse_next_cursor;
                    if (parse_index_q == 7'd127) begin
                        parse_active_q <= 1'b0;
                        phase_loaded_q <= 1'b1;
                        phase_poison_q <= parse_poison_q
                                        || parse_entry_poison
                                        || parse_terminal_poison;
                    end else begin
                        parse_index_q <= parse_index_q + 1'b1;
                    end
                end
                if (request_violation)
                    request_fault_q <= 1'b1;
            end

            if (m82_beat_accept) begin
                accepted_grace_q <= 1'b1;
                accepted_grace_kind_q <= service_kind;
                accepted_grace_pattern_q <= service_pattern;
                accepted_grace_source_q <= service_source;
                accepted_grace_block_q <= service_block;
                accepted_grace_beat_q <= service_beat;
                accepted_grace_negate_q <= service_negate;
                accepted_grace_tag_q <= service_tag;
                if (!transaction_active_q) begin
                    transaction_kind_q <= service_kind;
                    transaction_pattern_q <= service_pattern;
                    transaction_source_q <= service_source;
                    transaction_block_q <= service_block;
                    transaction_negate_q <= service_negate;
                    transaction_tag_q <= service_tag;
                    expected_beat_q <= 3'd1;
                    transaction_active_q <= !request_last;
                    output_kind_q <= service_kind;
                    output_negate_q <= service_kind == KIND_CORRECTION
                                     && service_negate;
                end else if (request_last) begin
                    transaction_active_q <= 1'b0;
                    expected_beat_q <= '0;
                end else begin
                    expected_beat_q <= expected_beat_q + 1'b1;
                end
            end
        end
    end

    always_comb begin : shared_slot_mapper
        selected_code = metadata_q[
            (service_pattern*8 + service_block)*3 +: 3];
        prefix_words = {1'b0, metadata_q[
            384 + service_pattern*13 +: 13]};
        for (int prior = 0; prior < 8; prior++)
            if (prior < service_block)
                prefix_words = prefix_words + {8'd0, words_for_code(
                    metadata_q[({service_pattern, prior[2:0]}*3) +: 3])};

        request_kind_valid = service_kind inside {
            KIND_PWP, KIND_CORRECTION, KIND_FALLBACK};
        request_code_valid = (service_kind == KIND_PWP
                              && selected_code <= 3)
                           || (service_kind == KIND_CORRECTION
                               && selected_code <= 3)
                           || (service_kind == KIND_FALLBACK
                               && selected_code == 4);
        descriptor_width = service_kind == KIND_PWP
                         ? 4'd8 + {1'b0, selected_code} : 4'd8;
        if (service_kind != KIND_PWP) begin
            descriptor_beats = 3;
        end else begin
            case (selected_code)
                3'd0: descriptor_beats = 3;
                3'd1, 3'd2: descriptor_beats = 4;
                3'd3: descriptor_beats = 5;
                default: descriptor_beats = 0;
            endcase
        end
        request_last = service_beat + 3'd1 == descriptor_beats;

        request_identity_valid = 1'b1;
        if (!transaction_active_q) begin
            request_identity_valid = service_beat == 0;
        end else begin
            request_identity_valid = service_kind == transaction_kind_q
                                  && service_pattern == transaction_pattern_q
                                  && service_source == transaction_source_q
                                  && service_block == transaction_block_q
                                  && service_beat == expected_beat_q
                                  && service_negate == transaction_negate_q
                                  && service_tag == transaction_tag_q;
        end
        if (service_kind != KIND_CORRECTION && service_negate)
            request_identity_valid = 1'b0;

        weight_base_word = {7'd0, service_source, service_block} * 14'd24;
        logical_base_word = service_kind == KIND_PWP
                          ? prefix_words + ({11'd0, service_beat} << 3)
                          : weight_base_word + ({11'd0, service_beat} << 3);
        request_semantically_valid = phase_loaded_q && !phase_poison_q
                                   && !request_fault_q && request_kind_valid
                                   && request_code_valid
                                   && request_identity_valid
                                   && service_beat < descriptor_beats
                                   && logical_base_word + 7 < (service_kind == KIND_PWP
                                                               ? BUFFER_WORDS
                                                               : 3072);

        base_bank = logical_base_word[2:0];
        base_row = logical_base_word[12:3];
        bank_row_addresses = '0;
        logical_words = '0;
        if (request_semantically_valid) begin
            for (int bank = 0; bank < 8; bank++)
                bank_row_addresses[bank*ROW_W +: ROW_W] =
                    base_row + (bank < base_bank);
            for (int word = 0; word < 8; word++)
                logical_words[word*32 +: 32] = bank_words[
                    (((base_bank + word) & 3'h7)*32) +: 32];
        end

        keep_words = 4'd8;
        if (service_kind == KIND_PWP && request_last) begin
            case (descriptor_width)
                4'd8: keep_words = 4'd8;
                4'd9: keep_words = 4'd3;
                4'd10: keep_words = 4'd6;
                4'd11: keep_words = 4'd1;
                default: keep_words = 4'd0;
            endcase
        end
        masked_words = logical_words;
        if (request_last)
            for (int word = 0; word < 8; word++)
                if (word >= keep_words)
                    masked_words[word*32 +: 32] = '0;
    end

    assign bank_select_pwp = service_kind == KIND_PWP;
    assign service_ready = request_semantically_valid && m82_beat_ready;
    assign accepted_grace_match = accepted_grace_q
                                && service_kind == accepted_grace_kind_q
                                && service_pattern == accepted_grace_pattern_q
                                && service_source == accepted_grace_source_q
                                && service_block == accepted_grace_block_q
                                && service_beat == accepted_grace_beat_q
                                && service_negate == accepted_grace_negate_q
                                && service_tag == accepted_grace_tag_q;
    assign request_violation = service_valid && !request_semantically_valid
                             && !accepted_grace_match;
    assign protocol_error = request_fault_q || phase_poison_q
                          || m82_protocol_error || request_violation;
    assign busy = parse_active_q || transaction_active_q || m82_busy;

    // A top-level metadata or request fault owns the externally visible
    // transaction.  M82 may still hold an older buffered result, but that
    // result is quarantined until reset and can never be accepted under fault.
    assign output_valid = !protocol_error && m82_output_valid;
    assign output_tag = output_valid ? m82_output_tag : '0;
    assign output_kind = output_valid ? output_kind_q : '0;
    assign output_width = output_valid ? m82_output_width : '0;
    assign output_escape = 1'b0;
    assign output_accept = output_valid && output_ready;
    always_comb begin : signed_correction_transform
        output_values = '0;
        if (output_valid) begin
            output_values = m82_output_values;
            if (output_negate_q)
                for (int lane = 0; lane < 96; lane++)
                    output_values[lane*12 +: 12] =
                        (~m82_output_values[lane*12 +: 12]) + 1'b1;
        end
    end

    zero_bubble_elastic_pwp_stream m82_stream (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .beat_valid(service_valid && request_semantically_valid),
        .beat_ready(m82_beat_ready),
        .beat_start(service_beat == 0),
        .beat_last(request_last),
        .beat_width(service_beat == 0 ? descriptor_width : 4'd0),
        .beat_tag(service_beat == 0 ? service_tag : '0),
        .beat_data(masked_words),
        .beat_accept(m82_beat_accept),
        .output_valid(m82_output_valid),
        .output_ready(output_ready && !protocol_error),
        .output_tag(m82_output_tag),
        .output_width(m82_output_width),
        .output_escape(m82_output_escape),
        .output_values(m82_output_values),
        .output_accept(m82_output_accept),
        .protocol_error(m82_protocol_error),
        .collecting(m82_collecting),
        .busy(m82_busy)
    );
endmodule

`default_nettype wire
