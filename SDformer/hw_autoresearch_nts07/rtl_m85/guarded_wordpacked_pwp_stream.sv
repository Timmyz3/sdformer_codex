`timescale 1ns/1ps
`default_nettype none

// M85 closes the previously untested seam between M83 word-packed records,
// M84 bank mapping, and the M82 zero-bubble signed-vector assembler.
//
// The complete per-phase metadata image is exactly 74 bytes:
//   bits [383:0]   : 128 canonical three-bit width codes;
//   bits [591:384] : sixteen packed 13-bit pattern-base word offsets.
// A phase-load audit poisons reserved codes, inconsistent bases, and an
// out-of-range terminal.  The final 256-bit beat is masked because a word-
// packed 9/10/11-bit PWP otherwise contains words from the following vector.
module guarded_wordpacked_pwp_stream #(
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

    input  logic                     lookup_valid,
    output logic                     lookup_ready,
    input  logic [3:0]               lookup_pattern,
    input  logic [2:0]               lookup_block,
    input  logic [2:0]               lookup_beat,
    input  logic [TAG_W-1:0]         lookup_tag,
    input  logic [255:0]             bank_words,
    output logic [8*ROW_W-1:0]       bank_row_addresses,

    output logic                     output_valid,
    input  logic                     output_ready,
    output logic [TAG_W-1:0]         output_tag,
    output logic [3:0]               output_width,
    output logic                     output_escape,
    output logic [96*12-1:0]         output_values,
    output logic                     output_accept,
    output logic                     protocol_error,
    output logic                     busy
);
    logic [591:0] metadata_q;
    logic phase_loaded_q, phase_poison_q, lookup_error_q;

    logic load_poison;
    logic [13:0] load_terminal_words;
    integer load_cursor;
    integer load_code;
    integer load_fetch_end;

    logic [2:0] selected_code;
    logic [13:0] prefix_words;
    logic [13:0] logical_base_word;
    logic [3:0] descriptor_width;
    logic [2:0] descriptor_beats;
    logic descriptor_escape, mapper_valid, mapper_last;
    logic [2:0] base_bank;
    logic [ROW_W-1:0] base_row;
    logic [255:0] logical_words, masked_words;
    integer lookup_prior_code;
    integer keep_words;

    logic m82_beat_ready, m82_beat_accept;
    logic m82_output_valid, m82_output_accept;
    logic m82_protocol_error, m82_collecting, m82_busy;

`ifndef SYNTHESIS
    initial begin
        if (ROW_W != 10 || TAG_W != 32 || BUFFER_WORDS != 3680)
            $fatal(1, "M85 frozen geometry drift");
    end
`endif

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            4: words_for_code = 0;
            default: words_for_code = 0;
        endcase
    endfunction

    // This audit is on the phase-load path, not the per-lookup critical path.
    always_comb begin : audit_packed_metadata
        load_poison = 1'b0;
        load_cursor = 0;
        for (int pattern = 0; pattern < 16; pattern++) begin
            if (phase_metadata[384 + pattern*13 +: 13]
                    != load_cursor[12:0])
                load_poison = 1'b1;
            for (int block = 0; block < 8; block++) begin
                load_code = phase_metadata[(pattern*8+block)*3 +: 3];
                if (load_code > 4)
                    load_poison = 1'b1;
                else begin
                    load_fetch_end = load_cursor;
                    case (load_code)
                        0: load_fetch_end = load_cursor + 24;
                        1, 2: load_fetch_end = load_cursor + 32;
                        3: load_fetch_end = load_cursor + 40;
                        default: load_fetch_end = load_cursor;
                    endcase
                    if (load_fetch_end > BUFFER_WORDS)
                        load_poison = 1'b1;
                    load_cursor = load_cursor + words_for_code(load_code);
                end
                if (load_cursor > BUFFER_WORDS)
                    load_poison = 1'b1;
            end
        end
        load_terminal_words = (load_cursor + 7) & 14'h3ff8;
        if (load_terminal_words == 0 || load_cursor > BUFFER_WORDS)
            load_poison = 1'b1;
    end

    assign phase_loaded = phase_loaded_q;
    assign metadata_error = phase_poison_q;
    assign phase_load_ready = !m82_busy;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            metadata_q <= '0;
            phase_loaded_q <= 1'b0;
            phase_poison_q <= 1'b0;
            lookup_error_q <= 1'b0;
        end else begin
            if (phase_load_valid && phase_load_ready) begin
                metadata_q <= phase_metadata;
                phase_loaded_q <= 1'b1;
                phase_poison_q <= load_poison;
                lookup_error_q <= 1'b0;
            end else if (lookup_valid && !mapper_valid) begin
                lookup_error_q <= 1'b1;
            end
        end
    end

    always_comb begin : guarded_lookup_and_final_mask
        selected_code = metadata_q[
            (lookup_pattern*8 + lookup_block)*3 +: 3];
        prefix_words = {1'b0, metadata_q[
            384 + lookup_pattern*13 +: 13]};
        for (int prior = 0; prior < 8; prior++) begin
            if (prior < lookup_block) begin
                lookup_prior_code = metadata_q[
                    (lookup_pattern*8 + prior)*3 +: 3];
                prefix_words = prefix_words
                             + words_for_code(lookup_prior_code);
            end
        end
        descriptor_escape = selected_code == 3'd4;
        descriptor_width = descriptor_escape ? 4'd12
                                             : 4'(8 + selected_code);
        case (selected_code)
            3'd0: descriptor_beats = 3;
            3'd1, 3'd2: descriptor_beats = 4;
            3'd3: descriptor_beats = 5;
            default: descriptor_beats = 0;
        endcase
        mapper_last = descriptor_escape
                    || lookup_beat + 1'b1 == descriptor_beats;
        logical_base_word = prefix_words + lookup_beat*8;
        mapper_valid = phase_loaded_q && !phase_poison_q && !lookup_error_q
                     && selected_code <= 4
                     && ((descriptor_escape && lookup_beat == 0)
                         || (!descriptor_escape
                             && lookup_beat < descriptor_beats
                             && logical_base_word + 7 < BUFFER_WORDS));
        base_bank = logical_base_word[2:0];
        base_row = logical_base_word[12:3];

        bank_row_addresses = '0;
        logical_words = '0;
        if (mapper_valid && !descriptor_escape) begin
            for (int bank = 0; bank < 8; bank++)
                bank_row_addresses[bank*ROW_W +: ROW_W] =
                    base_row + (bank < base_bank);
            for (int word = 0; word < 8; word++)
                logical_words[word*32 +: 32] = bank_words[
                    (((base_bank + word) & 3'h7)*32) +: 32];
        end

        keep_words = 8;
        if (mapper_last && !descriptor_escape) begin
            case (descriptor_width)
                8: keep_words = 8;
                9: keep_words = 3;
                10: keep_words = 6;
                11: keep_words = 1;
                default: keep_words = 0;
            endcase
        end
        masked_words = logical_words;
        if (mapper_last)
            for (int word = 0; word < 8; word++)
                if (word >= keep_words)
                    masked_words[word*32 +: 32] = '0;
        if (descriptor_escape)
            masked_words = '0;
    end

    assign lookup_ready = mapper_valid && m82_beat_ready;
    assign protocol_error = phase_poison_q || lookup_error_q
                          || m82_protocol_error;
    assign busy = m82_busy;
    assign output_valid = m82_output_valid;
    assign output_accept = m82_output_accept;

    zero_bubble_elastic_pwp_stream m82_stream (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .beat_valid(lookup_valid && mapper_valid),
        .beat_ready(m82_beat_ready),
        .beat_start(descriptor_escape || lookup_beat == 0),
        .beat_last(mapper_last),
        .beat_width((descriptor_escape || lookup_beat == 0)
                    ? descriptor_width : 4'd0),
        .beat_tag((descriptor_escape || lookup_beat == 0)
                  ? lookup_tag : '0),
        .beat_data(masked_words),
        .beat_accept(m82_beat_accept),
        .output_valid(m82_output_valid),
        .output_ready(output_ready),
        .output_tag(output_tag),
        .output_width(output_width),
        .output_escape(output_escape),
        .output_values(output_values),
        .output_accept(m82_output_accept),
        .protocol_error(m82_protocol_error),
        .collecting(m82_collecting),
        .busy(m82_busy)
    );
endmodule

`default_nettype wire
