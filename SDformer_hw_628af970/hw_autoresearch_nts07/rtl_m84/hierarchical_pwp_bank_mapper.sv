`timescale 1ns/1ps
`default_nettype none

// Random-access mapper for the M83/M84 word-packed PWP phase image.
//
// A flat 128-entry table would store {escape,width,start_word} in 16 bits per
// entry.  M84 keeps the canonical 3-bit width header plus one 13-bit base per
// pattern.  At lookup time a prefix over at most seven preceding blocks
// reconstructs the exact word offset.  Eight interleaved 32-bit banks then
// provide one 256-bit beat without a bank conflict; a rotate restores logical
// word order when start_word is not bank aligned.
module hierarchical_pwp_bank_mapper #(
    parameter int PATTERNS = 16,
    parameter int BLOCKS = 8,
    parameter int OFFSET_W = 13,
    parameter int ROW_W = 10
) (
    input  logic [PATTERNS*BLOCKS*3-1:0] width_header,
    input  logic [PATTERNS*OFFSET_W-1:0] pattern_base_words,
    input  logic [3:0]                    pattern_index,
    input  logic [2:0]                    block_index,
    input  logic [2:0]                    beat_index,
    input  logic [255:0]                  bank_words,

    output logic                          descriptor_valid,
    output logic                          descriptor_escape,
    output logic [3:0]                    descriptor_width,
    output logic [2:0]                    descriptor_beats,
    output logic [OFFSET_W-1:0]           start_word,
    output logic                          beat_index_valid,
    output logic [8*ROW_W-1:0]            bank_row_addresses,
    output logic [255:0]                  logical_words
);
    logic [2:0] selected_code;
    logic [OFFSET_W:0] prefix_words;
    logic [OFFSET_W:0] logical_base_word;
    logic [ROW_W-1:0] base_row;
    logic [2:0] base_bank;

`ifndef SYNTHESIS
    initial begin
        if (PATTERNS != 16 || BLOCKS != 8 || OFFSET_W != 13 || ROW_W != 10)
            $fatal(1, "M84 frozen geometry drift");
    end
`endif

    function automatic logic [5:0] words_for_code(input logic [2:0] code);
        case (code)
            3'd0: words_for_code = 6'd24;
            3'd1: words_for_code = 6'd27;
            3'd2: words_for_code = 6'd30;
            3'd3: words_for_code = 6'd33;
            3'd4: words_for_code = 6'd0;
            default: words_for_code = 6'd0;
        endcase
    endfunction

    always_comb begin : decode_and_map
        selected_code = width_header[
            (pattern_index * BLOCKS + block_index) * 3 +: 3];
        prefix_words = {1'b0, pattern_base_words[
            pattern_index * OFFSET_W +: OFFSET_W]};
        descriptor_valid = selected_code <= 3'd4;
        descriptor_escape = descriptor_valid && selected_code == 3'd4;
        descriptor_width = descriptor_escape ? 4'd12
                                             : {1'b1, selected_code};
        case (selected_code)
            3'd0: descriptor_beats = 3'd3;
            3'd1, 3'd2: descriptor_beats = 3'd4;
            3'd3: descriptor_beats = 3'd5;
            default: descriptor_beats = 3'd0;
        endcase

        for (int prior = 0; prior < BLOCKS; prior++) begin
            if (prior < block_index)
                prefix_words = prefix_words + words_for_code(width_header[
                    (pattern_index * BLOCKS + prior) * 3 +: 3]);
        end
        start_word = prefix_words[OFFSET_W-1:0];
        beat_index_valid = descriptor_valid && !descriptor_escape
                         && beat_index < descriptor_beats;
        logical_base_word = prefix_words + (beat_index * 8);
        base_bank = logical_base_word[2:0];
        base_row = logical_base_word[OFFSET_W:3];

        bank_row_addresses = '0;
        logical_words = '0;
        if (beat_index_valid) begin
            for (int bank = 0; bank < 8; bank++)
                bank_row_addresses[bank*ROW_W +: ROW_W] =
                    base_row + (bank < base_bank);
            for (int word = 0; word < 8; word++)
                logical_words[word*32 +: 32] = bank_words[
                    (((base_bank + word) & 3'h7) * 32) +: 32];
        end
    end
endmodule

`default_nettype wire
