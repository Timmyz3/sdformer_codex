`timescale 1ns/1ps
`default_nettype none

// M102 physical-denominator service island for the frozen M78/M88 bit-sparse
// model.  Each request is already one precompacted active-source/output-block
// vector operation.  Mask scanning, source enumeration, SRAM, DMA, phase
// scheduling and accumulation are explicit port cuts.
//
// A 96-byte INT8 weight vector is served through exactly three 256-bit beats.
// The eight 32-bit bank words are rotated into logical word order before the
// unchanged M82 elastic assembler sign-extends all 96 lanes to signed12.
module m102_bit_sparse_weight_stream #(
    parameter int ROW_W = 10,
    parameter int TAG_W = 32,
    parameter int SOURCES = 16,
    parameter int OUTPUT_BLOCKS = 8,
    parameter int LANES = 96,
    parameter int WGT_W = 8,
    parameter int OUT_W = 12
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         lookup_valid,
    output logic                         lookup_ready,
    input  logic [3:0]                   lookup_source,
    input  logic [2:0]                   lookup_block,
    input  logic [1:0]                   lookup_beat,
    input  logic [TAG_W-1:0]             lookup_tag,
    input  logic [255:0]                 bank_words,
    output logic [8*ROW_W-1:0]           bank_row_addresses,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [TAG_W-1:0]             output_tag,
    output logic [3:0]                   output_width,
    output logic                         output_escape,
    output logic [LANES*OUT_W-1:0]       output_values,
    output logic                         output_accept,
    output logic                         protocol_error,
    output logic                         busy
);
    localparam int WORDS_PER_VECTOR = LANES * WGT_W / 32;
    localparam int WORDS_PER_PHASE = SOURCES * OUTPUT_BLOCKS
                                   * WORDS_PER_VECTOR;

    logic transaction_active_q, request_fault_q;
    logic [3:0] transaction_source_q;
    logic [2:0] transaction_block_q;
    logic [1:0] expected_beat_q;
    logic [TAG_W-1:0] transaction_tag_q;
    logic accepted_grace_q;
    logic [3:0] accepted_grace_source_q;
    logic [2:0] accepted_grace_block_q;
    logic [1:0] accepted_grace_beat_q;
    logic [TAG_W-1:0] accepted_grace_tag_q;

    logic [6:0] vector_index;
    logic [11:0] base_word, logical_word;
    logic [12:0] fetch_end_word;
    logic [2:0] base_bank;
    logic [ROW_W-1:0] base_row;
    logic request_identity_valid, request_in_range;
    logic request_semantically_valid, accepted_grace_match;
    logic request_violation;
    logic [255:0] logical_words;
    logic [2:0] rotated_banks [0:7];

    logic m82_beat_ready, m82_beat_accept;
    logic m82_output_valid;
    logic [TAG_W-1:0] m82_output_tag;
    logic [3:0] m82_output_width;
    logic m82_protocol_error, m82_busy;
    logic [LANES*OUT_W-1:0] m82_output_values;

`ifndef SYNTHESIS
    initial begin
        if (ROW_W != 10 || TAG_W != 32 || SOURCES != 16
                || OUTPUT_BLOCKS != 8 || LANES != 96 || WGT_W != 8
                || OUT_W != 12 || WORDS_PER_VECTOR != 24
                || WORDS_PER_PHASE != 3072)
            $fatal(1, "M102 bit-sparse frozen geometry drift");
    end
`endif

    always_comb begin : weight_bank_mapper
        vector_index = {lookup_source, lookup_block};
        base_word = {5'd0, vector_index} * 12'd24;
        logical_word = base_word + ({10'd0, lookup_beat} << 3);
        fetch_end_word = {1'b0, logical_word} + 13'd7;
        base_bank = logical_word[2:0];
        base_row = {{(ROW_W-9){1'b0}}, logical_word[11:3]};

        if (!transaction_active_q) begin
            request_identity_valid = lookup_beat == 2'd0;
        end else begin
            request_identity_valid = lookup_beat == expected_beat_q
                                  && lookup_source == transaction_source_q
                                  && lookup_block == transaction_block_q
                                  && lookup_tag == transaction_tag_q;
        end
        request_in_range = lookup_beat < 2'd3
                        && fetch_end_word < 13'd3072;
        request_semantically_valid = request_identity_valid
                                   && request_in_range;

        bank_row_addresses = '0;
        logical_words = '0;
        for (int word = 0; word < 8; word++)
            rotated_banks[word] = base_bank + word[2:0];
        if (request_semantically_valid) begin
            for (int bank = 0; bank < 8; bank++)
                bank_row_addresses[bank*ROW_W +: ROW_W] =
                    base_row
                    + {{(ROW_W-1){1'b0}}, (bank[2:0] < base_bank)};
            for (int word = 0; word < 8; word++)
                logical_words[word*32 +: 32] = bank_words[
                    {rotated_banks[word], 5'b0} +: 32];
        end
    end

    assign accepted_grace_match = accepted_grace_q
                                && lookup_source == accepted_grace_source_q
                                && lookup_block == accepted_grace_block_q
                                && lookup_beat == accepted_grace_beat_q
                                && lookup_tag == accepted_grace_tag_q;
    assign request_violation = lookup_valid && !request_semantically_valid
                             && !accepted_grace_match;
    assign lookup_ready = !protocol_error
                        && request_semantically_valid && m82_beat_ready;
    assign protocol_error = request_fault_q || m82_protocol_error
                          || request_violation;
    assign busy = transaction_active_q || m82_busy;

    // A top-level protocol fault suppresses an already buffered M82 result.
    // Only reset may recover the transaction and buffered-output state.
    assign output_valid = !protocol_error && m82_output_valid;
    assign output_tag = output_valid ? m82_output_tag : '0;
    assign output_width = output_valid ? m82_output_width : '0;
    assign output_escape = 1'b0;
    assign output_values = output_valid ? m82_output_values : '0;
    assign output_accept = output_valid && output_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            transaction_active_q <= 1'b0;
            request_fault_q <= 1'b0;
            transaction_source_q <= '0;
            transaction_block_q <= '0;
            expected_beat_q <= '0;
            transaction_tag_q <= '0;
            accepted_grace_q <= 1'b0;
            accepted_grace_source_q <= '0;
            accepted_grace_block_q <= '0;
            accepted_grace_beat_q <= '0;
            accepted_grace_tag_q <= '0;
        end else begin
            if (!lookup_valid || !accepted_grace_match)
                accepted_grace_q <= 1'b0;
            if (m82_protocol_error || request_violation)
                request_fault_q <= 1'b1;

            if (m82_beat_accept) begin
                accepted_grace_q <= 1'b1;
                accepted_grace_source_q <= lookup_source;
                accepted_grace_block_q <= lookup_block;
                accepted_grace_beat_q <= lookup_beat;
                accepted_grace_tag_q <= lookup_tag;
                if (!transaction_active_q) begin
                    transaction_active_q <= 1'b1;
                    transaction_source_q <= lookup_source;
                    transaction_block_q <= lookup_block;
                    transaction_tag_q <= lookup_tag;
                    expected_beat_q <= 2'd1;
                end else if (lookup_beat == 2'd2) begin
                    transaction_active_q <= 1'b0;
                    expected_beat_q <= '0;
                end else begin
                    expected_beat_q <= expected_beat_q + 1'b1;
                end
            end
        end
    end

    zero_bubble_elastic_pwp_stream #(
        .LANES(LANES),
        .BEAT_W(256),
        .MAX_BEATS(5),
        .OUT_W(OUT_W),
        .TAG_W(TAG_W)
    ) m82_stream (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .beat_valid(lookup_valid && !protocol_error
                    && request_semantically_valid),
        .beat_ready(m82_beat_ready),
        .beat_start(lookup_beat == 0),
        .beat_last(lookup_beat == 2),
        .beat_width(lookup_beat == 0 ? 4'd8 : 4'd0),
        .beat_tag(lookup_beat == 0 ? lookup_tag : '0),
        .beat_data(logical_words),
        .beat_accept(m82_beat_accept),
        .output_valid(m82_output_valid),
        .output_ready(output_ready && !protocol_error),
        .output_tag(m82_output_tag),
        .output_width(m82_output_width),
        .output_escape(),
        .output_values(m82_output_values),
        .output_accept(),
        .protocol_error(m82_protocol_error),
        .collecting(),
        .busy(m82_busy)
    );
endmodule

`default_nettype wire
