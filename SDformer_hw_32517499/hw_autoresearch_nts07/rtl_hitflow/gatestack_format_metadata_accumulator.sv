`timescale 1ns/1ps
`default_nettype none

// Accumulates the exact per-head metadata needed by the typed format policy.
// A term is one unique {gate,lane} pair and must have at least one destination.
module gatestack_format_metadata_accumulator #(
    parameter int TAG_W             = 32,
    parameter int CLASS_COUNT_W     = 4,
    parameter int TERM_COUNT_W      = 8,
    parameter int EVENT_COUNT_W     = 13,
    parameter int DEST_COUNT_W      = 8,
    parameter int DEST_BYTES_W      = 13,
    parameter int BITMAP_THRESHOLD  = 21,
    parameter int COUNTER_W         = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         head_start_valid,
    output logic                         head_start_ready,
    input  logic [TAG_W-1:0]             head_start_tag,
    input  logic [CLASS_COUNT_W-1:0]     head_start_active_classes,

    input  logic                         term_valid,
    output logic                         term_ready,
    input  logic [DEST_COUNT_W-1:0]      term_destination_count,

    input  logic                         head_end_valid,
    output logic                         head_end_ready,
    input  logic                         head_end_builder_error,

    output logic                         metadata_valid,
    input  logic                         metadata_ready,
    output logic [TAG_W-1:0]             metadata_tag,
    output logic [CLASS_COUNT_W-1:0]     metadata_active_classes,
    output logic [TERM_COUNT_W-1:0]      metadata_term_count,
    output logic [EVENT_COUNT_W-1:0]     metadata_event_count,
    output logic [TERM_COUNT_W-1:0]      metadata_bitmap_term_count,
    output logic [DEST_BYTES_W-1:0]      metadata_fadc_destination_bytes,
    output logic                         metadata_overflow,

    output logic [COUNTER_W-1:0]         count_heads,
    output logic [COUNTER_W-1:0]         count_terms,
    output logic [COUNTER_W-1:0]         count_invalid_terms,
    output logic [COUNTER_W-1:0]         count_metadata_overflows
);

    localparam int TERM_EXT_W = TERM_COUNT_W + 1;
    localparam int EVENT_EXT_W = EVENT_COUNT_W + 1;
    localparam int DEST_EXT_W = DEST_BYTES_W + 1;

    logic active_q;
    logic metadata_valid_q;
    logic [TAG_W-1:0] tag_q;
    logic [CLASS_COUNT_W-1:0] active_classes_q;
    logic [TERM_COUNT_W-1:0] term_count_q;
    logic [EVENT_COUNT_W-1:0] event_count_q;
    logic [TERM_COUNT_W-1:0] bitmap_term_count_q;
    logic [DEST_BYTES_W-1:0] fadc_destination_bytes_q;
    logic overflow_q;

    logic head_start_fire;
    logic term_fire;
    logic head_end_fire;
    logic metadata_fire;
    logic [TERM_EXT_W-1:0] next_term_count_ext;
    logic [EVENT_EXT_W-1:0] next_event_count_ext;
    logic [DEST_EXT_W-1:0] next_destination_bytes_ext;
    logic [DEST_COUNT_W-1:0] encoded_destination_bytes;
    logic term_is_bitmap;
    logic term_is_invalid;

    assign head_start_ready = !active_q && !metadata_valid_q;
    assign term_ready = active_q && !metadata_valid_q && !head_end_valid;
    assign head_end_ready = active_q && !metadata_valid_q && !term_valid;

    assign head_start_fire = head_start_valid && head_start_ready;
    assign term_fire = term_valid && term_ready;
    assign head_end_fire = head_end_valid && head_end_ready;
    assign metadata_fire = metadata_valid_q && metadata_ready;

    assign metadata_valid = metadata_valid_q;
    assign metadata_tag = tag_q;
    assign metadata_active_classes = active_classes_q;
    assign metadata_term_count = term_count_q;
    assign metadata_event_count = event_count_q;
    assign metadata_bitmap_term_count = bitmap_term_count_q;
    assign metadata_fadc_destination_bytes = fadc_destination_bytes_q;
    assign metadata_overflow = overflow_q;

    assign term_is_invalid = term_destination_count == '0;
    assign term_is_bitmap = 32'(term_destination_count) > BITMAP_THRESHOLD;
    assign encoded_destination_bytes = term_is_bitmap ?
        DEST_COUNT_W'(BITMAP_THRESHOLD) : term_destination_count;
    assign next_term_count_ext =
        TERM_EXT_W'(term_count_q) + TERM_EXT_W'(1);
    assign next_event_count_ext =
        EVENT_EXT_W'(event_count_q) + EVENT_EXT_W'(term_destination_count);
    assign next_destination_bytes_ext =
        DEST_EXT_W'(fadc_destination_bytes_q) +
        DEST_EXT_W'(encoded_destination_bytes);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            metadata_valid_q <= 1'b0;
            tag_q <= '0;
            active_classes_q <= '0;
            term_count_q <= '0;
            event_count_q <= '0;
            bitmap_term_count_q <= '0;
            fadc_destination_bytes_q <= '0;
            overflow_q <= 1'b0;
            count_heads <= '0;
            count_terms <= '0;
            count_invalid_terms <= '0;
            count_metadata_overflows <= '0;
        end else begin
            if (metadata_fire) begin
                metadata_valid_q <= 1'b0;
            end

            if (head_start_fire) begin
                active_q <= 1'b1;
                tag_q <= head_start_tag;
                active_classes_q <= head_start_active_classes;
                term_count_q <= '0;
                event_count_q <= '0;
                bitmap_term_count_q <= '0;
                fadc_destination_bytes_q <= '0;
                overflow_q <= 1'b0;
                count_heads <= count_heads + 1'b1;
            end else if (term_fire) begin
                count_terms <= count_terms + 1'b1;
                if (term_is_invalid) begin
                    overflow_q <= 1'b1;
                    count_invalid_terms <= count_invalid_terms + 1'b1;
                end else begin
                    if (next_term_count_ext[TERM_COUNT_W]) begin
                        term_count_q <= '1;
                        overflow_q <= 1'b1;
                    end else begin
                        term_count_q <= next_term_count_ext[TERM_COUNT_W-1:0];
                    end

                    if (next_event_count_ext[EVENT_COUNT_W]) begin
                        event_count_q <= '1;
                        overflow_q <= 1'b1;
                    end else begin
                        event_count_q <=
                            next_event_count_ext[EVENT_COUNT_W-1:0];
                    end

                    if (next_destination_bytes_ext[DEST_BYTES_W]) begin
                        fadc_destination_bytes_q <= '1;
                        overflow_q <= 1'b1;
                    end else begin
                        fadc_destination_bytes_q <=
                            next_destination_bytes_ext[DEST_BYTES_W-1:0];
                    end

                    if (term_is_bitmap) begin
                        if (&bitmap_term_count_q) begin
                            overflow_q <= 1'b1;
                        end else begin
                            bitmap_term_count_q <=
                                bitmap_term_count_q + 1'b1;
                        end
                    end
                end
            end else if (head_end_fire) begin
                active_q <= 1'b0;
                metadata_valid_q <= 1'b1;
                if (head_end_builder_error) begin
                    overflow_q <= 1'b1;
                end
                if (overflow_q || head_end_builder_error) begin
                    count_metadata_overflows <=
                        count_metadata_overflows + 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
