`timescale 1ns/1ps
`default_nettype none

// In-order rendezvous between independently generated product values and the
// exact destination bitmap for the same term issue sequence.
module gatestack_product_bitmap_join #(
    parameter int TOKENS          = 162,
    parameter int OUT_TILE        = 8,
    parameter int PRODUCT_W       = 17,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int TAG_W           = 32,
    parameter int COUNTER_W       = 32
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              product_valid,
    output logic                              product_ready,
    input  logic [TAG_W-1:0]                  product_tag,
    input  logic [INPUT_CH_W-1:0]             product_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]          product_output_tile,
    input  logic [ISSUE_SEQ_W-1:0]            product_issue_seq,
    input  logic [(OUT_TILE*PRODUCT_W)-1:0]   product_values,

    input  logic                              bitmap_valid,
    output logic                              bitmap_ready,
    input  logic [TAG_W-1:0]                  bitmap_tag,
    input  logic [ISSUE_SEQ_W-1:0]            bitmap_issue_seq,
    input  logic [TOKENS-1:0]                 bitmap_destinations,

    output logic                              joined_valid,
    input  logic                              joined_ready,
    output logic [TAG_W-1:0]                  joined_tag,
    output logic [INPUT_CH_W-1:0]             joined_input_channel,
    output logic [OUTPUT_TILE_W-1:0]          joined_output_tile,
    output logic [ISSUE_SEQ_W-1:0]            joined_issue_seq,
    output logic [TOKENS-1:0]                 joined_destinations,
    output logic [(OUT_TILE*PRODUCT_W)-1:0]   joined_values,

    output logic                              protocol_error,
    output logic [COUNTER_W-1:0]              count_joined_terms,
    output logic [COUNTER_W-1:0]              count_product_wait_cycles,
    output logic [COUNTER_W-1:0]              count_bitmap_wait_cycles,
    output logic [COUNTER_W-1:0]              count_output_stall_cycles
);

    logic product_buffer_valid_q;
    logic [TAG_W-1:0] product_tag_q;
    logic [INPUT_CH_W-1:0] product_input_channel_q;
    logic [OUTPUT_TILE_W-1:0] product_output_tile_q;
    logic [ISSUE_SEQ_W-1:0] product_issue_seq_q;
    logic [(OUT_TILE*PRODUCT_W)-1:0] product_values_q;
    logic bitmap_buffer_valid_q;
    logic [TAG_W-1:0] bitmap_tag_q;
    logic [ISSUE_SEQ_W-1:0] bitmap_issue_seq_q;
    logic [TOKENS-1:0] bitmap_destinations_q;
    logic metadata_matches;
    logic product_fire;
    logic bitmap_fire;
    logic joined_fire;
    logic mismatch_drop;

    assign product_ready = !product_buffer_valid_q;
    assign bitmap_ready = !bitmap_buffer_valid_q;
    assign product_fire = product_valid && product_ready;
    assign bitmap_fire = bitmap_valid && bitmap_ready;
    assign metadata_matches = product_tag_q == bitmap_tag_q &&
                              product_issue_seq_q == bitmap_issue_seq_q;
    assign joined_valid = product_buffer_valid_q && bitmap_buffer_valid_q &&
                          metadata_matches;
    assign joined_tag = product_tag_q;
    assign joined_input_channel = product_input_channel_q;
    assign joined_output_tile = product_output_tile_q;
    assign joined_issue_seq = product_issue_seq_q;
    assign joined_destinations = bitmap_destinations_q;
    assign joined_values = product_values_q;
    assign joined_fire = joined_valid && joined_ready;
    assign mismatch_drop = product_buffer_valid_q && bitmap_buffer_valid_q &&
                           !metadata_matches;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            product_buffer_valid_q <= 1'b0;
            product_tag_q <= '0;
            product_input_channel_q <= '0;
            product_output_tile_q <= '0;
            product_issue_seq_q <= '0;
            product_values_q <= '0;
            bitmap_buffer_valid_q <= 1'b0;
            bitmap_tag_q <= '0;
            bitmap_issue_seq_q <= '0;
            bitmap_destinations_q <= '0;
            protocol_error <= 1'b0;
            count_joined_terms <= '0;
            count_product_wait_cycles <= '0;
            count_bitmap_wait_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (product_fire) begin
                product_buffer_valid_q <= 1'b1;
                product_tag_q <= product_tag;
                product_input_channel_q <= product_input_channel;
                product_output_tile_q <= product_output_tile;
                product_issue_seq_q <= product_issue_seq;
                product_values_q <= product_values;
            end
            if (bitmap_fire) begin
                bitmap_buffer_valid_q <= 1'b1;
                bitmap_tag_q <= bitmap_tag;
                bitmap_issue_seq_q <= bitmap_issue_seq;
                bitmap_destinations_q <= bitmap_destinations;
            end
            if (joined_fire) begin
                product_buffer_valid_q <= 1'b0;
                bitmap_buffer_valid_q <= 1'b0;
                count_joined_terms <= count_joined_terms + 1'b1;
            end else if (mismatch_drop) begin
                product_buffer_valid_q <= 1'b0;
                bitmap_buffer_valid_q <= 1'b0;
                protocol_error <= 1'b1;
            end
            if (product_buffer_valid_q && !bitmap_buffer_valid_q) begin
                count_bitmap_wait_cycles <= count_bitmap_wait_cycles + 1'b1;
            end
            if (bitmap_buffer_valid_q && !product_buffer_valid_q) begin
                count_product_wait_cycles <= count_product_wait_cycles + 1'b1;
            end
            if (joined_valid && !joined_ready) begin
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
