`timescale 1ns/1ps
`default_nettype none

// Lossless two-consumer fork for term metadata. A term is retired only after
// both the product path and bitmap path have independently accepted it.
module gatestack_term_fork #(
    parameter int TAG_W          = 32,
    parameter int LANE_ID_W      = 5,
    parameter int INPUT_CH_W     = 10,
    parameter int OUTPUT_TILE_W  = 8,
    parameter int ISSUE_SEQ_W    = 13,
    parameter int COUNTER_W      = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         term_valid,
    output logic                         term_ready,
    input  logic [TAG_W-1:0]             term_tag,
    input  logic [8:0]                   term_gate_code,
    input  logic [LANE_ID_W-1:0]         term_lane_id,
    input  logic [INPUT_CH_W-1:0]        term_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]     term_output_tile,
    input  logic [7:0]                   term_destination_count,
    input  logic [ISSUE_SEQ_W-1:0]       term_issue_seq,
    input  logic                         term_head_last,

    output logic                         product_term_valid,
    input  logic                         product_term_ready,
    output logic [TAG_W-1:0]             product_term_tag,
    output logic [8:0]                   product_term_gate_code,
    output logic [INPUT_CH_W-1:0]        product_term_input_channel,
    output logic [OUTPUT_TILE_W-1:0]     product_term_output_tile,
    output logic [ISSUE_SEQ_W-1:0]       product_term_issue_seq,

    output logic                         bitmap_term_valid,
    input  logic                         bitmap_term_ready,
    output logic [TAG_W-1:0]             bitmap_term_tag,
    output logic [8:0]                   bitmap_term_gate_code,
    output logic [LANE_ID_W-1:0]         bitmap_term_lane_id,
    output logic [7:0]                   bitmap_term_destination_count,
    output logic [ISSUE_SEQ_W-1:0]       bitmap_term_issue_seq,
    output logic                         bitmap_term_head_last,

    output logic [COUNTER_W-1:0]         count_terms,
    output logic [COUNTER_W-1:0]         count_product_wait_cycles,
    output logic [COUNTER_W-1:0]         count_bitmap_wait_cycles
);
    logic active_q;
    logic product_pending_q;
    logic bitmap_pending_q;
    logic [TAG_W-1:0] tag_q;
    logic [8:0] gate_q;
    logic [LANE_ID_W-1:0] lane_q;
    logic [INPUT_CH_W-1:0] input_channel_q;
    logic [OUTPUT_TILE_W-1:0] output_tile_q;
    logic [7:0] destination_count_q;
    logic [ISSUE_SEQ_W-1:0] issue_seq_q;
    logic head_last_q;
    logic term_fire;
    logic product_fire;
    logic bitmap_fire;

    assign term_ready = !active_q;
    assign term_fire = term_valid && term_ready;
    assign product_term_valid = active_q && product_pending_q;
    assign bitmap_term_valid = active_q && bitmap_pending_q;
    assign product_fire = product_term_valid && product_term_ready;
    assign bitmap_fire = bitmap_term_valid && bitmap_term_ready;

    assign product_term_tag = tag_q;
    assign product_term_gate_code = gate_q;
    assign product_term_input_channel = input_channel_q;
    assign product_term_output_tile = output_tile_q;
    assign product_term_issue_seq = issue_seq_q;
    assign bitmap_term_tag = tag_q;
    assign bitmap_term_gate_code = gate_q;
    assign bitmap_term_lane_id = lane_q;
    assign bitmap_term_destination_count = destination_count_q;
    assign bitmap_term_issue_seq = issue_seq_q;
    assign bitmap_term_head_last = head_last_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            product_pending_q <= 1'b0;
            bitmap_pending_q <= 1'b0;
            tag_q <= '0;
            gate_q <= '0;
            lane_q <= '0;
            input_channel_q <= '0;
            output_tile_q <= '0;
            destination_count_q <= '0;
            issue_seq_q <= '0;
            head_last_q <= 1'b0;
            count_terms <= '0;
            count_product_wait_cycles <= '0;
            count_bitmap_wait_cycles <= '0;
        end else begin
            if (term_fire) begin
                active_q <= 1'b1;
                product_pending_q <= 1'b1;
                bitmap_pending_q <= 1'b1;
                tag_q <= term_tag;
                gate_q <= term_gate_code;
                lane_q <= term_lane_id;
                input_channel_q <= term_input_channel;
                output_tile_q <= term_output_tile;
                destination_count_q <= term_destination_count;
                issue_seq_q <= term_issue_seq;
                head_last_q <= term_head_last;
                count_terms <= count_terms + 1'b1;
            end
            if (product_fire) begin
                product_pending_q <= 1'b0;
            end
            if (bitmap_fire) begin
                bitmap_pending_q <= 1'b0;
            end
            if (active_q &&
                (!product_pending_q || product_fire) &&
                (!bitmap_pending_q || bitmap_fire)) begin
                active_q <= 1'b0;
            end
            if (product_term_valid && !product_term_ready) begin
                count_product_wait_cycles <= count_product_wait_cycles + 1'b1;
            end
            if (bitmap_term_valid && !bitmap_term_ready) begin
                count_bitmap_wait_cycles <= count_bitmap_wait_cycles + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
