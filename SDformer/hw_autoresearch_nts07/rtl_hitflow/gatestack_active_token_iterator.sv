`timescale 1ns/1ps
`default_nettype none

// Segmented wrapper around OBI for a 162-bit active-token mask.
module gatestack_active_token_iterator #(
    parameter int TOKENS          = 162,
    parameter int SEGMENT_TOKENS  = 18,
    parameter int SEGMENTS        = (TOKENS + SEGMENT_TOKENS - 1) / SEGMENT_TOKENS,
    parameter int TAG_W           = 32,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int SEGMENT_ID_W    = (SEGMENTS <= 1) ? 1 : $clog2(SEGMENTS),
    parameter int OFFSET_ID_W     = (SEGMENT_TOKENS <= 1) ? 1 : $clog2(SEGMENT_TOKENS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [TAG_W-1:0]             load_tag,
    input  logic [TOKENS-1:0]            load_active_token_mask,
    output logic                         token_valid,
    input  logic                         token_ready,
    output logic [TAG_W-1:0]             token_tag,
    output logic [TOKEN_ID_W-1:0]        token_id,
    output logic                         token_last,
    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_W-1:0]             done_tag,
    output logic [COUNTER_W-1:0]         count_loads,
    output logic [COUNTER_W-1:0]         count_tokens,
    output logic [COUNTER_W-1:0]         count_stall_cycles
);

    localparam int PADDED_TOKENS = SEGMENTS * SEGMENT_TOKENS;

    logic [PADDED_TOKENS-1:0] padded_mask;
    logic [SEGMENT_ID_W-1:0] obi_segment;
    logic [OFFSET_ID_W-1:0] obi_offset;

    always_comb begin
        padded_mask = '0;
        padded_mask[TOKENS-1:0] = load_active_token_mask;
    end

    assign token_id = TOKEN_ID_W'(
        (32'(obi_segment) * SEGMENT_TOKENS) + 32'(obi_offset)
    );

    gatestack_obi_iterator #(
        .SLOTS(SEGMENTS),
        .LANES(SEGMENT_TOKENS),
        .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W),
        .SLOT_ID_W(SEGMENT_ID_W),
        .LANE_ID_W(OFFSET_ID_W)
    ) u_token_obi (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .load_valid(load_valid),
        .load_ready(load_ready),
        .load_tag(load_tag),
        .load_occupied_mask(padded_mask),
        .entry_valid(token_valid),
        .entry_ready(token_ready),
        .entry_tag(token_tag),
        .entry_slot_id(obi_segment),
        .entry_lane_id(obi_offset),
        .entry_last(token_last),
        .done_valid(done_valid),
        .done_ready(done_ready),
        .done_tag(done_tag),
        .count_loads(count_loads),
        .count_entries(count_tokens),
        .count_entry_stall_cycles(count_stall_cycles)
    );

endmodule

`default_nettype wire
