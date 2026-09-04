`timescale 1ns/1ps
`default_nettype none

// Test-only compatibility shell: reuse the independently exercised M201
// scoreboard and SVA without macro-renaming either design module.
module m201_fc2_raw4_to_descriptor4_stable_compactor #(
    parameter int TAG_BITS = 24,
    parameter int QUEUE_DEPTH = 8
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, output logic header_ready,
    input logic [TAG_BITS-1:0] header_token_tag,
    input logic [5:0] header_raw_beat_count,
    input logic [3:0] header_window_depth,
    output logic header_accept,
    input logic raw_valid, output logic raw_ready,
    input logic [3:0] raw_lane_valid,
    input logic [4:0] raw_beat_index [0:3],
    input logic [95:0] raw_bitmap [0:3],
    input logic raw_last, output logic raw_accept,
    output logic descriptor_valid, input logic descriptor_ready,
    output logic [2:0] descriptor_count,
    output logic [TAG_BITS-1:0] descriptor_token_tag,
    output logic [4:0] descriptor_beat_index [0:3],
    output logic [95:0] descriptor_bitmap [0:3],
    output logic [3:0] descriptor_window_last,
    output logic descriptor_accept,
    output logic token_done_valid, input logic token_done_ready,
    output logic [TAG_BITS-1:0] token_done_tag,
    output logic [5:0] token_done_descriptor_count,
    output logic token_done_accept,
    output logic protocol_error, output logic busy
);
    m202_fc2_raw4_to_descriptor4_fresh_bypass_compactor #(
        .TAG_BITS(TAG_BITS), .QUEUE_DEPTH(QUEUE_DEPTH)
    ) impl (.*);
endmodule

`default_nettype wire
