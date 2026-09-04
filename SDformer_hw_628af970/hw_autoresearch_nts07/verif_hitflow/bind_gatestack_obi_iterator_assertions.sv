`timescale 1ns/1ps
`default_nettype none

bind gatestack_obi_iterator gatestack_obi_iterator_assertions #(
    .SLOTS(SLOTS),
    .LANES(LANES),
    .TAG_W(TAG_W),
    .SLOT_ID_W(SLOT_ID_W),
    .LANE_ID_W(LANE_ID_W)
) u_gatestack_obi_iterator_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .load_valid(load_valid),
    .load_ready(load_ready),
    .entry_valid(entry_valid),
    .entry_ready(entry_ready),
    .entry_tag(entry_tag),
    .entry_slot_id(entry_slot_id),
    .entry_lane_id(entry_lane_id),
    .entry_last(entry_last),
    .done_valid(done_valid),
    .done_ready(done_ready),
    .done_tag(done_tag)
);

`default_nettype wire
