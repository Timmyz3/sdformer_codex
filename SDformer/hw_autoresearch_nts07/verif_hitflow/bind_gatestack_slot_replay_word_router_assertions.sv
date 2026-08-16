`timescale 1ns/1ps
`default_nettype none

bind gatestack_slot_replay_word_router
gatestack_slot_replay_word_router_assertions #(
    .TAG_W(TAG_W), .WORD_INDEX_W(WORD_INDEX_W), .FORMAT_W(FORMAT_W)
) u_gatestack_slot_replay_word_router_assertions (.*);

`default_nettype wire
