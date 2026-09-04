`timescale 1ns/1ps
`default_nettype none

bind gatestack_replay_atomic_commit
gatestack_replay_atomic_commit_assertions #(
    .FORMAT_W(FORMAT_W), .ROUTE_W(ROUTE_W),
    .WORD_INDEX_W(WORD_INDEX_W)
)
u_gatestack_replay_atomic_commit_assertions (.*);

`default_nettype wire
