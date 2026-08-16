`timescale 1ns/1ps
`default_nettype none
bind gatestack_replay_lifecycle_manager
 gatestack_replay_lifecycle_manager_assertions #(.TAG_W(TAG_W),
 .CONTEXT_ID_W(CONTEXT_ID_W),.HEAD_ID_W(HEAD_ID_W))
 i_gatestack_replay_lifecycle_manager_assertions(.*);
`default_nettype wire
