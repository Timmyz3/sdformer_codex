`timescale 1ns/1ps
`default_nettype none
bind gatestack_replay_launch_control gatestack_replay_launch_control_assertions #(
 .TAG_W(TAG_W),.EVENT_COUNT_W(EVENT_COUNT_W),.WORD_INDEX_W(WORD_INDEX_W)
) i_gatestack_replay_launch_control_assertions(.*);
`default_nettype wire
