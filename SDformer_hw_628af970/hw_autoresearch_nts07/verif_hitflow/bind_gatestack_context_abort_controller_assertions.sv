`timescale 1ns/1ps
`default_nettype none

bind gatestack_context_abort_controller
gatestack_context_abort_controller_assertions #(.TAG_W(TAG_W))
u_gatestack_context_abort_controller_assertions (.*);

`default_nettype wire
