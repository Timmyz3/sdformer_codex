`timescale 1ns/1ps
`default_nettype none

bind gatestack_backend_done_guard
gatestack_backend_done_guard_assertions #(.TAG_W(TAG_W))
u_gatestack_backend_done_guard_assertions (.*);

`default_nettype wire
