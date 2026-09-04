`timescale 1ns/1ps
`default_nettype none

bind hitflow_dptme_array hitflow_dptme_assertions #(
    .LANES(LANES),
    .SLOTS(SLOTS),
    .ACC_W(ACC_W),
    .TAG_W(TAG_W)
) u_hitflow_dptme_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .step_valid(step_valid),
    .step_ready(step_ready),
    .step_last(step_last),
    .protocol_error(protocol_error),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_events(out_events),
    .out_hidden(out_hidden),
    .out_slot_valid(out_slot_valid),
    .out_tag(out_tag)
);

`default_nettype wire
