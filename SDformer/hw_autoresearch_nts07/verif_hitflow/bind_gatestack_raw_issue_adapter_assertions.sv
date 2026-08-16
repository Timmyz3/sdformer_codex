`timescale 1ns/1ps
`default_nettype none

bind gatestack_raw_issue_adapter
    gatestack_raw_issue_adapter_assertions #(
        .EVENT_WAYS(EVENT_WAYS),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) i_gatestack_raw_issue_adapter_assertions (.*);

`default_nettype wire
