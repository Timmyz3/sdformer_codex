`timescale 1ns/1ps
`default_nettype none

bind gatestack_ppdi_dctf_term_fabric
    gatestack_ppdi_dctf_term_fabric_assertions #(
        .Q(Q), .GROUP_TAG_W(GROUP_TAG_W), .SEQUENCE_W(SEQUENCE_W),
        .TERM_ISSUE_SEQ_W(TERM_ISSUE_SEQ_W), .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .GATE_CODE_W(GATE_CODE_W), .LANE_ID_W(LANE_ID_W),
        .DEST_TOKEN_W(DEST_TOKEN_W)
    ) u_gatestack_ppdi_dctf_term_fabric_assertions (.*);

`default_nettype wire
