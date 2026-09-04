`timescale 1ns/1ps
`default_nettype none

bind gatestack_dctf_term_event_adapter
    gatestack_dctf_term_event_adapter_assertions #(
        .TAG_W(TAG_W),
        .GATE_CODE_W(GATE_CODE_W),
        .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W)
    ) i_gatestack_dctf_term_event_adapter_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(flush),
        .clear_error(clear_error),
        .term_ready(term_ready),
        .event_ready(event_ready),
        .event_fire(event_fire),
        .event_contract_ok(event_contract_ok),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_group_tag(cmd_group_tag),
        .cmd_sequence(cmd_sequence),
        .cmd_gate_code(cmd_gate_code),
        .cmd_lane_id(cmd_lane_id),
        .cmd_destination_token(cmd_destination_token),
        .cmd_term_issue_seq(cmd_term_issue_seq),
        .cmd_term_first(cmd_term_first),
        .cmd_term_last(cmd_term_last),
        .cmd_head_last(cmd_head_last),
        .protocol_error(protocol_error)
    );

`default_nettype wire
