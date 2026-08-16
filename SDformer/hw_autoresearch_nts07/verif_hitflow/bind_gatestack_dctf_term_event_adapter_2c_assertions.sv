`timescale 1ns/1ps
`default_nettype none

bind gatestack_dctf_term_event_adapter_2c
    gatestack_dctf_term_event_adapter_2c_assertions #(
        .TAG_W(TAG_W), .GATE_CODE_W(GATE_CODE_W),
        .LANE_ID_W(LANE_ID_W), .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .TOKEN_ID_W(TOKEN_ID_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W)
    ) i_gatestack_dctf_term_event_adapter_2c_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
        .clear_error(clear_error), .term_valid(term_valid),
        .term_ready(term_ready), .event_valid(event_valid),
        .event_ready(event_ready), .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready), .cmd_group_tag(cmd_group_tag),
        .cmd_sequence(cmd_sequence), .cmd_gate_code(cmd_gate_code),
        .cmd_lane_id(cmd_lane_id),
        .cmd_destination_token(cmd_destination_token),
        .cmd_term_issue_seq(cmd_term_issue_seq),
        .cmd_term_first(cmd_term_first), .cmd_term_last(cmd_term_last),
        .cmd_head_last(cmd_head_last),
        .cmd_input_channel_base(cmd_input_channel_base),
        .cmd_logical_supertile(cmd_logical_supertile), .idle(idle),
        .protocol_error(protocol_error), .context_valid_q(context_valid_q),
        .context_complete_q(context_complete_q),
        .fill_active_q(fill_active_q), .fill_drop_q(fill_drop_q),
        .fill_context_q(fill_context_q), .head_context_q(head_context_q),
        .tail_context_q(tail_context_q)
    );

`default_nettype wire
