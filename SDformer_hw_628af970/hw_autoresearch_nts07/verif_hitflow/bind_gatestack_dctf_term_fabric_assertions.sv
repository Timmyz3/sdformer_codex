`timescale 1ns/1ps
`default_nettype none

bind gatestack_dctf_term_fabric
    gatestack_dctf_term_fabric_assertions #(
        .Q(Q),
        .GROUP_TAG_W(GROUP_TAG_W),
        .SEQUENCE_W(SEQUENCE_W),
        .TERM_ISSUE_SEQ_W(TERM_ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .GATE_CODE_W(GATE_CODE_W),
        .LANE_ID_W(LANE_ID_W),
        .DEST_TOKEN_W(DEST_TOKEN_W),
        .COUNTER_W(COUNTER_W)
    ) i_gatestack_dctf_term_fabric_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(flush),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_sequence(cmd_sequence),
        .cmd_term_last(cmd_term_last),
        .cmd_head_last(cmd_head_last),
        .bank_valid(bank_valid),
        .bank_ready(bank_ready),
        .bank_group_tags(bank_group_tags),
        .bank_sequences(bank_sequences),
        .bank_term_issue_seqs(bank_term_issue_seqs),
        .bank_term_first(bank_term_first),
        .bank_term_last(bank_term_last),
        .bank_head_last(bank_head_last),
        .bank_input_channels(bank_input_channels),
        .bank_logical_supertiles(bank_logical_supertiles),
        .bank_gate_codes(bank_gate_codes),
        .bank_lane_ids(bank_lane_ids),
        .bank_destination_tokens(bank_destination_tokens),
        .retire_valid(retire_valid),
        .retire_sequence(retire_sequence),
        .occupancy(occupancy),
        .count_accepted(count_accepted),
        .count_bank_consumed(count_bank_consumed),
        .count_retired(count_retired)
    );

`default_nettype wire
