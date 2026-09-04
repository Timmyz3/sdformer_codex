`timescale 1ns/1ps
`default_nettype none

bind hitflow_banked_accumulator hitflow_banked_accumulator_assertions #(
    .TOKENS(TOKENS), .BANKS(BANKS), .ACC_W(ACC_W), .OUT_TILE(OUT_TILE),
    .TAG_W(TAG_W), .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W)
) u_hitflow_banked_accumulator_assertions (
    .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
    .protocol_error(protocol_error), .group_start_ready(group_start_ready),
    .update_valid(update_valid), .update_ready(update_ready),
    .final_valid(final_valid), .final_ready(final_ready),
    .final_token_ids(final_token_ids), .final_tag(final_tag),
    .final_values(final_values), .group_finish_ready(group_finish_ready),
    .count_updates(count_updates), .count_writes(count_writes),
    .count_bias_commits(count_bias_commits),
    .count_bank_stall_cycles(count_bank_stall_cycles),
    .count_final_stall_cycles(count_final_stall_cycles)
);

`default_nettype wire
