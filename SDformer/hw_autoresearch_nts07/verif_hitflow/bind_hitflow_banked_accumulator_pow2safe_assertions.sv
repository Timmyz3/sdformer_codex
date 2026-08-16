`timescale 1ns/1ps
`default_nettype none

// Bind GPT assertion module onto pow2safe accumulator (new file).
bind hitflow_banked_accumulator_pow2safe hitflow_banked_accumulator_assertions #(
    .TOKENS(TOKENS), .BANKS(BANKS), .ACC_W(ACC_W), .OUT_TILE(OUT_TILE),
    .TAG_W(TAG_W), .TOKEN_ID_W(TOKEN_ID_W)
) u_hitflow_banked_accumulator_pow2safe_assertions (
    .clk_core(clk_core), .rst_core(rst_core), .protocol_error(protocol_error),
    .update_valid(update_valid), .update_ready(update_ready),
    .final_valid(final_valid), .final_ready(final_ready),
    .final_token_ids(final_token_ids), .final_tag(final_tag),
    .final_values(final_values), .group_finish_ready(group_finish_ready)
);

`default_nettype wire
