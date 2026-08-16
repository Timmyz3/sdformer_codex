`timescale 1ns/1ps
`default_nettype none

bind hitflow_implicit_bias_finalizer_accumulator
    hitflow_implicit_bias_finalizer_assertions #(
        .BANKS(BANKS), .TOKEN_ID_W(TOKEN_ID_W), .OUT_TILE(OUT_TILE),
        .ACC_W(ACC_W), .TAG_W(TAG_W)
    ) u_ibf_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
        .final_valid(final_valid), .final_ready(final_ready),
        .final_token_ids(final_token_ids), .final_tag(final_tag),
        .final_values(final_values),
        .finalize_done_valid(finalize_done_valid),
        .finalize_done_ready(finalize_done_ready),
        .finalize_done_tag(finalize_done_tag)
    );

`default_nettype wire
