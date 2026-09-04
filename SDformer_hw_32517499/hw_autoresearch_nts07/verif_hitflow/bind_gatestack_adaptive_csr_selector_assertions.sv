`timescale 1ns/1ps
`default_nettype none

bind gatestack_adaptive_csr_replay_decoder
    gatestack_adaptive_csr_selector_assertions #(
        .WORD_INDEX_W(WORD_INDEX_W)
    ) u_adaptive_csr_selector_assertions (
        .word_magic(word_data[15:0]),
        .*
    );

`default_nettype wire
