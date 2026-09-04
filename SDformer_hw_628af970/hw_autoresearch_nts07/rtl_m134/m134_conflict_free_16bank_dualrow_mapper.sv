`timescale 1ns/1ps
`default_nettype none

// Conflict-free 16-bank adapter for one 512-bit PWP service beat.
//
// Logical 32-bit word w is stored in bank w[3:0], row w[11:4].  Therefore
// sixteen consecutive words use every bank exactly once, even when the
// request crosses a physical row boundary.  Compared with the eight-bank
// 256-bit organization, this is the executable mapping needed to supply two
// logical 256-bit rows without a second read port per bank.
//
// This module is a combinational bank-data port cut.  The sixteen SRAM macros,
// their read latency, wiring, clocking, area and energy are outside this RTL.
module m134_conflict_free_16bank_dualrow_mapper #(
    parameter int WORDS = 3680,
    parameter int BANKS = 16,
    parameter int WORD_W = 32,
    parameter int BASE_W = 12,
    parameter int ROW_W = 8
) (
    input  logic                         request_valid,
    input  logic [BASE_W-1:0]            logical_base_word,
    input  logic [BANKS*WORD_W-1:0]      bank_words,

    output logic                         request_legal,
    output logic [BANKS*ROW_W-1:0]       bank_row_addresses,
    output logic [BANKS*WORD_W-1:0]      logical_words,
    output logic [BANKS-1:0]             bank_use_mask,
    output logic                         conflict_free
);
    logic [BASE_W:0] request_end_word;
    logic [3:0] base_bank;
    logic [ROW_W-1:0] base_row;

`ifndef SYNTHESIS
    initial begin
        if (WORDS != 3680 || BANKS != 16 || WORD_W != 32
                || BASE_W != 12 || ROW_W != 8)
            $fatal(1, "M134 production geometry drift");
    end
`endif

    always_comb begin : map_sixteen_consecutive_words
        request_end_word = {1'b0, logical_base_word} + 13'd15;
        request_legal = request_valid && request_end_word < WORDS;
        base_bank = logical_base_word[3:0];
        base_row = logical_base_word[11:4];

        bank_row_addresses = '0;
        logical_words = '0;
        bank_use_mask = '0;
        conflict_free = 1'b0;
        if (request_legal) begin
            bank_use_mask = 16'hffff;
            conflict_free = 1'b1;
            for (int bank = 0; bank < BANKS; bank++) begin
                bank_row_addresses[bank*ROW_W +: ROW_W] =
                    base_row + (bank < base_bank);
            end
            for (int word = 0; word < BANKS; word++) begin
                logical_words[word*WORD_W +: WORD_W] = bank_words[
                    (((base_bank + word) & 4'hf)*WORD_W) +: WORD_W];
            end
        end
    end
endmodule

`default_nettype wire
