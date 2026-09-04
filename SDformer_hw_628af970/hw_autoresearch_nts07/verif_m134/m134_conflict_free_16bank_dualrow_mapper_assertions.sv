`timescale 1ns/1ps
`default_nettype none

module m134_conflict_free_16bank_dualrow_mapper_assertions #(
    parameter int BANKS = 16,
    parameter int ROW_W = 8,
    parameter int BASE_W = 12
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         request_valid,
    input logic [BASE_W-1:0]            logical_base_word,
    input logic                         request_legal,
    input logic [BANKS*ROW_W-1:0]       bank_row_addresses,
    input logic [BANKS*32-1:0]          logical_words,
    input logic [BANKS-1:0]             bank_use_mask,
    input logic                         conflict_free
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_legal_definition:
        assert property (request_legal
                         == (request_valid
                             && ({1'b0, logical_base_word} + 13'd15
                                 < 13'd3680)));
    ap_legal_uses_every_bank_once:
        assert property (request_legal
                         |-> bank_use_mask == 16'hffff && conflict_free);
    ap_illegal_quiet:
        assert property (!request_legal
                         |-> bank_row_addresses == 0
                             && logical_words == 0
                             && bank_use_mask == 0
                             && !conflict_free);

    for (genvar bank = 0; bank < BANKS; bank++) begin : bank_address_checks
        ap_exact_bank_row:
            assert property (request_legal
                             |-> bank_row_addresses[bank*ROW_W +: ROW_W]
                                 == logical_base_word[11:4]
                                    + (bank < logical_base_word[3:0]));
    end

    for (genvar offset = 0; offset < BANKS; offset++) begin : offset_covers
        cp_every_base_bank:
            cover property (request_legal
                            && logical_base_word[3:0] == offset);
    end
    cp_crosses_physical_row:
        cover property (request_legal && logical_base_word[3:0] != 0);
    cp_last_legal_window:
        cover property (request_legal && logical_base_word == 12'd3664);
    cp_first_illegal_window:
        cover property (request_valid && !request_legal
                        && logical_base_word == 12'd3665);
endmodule

`default_nettype wire
