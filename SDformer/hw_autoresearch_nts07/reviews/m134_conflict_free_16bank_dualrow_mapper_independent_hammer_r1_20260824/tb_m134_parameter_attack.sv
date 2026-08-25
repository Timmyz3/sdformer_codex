`timescale 1ns/1ps
`default_nettype none

module tb_m134_parameter_attack;
`ifdef ATTACK_WORDS
    localparam int TEST_WORDS = 3679;
`else
    localparam int TEST_WORDS = 3680;
`endif
`ifdef ATTACK_BANKS
    localparam int TEST_BANKS = 8;
`else
    localparam int TEST_BANKS = 16;
`endif
`ifdef ATTACK_WORD_W
    localparam int TEST_WORD_W = 16;
`else
    localparam int TEST_WORD_W = 32;
`endif
`ifdef ATTACK_BASE_W
    localparam int TEST_BASE_W = 11;
`else
    localparam int TEST_BASE_W = 12;
`endif
`ifdef ATTACK_ROW_W
    localparam int TEST_ROW_W = 7;
`else
    localparam int TEST_ROW_W = 8;
`endif

    logic request_valid;
    logic [TEST_BASE_W-1:0] logical_base_word;
    logic [TEST_BANKS*TEST_WORD_W-1:0] bank_words;
    logic request_legal;
    logic [TEST_BANKS*TEST_ROW_W-1:0] bank_row_addresses;
    logic [TEST_BANKS*TEST_WORD_W-1:0] logical_words;
    logic [TEST_BANKS-1:0] bank_use_mask;
    logic conflict_free;

    m134_conflict_free_16bank_dualrow_mapper #(
        .WORDS(TEST_WORDS),
        .BANKS(TEST_BANKS),
        .WORD_W(TEST_WORD_W),
        .BASE_W(TEST_BASE_W),
        .ROW_W(TEST_ROW_W)
    ) dut (.*);

    initial begin
        request_valid = 1'b1;
        logical_base_word = '0;
        bank_words = '0;
`ifdef SYNTHESIS
        // SYNTHESIS removes the production-geometry initial guard. BANKS=8
        // is then accepted even though the RTL retains modulo-16 constants.
        #1ps;
`ifdef ATTACK_BANKS
        logical_base_word = TEST_BASE_W'(8);
        for (int bank = 0; bank < TEST_BANKS; bank++)
            bank_words[bank*TEST_WORD_W +: TEST_WORD_W] = bank;
        #1ps;
        if (!request_legal || !conflict_free || bank_use_mask != '1)
            $fatal(1, "parameter guard bypass setup failed");
        if (!$isunknown(logical_words[0 +: TEST_WORD_W]))
            $fatal(1, "BANKS=8 hardcoded modulo-16 read did not expose unknown");
        $display("PASS M134 synthesis-define parameter guard bypass banks=8 guard_active=false hardcoded_modulo16_unknown=true production_geometry_only=true");
        $finish;
`else
        $fatal(1, "SYNTHESIS boundary run requires ATTACK_BANKS");
`endif
`else
        // Each deviation must be rejected by the simulation-only guard before
        // any result can be cited.
        #10ps;
        $fatal(1, "geometry attack escaped production guard");
`endif
    end
endmodule

`default_nettype wire
