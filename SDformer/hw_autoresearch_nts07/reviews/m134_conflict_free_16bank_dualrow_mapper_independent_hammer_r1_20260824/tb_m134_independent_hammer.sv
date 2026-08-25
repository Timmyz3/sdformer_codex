`timescale 1ns/1ps
`default_nettype none

module tb_m134_independent_hammer;
    localparam int WORDS = 3680;
    localparam int BANKS = 16;
    localparam int WORD_W = 32;
    localparam int ROW_W = 8;

    logic clk_core;
    logic rst_core;
    logic request_valid;
    logic [11:0] logical_base_word;
    logic [BANKS*WORD_W-1:0] bank_words;
    logic request_legal;
    logic [BANKS*ROW_W-1:0] bank_row_addresses;
    logic [BANKS*WORD_W-1:0] logical_words;
    logic [BANKS-1:0] bank_use_mask;
    logic conflict_free;

    int unsigned legal_windows;
    int unsigned illegal_windows;
    int unsigned logical_word_checks;
    int unsigned physical_address_checks;
    int unsigned one_read_per_bank_checks;
    int unsigned row_crossing_windows;
    int unsigned crossed_bank_address_checks;
    int unsigned valid_low_payload_checks;
    int unsigned stale_or_skewed_data_undetected;
    int unsigned x_base_not_fail_closed;
    int unsigned base_offset_hits [0:BANKS-1];

    m134_conflict_free_16bank_dualrow_mapper dut (.*);
    m134_conflict_free_16bank_dualrow_mapper_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [31:0] physical_payload(
        input int logical_word,
        input int salt
    );
        logic [31:0] mixed;
        begin
            mixed = logical_word * 32'h45d9f3b;
            physical_payload = 32'h9e3779b9 ^ mixed
                             ^ (logical_word << 17) ^ salt;
        end
    endfunction

    task automatic check_legal(input int base_word);
        int expected_row;
        int expected_bank;
        int expected_physical_word;
        int incremented_bank_count;
        logic [15:0] offset_bank_seen;
        logic [127:0] address_snapshot;
        begin
            @(negedge clk_core);
            request_valid = 1'b1;
            logical_base_word = base_word[11:0];
            bank_words = {16{32'hdeadbeef}};
            #1ps;
            if (!request_legal || !conflict_free
                    || bank_use_mask !== 16'hffff)
                $fatal(1, "independent legal rejection base=%0d", base_word);
            address_snapshot = bank_row_addresses;
            incremented_bank_count = 0;
            for (int bank = 0; bank < BANKS; bank++) begin
                expected_row = (base_word >> 4) + (bank < (base_word & 15));
                if (bank_row_addresses[bank*ROW_W +: ROW_W] !== expected_row[7:0])
                    $fatal(1, "independent row mismatch base=%0d bank=%0d got=%0d expected=%0d",
                           base_word, bank,
                           bank_row_addresses[bank*ROW_W +: ROW_W], expected_row);
                if (expected_row < 0 || expected_row >= 230)
                    $fatal(1, "independent row out of physical range base=%0d bank=%0d row=%0d",
                           base_word, bank, expected_row);
                expected_physical_word = expected_row * BANKS + bank;
                if (expected_physical_word < 0 || expected_physical_word >= WORDS)
                    $fatal(1, "independent physical word out of range base=%0d bank=%0d word=%0d",
                           base_word, bank, expected_physical_word);
                bank_words[bank*WORD_W +: WORD_W] =
                    physical_payload(expected_physical_word, base_word);
                physical_address_checks++;
                one_read_per_bank_checks++;
                if (expected_row == (base_word >> 4) + 1) begin
                    incremented_bank_count++;
                    crossed_bank_address_checks++;
                end
            end
            #1ps;
            if (bank_row_addresses !== address_snapshot)
                $fatal(1, "bank data affected address generation base=%0d", base_word);

            offset_bank_seen = '0;
            for (int offset = 0; offset < BANKS; offset++) begin
                expected_physical_word = base_word + offset;
                expected_bank = expected_physical_word & 15;
                expected_row = expected_physical_word >> 4;
                if (offset_bank_seen[expected_bank])
                    $fatal(1, "duplicate bank service base=%0d offset=%0d bank=%0d",
                           base_word, offset, expected_bank);
                offset_bank_seen[expected_bank] = 1'b1;
                if (bank_row_addresses[expected_bank*ROW_W +: ROW_W]
                        !== expected_row[7:0])
                    $fatal(1, "logical-to-physical address mismatch base=%0d offset=%0d",
                           base_word, offset);
                if (logical_words[offset*WORD_W +: WORD_W]
                        !== physical_payload(expected_physical_word, base_word))
                    $fatal(1, "logical reorder mismatch base=%0d offset=%0d bank=%0d row=%0d",
                           base_word, offset, expected_bank, expected_row);
                logical_word_checks++;
            end
            if (offset_bank_seen !== 16'hffff)
                $fatal(1, "not every bank used exactly once base=%0d mask=%h",
                       base_word, offset_bank_seen);
            if (incremented_bank_count != (base_word & 15))
                $fatal(1, "row crossing bank count mismatch base=%0d got=%0d expected=%0d",
                       base_word, incremented_bank_count, base_word & 15);
            if ((base_word & 15) != 0)
                row_crossing_windows++;
            base_offset_hits[base_word & 15]++;
            legal_windows++;
            @(posedge clk_core);
        end
    endtask

    task automatic check_illegal(input int base_word, input logic [511:0] data);
        begin
            @(negedge clk_core);
            request_valid = 1'b1;
            logical_base_word = base_word[11:0];
            bank_words = data;
            #1ps;
            if (request_legal !== 1'b0 || conflict_free !== 1'b0
                    || bank_use_mask !== 0 || bank_row_addresses !== 0
                    || logical_words !== 0)
                $fatal(1, "illegal request did not quarantine base=%0d", base_word);
            illegal_windows++;
            @(posedge clk_core);
        end
    endtask

    initial begin : exhaustive_sequence
        rst_core = 1'b1;
        request_valid = 1'b0;
        logical_base_word = '0;
        bank_words = '0;
        legal_windows = 0;
        illegal_windows = 0;
        logical_word_checks = 0;
        physical_address_checks = 0;
        one_read_per_bank_checks = 0;
        row_crossing_windows = 0;
        crossed_bank_address_checks = 0;
        valid_low_payload_checks = 0;
        stale_or_skewed_data_undetected = 0;
        x_base_not_fail_closed = 0;
        for (int bank = 0; bank < BANKS; bank++)
            base_offset_hits[bank] = 0;

        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (int base = 0; base <= WORDS-BANKS; base++)
            check_legal(base);
        for (int base = WORDS-BANKS+1; base < (1 << 12); base++)
            check_illegal(base, {16{32'hc001d00d ^ base}});

        // With valid low, every output must remain quiet for changing legal,
        // illegal, X-free payload and data patterns.
        for (int trial = 0; trial < 64; trial++) begin
            @(negedge clk_core);
            request_valid = 1'b0;
            logical_base_word = ((trial * 12'd719) ^ 12'ha5b);
            bank_words = {16{32'h13579bdf ^ trial ^ (trial << 23)}};
            #1ps;
            if (request_legal !== 1'b0 || conflict_free !== 1'b0
                    || bank_use_mask !== 0 || bank_row_addresses !== 0
                    || logical_words !== 0)
                $fatal(1, "valid-low payload dependence trial=%0d", trial);
            valid_low_payload_checks++;
            @(posedge clk_core);
        end

        // The port cut has no response row/tag/valid. Deliberately provide
        // stale or skewed bank data: the mapper legally rotates it and cannot
        // detect that it did not come from bank_row_addresses.
        @(negedge clk_core);
        request_valid = 1'b1;
        logical_base_word = 12'd31;
        for (int bank = 0; bank < BANKS; bank++)
            bank_words[bank*32 +: 32] = 32'hbad00000 | bank;
        #1ps;
        if (!request_legal || !conflict_free || logical_words[0 +: 32] !== 32'hbad0000f)
            $fatal(1, "port-cut stale-data boundary setup failed");
        stale_or_skewed_data_undetected++;

        // Four-state simulation boundary: an unknown base does not resolve
        // request_legal to fail-closed zero. Record, do not promote this to a
        // silicon functional failure.
        logical_base_word = 12'b0x00_0000_0000;
        #1ps;
        if (!$isunknown(request_legal))
            $fatal(1, "X-base boundary unexpectedly resolved legal=%b", request_legal);
        if (bank_row_addresses !== 0 || logical_words !== 0
                || bank_use_mask !== 0 || conflict_free !== 0)
            $fatal(1, "X-base boundary did not keep downstream outputs quiet");
        x_base_not_fail_closed++;

        if (legal_windows != 3665 || illegal_windows != 431
                || logical_word_checks != 58640
                || physical_address_checks != 58640
                || one_read_per_bank_checks != 58640
                || row_crossing_windows != 3435
                || crossed_bank_address_checks != 27480
                || valid_low_payload_checks != 64)
            $fatal(1, "independent counter mismatch legal=%0d illegal=%0d logical=%0d physical=%0d onebank=%0d crossing=%0d crossedbanks=%0d idle=%0d",
                   legal_windows, illegal_windows, logical_word_checks,
                   physical_address_checks, one_read_per_bank_checks,
                   row_crossing_windows, crossed_bank_address_checks,
                   valid_low_payload_checks);
        if (base_offset_hits[0] != 230)
            $fatal(1, "base offset 0 count mismatch %0d", base_offset_hits[0]);
        for (int bank = 1; bank < BANKS; bank++)
            if (base_offset_hits[bank] != 229)
                $fatal(1, "base offset count mismatch bank=%0d count=%0d",
                       bank, base_offset_hits[bank]);

        $display("PASS M134 independent hammer legal_windows=3665 illegal_windows=431 logical_words=58640 physical_addresses=58640 one_read_per_bank_checks=58640 row_crossings=3435 crossed_bank_addresses=27480 base_offset0=230 other_base_offsets=229 valid_low_payload_checks=64 stale_or_skewed_data_undetected=1 x_base_not_fail_closed=1 words=3680 rows_per_bank=230 banks=16 word_bits=32 service_bits=512 exposed_address_bits=128 exposed_bank_data_bits=512 macro=false macro_latency=false response_tag=false parameter_guard_synthesis_hard=false physical_speedup=false system_speedup=false headline=false");
        $finish;
    end

    initial begin
        #30000000;
        $fatal(1, "M134 independent hammer timeout");
    end
endmodule

`default_nettype wire
