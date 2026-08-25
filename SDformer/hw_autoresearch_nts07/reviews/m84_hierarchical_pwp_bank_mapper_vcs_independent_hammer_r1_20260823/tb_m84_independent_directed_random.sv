`timescale 1ns/1ps
`default_nettype none

// Independent M84 adversarial test.  This test intentionally does not reuse
// the production geometry-loop TB or its checking tasks.
module tb_m84_independent_directed_random;
    localparam int PATTERNS = 16;
    localparam int BLOCKS = 8;
    localparam int OFFSET_W = 13;
    localparam int ROW_W = 10;

    logic [PATTERNS*BLOCKS*3-1:0] width_header;
    logic [PATTERNS*OFFSET_W-1:0] pattern_base_words;
    logic [3:0] pattern_index;
    logic [2:0] block_index;
    logic [2:0] beat_index;
    logic [255:0] bank_words;
    logic descriptor_valid;
    logic descriptor_escape;
    logic [3:0] descriptor_width;
    logic [2:0] descriptor_beats;
    logic [OFFSET_W-1:0] start_word;
    logic beat_index_valid;
    logic [8*ROW_W-1:0] bank_row_addresses;
    logic [255:0] logical_words;

    integer directed_checks;
    integer random_checks;
    integer all_start_mod8_checks;
    integer pattern15_block7_checks;
    integer selected_reserved_blocked;
    integer prior_reserved_failopen_observed;
    integer overflow_wrap_observed;
    integer escape_neighbor_checks;
    integer seed;

    hierarchical_pwp_bank_mapper dut (.*);

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            default: words_for_code = 0;
        endcase
    endfunction

    function automatic integer beats_for_code(input integer code);
        case (code)
            0: beats_for_code = 3;
            1, 2: beats_for_code = 4;
            3: beats_for_code = 5;
            default: beats_for_code = 0;
        endcase
    endfunction

    task automatic clear_inputs;
        begin
            width_header = '0;
            pattern_base_words = '0;
            pattern_index = '0;
            block_index = '0;
            beat_index = '0;
            bank_words = '0;
            #1;
        end
    endtask

    task automatic put_code(
        input integer pattern,
        input integer block,
        input integer code
    );
        begin
            width_header[(pattern*BLOCKS+block)*3 +: 3] = code[2:0];
        end
    endtask

    task automatic put_base(input integer pattern, input integer base);
        begin
            pattern_base_words[pattern*OFFSET_W +: OFFSET_W] = base[12:0];
        end
    endtask

    task automatic drive_bank_tags;
        begin
            for (int bank = 0; bank < 8; bank++)
                bank_words[bank*32 +: 32] = 32'hc300_0000 | bank;
        end
    endtask

    task automatic check_safe_reference(
        input integer pattern,
        input integer block,
        input integer beat,
        input integer expected_base,
        input integer expected_code
    );
        integer prefix;
        integer expected_beats;
        integer logical_base;
        integer base_row;
        integer base_bank;
        integer expected_row;
        integer expected_bank;
        begin
            prefix = expected_base;
            for (int prior = 0; prior < block; prior++)
                prefix += words_for_code(
                    width_header[(pattern*BLOCKS+prior)*3 +: 3]);
            expected_beats = beats_for_code(expected_code);
            pattern_index = pattern[3:0];
            block_index = block[2:0];
            beat_index = beat[2:0];
            drive_bank_tags();
            #1;
            if (expected_code <= 4) begin
                if (!descriptor_valid)
                    $fatal(1, "independent: legal code rejected code=%0d", expected_code);
                if (descriptor_escape != (expected_code == 4))
                    $fatal(1, "independent: escape mismatch code=%0d", expected_code);
                if (descriptor_width != ((expected_code == 4) ? 12 : expected_code+8))
                    $fatal(1, "independent: width mismatch code=%0d got=%0d",
                           expected_code, descriptor_width);
                if (descriptor_beats != expected_beats)
                    $fatal(1, "independent: beat-count mismatch code=%0d", expected_code);
                if (start_word != prefix)
                    $fatal(1, "independent: prefix mismatch pattern=%0d block=%0d got=%0d expected=%0d",
                           pattern, block, start_word, prefix);
                if (beat_index_valid != (expected_code < 4 && beat < expected_beats))
                    $fatal(1, "independent: beat valid mismatch");
            end else begin
                if (descriptor_valid || beat_index_valid)
                    $fatal(1, "independent: selected reserved code accepted=%0d", expected_code);
            end
            if (beat_index_valid) begin
                logical_base = prefix + beat*8;
                base_row = logical_base / 8;
                base_bank = logical_base % 8;
                for (int bank = 0; bank < 8; bank++) begin
                    expected_row = base_row + (bank < base_bank);
                    if (bank_row_addresses[bank*ROW_W +: ROW_W] != expected_row)
                        $fatal(1, "independent: row mismatch bank=%0d got=%0d expected=%0d",
                               bank, bank_row_addresses[bank*ROW_W +: ROW_W], expected_row);
                end
                for (int word = 0; word < 8; word++) begin
                    expected_bank = (base_bank + word) & 7;
                    if (logical_words[word*32 +: 32]
                            != (32'hc300_0000 | expected_bank))
                        $fatal(1, "independent: rotate-direction mismatch word=%0d base_bank=%0d",
                               word, base_bank);
                end
            end
        end
    endtask

    initial begin
        integer code;
        integer base;
        integer selected_pattern;
        integer selected_block;
        integer selected_beat;
        integer raw_prefix;
        directed_checks = 0;
        random_checks = 0;
        all_start_mod8_checks = 0;
        pattern15_block7_checks = 0;
        selected_reserved_blocked = 0;
        prior_reserved_failopen_observed = 0;
        overflow_wrap_observed = 0;
        escape_neighbor_checks = 0;
        seed = 32'h6d84_a117;
        clear_inputs();

        // All eight starting banks and every legal width, including the last beat.
        for (int start_mod = 0; start_mod < 8; start_mod++) begin
            clear_inputs();
            code = start_mod % 4;
            put_base(0, 800 + start_mod);
            put_code(0, 0, code);
            check_safe_reference(0, 0, beats_for_code(code)-1,
                                 800 + start_mod, code);
            all_start_mod8_checks++;
            directed_checks++;
        end

        // pattern15/block7 with a nontrivial prefix containing two escapes.
        clear_inputs();
        put_base(15, 3300);
        put_code(15, 0, 0);
        put_code(15, 1, 4);
        put_code(15, 2, 1);
        put_code(15, 3, 2);
        put_code(15, 4, 3);
        put_code(15, 5, 4);
        put_code(15, 6, 0);
        put_code(15, 7, 3);
        for (int beat = 0; beat < 5; beat++) begin
            check_safe_reference(15, 7, beat, 3300, 3);
            pattern15_block7_checks++;
            directed_checks++;
        end

        // Escape must consume zero words: next block starts at the escape cursor.
        clear_inputs();
        put_base(6, 1003);
        put_code(6, 0, 0);
        put_code(6, 1, 4);
        put_code(6, 2, 1);
        check_safe_reference(6, 0, 0, 1003, 0);
        check_safe_reference(6, 1, 0, 1003, 4);
        check_safe_reference(6, 2, 0, 1003, 1);
        if (start_word != 1027)
            $fatal(1, "independent: subsequent cursor advanced by escape");
        escape_neighbor_checks += 3;
        directed_checks += 3;

        // Selected reserved codes 5/6/7 fail closed.
        for (int reserved = 5; reserved <= 7; reserved++) begin
            clear_inputs();
            put_base(2, 321);
            put_code(2, 0, reserved);
            check_safe_reference(2, 0, 0, 321, reserved);
            selected_reserved_blocked++;
            directed_checks++;
        end

        // A reserved code in the prefix is silently treated as zero.  Record
        // the fail-open behavior for review rather than declaring it legal.
        for (int reserved = 5; reserved <= 7; reserved++) begin
            clear_inputs();
            put_base(3, 777);
            put_code(3, 0, reserved);
            put_code(3, 1, 0);
            check_safe_reference(3, 1, 0, 777, 0);
            if (!descriptor_valid || start_word != 777)
                $fatal(1, "independent: unexpected prior-reserved behavior");
            prior_reserved_failopen_observed++;
            directed_checks++;
        end

        // Prefix overflow: base 8191 plus one 24-word predecessor wraps the
        // externally visible 13-bit start and truncates the 11-bit row to 10.
        clear_inputs();
        put_base(4, 8191);
        put_code(4, 0, 0);
        put_code(4, 1, 0);
        pattern_index = 4;
        block_index = 1;
        beat_index = 0;
        drive_bank_tags();
        #1;
        if (!descriptor_valid || start_word != 23
                || bank_row_addresses[0*ROW_W +: ROW_W] != 3)
            $fatal(1, "independent: prefix-overflow observation drift start=%0d row0=%0d",
                   start_word, bank_row_addresses[0*ROW_W +: ROW_W]);
        overflow_wrap_observed++;
        directed_checks++;

        // Row boundary: a legal 13-bit start at word 8184 reaches row 1024 on
        // its second beat and wraps the 10-bit bank row to zero.
        clear_inputs();
        put_base(5, 8184);
        put_code(5, 0, 0);
        pattern_index = 5;
        block_index = 0;
        beat_index = 1;
        drive_bank_tags();
        #1;
        if (!descriptor_valid || start_word != 8184
                || bank_row_addresses[0*ROW_W +: ROW_W] != 0)
            $fatal(1, "independent: row-overflow observation drift");
        overflow_wrap_observed++;
        directed_checks++;

        // Independent random reference: arbitrary safe bases, every code 0-7,
        // every pattern/block position and arbitrary beat_index.
        for (int trial = 0; trial < 4096; trial++) begin
            clear_inputs();
            selected_pattern = $urandom(seed) % 16;
            selected_block = $urandom(seed) % 8;
            selected_beat = $urandom(seed) % 8;
            base = $urandom(seed) % 3501;
            put_base(selected_pattern, base);
            for (int block = 0; block < 8; block++)
                put_code(selected_pattern, block, $urandom(seed) % 8);
            code = width_header[(selected_pattern*8+selected_block)*3 +: 3];
            check_safe_reference(selected_pattern, selected_block, selected_beat,
                                 base, code);
            random_checks++;
        end

        if (all_start_mod8_checks != 8 || pattern15_block7_checks != 5
                || selected_reserved_blocked != 3
                || prior_reserved_failopen_observed != 3
                || overflow_wrap_observed != 2
                || escape_neighbor_checks != 3 || random_checks != 4096)
            $fatal(1, "independent: coverage counter drift");
        $display("PASS M84 independent directed_random directed=%0d random=%0d start_mod8=%0d pattern15_block7=%0d escape_neighbor=%0d selected_reserved_blocked=%0d prior_reserved_failopen=%0d overflow_wrap=%0d",
                 directed_checks, random_checks, all_start_mod8_checks,
                 pattern15_block7_checks, escape_neighbor_checks,
                 selected_reserved_blocked, prior_reserved_failopen_observed,
                 overflow_wrap_observed);
        $finish;
    end
endmodule

`default_nettype wire
