`timescale 1ns/1ps
`default_nettype none

module tb_hierarchical_pwp_bank_mapper;
    localparam int PHASES = 1728;
    localparam int ENTRIES = 128;
    localparam int ROW_W = 10;

    logic clk_core, rst_core, sample_valid;
    logic [383:0] width_header;
    logic [207:0] pattern_base_words;
    logic [3:0] pattern_index;
    logic [2:0] block_index, beat_index;
    logic [255:0] bank_words;
    logic descriptor_valid, descriptor_escape, beat_index_valid;
    logic [3:0] descriptor_width;
    logic [2:0] descriptor_beats;
    logic [12:0] start_word;
    logic [8*ROW_W-1:0] bank_row_addresses;
    logic [255:0] logical_words;

    integer fd, byte_value;
    integer phase_count, entry_count, beat_count, escape_count;
    integer cross_row_count, invalid_attacks;
    string geometry_path;

    hierarchical_pwp_bank_mapper dut (.*);
    hierarchical_pwp_bank_mapper_assertions m84_sva (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #5000000;
        $fatal(1, "M84 watchdog timeout phases=%0d entries=%0d beats=%0d",
               phase_count, entry_count, beat_count);
    end

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            4: words_for_code = 0;
            default: words_for_code = -1;
        endcase
    endfunction

    task automatic sample_and_check(
        input integer expected_start,
        input integer expected_code,
        input integer expected_beat
    );
        integer logical_base, base_row, base_bank, expected_row;
        begin
            beat_index = expected_beat[2:0];
            logical_base = expected_start + expected_beat * 8;
            base_row = logical_base / 8;
            base_bank = logical_base % 8;
            bank_words = '0;
            for (int bank = 0; bank < 8; bank++) begin
                expected_row = base_row + (bank < base_bank);
                bank_words[bank*32 +: 32] = expected_row * 8 + bank;
            end
            @(negedge clk_core); sample_valid = 1'b1;
            @(posedge clk_core); #1;
            if (!descriptor_valid || descriptor_escape
                    || descriptor_width != expected_code + 8
                    || start_word != expected_start)
                $fatal(1, "M84 descriptor mismatch phase=%0d pattern=%0d block=%0d got_start=%0d expected_start=%0d code=%0d",
                       phase_count, pattern_index, block_index, start_word,
                       expected_start, expected_code);
            if (!beat_index_valid)
                $fatal(1, "M84 legal beat rejected");
            for (int bank = 0; bank < 8; bank++) begin
                expected_row = base_row + (bank < base_bank);
                if (bank_row_addresses[bank*ROW_W +: ROW_W] != expected_row)
                    $fatal(1, "M84 bank row mismatch bank=%0d got=%0d expected=%0d",
                           bank, bank_row_addresses[bank*ROW_W +: ROW_W],
                           expected_row);
            end
            for (int word = 0; word < 8; word++)
                if (logical_words[word*32 +: 32] != logical_base + word)
                    $fatal(1, "M84 barrel mismatch word=%0d got=%0d expected=%0d",
                           word, logical_words[word*32 +: 32],
                           logical_base + word);
            beat_count++;
            if (base_bank != 0) cross_row_count++;
        end
    endtask

    initial begin
        integer cursor, code, beats, stored_base, terminal;
        clk_core = 1'b0;
        rst_core = 1'b1;
        sample_valid = 1'b0;
        width_header = '0;
        pattern_base_words = '0;
        pattern_index = '0;
        block_index = '0;
        beat_index = '0;
        bank_words = '0;
        phase_count = 0;
        entry_count = 0;
        beat_count = 0;
        escape_count = 0;
        cross_row_count = 0;
        invalid_attacks = 0;
        if (!$value$plusargs("GEOMETRY_BIN=%s", geometry_path))
            $fatal(1, "M84 missing +GEOMETRY_BIN");
        fd = $fopen(geometry_path, "rb");
        if (fd == 0) $fatal(1, "M84 cannot open geometry binary");
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;

        for (int phase = 0; phase < PHASES; phase++) begin
            for (int byte_index = 0; byte_index < 48; byte_index++) begin
                byte_value = $fgetc(fd);
                if (byte_value < 0) $fatal(1, "M84 truncated header");
                width_header[byte_index*8 +: 8] = byte_value[7:0];
            end
            for (int pattern = 0; pattern < 16; pattern++) begin
                integer low_byte, high_byte;
                low_byte = $fgetc(fd); high_byte = $fgetc(fd);
                if (low_byte < 0 || high_byte < 0)
                    $fatal(1, "M84 truncated pattern base");
                pattern_base_words[pattern*13 +: 13] =
                    low_byte[7:0] | (high_byte[4:0] << 8);
                if ((high_byte & 8'he0) != 0)
                    $fatal(1, "M84 noncanonical pattern-base high bits");
            end
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                stored_base = pattern_base_words[pattern*13 +: 13];
                if (stored_base != cursor)
                    $fatal(1, "M84 pattern base mismatch phase=%0d pattern=%0d got=%0d expected=%0d",
                           phase, pattern, stored_base, cursor);
                for (int block = 0; block < 8; block++) begin
                    pattern_index = pattern[3:0];
                    block_index = block[2:0];
                    code = width_header[(pattern*8+block)*3 +: 3];
                    if (code == 4) begin
                        @(negedge clk_core); sample_valid = 1'b1;
                        @(posedge clk_core); #1;
                        if (!descriptor_valid || !descriptor_escape
                                || descriptor_width != 12 || start_word != cursor
                                || descriptor_beats != 0 || beat_index_valid)
                            $fatal(1, "M84 escape decode mismatch");
                        escape_count++;
                    end else begin
                        if (code < 0 || code > 3)
                            $fatal(1, "M84 illegal frozen code=%0d", code);
                        beats = (96 * (code + 8) + 255) / 256;
                        for (int beat = 0; beat < beats; beat++)
                            sample_and_check(cursor, code, beat);
                    end
                    cursor += words_for_code(code);
                    entry_count++;
                end
            end
            terminal = cursor;
            if (terminal <= 0 || terminal >= 8192)
                $fatal(1, "M84 invalid phase terminal=%0d", terminal);
            phase_count++;
        end
        byte_value = $fgetc(fd);
        if (byte_value != -1) $fatal(1, "M84 trailing geometry bytes");
        $fclose(fd);

        // Fail-closed attack: reserved width code 5 cannot issue a bank beat.
        width_header[2:0] = 3'd5;
        pattern_base_words[12:0] = 13'd0;
        pattern_index = 0; block_index = 0; beat_index = 0;
        @(negedge clk_core); sample_valid = 1'b1;
        @(posedge clk_core); #1;
        if (descriptor_valid || beat_index_valid)
            $fatal(1, "M84 reserved code was accepted");
        invalid_attacks++;
        @(negedge clk_core); sample_valid = 1'b0;

        if (phase_count != 1728 || entry_count != 221184
                || escape_count != 1 || cross_row_count == 0
                || invalid_attacks != 1)
            $fatal(1, "M84 coverage mismatch phases=%0d entries=%0d escape=%0d cross=%0d attacks=%0d",
                   phase_count, entry_count, escape_count,
                   cross_row_count, invalid_attacks);
        $display("PASS M84 exhaustive phases=1728 entries=221184 escape=1 beats=%0d cross_row=%0d invalid_attacks=1 metadata=74B_vs_256B",
                 beat_count, cross_row_count);
        $finish;
    end
endmodule

`default_nettype wire
