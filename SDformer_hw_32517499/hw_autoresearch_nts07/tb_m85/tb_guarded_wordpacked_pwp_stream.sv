`timescale 1ns/1ps
`default_nettype none

module tb_guarded_wordpacked_pwp_stream;
    localparam int PHASES = 1728;
    localparam int MAX_RECORD_BYTES = 14784;

    logic clk_core, rst_core;
    logic phase_load_valid, phase_load_ready, phase_loaded, metadata_error;
    logic [591:0] phase_metadata;
    logic lookup_valid, lookup_ready;
    logic [3:0] lookup_pattern;
    logic [2:0] lookup_block, lookup_beat;
    logic [31:0] lookup_tag;
    logic [255:0] bank_words;
    logic [79:0] bank_row_addresses;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [1151:0] output_values;
    logic protocol_error, busy;

    byte unsigned record_bytes [0:MAX_RECORD_BYTES-1];
    integer offsets [0:PHASES];
    integer records_fd, offsets_fd, metadata_fd;
    integer phase_count, entry_count, beat_count, output_count;
    integer escape_count, masked_nonzero_words, phase_stalls;
    integer previous_start_cycle, previous_beats, ii_checks, cycle_count;
    logic expected_pending, expected_escape;
    logic [31:0] expected_tag;
    logic [3:0] expected_width;
    logic [1151:0] expected_values;
    string records_path, offsets_path, metadata_path;

    guarded_wordpacked_pwp_stream dut (.*);
    guarded_wordpacked_pwp_stream_assertions m85_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;
    initial begin
        #8000000;
        $fatal(1, "M85 watchdog timeout phases=%0d entries=%0d outputs=%0d",
               phase_count, entry_count, output_count);
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

    function automatic logic [31:0] payload_word(
        input integer word_index, input integer terminal_words
    );
        integer byte_index;
        begin
            if (word_index >= terminal_words) begin
                payload_word = '0;
            end else begin
                byte_index = 48 + word_index * 4;
                payload_word = {record_bytes[byte_index+3],
                                record_bytes[byte_index+2],
                                record_bytes[byte_index+1],
                                record_bytes[byte_index+0]};
            end
        end
    endfunction

    task automatic build_expected(
        input integer start_word_value,
        input integer width_value,
        input integer terminal_words
    );
        logic [1279:0] packed_bits;
        begin
            packed_bits = '0;
            for (int word = 0; word < words_for_code(width_value-8); word++)
                packed_bits[word*32 +: 32] = payload_word(
                    start_word_value + word, terminal_words);
            expected_values = '0;
            for (int lane = 0; lane < 96; lane++) begin
                case (width_value)
                    8: expected_values[lane*12 +: 12] =
                        {{4{packed_bits[lane*8+7]}}, packed_bits[lane*8 +: 8]};
                    9: expected_values[lane*12 +: 12] =
                        {{3{packed_bits[lane*9+8]}}, packed_bits[lane*9 +: 9]};
                    10: expected_values[lane*12 +: 12] =
                        {{2{packed_bits[lane*10+9]}}, packed_bits[lane*10 +: 10]};
                    11: expected_values[lane*12 +: 12] =
                        {packed_bits[lane*11+10], packed_bits[lane*11 +: 11]};
                    default: $fatal(1, "M85 bad expected width");
                endcase
            end
        end
    endtask

    task automatic check_output_after_edge;
        begin
            #1;
            if (output_valid) begin
                if (!expected_pending)
                    $fatal(1, "M85 unexpected output tag=%0d", output_tag);
                if (output_tag != expected_tag
                        || output_width != expected_width
                        || output_escape != expected_escape
                        || output_values !== expected_values)
                    $fatal(1, "M85 output mismatch tag=%0d expected=%0d width=%0d expected_width=%0d escape=%0d expected_escape=%0d",
                           output_tag, expected_tag, output_width,
                           expected_width, output_escape, expected_escape);
                expected_pending = 1'b0;
                output_count++;
            end
            if (protocol_error)
                $fatal(1, "M85 protocol error during legal replay");
        end
    endtask

    task automatic drive_regular_beat(
        input integer pattern_value,
        input integer block_value,
        input integer beat_value,
        input integer beats_value,
        input integer tag_value,
        input integer start_word_value,
        input integer terminal_words
    );
        integer logical_base, base_row, base_bank, physical_word;
        begin
            logical_base = start_word_value + beat_value*8;
            base_row = logical_base / 8;
            base_bank = logical_base % 8;
            bank_words = '0;
            for (int bank = 0; bank < 8; bank++) begin
                physical_word = (base_row + (bank < base_bank))*8 + bank;
                bank_words[bank*32 +: 32] = payload_word(
                    physical_word, terminal_words);
            end
            @(negedge clk_core);
            lookup_valid = 1'b1;
            lookup_pattern = pattern_value[3:0];
            lookup_block = block_value[2:0];
            lookup_beat = beat_value[2:0];
            lookup_tag = beat_value == 0 ? tag_value : 0;
            do begin
                @(posedge clk_core);
                if (!lookup_ready) phase_stalls++;
            end while (!lookup_ready);
            if (beat_value == 0) begin
                if (previous_start_cycle >= 0
                        && cycle_count - previous_start_cycle != previous_beats)
                    $fatal(1, "M85 unexpected transaction bubble got=%0d expected=%0d",
                           cycle_count - previous_start_cycle, previous_beats);
                if (previous_start_cycle >= 0) ii_checks++;
                previous_start_cycle = cycle_count;
                previous_beats = beats_value;
            end
            beat_count++;
            check_output_after_edge();
        end
    endtask

    task automatic drive_escape(
        input integer pattern_value,
        input integer block_value,
        input integer tag_value
    );
        begin
            bank_words = '0;
            @(negedge clk_core);
            lookup_valid = 1'b1;
            lookup_pattern = pattern_value[3:0];
            lookup_block = block_value[2:0];
            lookup_beat = 0;
            lookup_tag = tag_value;
            do @(posedge clk_core); while (!lookup_ready);
            if (previous_start_cycle >= 0
                    && cycle_count - previous_start_cycle != previous_beats)
                $fatal(1, "M85 unexpected escape bubble");
            if (previous_start_cycle >= 0) ii_checks++;
            previous_start_cycle = cycle_count;
            previous_beats = 1;
            beat_count++;
            check_output_after_edge();
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            lookup_valid = 1'b0;
            phase_load_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core); rst_core = 1'b0;
        end
    endtask

    initial begin
        integer low, offset_value, record_length, bytes_read;
        integer cursor, code, width_value, beats_value, tag_value;
        integer phase_terminal;
        clk_core = 0;
        rst_core = 1;
        phase_load_valid = 0;
        phase_metadata = '0;
        lookup_valid = 0;
        lookup_pattern = 0;
        lookup_block = 0;
        lookup_beat = 0;
        lookup_tag = 0;
        bank_words = 0;
        output_ready = 1;
        phase_count = 0;
        entry_count = 0;
        beat_count = 0;
        output_count = 0;
        escape_count = 0;
        masked_nonzero_words = 0;
        phase_stalls = 0;
        previous_start_cycle = -1;
        previous_beats = 0;
        ii_checks = 0;
        cycle_count = 0;
        expected_pending = 0;
        if (!$value$plusargs("RECORDS_BIN=%s", records_path)
                || !$value$plusargs("OFFSETS_BIN=%s", offsets_path)
                || !$value$plusargs("METADATA_BIN=%s", metadata_path))
            $fatal(1, "M85 missing input plusargs");
        records_fd = $fopen(records_path, "rb");
        offsets_fd = $fopen(offsets_path, "rb");
        metadata_fd = $fopen(metadata_path, "rb");
        if (!records_fd || !offsets_fd || !metadata_fd)
            $fatal(1, "M85 cannot open input binary");
        for (int index = 0; index <= PHASES; index++) begin
            offset_value = 0;
            for (int byte_index = 0; byte_index < 4; byte_index++) begin
                low = $fgetc(offsets_fd);
                if (low < 0) $fatal(1, "M85 truncated offsets");
                offset_value |= low << (8*byte_index);
            end
            offsets[index] = offset_value;
        end
        if ($fgetc(offsets_fd) != -1) $fatal(1, "M85 trailing offsets");
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        tag_value = 1;
        for (int phase = 0; phase < PHASES; phase++) begin
            record_length = offsets[phase+1] - offsets[phase];
            if (record_length <= 0 || record_length > MAX_RECORD_BYTES)
                $fatal(1, "M85 bad record length=%0d", record_length);
            if ($fseek(records_fd, offsets[phase], 0) != 0)
                $fatal(1, "M85 record seek failed");
            bytes_read = $fread(record_bytes, records_fd, 0, record_length);
            if (bytes_read != record_length)
                $fatal(1, "M85 short record read got=%0d expected=%0d",
                       bytes_read, record_length);
            phase_metadata = '0;
            for (int byte_index = 0; byte_index < 74; byte_index++) begin
                low = $fgetc(metadata_fd);
                if (low < 0) $fatal(1, "M85 truncated metadata");
                phase_metadata[byte_index*8 +: 8] = low[7:0];
                if (byte_index < 48
                        && low[7:0] != record_bytes[byte_index])
                    $fatal(1, "M85 metadata/header mismatch");
            end
            lookup_valid = 0;
            do @(posedge clk_core); while (!phase_load_ready);
            @(negedge clk_core); phase_load_valid = 1;
            @(posedge clk_core);
            @(negedge clk_core); phase_load_valid = 0;
            if (!phase_loaded || metadata_error || protocol_error)
                $fatal(1, "M85 legal phase load rejected phase=%0d", phase);

            phase_terminal = 0;
            for (int entry = 0; entry < 128; entry++) begin
                code = phase_metadata[entry*3 +: 3];
                if (code < 0 || code > 4)
                    $fatal(1, "M85 frozen reserved code");
                phase_terminal += words_for_code(code);
            end
            cursor = 0;
            previous_start_cycle = -1;
            for (int pattern = 0; pattern < 16; pattern++) begin
                for (int block = 0; block < 8; block++) begin
                    code = phase_metadata[(pattern*8+block)*3 +: 3];
                    if (code == 4) begin
                        expected_pending = 1;
                        expected_tag = tag_value;
                        expected_width = 12;
                        expected_escape = 1;
                        expected_values = '0;
                        drive_escape(pattern, block, tag_value);
                        escape_count++;
                    end else begin
                        width_value = 8 + code;
                        beats_value = (96*width_value + 255)/256;
                        build_expected(cursor, width_value, phase_terminal);
                        expected_pending = 0;
                        for (int beat = 0; beat < beats_value; beat++) begin
                            if (beat == beats_value-1 && width_value != 8) begin
                                integer valid_words;
                                valid_words = words_for_code(code) - beat*8;
                                for (int word = valid_words; word < 8; word++)
                                    if (payload_word(cursor + beat*8 + word,
                                                     phase_terminal) != 0)
                                        masked_nonzero_words++;
                            end
                            if (beat == beats_value-1) begin
                                expected_pending = 1;
                                expected_tag = tag_value;
                                expected_width = width_value;
                                expected_escape = 0;
                            end
                            drive_regular_beat(pattern, block, beat,
                                beats_value, tag_value, cursor, phase_terminal);
                        end
                        cursor += words_for_code(code);
                    end
                    if (expected_pending)
                        $fatal(1, "M85 expected output did not arrive tag=%0d",
                               tag_value);
                    entry_count++;
                    tag_value++;
                end
            end
            // Retire the last output before replacing phase metadata.
            lookup_valid = 0;
            do @(posedge clk_core); while (busy);
            phase_count++;
        end
        if ($fgetc(metadata_fd) != -1)
            $fatal(1, "M85 trailing metadata bytes");
        $fclose(records_fd); $fclose(offsets_fd); $fclose(metadata_fd);

        // Poison attacks: reserved prior code, inconsistent base, overflow.
        for (int attack = 0; attack < 3; attack++) begin
            reset_dut();
            phase_metadata = '0;
            for (int entry = 0; entry < 128; entry++)
                phase_metadata[entry*3 +: 3] = 0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384+pattern*13 +: 13] = cursor;
                cursor += 8*24;
            end
            if (attack == 0) phase_metadata[0 +: 3] = 5;
            if (attack == 1) phase_metadata[384+4*13 +: 13] ^= 1;
            if (attack == 2) phase_metadata[384+15*13 +: 13] = 8191;
            @(negedge clk_core); phase_load_valid = 1;
            @(posedge clk_core);
            @(negedge clk_core); phase_load_valid = 0;
            lookup_valid = 1;
            lookup_pattern = 0; lookup_block = 1; lookup_beat = 0;
            @(posedge clk_core); #1;
            if (!metadata_error || !protocol_error || lookup_ready)
                $fatal(1, "M85 poison attack accepted attack=%0d", attack);
            lookup_valid = 0;
        end

        if (phase_count != 1728 || entry_count != 221184
                || output_count != 221184 || escape_count != 1
                || beat_count != 835383 || masked_nonzero_words == 0
                || ii_checks != 221184-1728 || phase_stalls != 0)
            $fatal(1, "M85 coverage mismatch phases=%0d entries=%0d outputs=%0d escape=%0d beats=%0d masked=%0d ii=%0d stalls=%0d",
                   phase_count, entry_count, output_count, escape_count,
                   beat_count, masked_nonzero_words, ii_checks, phase_stalls);
        $display("PASS M85 actual-record integration phases=1728 entries=221184 outputs=221184 escape=1 beats=835383 masked_nonzero_words=%0d ii_checks=219456 metadata_poison_attacks=3",
                 masked_nonzero_words);
        $finish;
    end
endmodule

`default_nettype wire
