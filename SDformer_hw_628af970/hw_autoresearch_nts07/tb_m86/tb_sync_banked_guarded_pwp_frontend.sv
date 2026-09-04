`timescale 1ns/1ps
`default_nettype none

module tb_sync_banked_guarded_pwp_frontend;
    localparam int PHASES = 1728;
    localparam int ROWS = 460;
    localparam int MAX_RECORD_BYTES = 14784;
    localparam int EXPECTED_DEPTH = 32;

    logic clk_core, rst_core;
    logic payload_load_valid, payload_load_ready, payload_load_accept;
    logic [9:0] payload_load_row;
    logic [255:0] payload_load_words;
    logic phase_load_valid, phase_load_ready, phase_loaded, metadata_error;
    logic [591:0] phase_metadata;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [3:0] descriptor_pattern;
    logic [2:0] descriptor_block;
    logic [31:0] descriptor_tag;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [1151:0] output_values;
    logic protocol_error, busy, bank_read_issue, bank_response_enqueue;
    logic [2:0] bank_read_beat, response_fifo_level;

    byte unsigned record_bytes [0:MAX_RECORD_BYTES-1];
    integer offsets [0:PHASES];
    logic [31:0] expected_tag_q [0:EXPECTED_DEPTH-1];
    logic [3:0] expected_width_q [0:EXPECTED_DEPTH-1];
    logic expected_escape_q [0:EXPECTED_DEPTH-1];
    logic [1151:0] expected_values_q [0:EXPECTED_DEPTH-1];
    integer expected_read_ptr, expected_write_ptr, expected_count;

    integer records_fd, offsets_fd, metadata_fd;
    integer phase_count, descriptor_count, output_count;
    integer bank_issue_count, bank_response_count, escape_count;
    integer always_ready_ii_checks, backpressure_cycles, fifo_full_cycles;
    integer previous_start_cycle, previous_transaction_beats, cycle_count;
    integer stress_phase_count;
    logic stress_mode;
    logic [31:0] lfsr_q;
    string records_path, offsets_path, metadata_path;

    sync_banked_guarded_pwp_frontend dut (.*);
    sync_banked_guarded_pwp_frontend_assertions m86_sva (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #15000000;
        $fatal(1, "M86 watchdog timeout phase=%0d descriptors=%0d outputs=%0d",
               phase_count, descriptor_count, output_count);
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
                byte_index = 48 + word_index*4;
                payload_word = {record_bytes[byte_index+3],
                                record_bytes[byte_index+2],
                                record_bytes[byte_index+1],
                                record_bytes[byte_index+0]};
            end
        end
    endfunction

    task automatic push_expected(
        input integer tag_value,
        input integer width_value,
        input logic escape_value,
        input logic [1151:0] values_value
    );
        begin
            if (expected_count >= EXPECTED_DEPTH)
                $fatal(1, "M86 expected queue overflow");
            expected_tag_q[expected_write_ptr] = tag_value;
            expected_width_q[expected_write_ptr] = width_value;
            expected_escape_q[expected_write_ptr] = escape_value;
            expected_values_q[expected_write_ptr] = values_value;
            expected_write_ptr = (expected_write_ptr + 1) % EXPECTED_DEPTH;
            expected_count++;
        end
    endtask

    task automatic build_and_push_expected(
        input integer start_word_value,
        input integer width_value,
        input integer terminal_words,
        input integer tag_value
    );
        logic [1279:0] packed_bits;
        logic [1151:0] values;
        begin
            packed_bits = '0;
            values = '0;
            for (int word = 0;
                    word < words_for_code(width_value-8); word++)
                packed_bits[word*32 +: 32] = payload_word(
                    start_word_value + word, terminal_words);
            for (int lane = 0; lane < 96; lane++) begin
                case (width_value)
                    8: values[lane*12 +: 12] = {
                        {4{packed_bits[lane*8+7]}},
                        packed_bits[lane*8 +: 8]};
                    9: values[lane*12 +: 12] = {
                        {3{packed_bits[lane*9+8]}},
                        packed_bits[lane*9 +: 9]};
                    10: values[lane*12 +: 12] = {
                        {2{packed_bits[lane*10+9]}},
                        packed_bits[lane*10 +: 10]};
                    11: values[lane*12 +: 12] = {
                        packed_bits[lane*11+10],
                        packed_bits[lane*11 +: 11]};
                    default: $fatal(1, "M86 bad expected width");
                endcase
            end
            push_expected(tag_value, width_value, 1'b0, values);
        end
    endtask

    task automatic drive_payload_row(
        input integer row_value, input integer terminal_words
    );
        begin
            @(negedge clk_core);
            payload_load_row = row_value;
            for (int bank = 0; bank < 8; bank++)
                payload_load_words[bank*32 +: 32] = payload_word(
                    row_value*8 + bank, terminal_words);
            payload_load_valid = 1'b1;
            do @(posedge clk_core); while (!payload_load_ready);
            #0.1 payload_load_valid = 1'b0;
        end
    endtask

    task automatic drive_phase_load;
        begin
            @(negedge clk_core); phase_load_valid = 1'b1;
            do @(posedge clk_core); while (!phase_load_ready);
            #0.1 phase_load_valid = 1'b0;
            #0.9;
            if (!phase_loaded || metadata_error || protocol_error)
                $fatal(1, "M86 legal phase rejected phase=%0d", phase_count);
        end
    endtask

    task automatic drive_descriptor(
        input integer pattern_value,
        input integer block_value,
        input integer tag_value
    );
        begin
            @(negedge clk_core);
            descriptor_pattern = pattern_value;
            descriptor_block = block_value;
            descriptor_tag = tag_value;
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_ready);
            #0.1 descriptor_valid = 1'b0;
        end
    endtask

    // Check at the accepting edge, before the DUT's nonblocking state update.
    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (descriptor_accept)
                descriptor_count = descriptor_count + 1;
            if (bank_read_issue) begin
                bank_issue_count = bank_issue_count + 1;
                if (bank_read_beat == 0 && !stress_mode) begin
                    if (previous_start_cycle >= 0) begin
                        if (cycle_count - previous_start_cycle
                                != previous_transaction_beats)
                            $fatal(1, "M86 always-ready start bubble got=%0d expected=%0d phase=%0d",
                                   cycle_count - previous_start_cycle,
                                   previous_transaction_beats, phase_count);
                        always_ready_ii_checks = always_ready_ii_checks + 1;
                    end
                    previous_start_cycle = cycle_count;
                    previous_transaction_beats = dut.issue_escape
                                               ? 1 : dut.issue_beats;
                end
            end
            if (bank_response_enqueue)
                bank_response_count = bank_response_count + 1;
            if (output_valid && !output_ready)
                backpressure_cycles = backpressure_cycles + 1;
            if (response_fifo_level == 4)
                fifo_full_cycles = fifo_full_cycles + 1;
            if (output_accept) begin
                if (expected_count <= 0)
                    $fatal(1, "M86 unexpected output tag=%0d", output_tag);
                if (output_tag != expected_tag_q[expected_read_ptr]
                        || output_width != expected_width_q[expected_read_ptr]
                        || output_escape != expected_escape_q[expected_read_ptr]
                        || output_values !== expected_values_q[expected_read_ptr])
                    $fatal(1, "M86 output mismatch got_tag=%0d expected_tag=%0d got_width=%0d expected_width=%0d",
                           output_tag, expected_tag_q[expected_read_ptr],
                           output_width, expected_width_q[expected_read_ptr]);
                expected_read_ptr = (expected_read_ptr + 1) % EXPECTED_DEPTH;
                expected_count = expected_count - 1;
                output_count = output_count + 1;
            end
            if (protocol_error && phase_count < PHASES)
                $fatal(1, "M86 protocol error during legal replay phase=%0d",
                       phase_count);
        end
    end

    always @(negedge clk_core) begin
        if (rst_core) begin
            output_ready <= 1'b1;
            lfsr_q <= 32'h1ace_b00c;
        end else if (stress_mode) begin
            lfsr_q <= {lfsr_q[30:0],
                       lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            output_ready <= lfsr_q[0] | lfsr_q[3];
        end else begin
            output_ready <= 1'b1;
        end
    end

    initial begin
        integer low, offset_value, record_length, bytes_read;
        integer cursor, code, width_value, tag_value, phase_terminal;
        clk_core = 0;
        rst_core = 1;
        payload_load_valid = 0;
        payload_load_row = 0;
        payload_load_words = 0;
        phase_load_valid = 0;
        phase_metadata = 0;
        descriptor_valid = 0;
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 0;
        output_ready = 1;
        stress_mode = 0;
        lfsr_q = 32'h1ace_b00c;
        expected_read_ptr = 0;
        expected_write_ptr = 0;
        expected_count = 0;
        phase_count = 0;
        descriptor_count = 0;
        output_count = 0;
        bank_issue_count = 0;
        bank_response_count = 0;
        escape_count = 0;
        always_ready_ii_checks = 0;
        backpressure_cycles = 0;
        fifo_full_cycles = 0;
        previous_start_cycle = -1;
        previous_transaction_beats = 0;
        cycle_count = 0;
        stress_phase_count = 0;

        if (!$value$plusargs("RECORDS_BIN=%s", records_path)
                || !$value$plusargs("OFFSETS_BIN=%s", offsets_path)
                || !$value$plusargs("METADATA_BIN=%s", metadata_path))
            $fatal(1, "M86 missing input plusargs");
        records_fd = $fopen(records_path, "rb");
        offsets_fd = $fopen(offsets_path, "rb");
        metadata_fd = $fopen(metadata_path, "rb");
        if (!records_fd || !offsets_fd || !metadata_fd)
            $fatal(1, "M86 cannot open input binary");
        for (int index = 0; index <= PHASES; index++) begin
            offset_value = 0;
            for (int byte_index = 0; byte_index < 4; byte_index++) begin
                low = $fgetc(offsets_fd);
                if (low < 0) $fatal(1, "M86 truncated offsets");
                offset_value |= low << (8*byte_index);
            end
            offsets[index] = offset_value;
        end
        if ($fgetc(offsets_fd) != -1) $fatal(1, "M86 trailing offsets");
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        tag_value = 1;
        for (int phase = 0; phase < PHASES; phase++) begin
            stress_mode = phase >= 1600;
            if (stress_mode) stress_phase_count++;
            previous_start_cycle = -1;
            record_length = offsets[phase+1] - offsets[phase];
            if (record_length <= 0 || record_length > MAX_RECORD_BYTES)
                $fatal(1, "M86 bad record length=%0d", record_length);
            if ($fseek(records_fd, offsets[phase], 0) != 0)
                $fatal(1, "M86 record seek failed");
            bytes_read = $fread(record_bytes, records_fd, 0, record_length);
            if (bytes_read != record_length)
                $fatal(1, "M86 short record read");
            phase_metadata = '0;
            for (int byte_index = 0; byte_index < 74; byte_index++) begin
                low = $fgetc(metadata_fd);
                if (low < 0) $fatal(1, "M86 truncated metadata");
                phase_metadata[byte_index*8 +: 8] = low[7:0];
                if (byte_index < 48 && low[7:0] != record_bytes[byte_index])
                    $fatal(1, "M86 metadata/header mismatch");
            end
            phase_terminal = 0;
            for (int entry = 0; entry < 128; entry++) begin
                code = phase_metadata[entry*3 +: 3];
                phase_terminal += words_for_code(code);
            end
            if (phase_terminal <= 0 || phase_terminal > 3680)
                $fatal(1, "M86 invalid terminal=%0d", phase_terminal);
            for (int row = 0; row < ROWS; row++)
                drive_payload_row(row, phase_terminal);
            drive_phase_load();

            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                for (int block = 0; block < 8; block++) begin
                    code = phase_metadata[(pattern*8+block)*3 +: 3];
                    if (code == 4) begin
                        push_expected(tag_value, 12, 1'b1, '0);
                        escape_count++;
                    end else begin
                        width_value = 8 + code;
                        build_and_push_expected(cursor, width_value,
                                                phase_terminal, tag_value);
                        cursor += words_for_code(code);
                    end
                    drive_descriptor(pattern, block, tag_value);
                    tag_value++;
                end
            end
            #0.1 descriptor_valid = 0;
            while (busy || expected_count != 0) @(posedge clk_core);
            if (protocol_error)
                $fatal(1, "M86 legal phase ended faulted");
            phase_count++;
        end
        if ($fgetc(metadata_fd) != -1)
            $fatal(1, "M86 trailing metadata");
        $fclose(records_fd);
        $fclose(offsets_fd);
        $fclose(metadata_fd);

        // Fail-closed loader attack: a duplicate row is not a complete image.
        @(negedge clk_core); rst_core = 1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;
        stress_mode = 0;
        payload_load_row = 0;
        payload_load_words = '0;
        payload_load_valid = 1;
        @(posedge clk_core);
        #0.1 payload_load_valid = 0;
        @(negedge clk_core); payload_load_valid = 1;
        @(posedge clk_core);
        #1;
        if (!protocol_error || phase_load_ready || descriptor_ready)
            $fatal(1, "M86 duplicate-row loader attack did not fail closed");
        payload_load_valid = 0;
        // Leave one sampling edge for the protocol-error cover property.
        @(posedge clk_core); #1;

        if (phase_count != 1728 || descriptor_count != 221184
                || output_count != 221184 || escape_count != 1
                || bank_issue_count != 835383
                || bank_response_count != bank_issue_count
                || stress_phase_count != 128
                || always_ready_ii_checks != 1600*127
                || backpressure_cycles == 0 || fifo_full_cycles == 0)
            $fatal(1, "M86 coverage mismatch phase=%0d desc=%0d out=%0d escape=%0d issue=%0d response=%0d stress=%0d ii=%0d bp=%0d full=%0d",
                   phase_count, descriptor_count, output_count, escape_count,
                   bank_issue_count, bank_response_count, stress_phase_count,
                   always_ready_ii_checks, backpressure_cycles,
                   fifo_full_cycles);
        $display("PASS M86 sync-bank actual-record replay phases=1728 descriptors=221184 outputs=221184 beats=835383 always_ready_ii_checks=203200 stress_phases=128 backpressure_cycles=%0d fifo_full_cycles=%0d duplicate_row_attacks=1",
                 backpressure_cycles, fifo_full_cycles);
        $finish;
    end
endmodule

`default_nettype wire
