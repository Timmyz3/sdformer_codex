`timescale 1ns/1ps
`default_nettype none

module tb_phase_fsm_sync_bank_actual_records_diff;
    localparam int PHASES = 1728;
    localparam int ROWS = 460;
    localparam int MAX_RECORD_BYTES = 14784;

    logic clk_core, rst_core;
    logic payload_load_valid, payload_load_ready, payload_load_accept;
    logic [9:0] payload_load_row;
    logic [255:0] payload_load_words;
    logic phase_load_valid, phase_load_ready, phase_load_accept;
    logic [591:0] phase_metadata;
    logic phase_loaded, metadata_error;
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
    logic payload_selected, phase_selected, descriptor_selected;
    logic [2:0] fsm_state;
    logic [8:0] accepted_rows, accepted_descriptors;

    logic ref_payload_ready, ref_payload_accept;
    logic ref_phase_ready, ref_phase_loaded, ref_metadata_error;
    logic ref_descriptor_ready, ref_descriptor_accept;
    logic ref_output_valid, ref_output_escape, ref_output_accept;
    logic [31:0] ref_output_tag;
    logic [3:0] ref_output_width;
    logic [1151:0] ref_output_values;
    logic ref_protocol_error, ref_busy, ref_bank_issue, ref_bank_response;
    logic [2:0] ref_bank_beat, ref_fifo_level;

    byte unsigned record_bytes [0:MAX_RECORD_BYTES-1];
    integer offsets [0:PHASES];
    integer records_fd, offsets_fd, metadata_fd;
    integer phase_count, descriptor_count, output_count;
    integer issue_count, response_count, escape_count;
    integer backpressure_cycles, fifo_full_cycles, cycle_count;
    integer stress_phase_count;
    logic stress_mode;
    logic [31:0] lfsr_q;
    string records_path, offsets_path, metadata_path;

    phase_fsm_sync_banked_guarded_pwp_frontend dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .payload_load_valid(payload_load_valid),
        .payload_load_ready(payload_load_ready),
        .payload_load_row(payload_load_row),
        .payload_load_words(payload_load_words),
        .payload_load_accept(payload_load_accept),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(phase_load_ready),
        .phase_metadata(phase_metadata),
        .phase_load_accept(phase_load_accept),
        .phase_loaded(phase_loaded), .metadata_error(metadata_error),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_pattern(descriptor_pattern),
        .descriptor_block(descriptor_block),
        .descriptor_tag(descriptor_tag),
        .descriptor_accept(descriptor_accept),
        .output_valid(output_valid), .output_ready(output_ready),
        .output_tag(output_tag), .output_width(output_width),
        .output_escape(output_escape), .output_values(output_values),
        .output_accept(output_accept), .protocol_error(protocol_error),
        .busy(busy), .bank_read_issue(bank_read_issue),
        .bank_read_beat(bank_read_beat),
        .bank_response_enqueue(bank_response_enqueue),
        .response_fifo_level(response_fifo_level),
        .payload_selected(payload_selected), .phase_selected(phase_selected),
        .descriptor_selected(descriptor_selected), .fsm_state(fsm_state),
        .accepted_rows(accepted_rows),
        .accepted_descriptors(accepted_descriptors)
    );
    phase_fsm_sync_banked_guarded_pwp_frontend_assertions r3_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .payload_load_valid(payload_load_valid),
        .payload_load_ready(payload_load_ready),
        .payload_load_accept(payload_load_accept),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(phase_load_ready),
        .phase_load_accept(phase_load_accept),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_accept(descriptor_accept),
        .output_valid(output_valid), .output_ready(output_ready),
        .output_accept(output_accept), .protocol_error(protocol_error),
        .busy(busy), .payload_selected(payload_selected),
        .phase_selected(phase_selected),
        .descriptor_selected(descriptor_selected), .fsm_state(fsm_state),
        .accepted_rows(accepted_rows),
        .accepted_descriptors(accepted_descriptors)
    );
    sync_banked_guarded_pwp_frontend reference_r1 (
        .clk_core(clk_core), .rst_core(rst_core),
        .payload_load_valid(payload_load_valid),
        .payload_load_ready(ref_payload_ready),
        .payload_load_row(payload_load_row),
        .payload_load_words(payload_load_words),
        .payload_load_accept(ref_payload_accept),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(ref_phase_ready),
        .phase_metadata(phase_metadata),
        .phase_loaded(ref_phase_loaded), .metadata_error(ref_metadata_error),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(ref_descriptor_ready),
        .descriptor_pattern(descriptor_pattern),
        .descriptor_block(descriptor_block),
        .descriptor_tag(descriptor_tag),
        .descriptor_accept(ref_descriptor_accept),
        .output_valid(ref_output_valid), .output_ready(output_ready),
        .output_tag(ref_output_tag), .output_width(ref_output_width),
        .output_escape(ref_output_escape),
        .output_values(ref_output_values),
        .output_accept(ref_output_accept),
        .protocol_error(ref_protocol_error), .busy(ref_busy),
        .bank_read_issue(ref_bank_issue),
        .bank_read_beat(ref_bank_beat),
        .bank_response_enqueue(ref_bank_response),
        .response_fifo_level(ref_fifo_level)
    );

    always #1.5 clk_core = ~clk_core;
    initial begin
        #18000000;
        $fatal(1, "M86-R3 actual-record watchdog phase=%0d desc=%0d out=%0d",
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

    task automatic drive_payload_row(
        input integer row_value, input integer terminal_words
    );
        begin
            @(negedge clk_core);
            payload_load_row = row_value;
            for (int bank = 0; bank < 8; bank++)
                payload_load_words[bank*32 +: 32] = payload_word(
                    row_value*8 + bank, terminal_words);
            payload_load_valid = 1;
            do @(posedge clk_core); while (!payload_load_ready);
            #0.1 payload_load_valid = 0;
        end
    endtask

    task automatic drive_phase_load;
        begin
            @(negedge clk_core); phase_load_valid = 1;
            do @(posedge clk_core); while (!phase_load_ready);
            #0.1 phase_load_valid = 0;
            #0.9;
            if (!phase_loaded || metadata_error || protocol_error)
                $fatal(1, "M86-R3 actual legal phase rejected phase=%0d",
                       phase_count);
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
            descriptor_valid = 1;
            do @(posedge clk_core); while (!descriptor_ready);
            #0.1 descriptor_valid = 0;
        end
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count++;
            if ((payload_load_valid
                        && payload_load_ready !== ref_payload_ready)
                    || payload_load_accept !== ref_payload_accept
                    || (phase_load_valid && phase_load_ready !== ref_phase_ready)
                    || phase_load_accept !== (phase_load_valid && ref_phase_ready)
                    || (descriptor_valid
                        && descriptor_ready !== ref_descriptor_ready)
                    || descriptor_accept !== ref_descriptor_accept)
                $fatal(1, "M86-R3/R1 request differential mismatch cycle=%0d state=%0d",
                       cycle_count, fsm_state);
            if (bank_read_issue !== ref_bank_issue
                    || (bank_read_issue && bank_read_beat !== ref_bank_beat)
                    || bank_response_enqueue !== ref_bank_response
                    || response_fifo_level !== ref_fifo_level)
                $fatal(1, "M86-R3/R1 bank differential mismatch cycle=%0d",
                       cycle_count);
            if (output_valid !== ref_output_valid
                    || output_accept !== ref_output_accept
                    || (output_valid && ({output_tag, output_width,
                                         output_escape, output_values}
                                      !== {ref_output_tag, ref_output_width,
                                          ref_output_escape,
                                          ref_output_values})))
                $fatal(1, "M86-R3/R1 output differential mismatch cycle=%0d",
                       cycle_count);
            if (protocol_error || metadata_error
                    || ref_protocol_error || ref_metadata_error)
                $fatal(1, "M86-R3/R1 legal replay fault phase=%0d", phase_count);
            if (descriptor_accept) descriptor_count++;
            if (bank_read_issue) issue_count++;
            if (bank_response_enqueue) response_count++;
            if (output_valid && !output_ready) backpressure_cycles++;
            if (response_fifo_level == 4) fifo_full_cycles++;
            if (output_accept) begin
                output_count++;
                if (output_escape) escape_count++;
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core) begin
            output_ready <= 1;
            lfsr_q <= 32'h8673_b00c;
        end else if (stress_mode) begin
            lfsr_q <= {lfsr_q[30:0],
                       lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            output_ready <= lfsr_q[0] | lfsr_q[3];
        end else begin
            output_ready <= 1;
        end
    end

    initial begin
        integer low, offset_value, record_length, bytes_read;
        integer code, tag_value, phase_terminal;
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
        phase_count = 0;
        descriptor_count = 0;
        output_count = 0;
        issue_count = 0;
        response_count = 0;
        escape_count = 0;
        backpressure_cycles = 0;
        fifo_full_cycles = 0;
        cycle_count = 0;
        stress_phase_count = 0;
        stress_mode = 0;
        lfsr_q = 32'h8673_b00c;

        if (!$value$plusargs("RECORDS_BIN=%s", records_path)
                || !$value$plusargs("OFFSETS_BIN=%s", offsets_path)
                || !$value$plusargs("METADATA_BIN=%s", metadata_path))
            $fatal(1, "M86-R3 missing actual-record inputs");
        records_fd = $fopen(records_path, "rb");
        offsets_fd = $fopen(offsets_path, "rb");
        metadata_fd = $fopen(metadata_path, "rb");
        if (!records_fd || !offsets_fd || !metadata_fd)
            $fatal(1, "M86-R3 cannot open actual-record inputs");
        for (int index = 0; index <= PHASES; index++) begin
            offset_value = 0;
            for (int byte_index = 0; byte_index < 4; byte_index++) begin
                low = $fgetc(offsets_fd);
                if (low < 0) $fatal(1, "M86-R3 truncated offsets");
                offset_value |= low << (8*byte_index);
            end
            offsets[index] = offset_value;
        end
        if ($fgetc(offsets_fd) != -1) $fatal(1, "M86-R3 trailing offsets");
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        tag_value = 1;
        for (int phase = 0; phase < PHASES; phase++) begin
            stress_mode = phase >= 1600;
            if (stress_mode) stress_phase_count++;
            record_length = offsets[phase+1] - offsets[phase];
            if (record_length <= 0 || record_length > MAX_RECORD_BYTES)
                $fatal(1, "M86-R3 record length=%0d", record_length);
            if ($fseek(records_fd, offsets[phase], 0) != 0)
                $fatal(1, "M86-R3 record seek failed");
            bytes_read = $fread(record_bytes, records_fd, 0, record_length);
            if (bytes_read != record_length)
                $fatal(1, "M86-R3 short record read");
            phase_metadata = '0;
            for (int byte_index = 0; byte_index < 74; byte_index++) begin
                low = $fgetc(metadata_fd);
                if (low < 0) $fatal(1, "M86-R3 truncated metadata");
                phase_metadata[byte_index*8 +: 8] = low[7:0];
                if (byte_index < 48 && low[7:0] != record_bytes[byte_index])
                    $fatal(1, "M86-R3 metadata/header mismatch");
            end
            phase_terminal = 0;
            for (int entry = 0; entry < 128; entry++) begin
                code = phase_metadata[entry*3 +: 3];
                phase_terminal += words_for_code(code);
            end
            if (phase_terminal <= 0 || phase_terminal > 3680)
                $fatal(1, "M86-R3 terminal=%0d", phase_terminal);
            for (int row = 0; row < ROWS; row++)
                drive_payload_row(row, phase_terminal);
            drive_phase_load();
            for (int pattern = 0; pattern < 16; pattern++) begin
                for (int block = 0; block < 8; block++) begin
                    drive_descriptor(pattern, block, tag_value);
                    tag_value++;
                end
            end
            while (busy || ref_busy) @(posedge clk_core);
            if (accepted_descriptors != 0 || fsm_state != 0)
                $fatal(1, "M86-R3 failed phase retirement phase=%0d state=%0d count=%0d",
                       phase, fsm_state, accepted_descriptors);
            phase_count++;
        end
        if ($fgetc(metadata_fd) != -1)
            $fatal(1, "M86-R3 trailing metadata");
        $fclose(records_fd);
        $fclose(offsets_fd);
        $fclose(metadata_fd);
        if (phase_count != PHASES || descriptor_count != PHASES*128
                || output_count != PHASES*128 || escape_count != 1
                || issue_count != 835383 || response_count != issue_count
                || stress_phase_count != 128 || backpressure_cycles == 0
                || fifo_full_cycles == 0)
            $fatal(1, "M86-R3 actual coverage mismatch phase=%0d desc=%0d out=%0d escape=%0d issue=%0d response=%0d stress=%0d bp=%0d full=%0d",
                   phase_count, descriptor_count, output_count, escape_count,
                   issue_count, response_count, stress_phase_count,
                   backpressure_cycles, fifo_full_cycles);
        $display("PASS M86-R3 actual-record differential phases=1728 descriptors=221184 outputs=221184 beats=835383 escape=1 stress_phases=128 backpressure_cycles=%0d fifo_full_cycles=%0d r1_cycle_mismatches=0", backpressure_cycles, fifo_full_cycles);
        $finish;
    end
endmodule

`default_nettype wire
