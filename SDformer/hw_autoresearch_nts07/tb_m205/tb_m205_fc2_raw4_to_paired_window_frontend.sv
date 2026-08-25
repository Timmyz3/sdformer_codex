`timescale 1ns/1ps
`default_nettype none

module tb_m205_fc2_raw4_to_paired_window_frontend;
    logic clk_core = 0, rst_core;
    always #1.5 clk_core = ~clk_core;

    logic header_valid, header_ready, header_accept;
    logic [23:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid, raw_ready, raw_accept;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic raw_last;
    logic group_valid, group_ready, group_accept;
    logic [23:0] group_tag;
    logic [2:0] group_output_block;
    logic [3:0] group_source_count;
    logic [7:0] group_bank_valid;
    logic [11:0] group_source_channel [0:7];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [23:0] token_done_tag;
    logic [5:0] token_done_descriptor_count;
    logic token_done_had_event, protocol_error, busy;

    logic [95:0] raw_payload [0:31];
    logic [95:0] model_bitmap [0:31];
    logic [4:0] model_index [0:31];
    logic [7:0] expected_mask [0:8191];
    logic [11:0] expected_channel [0:8191][0:7];
    logic [2:0] expected_block [0:8191];
    int expected_write, expected_read, token_expected_end;
    int active_descriptor_count, active_output_blocks;
    logic [23:0] active_tag;
    int legal_headers, accepted_raw_packets, accepted_groups, accepted_done;
    int group_stalls, raw_backpressure_cycles, protocol_attacks;
    int descriptor4_observations, paired_observations;
    int cycle_counter, forced_group_stall;
    int token_expected_start;
    time accepted_header_time;

    m205_fc2_raw4_to_paired_window_frontend dut (.*);

    function automatic logic [95:0] event_pattern(
        input int beat, input int seed
    );
        logic [95:0] value;
        int bank0, bank1, bank2, row0, row1, row2;
        begin
            value = 0;
            bank0 = (beat + seed) % 8;
            bank1 = (beat * 5 + seed + 3) % 8;
            bank2 = (beat * 3 + seed + 6) % 8;
            row0 = (beat * 3 + seed + 1) % 12;
            row1 = (beat * 7 + seed + 2) % 12;
            row2 = (beat * 11 + seed + 5) % 12;
            value[row0*8+bank0] = 1;
            value[row1*8+bank1] = 1;
            if ((beat + seed) % 3 == 0)
                value[row2*8+bank2] = 1;
            return value;
        end
    endfunction

    function automatic int popcount8(input logic [7:0] value);
        int count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 8; bit_index++)
                count += value[bit_index];
            return count;
        end
    endfunction

    task automatic derive_shape(
        input int blocks, output int raw_count, output int depth
    );
        begin
            case (blocks)
                1: begin raw_count = 4; depth = 2; end
                2: begin raw_count = 8; depth = 4; end
                4: begin raw_count = 16; depth = 8; end
                8: begin raw_count = 32; depth = 8; end
                default: $fatal(1, "bad test shape");
            endcase
        end
    endtask

    task automatic build_payload_and_reference(
        input int blocks, input int mode, input int seed
    );
        int raw_count, depth, descriptor_count;
        int start_entry, stop_entry, pair_capacity;
        logic found, any_event;
        logic [7:0] selected_mask;
        logic [11:0] selected_channel [0:7];
        begin
            derive_shape(blocks, raw_count, depth);
            descriptor_count = 0;
            for (int beat = 0; beat < raw_count; beat++) begin
                case (mode)
                    0: raw_payload[beat] = event_pattern(beat, seed);
                    1: raw_payload[beat] = ((beat + seed) % 3 == 0)
                        ? 0 : event_pattern(beat, seed);
                    2: raw_payload[beat] = 0;
                    default: raw_payload[beat] = ((beat + seed) % 2 == 0)
                        ? event_pattern(beat, seed) : 0;
                endcase
                if (raw_payload[beat] != 0) begin
                    model_bitmap[descriptor_count] = raw_payload[beat];
                    model_index[descriptor_count] = beat;
                    descriptor_count++;
                end
            end
            for (int beat = raw_count; beat < 32; beat++) raw_payload[beat] = 0;

            active_descriptor_count = descriptor_count;
            active_output_blocks = blocks;
            start_entry = 0;
            pair_capacity = blocks == 1 ? depth : 2 * depth;
            while (start_entry < descriptor_count) begin
                stop_entry = start_entry + pair_capacity;
                if (stop_entry > descriptor_count) stop_entry = descriptor_count;
                any_event = 1;
                while (any_event) begin
                    selected_mask = 0;
                    for (int bank = 0; bank < 8; bank++) begin
                        found = 0; selected_channel[bank] = 0;
                        for (int entry = start_entry;
                                entry < stop_entry; entry++) begin
                            for (int row = 0; row < 12; row++) begin
                                if (!found
                                        && model_bitmap[entry][row*8+bank]) begin
                                    selected_mask[bank] = 1;
                                    selected_channel[bank]
                                        = (model_index[entry]*12+row)*8+bank;
                                    model_bitmap[entry][row*8+bank] = 0;
                                    found = 1;
                                end
                            end
                        end
                    end
                    any_event = selected_mask != 0;
                    if (any_event) begin
                        for (int block = 0; block < blocks; block++) begin
                            expected_mask[expected_write] = selected_mask;
                            expected_block[expected_write] = block;
                            for (int bank = 0; bank < 8; bank++)
                                expected_channel[expected_write][bank]
                                    = selected_channel[bank];
                            expected_write++;
                        end
                    end
                end
                start_entry = stop_entry;
            end
            token_expected_end = expected_write;
        end
    endtask

    task automatic apply_reset;
        begin
            @(negedge clk_core);
            rst_core = 1; header_valid = 0; raw_valid = 0;
            raw_lane_valid = 0; raw_last = 0; forced_group_stall = 0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core); rst_core = 0;
        end
    endtask

    task automatic send_header(
        input logic [23:0] tag, input int blocks
    );
        int raw_count, depth;
        begin
            derive_shape(blocks, raw_count, depth);
            @(negedge clk_core);
            header_tag = tag; header_raw_beat_count = raw_count;
            header_window_depth = depth; header_output_blocks = blocks;
            header_valid = 1;
            do @(posedge clk_core); while (!header_accept);
            accepted_header_time = $time;
            legal_headers++;
            @(negedge clk_core); header_valid = 0;
        end
    endtask

    task automatic send_legal_token(
        input logic [23:0] tag, input int blocks,
        input int mode, input int seed, input int stall_cycles
    );
        int raw_count, depth;
        begin
            token_expected_start = expected_write;
            build_payload_and_reference(blocks, mode, seed);
            active_tag = tag;
            derive_shape(blocks, raw_count, depth);
            send_header(tag, blocks);
            forced_group_stall = stall_cycles;
            for (int base = 0; base < raw_count; base += 4) begin
                @(negedge clk_core);
                raw_lane_valid = 4'b1111;
                for (int lane = 0; lane < 4; lane++) begin
                    raw_beat_index[lane] = base + lane;
                    raw_bitmap[lane] = raw_payload[base + lane];
                end
                raw_last = base + 4 == raw_count;
                raw_valid = 1;
                do @(posedge clk_core); while (!raw_accept);
                accepted_raw_packets++;
                @(negedge clk_core); raw_valid = 0; raw_last = 0;
            end
            do @(posedge clk_core); while (!token_done_accept);
            $display("M205 TOKEN tag=%h blocks=%0d descriptors=%0d groups=%0d header_to_done_cycles=%0d",
                tag, blocks, active_descriptor_count,
                token_expected_end - token_expected_start,
                ($time - accepted_header_time) / 3);
            @(negedge clk_core);
            if (expected_read != token_expected_end)
                $fatal(1, "M205 token completed before expected groups read=%0d end=%0d",
                    expected_read, token_expected_end);
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            cycle_counter = 0; group_ready = 0; token_done_ready = 0;
        end else begin
            cycle_counter++;
            if (forced_group_stall > 0) begin
                group_ready = 0; forced_group_stall--;
            end else begin
                group_ready = (cycle_counter % 7 != 0)
                    && (cycle_counter % 11 != 0);
            end
            token_done_ready = cycle_counter % 5 != 0;
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (raw_valid && !raw_ready) raw_backpressure_cycles++;
            if (group_valid && !group_ready) group_stalls++;
            if (dut.descriptor_accept && dut.descriptor_count == 4)
                descriptor4_observations++;
            if (dut.paired_sink.pair_has_two) paired_observations++;
            if (group_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M205 unexpected group");
                if (group_tag !== active_tag
                        || group_output_block !== expected_block[expected_read]
                        || group_bank_valid !== expected_mask[expected_read]
                        || group_source_count
                            != popcount8(expected_mask[expected_read]))
                    $fatal(1, "M205 group metadata mismatch index=%0d",
                        expected_read);
                for (int bank = 0; bank < 8; bank++)
                    if (expected_mask[expected_read][bank]
                            && group_source_channel[bank]
                                !== expected_channel[expected_read][bank])
                        $fatal(1, "M205 group channel mismatch index=%0d bank=%0d",
                            expected_read, bank);
                expected_read++; accepted_groups++;
            end
            if (token_done_accept) begin
                if (token_done_tag !== active_tag
                        || token_done_descriptor_count
                            != active_descriptor_count
                        || token_done_had_event
                            !== (active_descriptor_count != 0))
                    $fatal(1, "M205 token done metadata mismatch");
                accepted_done++;
            end
        end
    end

    initial begin
        rst_core = 1; header_valid = 0; raw_valid = 0;
        raw_lane_valid = 0; raw_last = 0; group_ready = 0;
        token_done_ready = 0; expected_write = 0; expected_read = 0;
        legal_headers = 0; accepted_raw_packets = 0; accepted_groups = 0;
        accepted_done = 0; group_stalls = 0; raw_backpressure_cycles = 0;
        protocol_attacks = 0; descriptor4_observations = 0;
        paired_observations = 0; cycle_counter = 0; forced_group_stall = 0;
        repeat (3) @(posedge clk_core); @(negedge clk_core); rst_core = 0;

        send_legal_token(24'h205001, 2, 0, 1, 0);
        send_legal_token(24'h205002, 1, 1, 2, 0);
        send_legal_token(24'h205003, 4, 1, 3, 8);
        send_legal_token(24'h205004, 8, 0, 4, 40);
        send_legal_token(24'h205005, 2, 2, 5, 0);

        apply_reset(); @(negedge clk_core);
        header_tag = 24'hbad501; header_raw_beat_count = 8;
        header_window_depth = 8; header_output_blocks = 2;
        header_valid = 1; @(posedge clk_core);
        if (!protocol_error) $fatal(1, "M205 bad composite header missed");
        protocol_attacks++;
        @(negedge clk_core); header_valid = 0; @(posedge clk_core);

        apply_reset(); active_tag = 24'hbad502;
        send_header(active_tag, 2); @(negedge clk_core);
        raw_lane_valid = 4'b1011; raw_beat_index[0] = 0;
        raw_beat_index[1] = 1; raw_beat_index[2] = 2;
        raw_beat_index[3] = 3; raw_last = 0;
        for (int lane = 0; lane < 4; lane++)
            raw_bitmap[lane] = event_pattern(lane, 9);
        raw_valid = 1; @(posedge clk_core);
        if (!protocol_error) $fatal(1, "M205 bad raw prefix missed");
        protocol_attacks++;
        @(negedge clk_core); raw_valid = 0; @(posedge clk_core);

        if (descriptor4_observations == 0 || paired_observations == 0
                || raw_backpressure_cycles == 0 || group_stalls == 0)
            $fatal(1, "M205 required stress coverage missing");
        $display("PASS M205 M202-to-M204 cycle co-sim VCS legal_headers=%0d raw_packets=%0d groups=%0d done=%0d descriptor4=%0d paired=%0d group_stalls=%0d raw_backpressure=%0d protocol_attacks=%0d duplicate_storage=true complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            legal_headers, accepted_raw_packets, accepted_groups, accepted_done,
            descriptor4_observations, paired_observations, group_stalls,
            raw_backpressure_cycles, protocol_attacks);
        $finish;
    end
    initial begin #10000000; $fatal(1, "M205 watchdog timeout"); end
endmodule

`default_nettype wire
