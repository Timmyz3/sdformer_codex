`timescale 1ns/1ps
`default_nettype none

module tb_m184_fc2_dual_window_k8_fixed_bank_frontend;
    localparam int MAX_EXPECTED_GROUPS = 30000;
    localparam int MAX_EXPECTED_TOKENS = 24;
    logic clk_core, rst_core;
    logic header_valid, header_ready, header_accept;
    logic [23:0] header_tag;
    logic [3:0] header_output_blocks;
    logic [5:0] header_descriptor_count;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [4:0] descriptor_beat_index;
    logic [95:0] descriptor_bitmap;
    logic group_valid, group_ready, group_accept;
    logic [23:0] group_tag;
    logic [2:0] group_output_block;
    logic [3:0] group_source_count;
    logic [7:0] group_bank_valid;
    logic [11:0] group_source_channel [0:7];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [23:0] token_done_tag;
    logic token_done_had_event;
    logic protocol_error, busy;

    logic random_group_stall_mode, random_done_stall_mode;
    logic scoreboard_enabled;
    logic [95:0] token_bitmap [0:31];
    logic [4:0] token_index [0:31];
    logic [23:0] expected_group_tag [0:MAX_EXPECTED_GROUPS-1];
    logic [2:0] expected_group_block [0:MAX_EXPECTED_GROUPS-1];
    logic [3:0] expected_group_count [0:MAX_EXPECTED_GROUPS-1];
    logic [7:0] expected_group_valid [0:MAX_EXPECTED_GROUPS-1];
    logic [11:0] expected_group_channel
        [0:MAX_EXPECTED_GROUPS-1][0:7];
    logic [23:0] expected_done_tag [0:MAX_EXPECTED_TOKENS-1];
    logic expected_done_had_event [0:MAX_EXPECTED_TOKENS-1];

    integer expected_group_write, expected_group_read;
    integer expected_done_write, expected_done_read;
    integer accepted_headers, accepted_descriptors;
    integer accepted_groups, accepted_done;
    integer input_events, unique_groups, unique_terms;
    integer replayed_terms_expected, replayed_terms_observed;
    integer source_histogram [1:8];
    integer descriptor_stalls, group_stalls;
    integer both_windows_closed_cycles;
    integer release_refill_hits, window_replace_hits;
    integer cross_descriptor_groups;
    integer cycle_count, previous_group_cycle, consecutive_group_hits;
    integer final_headers, final_descriptors, final_groups, final_done;
    integer final_input_events, final_unique_groups, final_unique_terms;
    integer final_replayed_terms, final_descriptor_stalls;
    integer final_group_stalls, final_both_closed;
    integer final_release_refill, final_window_replace;
    integer final_cross_descriptor, final_consecutive;
    integer final_source_histogram [1:8];

    m184_fc2_dual_window_k8_fixed_bank_frontend dut (.*);
    bind m184_fc2_dual_window_k8_fixed_bank_frontend
        m184_fc2_dual_window_k8_fixed_bank_frontend_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [95:0] make_bitmap(
            input integer pattern, input integer descriptor);
        logic [95:0] value;
        integer forced_bit;
        integer source_count;
        begin
            value = '0;
            if (pattern >= 101 && pattern <= 108) begin
                source_count = pattern - 100;
                for (int bank = 0; bank < source_count; bank++)
                    value[bank] = 1'b1;
            end else if (pattern == 1) begin
                case (descriptor)
                    0: value[0] = 1'b1;
                    1: value[1] = 1'b1;
                    2: begin value[0] = 1'b1; value[8] = 1'b1; end
                    default: value[2] = 1'b1;
                endcase
            end else if (pattern == 2) begin
                case (descriptor)
                    0: value[0] = 1'b1;
                    1: value[1] = 1'b1;
                    2: value[2] = 1'b1;
                    3: value[3] = 1'b1;
                    4: value[4] = 1'b1;
                    5: value[5] = 1'b1;
                    6: value[6] = 1'b1;
                    default: begin
                        value[7] = 1'b1;
                        value[15] = 1'b1;
                        value[23] = 1'b1;
                    end
                endcase
            end else begin
                for (int bit_index = 0; bit_index < 96; bit_index++) begin
                    if (((bit_index + descriptor*7 + pattern*3)
                            % (11 + pattern*2)) == 0)
                        value[bit_index] = 1'b1;
                end
                forced_bit = (descriptor*11 + pattern*5) % 96;
                value[forced_bit] = 1'b1;
                if ((descriptor % 5) == 0) begin
                    value[descriptor % 8] = 1'b1;
                    value[8 + (descriptor % 8)] = 1'b1;
                end
            end
            return value;
        end
    endfunction

    task automatic plan_window(
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer first_descriptor,
            input integer descriptor_count);
        logic [95:0] work [0:7];
        integer event_count;
        integer chosen_count;
        logic found;
        logic [7:0] chosen_valid;
        logic [11:0] chosen_channel [0:7];
        begin
            for (int entry = 0; entry < 8; entry++)
                work[entry] = entry < descriptor_count
                    ? token_bitmap[first_descriptor + entry] : 96'b0;
            event_count = 0;
            for (int entry = 0; entry < descriptor_count; entry++)
                for (int bit_index = 0; bit_index < 96; bit_index++)
                    event_count += work[entry][bit_index];
            input_events += event_count;
            while (event_count != 0) begin
                chosen_count = 0;
                chosen_valid = '0;
                for (int bank = 0; bank < 8; bank++) begin
                    chosen_channel[bank] = '0;
                    found = 1'b0;
                    for (int entry = 0;
                            entry < descriptor_count; entry++) begin
                        for (int row = 0; row < 12; row++) begin
                            if (!found && work[entry][(row*8)+bank]) begin
                                work[entry][(row*8)+bank] = 1'b0;
                                chosen_valid[bank] = 1'b1;
                                chosen_channel[bank]
                                    = (((token_index[
                                            first_descriptor + entry] * 12)
                                        + row) * 8) + bank;
                                found = 1'b1;
                                chosen_count++;
                                event_count--;
                            end
                        end
                    end
                end
                if (chosen_count == 0)
                    $fatal(1, "M184 scoreboard selector made no progress");
                unique_groups++;
                unique_terms += chosen_count;
                replayed_terms_expected
                    += chosen_count * output_blocks_value;
                for (int block = 0; block < output_blocks_value; block++) begin
                    if (expected_group_write >= MAX_EXPECTED_GROUPS)
                        $fatal(1, "M184 expected group overflow");
                    expected_group_tag[expected_group_write] = tag_value;
                    expected_group_block[expected_group_write] = block;
                    expected_group_count[expected_group_write]
                        = chosen_count;
                    expected_group_valid[expected_group_write]
                        = chosen_valid;
                    for (int bank = 0; bank < 8; bank++)
                        expected_group_channel[expected_group_write][bank]
                            = chosen_channel[bank];
                    expected_group_write++;
                end
            end
        end
    endtask

    task automatic plan_token(
            input integer pattern,
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer descriptor_count);
        integer window_size;
        integer count_here;
        begin
            case (output_blocks_value)
                1: window_size = 2;
                2: window_size = 4;
                default: window_size = 8;
            endcase
            for (int descriptor = 0;
                    descriptor < descriptor_count; descriptor++) begin
                token_index[descriptor] = descriptor[4:0];
                token_bitmap[descriptor] = make_bitmap(pattern, descriptor);
                if (token_bitmap[descriptor] == 0)
                    $fatal(1, "M184 generated an empty descriptor");
            end
            for (int first = 0; first < descriptor_count;
                    first += window_size) begin
                count_here = descriptor_count - first;
                if (count_here > window_size)
                    count_here = window_size;
                plan_window(tag_value, output_blocks_value,
                    first, count_here);
            end
            expected_done_tag[expected_done_write] = tag_value;
            expected_done_had_event[expected_done_write]
                = descriptor_count != 0;
            expected_done_write++;
        end
    endtask

    task automatic send_header(
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer descriptor_count);
        begin
            @(negedge clk_core);
            header_tag = tag_value;
            header_output_blocks = output_blocks_value;
            header_descriptor_count = descriptor_count;
            header_valid = 1'b1;
            do @(posedge clk_core); while (!header_accept);
            @(negedge clk_core);
            header_valid = 1'b0;
        end
    endtask

    task automatic send_descriptor_stream(input integer descriptor_count);
        integer descriptor;
        begin
            if (descriptor_count != 0) begin
                descriptor = 0;
                @(negedge clk_core);
                descriptor_beat_index = token_index[0];
                descriptor_bitmap = token_bitmap[0];
                descriptor_valid = 1'b1;
                while (descriptor < descriptor_count) begin
                    @(posedge clk_core);
                    if (descriptor_accept) begin
                        descriptor++;
                        if (descriptor < descriptor_count) begin
                            @(negedge clk_core);
                            descriptor_beat_index = token_index[descriptor];
                            descriptor_bitmap = token_bitmap[descriptor];
                        end
                    end
                end
                @(negedge clk_core);
                descriptor_valid = 1'b0;
            end
        end
    endtask

    task automatic drive_token(
            input integer pattern,
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer descriptor_count);
        begin
            plan_token(pattern, tag_value,
                output_blocks_value, descriptor_count);
            send_header(tag_value, output_blocks_value, descriptor_count);
            send_descriptor_stream(descriptor_count);
            wait (expected_done_read == expected_done_write);
            @(negedge clk_core);
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core)
            group_ready <= 1'b0;
        else if (random_group_stall_mode)
            group_ready <= ($urandom_range(0, 5) != 0);
        else
            group_ready <= 1'b1;
        if (rst_core)
            token_done_ready <= 1'b0;
        else if (random_done_stall_mode)
            token_done_ready <= ($urandom_range(0, 4) != 0);
        else
            token_done_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        logic cross_entry;
        if (rst_core) begin
            accepted_headers <= 0;
            accepted_descriptors <= 0;
            accepted_groups <= 0;
            accepted_done <= 0;
            replayed_terms_observed <= 0;
            descriptor_stalls <= 0;
            group_stalls <= 0;
            both_windows_closed_cycles <= 0;
            release_refill_hits <= 0;
            window_replace_hits <= 0;
            cross_descriptor_groups <= 0;
            consecutive_group_hits <= 0;
            previous_group_cycle <= -1;
            cycle_count <= 0;
            for (int count = 1; count <= 8; count++)
                source_histogram[count] <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (header_accept)
                accepted_headers <= accepted_headers + 1;
            if (descriptor_accept)
                accepted_descriptors <= accepted_descriptors + 1;
            if (descriptor_valid && !descriptor_ready)
                descriptor_stalls <= descriptor_stalls + 1;
            if (group_valid && !group_ready)
                group_stalls <= group_stalls + 1;
            if (dut.window_closed_q[0] && dut.window_closed_q[1])
                both_windows_closed_cycles <= both_windows_closed_cycles + 1;
            if (dut.current_window_release && descriptor_accept
                    && dut.fill_window_releasing)
                release_refill_hits <= release_refill_hits + 1;
            if (dut.current_window_release && dut.candidate_load)
                window_replace_hits <= window_replace_hits + 1;
            if (dut.candidate_load) begin
                cross_entry = 1'b0;
                for (int left = 0; left < 8; left++) begin
                    for (int right = left + 1; right < 8; right++) begin
                        if (dut.selected_valid[left]
                                && dut.selected_valid[right]
                                && dut.selected_entry[left]
                                    != dut.selected_entry[right])
                            cross_entry = 1'b1;
                    end
                end
                if (cross_entry)
                    cross_descriptor_groups <= cross_descriptor_groups + 1;
            end
            if (group_accept && scoreboard_enabled) begin
                if (expected_group_read >= expected_group_write)
                    $fatal(1, "M184 unexpected group");
                if (group_tag !== expected_group_tag[expected_group_read]
                        || group_output_block
                            !== expected_group_block[expected_group_read]
                        || group_source_count
                            !== expected_group_count[expected_group_read]
                        || group_bank_valid
                            !== expected_group_valid[expected_group_read])
                    $fatal(1, "M184 group header mismatch index=%0d",
                        expected_group_read);
                for (int bank = 0; bank < 8; bank++) begin
                    if (group_source_channel[bank]
                            !== expected_group_channel[
                                expected_group_read][bank])
                        $fatal(1,
                            "M184 group payload mismatch index=%0d bank=%0d got_channel=%0d expected_channel=%0d",
                            expected_group_read, bank,
                            group_source_channel[bank],
                            expected_group_channel[expected_group_read][bank]);
                end
                accepted_groups <= accepted_groups + 1;
                source_histogram[group_source_count]
                    <= source_histogram[group_source_count] + 1;
                replayed_terms_observed
                    <= replayed_terms_observed + group_source_count;
                if (previous_group_cycle >= 0
                        && cycle_count - previous_group_cycle == 1)
                    consecutive_group_hits <= consecutive_group_hits + 1;
                previous_group_cycle <= cycle_count;
                expected_group_read <= expected_group_read + 1;
            end
            if (token_done_accept && scoreboard_enabled) begin
                if (expected_done_read >= expected_done_write
                        || token_done_tag
                            !== expected_done_tag[expected_done_read]
                        || token_done_had_event
                            !== expected_done_had_event[expected_done_read])
                    $fatal(1, "M184 done mismatch index=%0d",
                        expected_done_read);
                accepted_done <= accepted_done + 1;
                expected_done_read <= expected_done_read + 1;
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        header_valid = 1'b0;
        header_tag = '0;
        header_output_blocks = 4'd1;
        header_descriptor_count = '0;
        descriptor_valid = 1'b0;
        descriptor_beat_index = '0;
        descriptor_bitmap = '0;
        group_ready = 1'b0;
        token_done_ready = 1'b0;
        random_group_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        scoreboard_enabled = 1'b1;
        expected_group_write = 0;
        expected_group_read = 0;
        expected_done_write = 0;
        expected_done_read = 0;
        input_events = 0;
        unique_groups = 0;
        unique_terms = 0;
        replayed_terms_expected = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        drive_token(0, 24'h900000, 1, 0);
        for (int count = 1; count <= 8; count++)
            drive_token(100 + count, 24'h910000 + count, 1, 1);
        drive_token(1, 24'h920000, 1, 4);
        random_group_stall_mode = 1'b1;
        random_done_stall_mode = 1'b1;
        drive_token(2, 24'h930000, 2, 8);
        drive_token(3, 24'h940000, 4, 16);
        drive_token(4, 24'h950000, 8, 24);
        random_group_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        wait (expected_group_read == expected_group_write);
        wait (expected_done_read == expected_done_write);
        wait (!busy);

        // Same-cycle zero-token header rearm.
        plan_token(0, 24'h960000, 1, 0);
        send_header(24'h960000, 1, 0);
        wait (token_done_valid);
        plan_token(0, 24'h970000, 1, 0);
        @(negedge clk_core);
        header_valid = 1'b1;
        header_tag = 24'h970000;
        header_output_blocks = 4'd1;
        header_descriptor_count = 0;
        @(posedge clk_core);
        if (!token_done_accept || !header_accept)
            $fatal(1, "M184 same-cycle header rearm missing");
        @(negedge clk_core);
        header_valid = 1'b0;
        wait (expected_done_read == expected_done_write);
        wait (!busy);

        if (accepted_headers != 15 || accepted_descriptors != 60
                || accepted_done != 15
                || accepted_groups != expected_group_write
                || expected_group_read != expected_group_write
                || unique_terms != input_events
                || replayed_terms_observed != replayed_terms_expected)
            $fatal(1, "M184 conservation mismatch headers=%0d desc=%0d done=%0d groups=%0d/%0d terms=%0d/%0d replay=%0d/%0d",
                accepted_headers, accepted_descriptors, accepted_done,
                accepted_groups, expected_group_write,
                unique_terms, input_events,
                replayed_terms_observed, replayed_terms_expected);
        for (int count = 1; count <= 8; count++) begin
            if (source_histogram[count] == 0)
                $fatal(1, "M184 missing source count=%0d", count);
        end
        if (descriptor_stalls == 0 || group_stalls == 0
                || both_windows_closed_cycles == 0
                || release_refill_hits == 0 || window_replace_hits == 0
                || cross_descriptor_groups == 0
                || consecutive_group_hits == 0)
            $fatal(1, "M184 coverage counters missing");

        final_headers = accepted_headers;
        final_descriptors = accepted_descriptors;
        final_groups = accepted_groups;
        final_done = accepted_done;
        final_input_events = input_events;
        final_unique_groups = unique_groups;
        final_unique_terms = unique_terms;
        final_replayed_terms = replayed_terms_observed;
        final_descriptor_stalls = descriptor_stalls;
        final_group_stalls = group_stalls;
        final_both_closed = both_windows_closed_cycles;
        final_release_refill = release_refill_hits;
        final_window_replace = window_replace_hits;
        final_cross_descriptor = cross_descriptor_groups;
        final_consecutive = consecutive_group_hits;
        for (int count = 1; count <= 8; count++)
            final_source_histogram[count] = source_histogram[count];
        scoreboard_enabled = 1'b0;

        // Attack 1: illegal output-block geometry.
        @(negedge clk_core);
        header_valid = 1'b1;
        header_output_blocks = 4'd3;
        header_descriptor_count = 0;
        @(posedge clk_core);
        @(negedge clk_core); header_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || header_ready || descriptor_ready)
            $fatal(1, "M184 malformed header fail-close missing");

        // Attack 2: descriptor count exceeds the stage0 raw-beat extent.
        @(negedge clk_core); rst_core = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        @(negedge clk_core);
        header_valid = 1'b1;
        header_output_blocks = 4'd1;
        header_descriptor_count = 6'd5;
        @(posedge clk_core);
        @(negedge clk_core); header_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || header_ready || descriptor_ready)
            $fatal(1, "M184 extent-overflow header fail-close missing");

        // Attack 3: zero payload in the nonzero descriptor stream.
        @(negedge clk_core); rst_core = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        send_header(24'hff0003, 1, 1);
        @(negedge clk_core);
        descriptor_valid = 1'b1;
        descriptor_beat_index = 0;
        descriptor_bitmap = 0;
        @(posedge clk_core);
        @(negedge clk_core); descriptor_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || header_ready || descriptor_ready)
            $fatal(1, "M184 zero descriptor fail-close missing");

        // Attack 4: backward index after one accepted descriptor.
        @(negedge clk_core); rst_core = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        send_header(24'hff0004, 2, 2);
        token_index[0] = 5'd2; token_bitmap[0] = 96'h1;
        send_descriptor_stream(1);
        @(negedge clk_core);
        descriptor_valid = 1'b1;
        descriptor_beat_index = 5'd1;
        descriptor_bitmap = 96'h2;
        @(posedge clk_core);
        @(negedge clk_core); descriptor_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || header_ready || descriptor_ready)
            $fatal(1, "M184 backward index fail-close missing");

        $display("PASS M184 FC2 dual-window K8 fixed-bank frontend VCS headers=%0d descriptors=%0d tokens=%0d bitmap_events=%0d unique_groups=%0d unique_source_terms=%0d replayed_group_results=%0d replayed_source_terms=%0d one_source_groups=%0d two_source_groups=%0d three_source_groups=%0d four_source_groups=%0d five_source_groups=%0d six_source_groups=%0d seven_source_groups=%0d eight_source_groups=%0d descriptor_stall_cycles=%0d group_stall_cycles=%0d both_windows_closed_cycles=%0d release_refill_hits=%0d window_replace_hits=%0d cross_descriptor_groups=%0d consecutive_group_hits=%0d stage_windows=2,4,8,8 max_two_buffer_bitmap_bits=1536 protocol_attacks=4 extent_overflow_attacks=1 same_cycle_header_rearm=1 global_topk_sort=false bank_id_payload=false fixed_bank_valid_mask=true token_directory=true native_or_preindexed_source_required=true posthoc_scanner_speedup=false weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            final_headers, final_descriptors, final_done,
            final_input_events, final_unique_groups, final_unique_terms,
            final_groups, final_replayed_terms,
            final_source_histogram[1], final_source_histogram[2],
            final_source_histogram[3], final_source_histogram[4],
            final_source_histogram[5], final_source_histogram[6],
            final_source_histogram[7], final_source_histogram[8],
            final_descriptor_stalls, final_group_stalls,
            final_both_closed, final_release_refill,
            final_window_replace, final_cross_descriptor,
            final_consecutive);
        $finish;
    end

    initial begin
        #3000000;
        $fatal(1, "M184 watchdog timeout");
    end
endmodule

`default_nettype wire
