`timescale 1ns/1ps
`default_nettype none

module tb_m175_fc2_bitmap96_hierarchical_k4_replay_frontend;
    localparam int MAX_GROUP_RESULTS = 30000;
    localparam int MAX_TOKENS = 16;
    logic clk_core, rst_core;
    logic scan_valid, scan_ready, scan_last, scan_accept;
    logic [23:0] scan_tag;
    logic [3:0] scan_output_blocks;
    logic [8:0] scan_base_row;
    logic [95:0] scan_bitmap;
    logic group_valid, group_ready, group_accept;
    logic [23:0] group_tag;
    logic [2:0] group_output_block, group_source_count;
    logic [2:0] group_bank_id [0:3];
    logic [11:0] group_source_channel [0:3];
    logic token_done_valid, token_done_ready, token_done_had_event;
    logic token_done_accept;
    logic [23:0] token_done_tag;
    logic protocol_error, busy;

    logic random_group_stall_mode, random_done_stall_mode;
    integer expected_group_write, expected_group_read;
    integer expected_done_write, expected_done_read;
    integer accepted_scan_beats, accepted_group_results;
    integer accepted_done_tokens, unique_group_count;
    integer input_event_count, unique_source_terms;
    integer expected_replayed_source_terms, observed_replayed_source_terms;
    integer zero_scan_beats, output_stall_cycles, prefetch_accepts;
    integer consecutive_stream_hits, stage0_consecutive_hits;
    integer same_cycle_token_rearms;
    integer previous_group_cycle, previous_stage0_cycle, cycle_count;

    logic [23:0] expected_group_tag [0:MAX_GROUP_RESULTS-1];
    logic [2:0] expected_group_block [0:MAX_GROUP_RESULTS-1];
    logic [2:0] expected_group_count [0:MAX_GROUP_RESULTS-1];
    logic [2:0] expected_group_bank [0:MAX_GROUP_RESULTS-1][0:3];
    logic [11:0] expected_group_channel [0:MAX_GROUP_RESULTS-1][0:3];
    logic [23:0] expected_done_tag [0:MAX_TOKENS-1];
    logic expected_done_had_event [0:MAX_TOKENS-1];

    m175_fc2_bitmap96_hierarchical_k4_replay_frontend dut (.*);
    bind m175_fc2_bitmap96_hierarchical_k4_replay_frontend
        m175_fc2_bitmap96_hierarchical_k4_replay_frontend_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [95:0] make_bitmap(
            input integer token_index, input integer beat_index);
        logic [95:0] value;
        begin
            value = '0;
            if (token_index == 0) begin
                case (beat_index)
                    0: value = {96{1'b1}};
                    1: value = 96'h0808_0808_0808_0808_0808_0808;
                    default: value
                        = 96'h8000_0000_0000_0000_0000_000f;
                endcase
            end else if (token_index == 4) begin
                value = '0;
            end else begin
                for (int bit_index = 0; bit_index < 96; bit_index++) begin
                    if (((bit_index + 3*beat_index + 5*token_index) %
                            (13 + 2*token_index)) == 0)
                        value[bit_index] = 1'b1;
                    if (beat_index == token_index
                            && bit_index[2:0] == token_index[2:0])
                        value[bit_index] = 1'b1;
                end
                if (beat_index == 0)
                    value[7:0] = 8'hff;
                if ((beat_index % 7) == 4)
                    value = '0;
            end
            make_bitmap = value;
        end
    endfunction

    task automatic enqueue_groups(
            input logic [95:0] bitmap_value,
            input integer base_row_value,
            input integer output_blocks_value,
            input logic [23:0] tag_value);
        logic [95:0] work;
        logic found;
        integer selected, event_total;
        logic [2:0] selected_bank [0:3];
        logic [11:0] selected_channel [0:3];
        begin
            work = bitmap_value;
            event_total = 0;
            for (int bit_index = 0; bit_index < 96; bit_index++)
                if (bitmap_value[bit_index])
                    event_total = event_total + 1;
            input_event_count = input_event_count + event_total;
            while (work != 0) begin
                selected = 0;
                for (int slot = 0; slot < 4; slot++) begin
                    selected_bank[slot] = '0;
                    selected_channel[slot] = '0;
                end
                for (int bank = 0; bank < 8; bank++) begin
                    found = 1'b0;
                    for (int row = 0; row < 12; row++) begin
                        if (!found && selected < 4
                                && work[(row*8)+bank]) begin
                            selected_bank[selected] = bank[2:0];
                            selected_channel[selected]
                                = ((base_row_value + row) << 3) + bank;
                            work[(row*8)+bank] = 1'b0;
                            selected = selected + 1;
                            found = 1'b1;
                        end
                    end
                end
                if (selected == 0)
                    $fatal(1, "M175 scoreboard selector failed");
                unique_group_count = unique_group_count + 1;
                unique_source_terms = unique_source_terms + selected;
                expected_replayed_source_terms
                    = expected_replayed_source_terms
                        + selected * output_blocks_value;
                for (int block = 0; block < output_blocks_value; block++) begin
                    if (expected_group_write >= MAX_GROUP_RESULTS)
                        $fatal(1, "M175 expected group overflow");
                    expected_group_tag[expected_group_write] = tag_value;
                    expected_group_block[expected_group_write] = block;
                    expected_group_count[expected_group_write] = selected;
                    for (int slot = 0; slot < 4; slot++) begin
                        expected_group_bank[expected_group_write][slot]
                            = selected_bank[slot];
                        expected_group_channel[expected_group_write][slot]
                            = selected_channel[slot];
                    end
                    expected_group_write = expected_group_write + 1;
                end
            end
        end
    endtask

    task automatic drive_token(
            input integer token_index,
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer beat_count);
        logic [95:0] bitmap_value;
        logic had_event;
        begin
            had_event = 1'b0;
            expected_done_tag[expected_done_write] = tag_value;
            for (int beat = 0; beat < beat_count; beat++) begin
                bitmap_value = make_bitmap(token_index, beat);
                had_event |= |bitmap_value;
                enqueue_groups(bitmap_value, beat*12,
                    output_blocks_value, tag_value);
                @(negedge clk_core);
                scan_tag = tag_value;
                scan_output_blocks = output_blocks_value;
                scan_base_row = beat*12;
                scan_bitmap = bitmap_value;
                scan_last = beat == beat_count-1;
                scan_valid = 1'b1;
                do @(posedge clk_core); while (!scan_accept);
            end
            expected_done_had_event[expected_done_write] = had_event;
            expected_done_write = expected_done_write + 1;
            @(negedge clk_core);
            scan_valid = 1'b0;
            scan_last = 1'b0;
            wait (expected_done_read == expected_done_write);
            @(negedge clk_core);
        end
    endtask

    task automatic drive_same_cycle_rearm_pair;
        logic [95:0] first_bitmap;
        logic [95:0] second_bitmap;
        begin
            first_bitmap = 96'h1;
            second_bitmap = 96'h80;
            enqueue_groups(first_bitmap, 0, 1, 24'h560000);
            expected_done_tag[expected_done_write] = 24'h560000;
            expected_done_had_event[expected_done_write] = 1'b1;
            expected_done_write = expected_done_write + 1;
            @(negedge clk_core);
            scan_tag = 24'h560000;
            scan_output_blocks = 4'd1;
            scan_base_row = '0;
            scan_bitmap = first_bitmap;
            scan_last = 1'b1;
            scan_valid = 1'b1;
            do @(posedge clk_core); while (!scan_accept);
            @(negedge clk_core);
            scan_valid = 1'b0;
            wait (token_done_valid);

            enqueue_groups(second_bitmap, 0, 1, 24'h570000);
            expected_done_tag[expected_done_write] = 24'h570000;
            expected_done_had_event[expected_done_write] = 1'b1;
            expected_done_write = expected_done_write + 1;
            @(negedge clk_core);
            scan_tag = 24'h570000;
            scan_output_blocks = 4'd1;
            scan_base_row = '0;
            scan_bitmap = second_bitmap;
            scan_last = 1'b1;
            scan_valid = 1'b1;
            @(posedge clk_core);
            if (!token_done_accept || !scan_accept)
                $fatal(1, "M175 same-cycle token rearm missing");
            @(negedge clk_core);
            scan_valid = 1'b0;
            wait (expected_done_read == expected_done_write);
            @(negedge clk_core);
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core)
            group_ready <= 1'b0;
        else if (random_group_stall_mode)
            group_ready <= ($urandom_range(0, 4) != 0);
        else
            group_ready <= 1'b1;
        if (rst_core)
            token_done_ready <= 1'b0;
        else if (random_done_stall_mode)
            token_done_ready <= ($urandom_range(0, 3) != 0);
        else
            token_done_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            accepted_scan_beats <= 0;
            accepted_group_results <= 0;
            accepted_done_tokens <= 0;
            observed_replayed_source_terms <= 0;
            zero_scan_beats <= 0;
            output_stall_cycles <= 0;
            prefetch_accepts <= 0;
            consecutive_stream_hits <= 0;
            stage0_consecutive_hits <= 0;
            same_cycle_token_rearms <= 0;
            previous_group_cycle <= -1;
            previous_stage0_cycle <= -1;
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (scan_accept) begin
                accepted_scan_beats <= accepted_scan_beats + 1;
                if (scan_bitmap == 0)
                    zero_scan_beats <= zero_scan_beats + 1;
                if (group_valid && !dut.group_final_accept)
                    prefetch_accepts <= prefetch_accepts + 1;
            end
            if (group_valid && !group_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (group_accept) begin
                if (expected_group_read >= expected_group_write)
                    $fatal(1, "M175 unexpected group");
                if (group_tag !== expected_group_tag[expected_group_read]
                        || group_output_block
                            !== expected_group_block[expected_group_read]
                        || group_source_count
                            !== expected_group_count[expected_group_read])
                    $fatal(1, "M175 group header mismatch index=%0d",
                        expected_group_read);
                for (int slot = 0; slot < 4; slot++) begin
                    if (group_bank_id[slot]
                                !== expected_group_bank[expected_group_read][slot]
                            || group_source_channel[slot]
                                !== expected_group_channel[expected_group_read][slot])
                        $fatal(1,
                            "M175 descriptor mismatch index=%0d slot=%0d",
                            expected_group_read, slot);
                end
                accepted_group_results <= accepted_group_results + 1;
                observed_replayed_source_terms
                    <= observed_replayed_source_terms + group_source_count;
                if (previous_group_cycle >= 0
                        && cycle_count - previous_group_cycle == 1)
                    consecutive_stream_hits <= consecutive_stream_hits + 1;
                previous_group_cycle <= cycle_count;
                if (group_tag == 24'h510000) begin
                    if (previous_stage0_cycle >= 0
                            && cycle_count - previous_stage0_cycle == 1)
                        stage0_consecutive_hits
                            <= stage0_consecutive_hits + 1;
                    previous_stage0_cycle <= cycle_count;
                end
                expected_group_read <= expected_group_read + 1;
            end
            if (token_done_accept) begin
                if (expected_done_read >= expected_done_write
                        || token_done_tag
                            !== expected_done_tag[expected_done_read]
                        || token_done_had_event
                            !== expected_done_had_event[expected_done_read])
                    $fatal(1, "M175 token done mismatch index=%0d",
                        expected_done_read);
                accepted_done_tokens <= accepted_done_tokens + 1;
                expected_done_read <= expected_done_read + 1;
            end
            if (token_done_accept && scan_accept)
                same_cycle_token_rearms <= same_cycle_token_rearms + 1;
        end
    end

    initial begin
        rst_core = 1'b1;
        scan_valid = 1'b0;
        scan_tag = '0;
        scan_output_blocks = 4'd1;
        scan_base_row = '0;
        scan_bitmap = '0;
        scan_last = 1'b0;
        group_ready = 1'b0;
        token_done_ready = 1'b0;
        random_group_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        expected_group_write = 0;
        expected_group_read = 0;
        expected_done_write = 0;
        expected_done_read = 0;
        unique_group_count = 0;
        input_event_count = 0;
        unique_source_terms = 0;
        expected_replayed_source_terms = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        drive_token(0, 24'h510000, 1, 3);
        random_group_stall_mode = 1'b1;
        random_done_stall_mode = 1'b1;
        drive_token(1, 24'h520000, 2, 6);
        drive_token(2, 24'h530000, 4, 12);
        drive_token(3, 24'h540000, 8, 24);
        drive_token(4, 24'h550000, 1, 3);
        random_group_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        wait (expected_group_read == expected_group_write);
        wait (expected_done_read == expected_done_write);
        wait (!busy);
        drive_same_cycle_rearm_pair();
        wait (expected_group_read == expected_group_write);
        wait (expected_done_read == expected_done_write);
        wait (!busy);

        @(negedge clk_core);
        scan_valid = 1'b1;
        scan_tag = 24'hffffff;
        scan_output_blocks = 4'd1;
        scan_base_row = 9'd12;
        scan_bitmap = 96'h1;
        scan_last = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        scan_valid = 1'b0;
        repeat (2) @(posedge clk_core);

        if (!protocol_error || scan_ready)
            $fatal(1, "M175 protocol fail-close missing");
        if (accepted_scan_beats != 50 || accepted_done_tokens != 7
                || accepted_group_results != expected_group_write
                || expected_group_read != expected_group_write
                || unique_source_terms != input_event_count
                || observed_replayed_source_terms
                    != expected_replayed_source_terms)
            $fatal(1, "M175 conservation mismatch scan=%0d done=%0d groups=%0d/%0d events=%0d unique=%0d replay=%0d/%0d",
                accepted_scan_beats, accepted_done_tokens,
                accepted_group_results, expected_group_write,
                input_event_count, unique_source_terms,
                observed_replayed_source_terms,
                expected_replayed_source_terms);
        if (stage0_consecutive_hits < 23 || consecutive_stream_hits == 0
                || output_stall_cycles == 0 || prefetch_accepts == 0
                || zero_scan_beats == 0 || same_cycle_token_rearms != 1)
            $fatal(1, "M175 coverage counters missing");

        $display("PASS M175 FC2 bitmap96 hierarchical K4 replay frontend VCS scan_beats=%0d tokens=7 bitmap_events=%0d unique_groups=%0d unique_source_terms=%0d replayed_group_results=%0d replayed_source_terms=%0d zero_scan_beats=%0d zero_tokens=1 output_stall_cycles=%0d prefetch_accepts=%0d stage0_consecutive_group_hits=%0d consecutive_group_stream_hits=%0d same_cycle_token_rearms=%0d output_block_extents=1,2,4,8 protocol_attacks=1 scan_width_bits=96 max_sources_per_group=4 shared_hierarchical_selector=true source_group_held_across_output_blocks=true one_raw_beat_prefetch=true weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            accepted_scan_beats, input_event_count, unique_group_count,
            unique_source_terms, accepted_group_results,
            observed_replayed_source_terms, zero_scan_beats,
            output_stall_cycles, prefetch_accepts,
            stage0_consecutive_hits, consecutive_stream_hits,
            same_cycle_token_rearms);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "M175 watchdog timeout");
    end
endmodule

`default_nettype wire
