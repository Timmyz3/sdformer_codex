`timescale 1ns/1ps
`default_nettype none

module tb_m177_fc2_indexed_nonzero96_k4_replay_frontend;
    localparam int MAX_GROUP_RESULTS = 30000;
    localparam int MAX_TOKENS = 16;
    logic clk_core, rst_core;
    logic scan_valid, scan_ready, scan_eot, scan_accept;
    logic [23:0] scan_tag;
    logic [3:0] scan_output_blocks;
    logic [4:0] scan_beat_index;
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
    logic force_done_stall_mode, scoreboard_enabled;
    integer expected_group_write, expected_group_read;
    integer expected_done_write, expected_done_read;
    integer accepted_descriptors, accepted_payload_descriptors;
    integer accepted_eot_descriptors, accepted_group_results;
    integer accepted_done_tokens, unique_group_count;
    integer input_event_count, unique_source_terms;
    integer expected_replayed_source_terms, observed_replayed_source_terms;
    integer raw_beat_opportunities, zero_beats_elided;
    integer output_stall_cycles, prefetch_accepts;
    integer consecutive_stream_hits, stage0_consecutive_hits;
    integer indexed_gap_accepts, eot_with_pending_accepts;
    integer same_cycle_token_rearms;
    integer previous_group_cycle, previous_stage0_cycle, cycle_count;
    integer final_accepted_descriptors, final_payload_descriptors;
    integer final_eot_descriptors, final_group_results;
    integer final_done_tokens, final_output_stalls, final_prefetch_accepts;
    integer final_indexed_gaps, final_eot_pending;
    integer final_stage0_hits, final_stream_hits, final_token_rearms;
    integer final_bitmap_events, final_unique_groups;
    integer final_unique_source_terms, final_replayed_source_terms;
    integer final_raw_beats, final_zero_beats;

    logic [23:0] expected_group_tag [0:MAX_GROUP_RESULTS-1];
    logic [2:0] expected_group_block [0:MAX_GROUP_RESULTS-1];
    logic [2:0] expected_group_count [0:MAX_GROUP_RESULTS-1];
    logic [2:0] expected_group_bank [0:MAX_GROUP_RESULTS-1][0:3];
    logic [11:0] expected_group_channel [0:MAX_GROUP_RESULTS-1][0:3];
    logic [23:0] expected_done_tag [0:MAX_TOKENS-1];
    logic expected_done_had_event [0:MAX_TOKENS-1];

    m177_fc2_indexed_nonzero96_k4_replay_frontend dut (.*);
    bind m177_fc2_indexed_nonzero96_k4_replay_frontend
        m177_fc2_indexed_nonzero96_k4_replay_frontend_assertions sva (.*);

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
                    1: value = '0;
                    2: value = 96'h0000_0000_0000_0000_0000_00ff;
                    default: value = 96'h8000_0000_0000_0000_0000_0001;
                endcase
            end else if (token_index == 4) begin
                value = '0;
            end else begin
                for (int bit_index = 0; bit_index < 96; bit_index++) begin
                    if (((bit_index + 5*beat_index + 3*token_index) %
                            (11 + 2*token_index)) == 0)
                        value[bit_index] = 1'b1;
                    if (beat_index == token_index
                            && bit_index[2:0] == token_index[2:0])
                        value[bit_index] = 1'b1;
                end
                if (beat_index == 0)
                    value[7:0] = 8'hff;
                if (token_index == 1 && beat_index == 0)
                    value = 96'hf;
                if ((beat_index % 3) == 1 || (beat_index % 11) == 7)
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
                    $fatal(1, "M177 scoreboard selector failed");
                unique_group_count = unique_group_count + 1;
                unique_source_terms = unique_source_terms + selected;
                expected_replayed_source_terms
                    = expected_replayed_source_terms
                        + selected * output_blocks_value;
                for (int block = 0; block < output_blocks_value; block++) begin
                    if (expected_group_write >= MAX_GROUP_RESULTS)
                        $fatal(1, "M177 expected group overflow");
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

    task automatic send_descriptor(
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer beat_index_value,
            input logic [95:0] bitmap_value,
            input logic eot_value,
            input logic drop_valid_after);
        begin
            @(negedge clk_core);
            scan_tag = tag_value;
            scan_output_blocks = output_blocks_value;
            scan_beat_index = beat_index_value;
            scan_bitmap = bitmap_value;
            scan_eot = eot_value;
            scan_valid = 1'b1;
            do @(posedge clk_core); while (!scan_accept);
            if (drop_valid_after) begin
                @(negedge clk_core);
                scan_valid = 1'b0;
                scan_eot = 1'b0;
            end
        end
    endtask

    task automatic drive_token(
            input integer token_index,
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer raw_beat_count);
        logic [95:0] bitmap_value;
        logic had_event;
        begin
            had_event = 1'b0;
            expected_done_tag[expected_done_write] = tag_value;
            for (int beat = 0; beat < raw_beat_count; beat++) begin
                bitmap_value = make_bitmap(token_index, beat);
                raw_beat_opportunities = raw_beat_opportunities + 1;
                if (bitmap_value == 0) begin
                    zero_beats_elided = zero_beats_elided + 1;
                end else begin
                    had_event = 1'b1;
                    enqueue_groups(bitmap_value, beat*12,
                        output_blocks_value, tag_value);
                    send_descriptor(tag_value, output_blocks_value,
                        beat, bitmap_value, 1'b0, 1'b0);
                end
            end
            expected_done_had_event[expected_done_write] = had_event;
            expected_done_write = expected_done_write + 1;
            send_descriptor(tag_value, output_blocks_value,
                0, 96'b0, 1'b1, 1'b1);
            wait (expected_done_read == expected_done_write);
            @(negedge clk_core);
        end
    endtask

    task automatic drive_same_cycle_rearm_pair;
        begin
            raw_beat_opportunities = raw_beat_opportunities + 2;
            enqueue_groups(96'h1, 0, 1, 24'h660000);
            expected_done_tag[expected_done_write] = 24'h660000;
            expected_done_had_event[expected_done_write] = 1'b1;
            expected_done_write = expected_done_write + 1;
            send_descriptor(24'h660000, 1, 0, 96'h1, 1'b0, 1'b0);
            send_descriptor(24'h660000, 1, 0, 96'b0, 1'b1, 1'b1);
            wait (token_done_valid);

            enqueue_groups(96'h80, 24, 1, 24'h670000);
            expected_done_tag[expected_done_write] = 24'h670000;
            expected_done_had_event[expected_done_write] = 1'b1;
            expected_done_write = expected_done_write + 1;
            @(negedge clk_core);
            scan_tag = 24'h670000;
            scan_output_blocks = 4'd1;
            scan_beat_index = 5'd2;
            scan_bitmap = 96'h80;
            scan_eot = 1'b0;
            scan_valid = 1'b1;
            @(posedge clk_core);
            if (!token_done_accept || !scan_accept)
                $fatal(1, "M177 same-cycle token rearm missing");
            @(negedge clk_core);
            scan_valid = 1'b0;
            send_descriptor(24'h670000, 1, 0, 96'b0, 1'b1, 1'b1);
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
        if (rst_core || force_done_stall_mode)
            token_done_ready <= 1'b0;
        else if (random_done_stall_mode)
            token_done_ready <= ($urandom_range(0, 3) != 0);
        else
            token_done_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            accepted_descriptors <= 0;
            accepted_payload_descriptors <= 0;
            accepted_eot_descriptors <= 0;
            accepted_group_results <= 0;
            accepted_done_tokens <= 0;
            observed_replayed_source_terms <= 0;
            output_stall_cycles <= 0;
            prefetch_accepts <= 0;
            consecutive_stream_hits <= 0;
            stage0_consecutive_hits <= 0;
            indexed_gap_accepts <= 0;
            eot_with_pending_accepts <= 0;
            same_cycle_token_rearms <= 0;
            previous_group_cycle <= -1;
            previous_stage0_cycle <= -1;
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (scan_accept) begin
                accepted_descriptors <= accepted_descriptors + 1;
                if (scan_eot) begin
                    accepted_eot_descriptors
                        <= accepted_eot_descriptors + 1;
                    if (group_valid || dut.residual_valid_q)
                        eot_with_pending_accepts
                            <= eot_with_pending_accepts + 1;
                end else begin
                    accepted_payload_descriptors
                        <= accepted_payload_descriptors + 1;
                    if (dut.token_has_base_q
                            && scan_beat_index > dut.last_beat_index_q + 1)
                        indexed_gap_accepts <= indexed_gap_accepts + 1;
                    if (group_valid && !dut.group_final_accept)
                        prefetch_accepts <= prefetch_accepts + 1;
                end
            end
            if (group_valid && !group_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (group_accept && scoreboard_enabled) begin
                if (expected_group_read >= expected_group_write)
                    $fatal(1, "M177 unexpected group");
                if (group_tag !== expected_group_tag[expected_group_read]
                        || group_output_block
                            !== expected_group_block[expected_group_read]
                        || group_source_count
                            !== expected_group_count[expected_group_read])
                    $fatal(1, "M177 group header mismatch index=%0d",
                        expected_group_read);
                for (int slot = 0; slot < 4; slot++) begin
                    if (group_bank_id[slot]
                                !== expected_group_bank[expected_group_read][slot]
                            || group_source_channel[slot]
                                !== expected_group_channel[expected_group_read][slot])
                        $fatal(1,
                            "M177 descriptor mismatch index=%0d slot=%0d",
                            expected_group_read, slot);
                end
                accepted_group_results <= accepted_group_results + 1;
                observed_replayed_source_terms
                    <= observed_replayed_source_terms + group_source_count;
                if (previous_group_cycle >= 0
                        && cycle_count - previous_group_cycle == 1)
                    consecutive_stream_hits <= consecutive_stream_hits + 1;
                previous_group_cycle <= cycle_count;
                if (group_tag == 24'h610000) begin
                    if (previous_stage0_cycle >= 0
                            && cycle_count - previous_stage0_cycle == 1)
                        stage0_consecutive_hits
                            <= stage0_consecutive_hits + 1;
                    previous_stage0_cycle <= cycle_count;
                end
                expected_group_read <= expected_group_read + 1;
            end
            if (token_done_accept && scoreboard_enabled) begin
                if (expected_done_read >= expected_done_write
                        || token_done_tag
                            !== expected_done_tag[expected_done_read]
                        || token_done_had_event
                            !== expected_done_had_event[expected_done_read])
                    $fatal(1, "M177 token done mismatch index=%0d",
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
        scan_beat_index = '0;
        scan_bitmap = '0;
        scan_eot = 1'b0;
        group_ready = 1'b0;
        token_done_ready = 1'b0;
        random_group_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        force_done_stall_mode = 1'b0;
        scoreboard_enabled = 1'b1;
        expected_group_write = 0;
        expected_group_read = 0;
        expected_done_write = 0;
        expected_done_read = 0;
        unique_group_count = 0;
        input_event_count = 0;
        unique_source_terms = 0;
        expected_replayed_source_terms = 0;
        raw_beat_opportunities = 0;
        zero_beats_elided = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        drive_token(0, 24'h610000, 1, 4);
        random_group_stall_mode = 1'b1;
        random_done_stall_mode = 1'b1;
        drive_token(1, 24'h620000, 2, 8);
        drive_token(2, 24'h630000, 4, 16);
        drive_token(3, 24'h640000, 8, 24);
        drive_token(4, 24'h650000, 1, 4);
        random_group_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        wait (expected_group_read == expected_group_write);
        wait (expected_done_read == expected_done_write);
        wait (!busy);
        drive_same_cycle_rearm_pair();
        wait (expected_group_read == expected_group_write);
        wait (expected_done_read == expected_done_write);
        wait (!busy);

        if (accepted_done_tokens != 7
                || accepted_eot_descriptors != 7
                || accepted_group_results != expected_group_write
                || expected_group_read != expected_group_write
                || unique_source_terms != input_event_count
                || observed_replayed_source_terms
                    != expected_replayed_source_terms
                || accepted_payload_descriptors + zero_beats_elided
                    != raw_beat_opportunities)
            $fatal(1, "M177 conservation mismatch descriptors=%0d payload=%0d eot=%0d raw=%0d zero=%0d done=%0d groups=%0d/%0d events=%0d unique=%0d replay=%0d/%0d",
                accepted_descriptors, accepted_payload_descriptors,
                accepted_eot_descriptors, raw_beat_opportunities,
                zero_beats_elided, accepted_done_tokens,
                accepted_group_results, expected_group_write,
                input_event_count, unique_source_terms,
                observed_replayed_source_terms,
                expected_replayed_source_terms);
        if (stage0_consecutive_hits == 0 || consecutive_stream_hits == 0
                || output_stall_cycles == 0 || prefetch_accepts == 0
                || indexed_gap_accepts == 0 || eot_with_pending_accepts == 0
                || same_cycle_token_rearms != 1)
            $fatal(1, "M177 coverage counters missing");

        final_raw_beats = raw_beat_opportunities;
        final_zero_beats = zero_beats_elided;
        final_payload_descriptors = accepted_payload_descriptors;
        final_eot_descriptors = accepted_eot_descriptors;
        final_accepted_descriptors = accepted_descriptors;
        final_done_tokens = accepted_done_tokens;
        final_bitmap_events = input_event_count;
        final_unique_groups = unique_group_count;
        final_unique_source_terms = unique_source_terms;
        final_group_results = accepted_group_results;
        final_replayed_source_terms = observed_replayed_source_terms;
        final_output_stalls = output_stall_cycles;
        final_prefetch_accepts = prefetch_accepts;
        final_indexed_gaps = indexed_gap_accepts;
        final_eot_pending = eot_with_pending_accepts;
        final_stage0_hits = stage0_consecutive_hits;
        final_stream_hits = consecutive_stream_hits;
        final_token_rearms = same_cycle_token_rearms;
        scoreboard_enabled = 1'b0;

        // Attack 1: malformed zero non-EOT payload.
        @(negedge clk_core);
        scan_valid = 1'b1;
        scan_tag = 24'hffffff;
        scan_output_blocks = 4'd1;
        scan_beat_index = 5'd0;
        scan_bitmap = 96'b0;
        scan_eot = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core);
        scan_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || scan_ready)
            $fatal(1, "M177 zero non-EOT fail-close missing");

        // Attack 2: stage0 index four is outside the legal [0,3] extent.
        @(negedge clk_core); rst_core = 1'b1; scan_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        scan_valid = 1'b1; scan_tag = 24'hff0002;
        scan_output_blocks = 4'd1; scan_beat_index = 5'd4;
        scan_bitmap = 96'h1; scan_eot = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core); scan_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || scan_ready)
            $fatal(1, "M177 out-of-stage index fail-close missing");

        // Attack 3: a backward index after a legal index two payload.
        @(negedge clk_core); rst_core = 1'b1; scan_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        send_descriptor(24'hff0003, 1, 2, 96'h1, 1'b0, 1'b1);
        scan_valid = 1'b1; scan_tag = 24'hff0003;
        scan_output_blocks = 4'd1; scan_beat_index = 5'd1;
        scan_bitmap = 96'h2; scan_eot = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core); scan_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || scan_ready)
            $fatal(1, "M177 backward index fail-close missing");

        // Attack 4: EOT before the prior token's done acceptance.
        @(negedge clk_core); rst_core = 1'b1; scan_valid = 1'b0;
        force_done_stall_mode = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        send_descriptor(24'hff0004, 1, 0, 96'b0, 1'b1, 1'b1);
        wait (token_done_valid);
        @(negedge clk_core);
        scan_valid = 1'b1; scan_tag = 24'hff0004;
        scan_output_blocks = 4'd1; scan_beat_index = 5'd0;
        scan_bitmap = 96'b0; scan_eot = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core); scan_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || scan_ready)
            $fatal(1, "M177 premature EOT fail-close missing");

        // Legal boundary: done-accept plus EOT starts a new all-zero token.
        @(negedge clk_core); rst_core = 1'b1; scan_valid = 1'b0;
        force_done_stall_mode = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        send_descriptor(24'hfe0001, 1, 0, 96'b0, 1'b1, 1'b1);
        wait (token_done_valid);
        @(negedge clk_core);
        scan_valid = 1'b1; scan_tag = 24'hfe0002;
        scan_output_blocks = 4'd1; scan_beat_index = 5'd0;
        scan_bitmap = 96'b0; scan_eot = 1'b1;
        @(posedge clk_core);
        if (!token_done_accept || !scan_accept)
            $fatal(1, "M177 legal done plus EOT rearm missing");
        @(negedge clk_core); scan_valid = 1'b0; scan_eot = 1'b0;
        repeat (4) @(posedge clk_core);
        if (protocol_error)
            $fatal(1, "M177 legal done plus EOT raised protocol error");

        $display("PASS M177 FC2 indexed-nonzero96 K4 replay frontend VCS raw_beat_opportunities=%0d zero_beats_elided=%0d payload_descriptors=%0d eot_descriptors=%0d descriptors=%0d tokens=7 bitmap_events=%0d unique_groups=%0d unique_source_terms=%0d replayed_group_results=%0d replayed_source_terms=%0d output_stall_cycles=%0d prefetch_accepts=%0d indexed_gap_accepts=%0d eot_with_pending_accepts=%0d stage0_consecutive_group_hits=%0d consecutive_group_stream_hits=%0d same_cycle_token_rearms=%0d output_block_extents=1,2,4,8 protocol_attacks=4 legal_done_eot_rearm=1 bitmap_width_bits=96 beat_index_bits=5 stage_extents=4,8,16,32 explicit_eot=true future_prediction=false cross_beat_grouping=false weight_sram_response=false arithmetic=false complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            final_raw_beats, final_zero_beats,
            final_payload_descriptors, final_eot_descriptors,
            final_accepted_descriptors, final_bitmap_events,
            final_unique_groups, final_unique_source_terms,
            final_group_results, final_replayed_source_terms,
            final_output_stalls, final_prefetch_accepts,
            final_indexed_gaps, final_eot_pending,
            final_stage0_hits, final_stream_hits, final_token_rearms);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "M177 watchdog timeout");
    end
endmodule

`default_nettype wire
