`timescale 1ns/1ps
`default_nettype none

module tb_m523_c2d_k8_polyphase_tap_bundler;
    localparam int TAG_BITS = 8;
    localparam int CHANNEL_BITS = 8;
    localparam int COORD_BITS = 6;
    localparam int TIME_BITS = 4;
    localparam int BUNDLE_LANES = 8;
    localparam int FIFO_DEPTH = 18;
    localparam int MAX_EXPECTED = 256;

    logic clk_core = 0;
    logic rst_core = 1;
    always #1.5 clk_core = ~clk_core;

    logic event_valid, event_ready, event_accept;
    logic [TAG_BITS-1:0] event_tag;
    logic [TIME_BITS-1:0] event_time;
    logic [CHANNEL_BITS-1:0] event_source_channel;
    logic [COORD_BITS-1:0] event_source_y, event_source_x;
    logic [COORD_BITS-1:0] event_input_height, event_input_width;
    logic event_last;

    logic bundle_valid, bundle_ready, bundle_accept;
    logic [TAG_BITS-1:0] bundle_tag;
    logic [TIME_BITS-1:0] bundle_time;
    logic [3:0] bundle_count;
    logic [BUNDLE_LANES-1:0] tap_lane_valid, tap_event_last;
    logic [CHANNEL_BITS-1:0] tap_source_channel [0:BUNDLE_LANES-1];
    logic [COORD_BITS-1:0] tap_source_y [0:BUNDLE_LANES-1];
    logic [COORD_BITS-1:0] tap_source_x [0:BUNDLE_LANES-1];
    logic [1:0] tap_kernel_y [0:BUNDLE_LANES-1];
    logic [1:0] tap_kernel_x [0:BUNDLE_LANES-1];
    logic [3:0] tap_kernel_index [0:BUNDLE_LANES-1];
    logic [COORD_BITS-1:0] tap_destination_y [0:BUNDLE_LANES-1];
    logic [COORD_BITS-1:0] tap_destination_x [0:BUNDLE_LANES-1];
    logic [1:0] tap_phase_bank [0:BUNDLE_LANES-1];
    logic bundle_last_for_event, bundle_stream_last;
    logic protocol_error, busy;
    logic [31:0] debug_event_count, debug_bundle_count, debug_tap_count;
    logic [5:0] debug_fifo_count;

    logic [TAG_BITS-1:0] exp_tag [0:MAX_EXPECTED-1];
    logic [TIME_BITS-1:0] exp_time [0:MAX_EXPECTED-1];
    logic [CHANNEL_BITS-1:0] exp_channel [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_sy [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_sx [0:MAX_EXPECTED-1];
    logic [1:0] exp_ky [0:MAX_EXPECTED-1];
    logic [1:0] exp_kx [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_dy [0:MAX_EXPECTED-1];
    logic [COORD_BITS-1:0] exp_dx [0:MAX_EXPECTED-1];
    logic [1:0] exp_phase [0:MAX_EXPECTED-1];
    logic exp_last [0:MAX_EXPECTED-1];
    logic exp_stream_last [0:MAX_EXPECTED-1];

    integer expected_count = 0;
    integer observed_count = 0;
    integer observed_bundles = 0;
    integer full8_bundles = 0;
    integer one_tap_flushes = 0;
    integer replacement_count = 0;
    integer stall_count = 0;
    integer event_boundary_count = 0;
    integer cross_event_bundles = 0;
    integer tag_boundary_flushes = 0;
    integer time_boundary_flushes = 0;
    integer stream_last_flushes = 0;
    integer stream_last_same_context_isolation = 0;
    integer protocol_attack_count = 0;
    integer maximum_fifo_count = 0;
    integer phase_count [0:3];
    logic [15:0] lfsr = 16'h523d;
    logic force_stall = 0;

    m523_c2d_k8_polyphase_tap_bundler #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .COORD_BITS(COORD_BITS), .TIME_BITS(TIME_BITS),
        .BUNDLE_LANES(BUNDLE_LANES), .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (.*);

    m523_c2d_k8_polyphase_tap_bundler_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .COORD_BITS(COORD_BITS), .TIME_BITS(TIME_BITS),
        .BUNDLE_LANES(BUNDLE_LANES), .FIFO_DEPTH(FIFO_DEPTH)
    ) sva (.*);

    function automatic logic slot_valid(
        input integer slot, input integer sy, input integer sx);
        case (slot)
            0: slot_valid = sy != 0 && sx != 0;
            1, 4: slot_valid = sy != 0;
            2, 6: slot_valid = sx != 0;
            default: slot_valid = 1'b1;
        endcase
    endfunction

    task automatic slot_to_kernel(
        input integer slot, output integer ky, output integer kx);
        case (slot)
            0: begin ky = 0; kx = 0; end
            1: begin ky = 0; kx = 2; end
            2: begin ky = 2; kx = 0; end
            3: begin ky = 2; kx = 2; end
            4: begin ky = 0; kx = 1; end
            5: begin ky = 2; kx = 1; end
            6: begin ky = 1; kx = 0; end
            7: begin ky = 1; kx = 2; end
            default: begin ky = 1; kx = 1; end
        endcase
    endtask

    task automatic enqueue_event(
        input integer tag_value, input integer time_value,
        input integer channel_value, input integer sy, input integer sx,
        input logic stream_last);
        integer slot, ky, kx, last_slot;
        begin
            last_slot = -1;
            for (slot = 0; slot < 9; slot = slot + 1)
                if (slot_valid(slot, sy, sx)) last_slot = slot;
            for (slot = 0; slot < 9; slot = slot + 1) begin
                if (slot_valid(slot, sy, sx)) begin
                    if (expected_count >= MAX_EXPECTED)
                        $fatal(1, "M523 expected queue overflow");
                    slot_to_kernel(slot, ky, kx);
                    exp_tag[expected_count] = tag_value;
                    exp_time[expected_count] = time_value;
                    exp_channel[expected_count] = channel_value;
                    exp_sy[expected_count] = sy;
                    exp_sx[expected_count] = sx;
                    exp_ky[expected_count] = ky;
                    exp_kx[expected_count] = kx;
                    exp_dy[expected_count] = (2 * sy) - 1 + ky;
                    exp_dx[expected_count] = (2 * sx) - 1 + kx;
                    exp_phase[expected_count] = {
                        exp_dy[expected_count][0],
                        exp_dx[expected_count][0]};
                    exp_last[expected_count] = slot == last_slot;
                    exp_stream_last[expected_count] = stream_last
                        && slot == last_slot;
                    expected_count = expected_count + 1;
                end
            end
        end
    endtask

    task automatic drive_event(
        input integer tag_value, input integer time_value,
        input integer channel_value, input integer sy, input integer sx,
        input integer height, input integer width,
        input logic stream_last);
        begin
            enqueue_event(tag_value, time_value, channel_value, sy, sx,
                          stream_last);
            @(negedge clk_core);
            event_tag = tag_value;
            event_time = time_value;
            event_source_channel = channel_value;
            event_source_y = sy;
            event_source_x = sx;
            event_input_height = height;
            event_input_width = width;
            event_last = stream_last;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    // Variant used only when the caller is already at a negative edge.  It
    // prevents an unintended free output cycle while releasing force_stall.
    task automatic drive_event_now(
        input integer tag_value, input integer time_value,
        input integer channel_value, input integer sy, input integer sx,
        input integer height, input integer width,
        input logic stream_last);
        begin
            enqueue_event(tag_value, time_value, channel_value, sy, sx,
                          stream_last);
            event_tag = tag_value;
            event_time = time_value;
            event_source_channel = channel_value;
            event_source_y = sy;
            event_source_x = sx;
            event_input_height = height;
            event_input_width = width;
            event_last = stream_last;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            lfsr <= 16'h523d;
            bundle_ready <= 1'b0;
        end else if (force_stall) begin
            bundle_ready <= 1'b0;
        end else begin
            lfsr <= {lfsr[14:0],
                     lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            bundle_ready <= lfsr[0] | lfsr[2];
        end
    end

    always @(posedge clk_core) begin : scoreboard
        integer lane;
        integer final_index;
        logic crossed_event;
        if (!rst_core && debug_fifo_count > maximum_fifo_count)
            maximum_fifo_count = debug_fifo_count;
        if (!rst_core && bundle_valid && !bundle_ready)
            stall_count = stall_count + 1;
        if (!rst_core && event_accept && bundle_accept)
            replacement_count = replacement_count + 1;
        if (!rst_core && bundle_accept) begin
            if (bundle_count == 8) full8_bundles = full8_bundles + 1;
            if (bundle_count == 1) one_tap_flushes = one_tap_flushes + 1;
            crossed_event = 1'b0;
            for (lane = 0; lane < BUNDLE_LANES; lane = lane + 1) begin
                if (lane < bundle_count) begin
                    if (!tap_lane_valid[lane]
                            || observed_count >= expected_count)
                        $fatal(1, "M523 missing/unexpected active lane");
                    if (bundle_tag !== exp_tag[observed_count]
                            || bundle_time !== exp_time[observed_count]
                            || tap_source_channel[lane]
                                !== exp_channel[observed_count]
                            || tap_source_y[lane] !== exp_sy[observed_count]
                            || tap_source_x[lane] !== exp_sx[observed_count]
                            || tap_kernel_y[lane] !== exp_ky[observed_count]
                            || tap_kernel_x[lane] !== exp_kx[observed_count]
                            || tap_kernel_index[lane]
                                !== exp_ky[observed_count] * 3
                                    + exp_kx[observed_count]
                            || tap_destination_y[lane]
                                !== exp_dy[observed_count]
                            || tap_destination_x[lane]
                                !== exp_dx[observed_count]
                            || tap_phase_bank[lane]
                                !== exp_phase[observed_count]
                            || tap_event_last[lane]
                                !== exp_last[observed_count])
                        $fatal(1, "M523 tuple/boundary mismatch index=%0d lane=%0d",
                               observed_count, lane);
                    if (tap_event_last[lane]) begin
                        event_boundary_count = event_boundary_count + 1;
                        if (lane + 1 < bundle_count)
                            crossed_event = 1'b1;
                    end
                    phase_count[tap_phase_bank[lane]] =
                        phase_count[tap_phase_bank[lane]] + 1;
                    observed_count = observed_count + 1;
                end else if (tap_lane_valid[lane]
                        || tap_event_last[lane]) begin
                    $fatal(1, "M523 non-prefix lane metadata lane=%0d", lane);
                end
            end
            if (crossed_event)
                cross_event_bundles = cross_event_bundles + 1;
            final_index = observed_count - 1;
            if (bundle_last_for_event !== exp_last[final_index]
                    || bundle_stream_last !== exp_stream_last[final_index])
                $fatal(1, "M523 bundle fence mismatch count=%0d",
                       bundle_count);
            if (bundle_count < 8 && exp_last[final_index]
                    && !exp_stream_last[final_index]
                    && observed_count < expected_count) begin
                if (exp_tag[observed_count] != exp_tag[final_index])
                    tag_boundary_flushes = tag_boundary_flushes + 1;
                else if (exp_time[observed_count] != exp_time[final_index])
                    time_boundary_flushes = time_boundary_flushes + 1;
                else
                    $fatal(1, "M523 illegal partial same-tag/time flush");
            end
            if (bundle_stream_last) begin
                stream_last_flushes = stream_last_flushes + 1;
                if (observed_count < expected_count
                        && exp_tag[observed_count] == exp_tag[final_index]
                        && exp_time[observed_count] == exp_time[final_index])
                    stream_last_same_context_isolation =
                        stream_last_same_context_isolation + 1;
            end
            observed_bundles = observed_bundles + 1;
        end
    end

    initial begin
        event_valid = 0;
        event_tag = 0;
        event_time = 0;
        event_source_channel = 0;
        event_source_y = 0;
        event_source_x = 0;
        event_input_height = 0;
        event_input_width = 0;
        event_last = 0;
        bundle_ready = 0;
        for (int phase = 0; phase < 4; phase++) phase_count[phase] = 0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;

        // Same tag/time packs the first 4-tap event into the following
        // 6-tap event.  The third event has a different tag and is stream
        // last, so the 2-tap tail and then the 6-tap event flush separately.
        drive_event(1, 0, 3, 0, 0, 3, 4, 0);
        drive_event(1, 0, 5, 0, 2, 3, 4, 0);
        drive_event(2, 0, 7, 2, 0, 3, 4, 1);

        // Once the first cross-event full bundle has retired, freeze output.
        // Event four has the same tag/time as stream-last event three, so
        // their simultaneous residency proves stream-last alone forbids a
        // crossing.  Event five differs only in time.  Releasing output while
        // presenting it immediately makes the stream flush and atomic push
        // reach the exact 18-entry FIFO boundary.
        while (observed_bundles < 1) @(posedge clk_core);
        force_stall = 1'b1;
        @(negedge clk_core);
        drive_event(2, 0, 9, 1, 2, 3, 4, 0);
        force_stall = 1'b0;
        drive_event_now(2, 1, 11, 2, 3, 3, 4, 0);
        if (debug_fifo_count != FIFO_DEPTH)
            $fatal(1, "M523 FIFO near-full atomic admission missing");

        // The sixth event shares tag/time with event five and is stream last.
        // It waits as a stable valid request until output frees nine slots.
        force_stall = 1'b0;
        drive_event(2, 1, 13, 31, 31, 32, 32, 1);

        // Attack an illegal successor while accepted descriptors are stalled.
        // Sticky fault must block new events but allow all 43 taps to drain.
        @(posedge clk_core);
        force_stall = 1'b1;
        while (!(bundle_valid && !bundle_ready)) @(posedge clk_core);
        @(negedge clk_core);
        event_tag = 8'hff;
        event_time = 4'hf;
        event_source_y = 32;
        event_source_x = 0;
        event_input_height = 32;
        event_input_width = 32;
        event_last = 0;
        event_valid = 1;
        protocol_attack_count = protocol_attack_count + 1;
        @(posedge clk_core);
        @(negedge clk_core);
        event_valid = 0;
        force_stall = 0;

        while (observed_count != expected_count || busy)
            @(posedge clk_core);
        @(negedge clk_core);

        if (expected_count != 43 || observed_count != 43
                || observed_bundles != 8)
            $fatal(1, "M523 final tap/bundle total mismatch");
        if (full8_bundles != 4 || one_tap_flushes != 1
                || event_boundary_count != 6 || cross_event_bundles != 2
                || tag_boundary_flushes != 1
                || time_boundary_flushes != 1
                || stream_last_flushes != 2
                || stream_last_same_context_isolation != 1
                || maximum_fifo_count != FIFO_DEPTH
                || replacement_count == 0 || stall_count == 0
                || protocol_attack_count != 1)
            $fatal(1, "M523 final packing/protocol ledger mismatch");
        if (phase_count[0] != 6 || phase_count[1] != 10
                || phase_count[2] != 10 || phase_count[3] != 17)
            $fatal(1, "M523 exact phase totals mismatch");
        if (!protocol_error || event_ready || bundle_valid || busy
                || debug_fifo_count != 0)
            $fatal(1, "M523 illegal successor did not lock then drain");
        if (debug_event_count != 6 || debug_bundle_count != 8
                || debug_tap_count != 43)
            $fatal(1, "M523 debug counters mismatch");

        $display("PASS M523 events=%0d bundles=%0d taps=%0d full8=%0d tails1=%0d stalls=%0d replacements=%0d boundaries=%0d cross_event=%0d tag_flush=%0d time_flush=%0d stream_flush=%0d stream_iso=%0d fifo_max=%0d phases=%0d/%0d/%0d/%0d protocol_attack=%0d",
                 debug_event_count, debug_bundle_count, debug_tap_count,
                 full8_bundles, one_tap_flushes, stall_count,
                 replacement_count, event_boundary_count,
                 cross_event_bundles, tag_boundary_flushes,
                 time_boundary_flushes, stream_last_flushes,
                 stream_last_same_context_isolation, maximum_fifo_count,
                 phase_count[0], phase_count[1],
                 phase_count[2], phase_count[3], protocol_attack_count);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "M523 timeout");
    end
endmodule

`default_nettype wire
