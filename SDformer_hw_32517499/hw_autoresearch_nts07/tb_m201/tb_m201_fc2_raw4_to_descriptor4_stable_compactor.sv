`timescale 1ns/1ps
`default_nettype none

module tb_m201_fc2_raw4_to_descriptor4_stable_compactor;
    localparam int TAG_BITS = 24;
    localparam int QUEUE_DEPTH = 8;
    localparam int MAX_DESC = 64;
    logic clk_core, rst_core;
    logic header_valid, header_ready, header_accept;
    logic [TAG_BITS-1:0] header_token_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth;
    logic raw_valid, raw_ready, raw_last, raw_accept;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [2:0] descriptor_count;
    logic [TAG_BITS-1:0] descriptor_token_tag;
    logic [4:0] descriptor_beat_index [0:3];
    logic [95:0] descriptor_bitmap [0:3];
    logic [3:0] descriptor_window_last;
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [TAG_BITS-1:0] token_done_tag;
    logic [5:0] token_done_descriptor_count;
    logic protocol_error, busy;
    logic [95:0] token_raw [0:31];
    logic [95:0] expected_bitmap [0:MAX_DESC-1];
    logic [4:0] expected_index [0:MAX_DESC-1];
    logic expected_window_last [0:MAX_DESC-1];
    integer expected_count, expected_read, token_expected_count;
    logic [TAG_BITS-1:0] current_tag;
    integer accepted_headers, accepted_raw_packets, accepted_raw_beats;
    integer accepted_descriptors, accepted_descriptor_packets;
    integer accepted_done, descriptor_stalls, raw_backpressure;
    integer simultaneous_push_pop, full4_packets, zero_tokens;
    integer protocol_attacks;
    logic scoreboard_enabled, random_stalls_enabled;

    m201_fc2_raw4_to_descriptor4_stable_compactor dut (.*);
    bind m201_fc2_raw4_to_descriptor4_stable_compactor
        m201_fc2_raw4_to_descriptor4_stable_compactor_assertions sva (.*);
    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    task automatic clear_inputs;
        begin
            header_valid = 0; header_token_tag = 0;
            header_raw_beat_count = 0; header_window_depth = 0;
            raw_valid = 0; raw_lane_valid = 0; raw_last = 0;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = 0; raw_bitmap[lane] = 0;
            end
        end
    endtask
    task automatic apply_reset;
        begin
            @(negedge clk_core); rst_core = 1; clear_inputs();
            descriptor_ready = 1; token_done_ready = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core); rst_core = 0;
        end
    endtask
    function automatic logic [95:0] make_bitmap(
            input integer token_ordinal, input integer beat,
            input integer mode);
        logic [95:0] value;
        begin
            value = 0;
            if (mode == 1 || (mode == 2 && (beat % 5) != 0)
                    || (mode == 3 && ((beat + token_ordinal) % 3) == 1)) begin
                value = {
                    32'h80000001 ^ (token_ordinal * 17 + beat),
                    32'h01020408 ^ (token_ordinal * 29 + beat * 3),
                    32'h10204080 ^ (token_ordinal * 43 + beat * 7)
                };
                if (value == 0) value[beat % 96] = 1;
            end
            return value;
        end
    endfunction
    task automatic drive_token(
            input integer token_ordinal, input integer raw_beats,
            input integer window_depth, input integer mode);
        integer nonzero_ordinal, base, lanes;
        begin
            expected_count = 0; expected_read = 0; nonzero_ordinal = 0;
            current_tag = 24'hc90000 + token_ordinal;
            for (int beat = 0; beat < raw_beats; beat++) begin
                token_raw[beat] = make_bitmap(token_ordinal, beat, mode);
                if (token_raw[beat] != 0) begin
                    expected_bitmap[expected_count] = token_raw[beat];
                    expected_index[expected_count] = beat[4:0];
                    nonzero_ordinal = nonzero_ordinal + 1;
                    expected_window_last[expected_count]
                        = (nonzero_ordinal % window_depth) == 0;
                    expected_count = expected_count + 1;
                end
            end
            token_expected_count = expected_count;
            if (expected_count == 0) zero_tokens = zero_tokens + 1;
            @(negedge clk_core);
            header_token_tag = current_tag;
            header_raw_beat_count = raw_beats[5:0];
            header_window_depth = window_depth[3:0]; header_valid = 1;
            do @(posedge clk_core); while (!header_accept);
            @(negedge clk_core); header_valid = 0;
            base = 0;
            while (base < raw_beats) begin
                lanes = (raw_beats - base >= 4) ? 4 : raw_beats - base;
                @(negedge clk_core); raw_lane_valid = (1 << lanes) - 1;
                for (int lane = 0; lane < 4; lane++) begin
                    raw_beat_index[lane] = (base + lane) & 31;
                    raw_bitmap[lane] = lane < lanes
                        ? token_raw[base + lane] : 0;
                end
                raw_last = base + lanes == raw_beats; raw_valid = 1;
                do @(posedge clk_core); while (!raw_accept);
                base = base + lanes;
            end
            @(negedge clk_core); raw_valid = 0;
            raw_lane_valid = 0; raw_last = 0;
            do @(posedge clk_core); while (!token_done_accept);
            if (expected_read != expected_count)
                $fatal(1, "M201 token done before drain");
            @(negedge clk_core);
        end
    endtask
    always @(negedge clk_core) begin
        if (rst_core || !random_stalls_enabled) begin
            descriptor_ready <= 1; token_done_ready <= 1;
        end else begin
            descriptor_ready <= ($urandom_range(0, 4) != 0);
            token_done_ready <= ($urandom_range(0, 5) != 0);
        end
    end
    always @(posedge clk_core) begin : scoreboard
        integer packet_nonzero;
        if (!rst_core && scoreboard_enabled) begin
            if (header_accept) accepted_headers = accepted_headers + 1;
            if (raw_valid && !raw_ready && !protocol_error)
                raw_backpressure = raw_backpressure + 1;
            if (descriptor_valid && !descriptor_ready)
                descriptor_stalls = descriptor_stalls + 1;
            if (raw_accept) begin
                accepted_raw_packets = accepted_raw_packets + 1;
                packet_nonzero = 0;
                for (int lane = 0; lane < 4; lane++) if (raw_lane_valid[lane]) begin
                    accepted_raw_beats = accepted_raw_beats + 1;
                    if (raw_bitmap[lane] != 0)
                        packet_nonzero = packet_nonzero + 1;
                end
                if (raw_lane_valid == 4'hf && packet_nonzero == 4)
                    full4_packets = full4_packets + 1;
            end
            if (descriptor_accept) begin
                if (descriptor_token_tag !== current_tag)
                    $fatal(1, "M201 descriptor tag mismatch");
                if (descriptor_count < 1 || descriptor_count > 4)
                    $fatal(1, "M201 descriptor count illegal");
                for (int lane = 0; lane < descriptor_count; lane++) begin
                    if (expected_read >= expected_count
                            || descriptor_bitmap[lane]
                                !== expected_bitmap[expected_read]
                            || descriptor_beat_index[lane]
                                !== expected_index[expected_read]
                            || descriptor_window_last[lane]
                                !== expected_window_last[expected_read])
                        $fatal(1, "M201 descriptor mismatch index=%0d",
                            expected_read);
                    expected_read = expected_read + 1;
                    accepted_descriptors = accepted_descriptors + 1;
                end
                accepted_descriptor_packets
                    = accepted_descriptor_packets + 1;
            end
            if (token_done_accept) begin
                if (token_done_tag !== current_tag
                        || token_done_descriptor_count
                            !== token_expected_count)
                    $fatal(1, "M201 done mismatch");
                accepted_done = accepted_done + 1;
            end
            if (raw_accept && descriptor_accept)
                simultaneous_push_pop = simultaneous_push_pop + 1;
        end
    end
    initial begin : stimulus
        rst_core = 1; descriptor_ready = 1; token_done_ready = 1;
        scoreboard_enabled = 0; random_stalls_enabled = 0;
        accepted_headers = 0; accepted_raw_packets = 0;
        accepted_raw_beats = 0; accepted_descriptors = 0;
        accepted_descriptor_packets = 0; accepted_done = 0;
        descriptor_stalls = 0; raw_backpressure = 0;
        simultaneous_push_pop = 0; full4_packets = 0;
        zero_tokens = 0; protocol_attacks = 0;
        clear_inputs(); apply_reset(); scoreboard_enabled = 1;
        random_stalls_enabled = 1;
        drive_token(0, 4, 2, 0); drive_token(1, 32, 8, 1);
        drive_token(2, 31, 4, 2);
        for (int token = 3; token < 241; token++) begin
            case (token % 4)
                0: drive_token(token, 4, 2, 3);
                1: drive_token(token, 8, 4, 2);
                2: drive_token(token, 16, 8, 3);
                default: drive_token(token, 32, 8, 2);
            endcase
        end
        if (accepted_headers != 241 || accepted_done != 241
                || accepted_descriptors == 0 || full4_packets == 0
                || descriptor_stalls == 0 || raw_backpressure == 0
                || simultaneous_push_pop == 0 || zero_tokens == 0)
            $fatal(1, "M201 coverage/conservation precondition missing");
        scoreboard_enabled = 0; random_stalls_enabled = 0;
        apply_reset(); @(negedge clk_core);
        header_raw_beat_count = 0; header_window_depth = 4;
        header_valid = 1; @(posedge clk_core);
        if (!protocol_error) $fatal(1, "M201 bad header accepted");
        protocol_attacks = protocol_attacks + 1;
        @(negedge clk_core); header_valid = 0;
        apply_reset(); @(negedge clk_core);
        header_token_tag = 24'hbad101; header_raw_beat_count = 4;
        header_window_depth = 2; header_valid = 1;
        do @(posedge clk_core); while (!header_accept);
        @(negedge clk_core); header_valid = 0;
        raw_lane_valid = 4'b0101; raw_valid = 1; @(posedge clk_core);
        if (!protocol_error) $fatal(1, "M201 nonprefix accepted");
        protocol_attacks = protocol_attacks + 1;
        @(negedge clk_core); raw_valid = 0;
        apply_reset(); @(negedge clk_core);
        header_token_tag = 24'hbad102; header_raw_beat_count = 4;
        header_window_depth = 4; header_valid = 1;
        do @(posedge clk_core); while (!header_accept);
        @(negedge clk_core); header_valid = 0;
        raw_lane_valid = 4'hf;
        for (int lane = 0; lane < 4; lane++) begin
            raw_beat_index[lane] = lane + 1;
            raw_bitmap[lane] = 96'h1 << lane;
        end
        raw_last = 1; raw_valid = 1; @(posedge clk_core);
        if (!protocol_error) $fatal(1, "M201 bad index accepted");
        protocol_attacks = protocol_attacks + 1;
        @(negedge clk_core); raw_valid = 0;
        apply_reset(); @(negedge clk_core);
        header_token_tag = 24'hbad103; header_raw_beat_count = 4;
        header_window_depth = 8; header_valid = 1;
        do @(posedge clk_core); while (!header_accept);
        @(negedge clk_core); header_valid = 0;
        raw_lane_valid = 4'hf;
        for (int lane = 0; lane < 4; lane++) begin
            raw_beat_index[lane] = lane; raw_bitmap[lane] = 0;
        end
        raw_last = 0; raw_valid = 1; @(posedge clk_core);
        if (!protocol_error) $fatal(1, "M201 wrong last accepted");
        protocol_attacks = protocol_attacks + 1;
        $display("PASS M201 raw4-to-descriptor4 matched compactor VCS tokens=%0d raw_packets=%0d raw_beats=%0d descriptors=%0d descriptor_packets=%0d descriptor_stalls=%0d raw_backpressure=%0d simultaneous_push_pop=%0d full4=%0d zero_tokens=%0d protocol_attacks=%0d queue_depth=8 physical_speedup=false complete_fc2=false system_speedup=false headline=false",
            accepted_done, accepted_raw_packets, accepted_raw_beats,
            accepted_descriptors, accepted_descriptor_packets,
            descriptor_stalls, raw_backpressure, simultaneous_push_pop,
            full4_packets, zero_tokens, protocol_attacks);
        $finish;
    end
    initial begin #3000000; $fatal(1, "M201 watchdog timeout"); end
endmodule

`default_nettype wire
