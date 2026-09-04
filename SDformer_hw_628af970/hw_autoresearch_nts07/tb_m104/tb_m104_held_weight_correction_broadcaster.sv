`timescale 1ns/1ps
`default_nettype none

module tb_m104_held_weight_correction_broadcaster;
    logic clk_core, rst_core;
    logic load_valid, load_ready;
    logic [3:0] load_source;
    logic [2:0] load_block;
    logic [1:0] load_beat;
    logic [255:0] load_data;
    logic load_accept;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic event_negate, event_last_for_key;
    logic [31:0] event_tag;
    logic event_accept;
    logic output_valid, output_ready;
    logic [31:0] output_tag;
    logic [3:0] output_source;
    logic [2:0] output_block;
    logic output_negate;
    logic [1151:0] output_values;
    logic output_accept;
    logic held_valid, collecting;
    logic [1:0] expected_load_beat;
    logic protocol_error, busy;

    integer cycle_count, loaded_groups, accepted_load_beats;
    integer accepted_events, consecutive_event_pairs, stall_cycles;
    integer protocol_attacks, continuation_attacks, buffered_fault_attacks;
    integer accepted_event_grace_holds;

    logic [767:0] weight_payload;
    logic [1151:0] expected_positive, expected_negative;

    m104_held_weight_correction_broadcaster dut (.*);
    m104_held_weight_correction_broadcaster_assertions dut_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid), .load_ready(load_ready),
        .load_beat(load_beat), .load_accept(load_accept),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_negate(event_negate),
        .event_last_for_key(event_last_for_key),
        .event_accept(event_accept), .output_valid(output_valid),
        .output_ready(output_ready), .output_tag(output_tag),
        .output_source(output_source), .output_block(output_block),
        .output_negate(output_negate), .output_values(output_values),
        .output_accept(output_accept), .held_valid(held_valid),
        .collecting(collecting), .expected_load_beat(expected_load_beat),
        .protocol_error(protocol_error),
        .illegal_request(dut.illegal_request),
        .accepted_event_grace_match(dut.accepted_event_grace_match),
        .request_fault(dut.request_fault_q),
        .output_valid_q(dut.output_valid_q)
    );

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    initial begin
        #200000;
        $fatal(1, "M104 watchdog timeout cycle=%0d", cycle_count);
    end

    task automatic build_weight(input integer seed);
        integer signed value;
        logic signed [11:0] extended;
        begin
            weight_payload = '0;
            expected_positive = '0;
            expected_negative = '0;
            for (int lane = 0; lane < 96; lane++) begin
                case (lane % 8)
                    0: value = -128;
                    1: value = 127;
                    default: value = ((lane*19 + seed*13) % 255) - 127;
                endcase
                weight_payload[lane*8 +: 8] = value[7:0];
                extended = {{4{value[7]}}, value[7:0]};
                expected_positive[lane*12 +: 12] = extended;
                expected_negative[lane*12 +: 12] = -extended;
            end
        end
    endtask

    task automatic clear_drives;
        begin
            load_valid = 1'b0;
            load_source = '0;
            load_block = '0;
            load_beat = '0;
            load_data = '0;
            event_valid = 1'b0;
            event_source = '0;
            event_block = '0;
            event_negate = 1'b0;
            event_last_for_key = 1'b0;
            event_tag = '0;
            output_ready = 1'b1;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            clear_drives();
            rst_core = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic load_key(
        input integer source,
        input integer block,
        input integer seed
    );
        logic accepted_now;
        begin
            build_weight(seed);
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                load_valid = 1'b1;
                load_source = source[3:0];
                load_block = block[2:0];
                load_beat = beat[1:0];
                load_data = weight_payload[beat*256 +: 256];
                do begin
                    @(posedge clk_core);
                    accepted_now = load_ready;
                    if (accepted_now) begin
                        #0.01;
                        load_valid = 1'b0;
                    end
                    #0.49;
                    if (protocol_error)
                        $fatal(1, "M104 legal load fault beat=%0d", beat);
                end while (!accepted_now);
                accepted_load_beats++;
            end
            @(negedge clk_core);
            load_valid = 1'b0;
            #0.1;
            if (!held_valid || collecting || expected_load_beat != 0)
                $fatal(1, "M104 key did not become held");
            loaded_groups++;
        end
    endtask

    task automatic attack_event(
        input integer source,
        input integer block,
        input logic buffered_expected
    );
        begin
            @(negedge clk_core);
            event_valid = 1'b1;
            event_source = source[3:0];
            event_block = block[2:0];
            event_tag = 32'hbad10000 | protocol_attacks;
            #0.1;
            if (event_ready)
                $fatal(1, "M104 invalid event became ready");
            if (buffered_expected && (output_valid || !dut.output_valid_q))
                $fatal(1, "M104 same-cycle fault failed output quarantine");
            @(posedge clk_core); #0.5;
            if (!protocol_error || event_ready || output_valid || output_accept)
                $fatal(1, "M104 invalid event was not fail-closed");
            if (buffered_expected && !dut.output_valid_q)
                $fatal(1, "M104 buffered output was incorrectly retired");
            protocol_attacks++;
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic attack_load_first(
        input integer beat,
        input integer source,
        input integer block
    );
        begin
            @(negedge clk_core);
            load_valid = 1'b1;
            load_source = source[3:0];
            load_block = block[2:0];
            load_beat = beat[1:0];
            load_data = '0;
            #0.1;
            if (load_ready)
                $fatal(1, "M104 invalid load became ready beat=%0d", beat);
            @(posedge clk_core); #0.5;
            if (!protocol_error || load_ready)
                $fatal(1, "M104 invalid load was not fail-closed beat=%0d", beat);
            protocol_attacks++;
            @(negedge clk_core);
            load_valid = 1'b0;
        end
    endtask

    task automatic attack_load_continuation(input integer mutation);
        logic accepted_now;
        begin
            reset_dut();
            build_weight(40 + mutation);
            @(negedge clk_core);
            load_valid = 1'b1;
            load_source = 4'd6;
            load_block = 3'd2;
            load_beat = 0;
            load_data = weight_payload[255:0];
            @(posedge clk_core);
            accepted_now = load_ready;
            if (accepted_now) begin
                #0.01;
                load_valid = 1'b0;
            end
            #0.49;
            if (!accepted_now || protocol_error || !collecting)
                $fatal(1, "M104 continuation attack first beat failed");
            accepted_load_beats++;
            @(negedge clk_core);
            load_valid = 1'b1;
            load_beat = 1;
            load_data = weight_payload[511:256];
            case (mutation)
                0: load_source = 4'd7;
                1: load_block = 3'd3;
                2: load_beat = 2;
                default: $fatal(1, "M104 unknown continuation mutation");
            endcase
            #0.1;
            if (load_ready)
                $fatal(1, "M104 mutated continuation became ready");
            @(posedge clk_core); #0.5;
            if (!protocol_error || load_ready)
                $fatal(1, "M104 continuation mutation was not fail-closed");
            protocol_attacks++;
            continuation_attacks++;
            @(negedge clk_core);
            load_valid = 1'b0;
        end
    endtask

    initial begin : directed_test
        integer previous_accept_cycle;
        logic sign, last;
        logic [31:0] tag;
        logic [1151:0] held_output;

        clk_core = 1'b0;
        rst_core = 1'b1;
        cycle_count = 0;
        loaded_groups = 0;
        accepted_load_beats = 0;
        accepted_events = 0;
        consecutive_event_pairs = 0;
        stall_cycles = 0;
        protocol_attacks = 0;
        continuation_attacks = 0;
        buffered_fault_attacks = 0;
        accepted_event_grace_holds = 0;
        clear_drives();
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // One three-beat load feeds six consecutive destination descriptors.
        load_key(3, 5, 11);
        previous_accept_cycle = -1;
        for (int event_index = 0; event_index < 6; event_index++) begin
            @(negedge clk_core);
            event_valid = 1'b1;
            event_source = 4'd3;
            event_block = 3'd5;
            event_negate = event_index[0];
            event_last_for_key = event_index == 5;
            event_tag = 32'h10400000 + event_index;
            @(posedge clk_core);
            if (!event_ready)
                $fatal(1, "M104 legal event not ready index=%0d", event_index);
            if (event_index == 5) begin
                #0.01;
                event_valid = 1'b0;
            end
            #0.49;
            if (!output_valid || output_tag != 32'h10400000 + event_index
                    || output_source != 3 || output_block != 5
                    || output_negate != event_index[0]
                    || output_values !== (event_index[0]
                                          ? expected_negative
                                          : expected_positive))
                $fatal(1, "M104 streamed output mismatch index=%0d", event_index);
            if (previous_accept_cycle >= 0) begin
                if (cycle_count - previous_accept_cycle != 1)
                    $fatal(1, "M104 event II drift index=%0d", event_index);
                consecutive_event_pairs++;
            end
            previous_accept_cycle = cycle_count;
            accepted_events++;
        end
        @(negedge clk_core);
        event_valid = 1'b0;
        @(posedge clk_core); #0.5;
        if (output_valid || held_valid)
            $fatal(1, "M104 final event did not drain/release key");

        // Backpressure holds the wide result; the held key services another
        // destination without a second weight load.
        load_key(9, 1, 23);
        @(negedge clk_core);
        output_ready = 1'b0;
        event_valid = 1'b1;
        event_source = 4'd9;
        event_block = 3'd1;
        event_negate = 1'b1;
        event_last_for_key = 1'b0;
        event_tag = 32'h10410000;
        @(posedge clk_core);
        if (!event_ready)
            $fatal(1, "M104 stalled output event was not accepted");
        accepted_events++;
        @(negedge clk_core);
        event_valid = 1'b0;
        held_output = output_values;
        repeat (3) begin
            @(posedge clk_core); #0.5;
            if (!output_valid || output_values !== held_output
                    || output_tag != 32'h10410000)
                $fatal(1, "M104 output changed under stall");
            stall_cycles++;
        end
        @(negedge clk_core);
        output_ready = 1'b1;
        @(posedge clk_core); #0.5;
        if (output_valid || !held_valid)
            $fatal(1, "M104 stalled output did not drain cleanly");

        @(negedge clk_core);
        event_valid = 1'b1;
        event_source = 4'd9;
        event_block = 3'd1;
        event_negate = 1'b0;
        event_last_for_key = 1'b1;
        event_tag = 32'h10410001;
        @(posedge clk_core);
        if (!event_ready)
            $fatal(1, "M104 last event after stall was not accepted");
        #0.01;
        event_valid = 1'b0;
        accepted_events++;
        @(negedge clk_core);
        event_valid = 1'b0;
        @(posedge clk_core); #0.5;
        if (output_valid || held_valid)
            $fatal(1, "M104 second key did not retire");

        // A source may legally withdraw valid only after observing the
        // accepting edge.  Hold an accepted last-event request for a full
        // extra cycle: it must not be accepted twice or become a false
        // sticky protocol fault after the key is released.
        reset_dut();
        load_key(10, 2, 71);
        @(negedge clk_core);
        output_ready = 1'b0;
        event_valid = 1'b1;
        event_source = 4'd10;
        event_block = 3'd2;
        event_negate = 1'b1;
        event_last_for_key = 1'b1;
        event_tag = 32'h10420001;
        @(posedge clk_core);
        if (!event_ready)
            $fatal(1, "M104 grace setup last event was not accepted");
        accepted_events++;
        #0.1;
        held_output = output_values;
        if (!dut.accepted_event_grace_match || protocol_error || event_ready
                || event_accept || !output_valid || held_valid)
            $fatal(1, "M104 accepted last-event grace failed after transfer");
        @(posedge clk_core); #0.1;
        if (!dut.accepted_event_grace_match || protocol_error || event_ready
                || event_accept || !output_valid
                || output_values !== held_output)
            $fatal(1, "M104 accepted last-event grace was not stable");
        @(negedge clk_core);
        event_valid = 1'b0;
        output_ready = 1'b1;
        accepted_event_grace_holds++;
        @(posedge clk_core); #0.5;
        if (output_valid || held_valid || protocol_error)
            $fatal(1, "M104 accepted last-event grace did not drain cleanly");

        // Event with no held key.
        reset_dut();
        attack_event(1, 1, 1'b0);

        // Wrong event source and block under a legal held key.
        reset_dut();
        load_key(2, 4, 31);
        attack_event(3, 4, 1'b0);
        reset_dut();
        load_key(2, 4, 32);
        attack_event(2, 5, 1'b0);

        // Bad first beat and three continuation identity mutations.
        reset_dut();
        attack_load_first(1, 4, 2);
        for (int mutation = 0; mutation < 3; mutation++)
            attack_load_continuation(mutation);

        // Simultaneous load/event requests are a protocol collision.
        reset_dut();
        @(negedge clk_core);
        load_valid = 1'b1;
        load_source = 4'd1;
        load_block = 3'd1;
        load_beat = 0;
        event_valid = 1'b1;
        event_source = 4'd1;
        event_block = 3'd1;
        #0.1;
        if (load_ready || event_ready)
            $fatal(1, "M104 collision exposed ready");
        @(posedge clk_core); #0.5;
        if (!protocol_error)
            $fatal(1, "M104 collision did not fault");
        protocol_attacks++;

        // A new load cannot overwrite a live held vector.
        reset_dut();
        load_key(5, 6, 51);
        attack_load_first(0, 5, 6);

        // Exact fail-closed attack: retain an older stalled output internally,
        // but quarantine it immediately when a bad new event arrives.
        reset_dut();
        load_key(7, 3, 61);
        @(negedge clk_core);
        output_ready = 1'b0;
        event_valid = 1'b1;
        event_source = 4'd7;
        event_block = 3'd3;
        event_negate = 1'b0;
        event_last_for_key = 1'b0;
        event_tag = 32'h104f0001;
        @(posedge clk_core);
        if (!event_ready)
            $fatal(1, "M104 buffered-fault setup event not accepted");
        accepted_events++;
        @(negedge clk_core);
        event_valid = 1'b0;
        #0.1;
        if (!output_valid || !dut.output_valid_q)
            $fatal(1, "M104 failed to create stalled buffered output");
        attack_event(8, 3, 1'b1);
        buffered_fault_attacks++;
        @(negedge clk_core);
        output_ready = 1'b1;
        repeat (2) begin
            @(posedge clk_core); #0.5;
            if (!protocol_error || output_valid || output_accept
                    || !dut.output_valid_q)
                $fatal(1, "M104 buffered result escaped sticky fault");
        end

        if (loaded_groups != 7 || accepted_load_beats != 24
                || accepted_events != 10 || consecutive_event_pairs != 5
                || stall_cycles != 3 || protocol_attacks != 10
                || continuation_attacks != 3 || buffered_fault_attacks != 1
                || accepted_event_grace_holds != 1)
            $fatal(1, "M104 counter mismatch groups=%0d load_beats=%0d events=%0d ii1=%0d stalls=%0d attacks=%0d continuation=%0d buffered=%0d grace=%0d",
                   loaded_groups, accepted_load_beats, accepted_events,
                   consecutive_event_pairs, stall_cycles, protocol_attacks,
                   continuation_attacks, buffered_fault_attacks,
                   accepted_event_grace_holds);

        $display("PASS M104 r3 symmetric event grace groups=%0d load_beats=%0d events=%0d ii1_pairs=%0d stalls=%0d protocol_attacks=%0d continuation_attacks=%0d buffered_fault_attacks=%0d accepted_event_grace_holds=%0d lanes=96 macros=0",
                 loaded_groups, accepted_load_beats, accepted_events,
                 consecutive_event_pairs, stall_cycles, protocol_attacks,
                 continuation_attacks, buffered_fault_attacks,
                 accepted_event_grace_holds);
        $finish;
    end
endmodule

`default_nettype wire
