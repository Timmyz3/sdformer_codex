`timescale 1ns/1ps
`default_nettype none

module tb_m104_r3_independent_hammer;
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

    logic [767:0] weight_payload;
    logic [1151:0] expected_positive, expected_negative;
    integer accepted_events;
    integer last_lingers, nonlast_lingers;
    integer between_edge_low_high;
    integer mutation_attacks, sticky_checks;
    integer older_buffer_quarantines, reset_recoveries;

    m104_held_weight_correction_broadcaster dut (.*);
    m104_r3_independent_hammer_assertions hammer_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_source(event_source), .event_block(event_block),
        .event_negate(event_negate),
        .event_last_for_key(event_last_for_key), .event_tag(event_tag),
        .event_accept(event_accept), .output_valid(output_valid),
        .output_ready(output_ready), .output_tag(output_tag),
        .output_source(output_source), .output_block(output_block),
        .output_negate(output_negate), .output_values(output_values),
        .output_accept(output_accept), .protocol_error(protocol_error),
        .illegal_request(dut.illegal_request),
        .accepted_event_grace_match(dut.accepted_event_grace_match),
        .request_fault(dut.request_fault_q),
        .output_valid_q(dut.output_valid_q)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #100000;
        $fatal(1, "M104 r3 independent hammer watchdog timeout");
    end

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

    task automatic last_linger_with_between_edge_low_high;
        begin
            reset_dut();
            load_key(4'd12, 3'd6, 73);
            @(negedge clk_core);
            output_ready = 1'b0;
            event_valid = 1'b1;
            event_source = 4'd12;
            event_block = 3'd6;
            event_negate = 1'b0;
            event_last_for_key = 1'b1;
            event_tag = 32'h1042_0003;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M104 between-edge setup was not accepted");
            accepted_events++;
            #0.30;
            event_valid = 1'b0;
            #0.30;
            event_valid = 1'b1;
            #0.10;
            if (!dut.accepted_event_grace_match || protocol_error
                    || event_ready || event_accept || !output_valid)
                $fatal(1, "M104 between-edge low/high disturbed grace");
            @(posedge clk_core);
            if (!dut.accepted_event_grace_match || protocol_error
                    || event_ready || event_accept)
                $fatal(1, "M104 between-edge low/high was sampled as new request");
            between_edge_low_high++;
            @(negedge clk_core);
            event_valid = 1'b0;
            output_ready = 1'b1;
            @(posedge clk_core);
            if (!output_accept)
                $fatal(1, "M104 between-edge result did not retire");
            #0.1;
            if (protocol_error || output_valid || held_valid)
                $fatal(1, "M104 between-edge test did not drain");
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
            #0.1;
            if (protocol_error || held_valid || collecting || output_valid
                    || dut.output_valid_q || dut.request_fault_q)
                $fatal(1, "M104 reset failed to clear state");
            reset_recoveries++;
        end
    endtask

    task automatic build_weight(input integer seed);
        integer signed value;
        logic signed [11:0] extended;
        begin
            weight_payload = '0;
            expected_positive = '0;
            expected_negative = '0;
            for (int lane = 0; lane < 96; lane++) begin
                case (lane % 7)
                    0: value = -128;
                    1: value = 127;
                    default: value = ((lane*23 + seed*17) % 255) - 127;
                endcase
                weight_payload[lane*8 +: 8] = value[7:0];
                extended = {{4{value[7]}}, value[7:0]};
                expected_positive[lane*12 +: 12] = extended;
                expected_negative[lane*12 +: 12] = -extended;
            end
        end
    endtask

    task automatic load_key(
        input logic [3:0] source,
        input logic [2:0] block,
        input integer seed
    );
        begin
            build_weight(seed);
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                load_valid = 1'b1;
                load_source = source;
                load_block = block;
                load_beat = beat[1:0];
                load_data = weight_payload[beat*256 +: 256];
                #0.1;
                if (!load_ready || protocol_error)
                    $fatal(1, "M104 independent legal load not ready beat=%0d", beat);
                @(posedge clk_core);
                if (!load_accept)
                    $fatal(1, "M104 independent legal load not accepted beat=%0d", beat);
            end
            @(negedge clk_core);
            load_valid = 1'b0;
            #0.1;
            if (!held_valid || collecting || expected_load_beat != 0)
                $fatal(1, "M104 independent key load did not complete");
        end
    endtask

    task automatic exact_linger(
        input logic last,
        input logic [3:0] source,
        input logic [2:0] block,
        input logic negate,
        input logic [31:0] tag,
        input integer seed
    );
        logic [1151:0] captured_values;
        begin
            reset_dut();
            load_key(source, block, seed);
            @(negedge clk_core);
            output_ready = 1'b0;
            event_valid = 1'b1;
            event_source = source;
            event_block = block;
            event_negate = negate;
            event_last_for_key = last;
            event_tag = tag;
            #0.1;
            if (!event_ready || protocol_error)
                $fatal(1, "M104 independent linger setup not ready last=%0d", last);
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M104 independent linger setup not accepted last=%0d", last);
            accepted_events++;
            #0.1;
            captured_values = output_values;
            if (!dut.accepted_event_grace_match || protocol_error
                    || event_ready || event_accept || !output_valid
                    || output_tag != tag || output_source != source
                    || output_block != block || output_negate != negate
                    || output_values !== (negate
                                          ? expected_negative
                                          : expected_positive)
                    || held_valid == last)
                $fatal(1, "M104 grace state mismatch immediately after accept last=%0d", last);

            // Keep the exact accepted request asserted across the entire next
            // edge.  It is neither a new request nor an illegal request.
            @(posedge clk_core);
            if (!dut.accepted_event_grace_match || protocol_error
                    || event_ready || event_accept)
                $fatal(1, "M104 exact linger was faulted or reaccepted last=%0d", last);
            #0.1;
            if (!output_valid || output_values !== captured_values
                    || output_tag != tag || dut.request_fault_q)
                $fatal(1, "M104 stalled result changed during exact linger last=%0d", last);
            if (last) last_lingers++;
            else nonlast_lingers++;

            @(negedge clk_core);
            event_valid = 1'b0;
            #0.1;
            if (protocol_error || !output_valid)
                $fatal(1, "M104 exact linger withdrawal faulted last=%0d", last);
            @(negedge clk_core);
            output_ready = 1'b1;
            @(posedge clk_core);
            if (!output_accept)
                $fatal(1, "M104 exact linger result did not retire last=%0d", last);
            #0.1;
            if (output_valid || protocol_error)
                $fatal(1, "M104 exact linger result remained after retire last=%0d", last);
        end
    endtask

    task automatic mutate_last_grace(input integer field_index);
        logic [3:0] source;
        logic [2:0] block;
        logic negate;
        logic [31:0] tag;
        begin
            source = 4'd9;
            block = 3'd3;
            negate = field_index[0];
            tag = 32'h1043_0000 + field_index;
            reset_dut();
            load_key(source, block, 80 + field_index);
            @(negedge clk_core);
            output_ready = 1'b0;
            event_valid = 1'b1;
            event_source = source;
            event_block = block;
            event_negate = negate;
            event_last_for_key = 1'b1;
            event_tag = tag;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M104 mutation setup did not accept field=%0d", field_index);
            accepted_events++;
            #0.1;
            if (!dut.accepted_event_grace_match || !dut.output_valid_q
                    || !output_valid || held_valid)
                $fatal(1, "M104 mutation setup state bad field=%0d", field_index);

            @(negedge clk_core);
            case (field_index)
                0: event_source = source ^ 4'h1;
                1: event_block = block ^ 3'h1;
                2: event_negate = ~negate;
                3: event_last_for_key = 1'b0;
                4: event_tag = tag ^ 32'h0000_0001;
                default: $fatal(1, "M104 unknown identity field=%0d", field_index);
            endcase
            output_ready = 1'b1;
            #0.1;
            if (!dut.illegal_request || !protocol_error || event_ready
                    || event_accept || output_valid || output_accept
                    || !dut.output_valid_q)
                $fatal(1, "M104 mutation did not fail closed pre-edge field=%0d", field_index);
            @(posedge clk_core);
            #0.1;
            if (!dut.request_fault_q || !protocol_error || output_valid
                    || output_accept || !dut.output_valid_q)
                $fatal(1, "M104 mutation did not latch sticky fault field=%0d", field_index);
            mutation_attacks++;

            @(negedge clk_core);
            event_valid = 1'b0;
            repeat (2) begin
                @(posedge clk_core); #0.1;
                if (!protocol_error || output_valid || output_accept
                        || !dut.output_valid_q)
                    $fatal(1, "M104 sticky fault escaped without reset field=%0d", field_index);
                sticky_checks++;
            end
        end
    endtask

    task automatic older_buffer_fault_quarantine;
        begin
            reset_dut();
            load_key(4'd6, 3'd5, 99);
            @(negedge clk_core);
            output_ready = 1'b0;
            event_valid = 1'b1;
            event_source = 4'd6;
            event_block = 3'd5;
            event_negate = 1'b0;
            event_last_for_key = 1'b0;
            event_tag = 32'h1044_0001;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M104 older-buffer setup event was not accepted");
            accepted_events++;
            @(negedge clk_core);
            event_valid = 1'b0;
            #0.1;
            if (!output_valid || !dut.output_valid_q)
                $fatal(1, "M104 older-buffer setup result not held");
            @(posedge clk_core); #0.1;
            if (!output_valid || protocol_error)
                $fatal(1, "M104 older-buffer result did not survive stall");

            // Present an unrelated bad request while simultaneously opening
            // the sink.  The old buffered result must not escape on this edge.
            @(negedge clk_core);
            event_valid = 1'b1;
            event_source = 4'd7;
            event_block = 3'd5;
            event_negate = 1'b1;
            event_last_for_key = 1'b1;
            event_tag = 32'h1044_bad0;
            output_ready = 1'b1;
            #0.1;
            if (!dut.illegal_request || !protocol_error || output_valid
                    || output_accept || !dut.output_valid_q)
                $fatal(1, "M104 older buffered output escaped same-cycle quarantine");
            @(posedge clk_core); #0.1;
            if (!dut.request_fault_q || !protocol_error || output_valid
                    || output_accept || !dut.output_valid_q)
                $fatal(1, "M104 older buffered output escaped sticky quarantine");
            older_buffer_quarantines++;
            mutation_attacks++;
            @(negedge clk_core);
            event_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (!protocol_error || output_valid || output_accept
                    || !dut.output_valid_q)
                $fatal(1, "M104 older buffer escaped before reset");
            sticky_checks++;
        end
    endtask

    initial begin : independent_hammer
        clk_core = 1'b0;
        rst_core = 1'b1;
        accepted_events = 0;
        last_lingers = 0;
        nonlast_lingers = 0;
        between_edge_low_high = 0;
        mutation_attacks = 0;
        sticky_checks = 0;
        older_buffer_quarantines = 0;
        reset_recoveries = 0;
        clear_drives();
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        exact_linger(1'b1, 4'd10, 3'd2, 1'b1, 32'h1042_0001, 71);
        exact_linger(1'b0, 4'd11, 3'd4, 1'b0, 32'h1042_0002, 72);
        last_linger_with_between_edge_low_high();

        for (int field_index = 0; field_index < 5; field_index++)
            mutate_last_grace(field_index);

        older_buffer_fault_quarantine();

        // Reset is the only recovery path.  After it, a fresh legal key and
        // event must work again without any residue from the quarantined word.
        reset_dut();
        load_key(4'd2, 3'd1, 111);
        @(negedge clk_core);
        event_valid = 1'b1;
        event_source = 4'd2;
        event_block = 3'd1;
        event_negate = 1'b1;
        event_last_for_key = 1'b1;
        event_tag = 32'h1045_0001;
        output_ready = 1'b1;
        @(posedge clk_core);
        if (!event_accept)
            $fatal(1, "M104 post-reset recovery event not accepted");
        accepted_events++;
        @(negedge clk_core);
        event_valid = 1'b0;
        @(posedge clk_core);
        if (!output_accept)
            $fatal(1, "M104 post-reset recovery output not accepted");
        #0.1;
        if (protocol_error || held_valid || output_valid)
            $fatal(1, "M104 post-reset recovery left bad state");

        if (accepted_events != 10 || last_lingers != 1
                || nonlast_lingers != 1 || mutation_attacks != 6
                || sticky_checks != 11 || older_buffer_quarantines != 1
                || reset_recoveries != 10 || between_edge_low_high != 1)
            $fatal(1, "M104 independent counter mismatch events=%0d last=%0d nonlast=%0d low_high=%0d mutations=%0d sticky=%0d older=%0d resets=%0d",
                   accepted_events, last_lingers, nonlast_lingers,
                   between_edge_low_high,
                   mutation_attacks, sticky_checks,
                   older_buffer_quarantines, reset_recoveries);

        $display("PASS M104 r3 independent VCS last_linger=%0d nonlast_linger=%0d between_edge_low_high=%0d identity_mutations=5 older_buffer_quarantine=%0d sticky_checks=%0d reset_recoveries=%0d accepted_events=%0d macros=0",
                 last_lingers, nonlast_lingers, between_edge_low_high,
                 older_buffer_quarantines, sticky_checks,
                 reset_recoveries, accepted_events);
        $finish;
    end
endmodule

`default_nettype wire
