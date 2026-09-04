`timescale 1ns/1ps
`default_nettype none

module tb_m106_independent_adversarial;
    localparam int WIN_ROWS = 64;
    localparam int ROW_W = 6;
    localparam int BASE_W = 12;
    localparam int CONTEXT_W = 16;
    localparam int MAX_EXPECTED = 10000;

    logic clk_core, rst_core;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic [ROW_W-1:0] event_row_offset;
    logic event_negate;
    logic [BASE_W-1:0] window_base_row;
    logic [CONTEXT_W-1:0] window_context;
    logic event_accept;
    logic window_close_valid, window_close_ready, window_close_accept;
    logic service_valid, service_ready, service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [ROW_W-1:0] service_row_offset;
    logic [BASE_W-1:0] service_destination_row;
    logic service_negate, service_last_for_key;
    logic [CONTEXT_W-1:0] service_context;
    logic service_accept;
    logic fill_bank, drain_bank;
    logic [1:0] bank_ready;
    logic protocol_error, busy;

    logic expected_is_event [0:MAX_EXPECTED-1];
    logic [3:0] expected_source [0:MAX_EXPECTED-1];
    logic [2:0] expected_block [0:MAX_EXPECTED-1];
    logic [1:0] expected_beat [0:MAX_EXPECTED-1];
    logic [ROW_W-1:0] expected_row [0:MAX_EXPECTED-1];
    logic [BASE_W-1:0] expected_destination [0:MAX_EXPECTED-1];
    logic expected_negate [0:MAX_EXPECTED-1];
    logic expected_last [0:MAX_EXPECTED-1];
    logic [CONTEXT_W-1:0] expected_context [0:MAX_EXPECTED-1];

    integer expected_write, expected_read;
    integer accepted_events, accepted_closes, accepted_services;
    integer load_tokens, event_tokens, stall_cycles;
    integer overlap_accepts, event_grace_cycles, close_grace_cycles;
    integer protocol_attacks, reset_recoveries, empty_windows;
    integer close_accept_observed, event_accept_observed;
    logic scoreboard_enable;
    logic [8*32-1:0] test_case;

    m106_bounded_bitmap_transpose_scheduler dut (.*);

    m106_independent_adversarial_assertions hammer_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_accept(event_accept),
        .window_close_valid(window_close_valid),
        .window_close_ready(window_close_ready),
        .window_close_accept(window_close_accept),
        .service_valid(service_valid), .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source), .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_destination_row(service_destination_row),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_context(service_context),
        .service_accept(service_accept), .protocol_error(protocol_error),
        .illegal_request(dut.illegal_request),
        .fill_available(dut.fill_available_q),
        .drain_active(dut.drain_active_q),
        .fill_bank(fill_bank), .drain_bank(drain_bank),
        .accepted_event_grace(dut.accepted_event_grace_q),
        .accepted_close_grace(dut.accepted_close_grace_q),
        .accepted_event_grace_match(dut.accepted_event_grace_match),
        .accepted_close_grace_match(dut.accepted_close_grace_match)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #2000000;
        $fatal(1, "M106 independent watchdog timeout case=%s expected=%0d/%0d fault=%0d",
               test_case, expected_read, expected_write, protocol_error);
    end

    initial begin : geometry_audit
        if ($bits(dut.row_valid_q) != 16384)
            $fatal(1, "M106 presence bitmap geometry mismatch bits=%0d",
                   $bits(dut.row_valid_q));
        if ($bits(dut.row_negate_q) != 16384)
            $fatal(1, "M106 direction bitmap geometry mismatch bits=%0d",
                   $bits(dut.row_negate_q));
        if ($bits(dut.active_key_q) != 256
                || $bits(dut.bank_base_q) != 24
                || $bits(dut.bank_context_q) != 32
                || $bits(dut.identity_valid_q) != 2)
            $fatal(1, "M106 bank metadata geometry mismatch active=%0d base=%0d context=%0d identity=%0d",
                   $bits(dut.active_key_q), $bits(dut.bank_base_q),
                   $bits(dut.bank_context_q),
                   $bits(dut.identity_valid_q));
    end

    task automatic push_load(
        input integer source,
        input integer block,
        input integer beat,
        input integer context_value
    );
        begin
            if (expected_write >= MAX_EXPECTED)
                $fatal(1, "M106 independent expected queue overflow");
            expected_is_event[expected_write] = 1'b0;
            expected_source[expected_write] = source;
            expected_block[expected_write] = block;
            expected_beat[expected_write] = beat;
            expected_row[expected_write] = '0;
            expected_destination[expected_write] = '0;
            expected_negate[expected_write] = 1'b0;
            expected_last[expected_write] = 1'b0;
            expected_context[expected_write] = context_value;
            expected_write++;
        end
    endtask

    task automatic push_event(
        input integer source,
        input integer block,
        input integer row,
        input logic negate,
        input logic last,
        input integer base,
        input integer context_value
    );
        begin
            if (expected_write >= MAX_EXPECTED)
                $fatal(1, "M106 independent expected queue overflow");
            expected_is_event[expected_write] = 1'b1;
            expected_source[expected_write] = source;
            expected_block[expected_write] = block;
            expected_beat[expected_write] = '0;
            expected_row[expected_write] = row;
            expected_destination[expected_write] = base + row;
            expected_negate[expected_write] = negate;
            expected_last[expected_write] = last;
            expected_context[expected_write] = context_value;
            expected_write++;
        end
    endtask

    task automatic push_key(input integer key, input integer context_value);
        begin
            push_load(key >> 3, key & 7, 0, context_value);
            push_load(key >> 3, key & 7, 1, context_value);
            push_load(key >> 3, key & 7, 2, context_value);
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            event_valid = 1'b0;
            window_close_valid = 1'b0;
            service_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            @(posedge clk_core); #0.1;
            if (protocol_error || service_valid || !event_ready
                    || window_close_accept || event_accept)
                $fatal(1, "M106 independent reset recovery failed");
            reset_recoveries++;
        end
    endtask

    task automatic drive_event_once(
        input integer source,
        input integer block,
        input integer row,
        input logic negate,
        input integer base,
        input integer context_value,
        input integer exact_hold_edges
    );
        begin
            @(negedge clk_core);
            event_source = source;
            event_block = block;
            event_row_offset = row;
            event_negate = negate;
            window_base_row = base;
            window_context = context_value;
            event_valid = 1'b0;
            #0.1;
            while (!event_ready) begin
                @(posedge clk_core); #0.1;
                if (protocol_error)
                    $fatal(1, "M106 fault while waiting for ingress availability key=%0d:%0d row=%0d",
                           source, block, row);
            end
            @(negedge clk_core);
            event_valid = 1'b1;
            while (1) begin
                @(posedge clk_core);
                if (event_accept)
                    break;
                #0.1;
                if (protocol_error)
                    $fatal(1, "M106 unexpected ingress fault key=%0d:%0d row=%0d",
                           source, block, row);
            end
            #0.1;
            accepted_events++;
            repeat (exact_hold_edges) begin
                @(posedge clk_core); #0.1;
                if (!dut.accepted_event_grace_match || event_ready
                        || event_accept || protocol_error)
                    $fatal(1, "M106 exact event grace failed");
                event_grace_cycles++;
            end
            @(negedge clk_core);
            event_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (dut.accepted_event_grace_q)
                $fatal(1, "M106 event grace did not clear after valid-low edge");
        end
    endtask

    task automatic close_window(
        input integer base,
        input integer context_value,
        input integer exact_hold_edges
    );
        begin
            @(negedge clk_core);
            window_base_row = base;
            window_context = context_value;
            window_close_valid = 1'b0;
            #0.1;
            while (!window_close_ready) begin
                @(posedge clk_core); #0.1;
                if (protocol_error)
                    $fatal(1, "M106 fault while waiting for close availability base=%0d context=%0h",
                           base, context_value);
            end
            @(negedge clk_core);
            window_close_valid = 1'b1;
            while (1) begin
                @(posedge clk_core);
                if (window_close_accept)
                    break;
                #0.1;
                if (protocol_error)
                    $fatal(1, "M106 unexpected close fault base=%0d context=%0h",
                           base, context_value);
            end
            #0.1;
            accepted_closes++;
            repeat (exact_hold_edges) begin
                @(posedge clk_core); #0.1;
                if (!dut.accepted_close_grace_match || window_close_ready
                        || window_close_accept || protocol_error)
                    $fatal(1, "M106 exact close grace failed");
                close_grace_cycles++;
            end
            @(negedge clk_core);
            window_close_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (dut.accepted_close_grace_q)
                $fatal(1, "M106 close grace did not clear after valid-low edge");
        end
    endtask

    task automatic expect_full_window(
        input integer base,
        input integer context_value
    );
        begin
            for (int key = 0; key < 128; key++) begin
                push_key(key, context_value);
                for (int row = 0; row < WIN_ROWS; row++) begin
                    push_event(key >> 3, key & 7, row,
                               (key ^ row) & 1, row == WIN_ROWS - 1,
                               base, context_value);
                end
            end
        end
    endtask

    task automatic fill_full_window_reverse(
        input integer base,
        input integer context_value
    );
        begin
            // First event is held exactly for grace.  Every subsequent request
            // has a sampled valid-low edge before it is issued.
            drive_event_once(15, 7, 63, (127 ^ 63) & 1,
                             base, context_value, 2);
            for (int key = 127; key >= 0; key--) begin
                for (int row = 63; row >= 0; row--) begin
                    if (!(key == 127 && row == 63))
                        drive_event_once(key >> 3, key & 7, row,
                                         (key ^ row) & 1,
                                         base, context_value, 0);
                end
            end
        end
    endtask

    task automatic prove_fault_sticky_then_reset(input [8*48-1:0] label);
        begin
            repeat (4) begin
                @(posedge clk_core); #0.1;
                if (!protocol_error || event_ready || window_close_ready
                        || service_valid || service_accept)
                    $fatal(1, "M106 fault quarantine/stickiness failed: %s",
                           label);
            end
            protocol_attacks++;
            reset_dut();
        end
    endtask

    task automatic run_positive;
        integer full_closed_bank;
        integer full_fill_bank;
        begin
            scoreboard_enable = 1'b1;
            reset_dut();

            // Empty window: exactly one close control action and no service.
            close_window(0, 16'h1000, 0);
            empty_windows++;

            // Capacity witness: all 128 keys and every one of the 64 rows.
            expect_full_window(64, 16'h1100);
            fill_full_window_reverse(64, 16'h1100);
            full_closed_bank = fill_bank;
            close_window(64, 16'h1100, 0);
            #0.1;
            if (dut.active_key_q[full_closed_bank] !== {128{1'b1}})
                $fatal(1, "M106 did not retain all 128 active keys");
            for (int key = 0; key < 128; key++) begin
                if (dut.row_valid_q[full_closed_bank][key]
                        !== {64{1'b1}})
                    $fatal(1, "M106 full-row bitmap mismatch key=%0d", key);
            end
            full_fill_bank = full_closed_bank;

            // Sparse window behind the full drain.  Keys 0 and 8 revisit the
            // same output-block bank, with three load slots before the revisit.
            push_key(0, 16'h1200);
            push_event(0, 0, 0, 1'b0, 1'b0, 128, 16'h1200);
            push_event(0, 0, 63, 1'b1, 1'b1, 128, 16'h1200);
            push_key(7, 16'h1200);
            push_event(0, 7, 5, 1'b1, 1'b1, 128, 16'h1200);
            push_key(8, 16'h1200);
            push_event(1, 0, 0, 1'b1, 1'b1, 128, 16'h1200);

            fork
                begin : stalls
                    wait (service_valid && !service_is_event);
                    @(negedge clk_core); service_ready = 1'b0;
                    repeat (3) @(posedge clk_core);
                    @(negedge clk_core); service_ready = 1'b1;
                    wait (service_valid && service_is_event);
                    @(negedge clk_core); service_ready = 1'b0;
                    repeat (4) @(posedge clk_core);
                    @(negedge clk_core); service_ready = 1'b1;
                end
                begin : sparse_fill
                    drive_event_once(1, 0, 0, 1'b1, 128, 16'h1200, 0);
                    drive_event_once(0, 7, 5, 1'b1, 128, 16'h1200, 0);
                    drive_event_once(0, 0, 63, 1'b1, 128, 16'h1200, 0);
                    drive_event_once(0, 0, 0, 1'b0, 128, 16'h1200, 0);
                    // The alternate bank is still draining the 8K-event full
                    // window, so exact held close must not be reaccepted.
                    close_window(128, 16'h1200, 2);
                end
            join

            // Once the original full bank is reclaimed, fill it again while
            // the sparse successor drains.  This is the bank-reuse witness.
            push_key(2, 16'h1300);
            push_event(0, 2, 1, 1'b0, 1'b1, 192, 16'h1300);
            push_key(127, 16'h1300);
            push_event(15, 7, 63, 1'b1, 1'b1, 192, 16'h1300);
            drive_event_once(15, 7, 63, 1'b1, 192, 16'h1300, 0);
            drive_event_once(0, 2, 1, 1'b0, 192, 16'h1300, 0);
            if (fill_bank != full_fill_bank)
                $fatal(1, "M106 full bank was not safely reused expected=%0d got=%0d",
                       full_fill_bank, fill_bank);
            close_window(192, 16'h1300, 0);

            wait (expected_read == expected_write && !service_valid
                  && bank_ready == 0 && !busy);
            repeat (3) @(posedge clk_core);
            if (expected_write != 8597 || accepted_services != 8597
                    || load_tokens != 399 || event_tokens != 8198
                    || accepted_events != 8198 || accepted_closes != 4
                    || stall_cycles != 7 || overlap_accepts == 0
                    || event_grace_cycles != 2 || close_grace_cycles != 2
                    || protocol_error)
                $fatal(1, "M106 positive totals mismatch expected=%0d service=%0d load=%0d event=%0d ingress=%0d close=%0d stall=%0d overlap=%0d egrace=%0d cgrace=%0d fault=%0d",
                       expected_write, accepted_services, load_tokens,
                       event_tokens, accepted_events, accepted_closes,
                       stall_cycles, overlap_accepts, event_grace_cycles,
                       close_grace_cycles, protocol_error);

            scoreboard_enable = 1'b0;

            // Duplicate (key,row) after a sampled valid-low edge.
            reset_dut();
            drive_event_once(2, 3, 4, 1'b0, 256, 16'h2000, 0);
            @(negedge clk_core);
            event_source = 2; event_block = 3; event_row_offset = 4;
            event_negate = 1'b1; window_base_row = 256;
            window_context = 16'h2000; event_valid = 1'b1;
            #0.1;
            if (!dut.duplicate_event || !protocol_error || event_ready)
                $fatal(1, "M106 duplicate did not fail closed");
            prove_fault_sticky_then_reset("duplicate key,row");

            // Context mutation inside one open window.
            drive_event_once(4, 2, 6, 1'b0, 320, 16'h2100, 0);
            @(negedge clk_core);
            event_source = 4; event_block = 2; event_row_offset = 7;
            event_negate = 1'b0; window_base_row = 320;
            window_context = 16'h2101; event_valid = 1'b1;
            #0.1;
            if (!dut.event_violation || !protocol_error || event_ready)
                $fatal(1, "M106 context mutation did not fail closed");
            prove_fault_sticky_then_reset("context mutation");

            // Base mutation inside one open window.
            drive_event_once(5, 1, 2, 1'b1, 384, 16'h2200, 0);
            @(negedge clk_core);
            event_source = 5; event_block = 1; event_row_offset = 3;
            event_negate = 1'b0; window_base_row = 448;
            window_context = 16'h2200; event_valid = 1'b1;
            #0.1;
            if (!dut.event_violation || !protocol_error || event_ready)
                $fatal(1, "M106 base mutation did not fail closed");
            prove_fault_sticky_then_reset("base mutation");

            // Ingress/close collision releases a previously stalled service;
            // the old service must be quarantined combinationally.
            drive_event_once(3, 1, 8, 1'b1, 512, 16'h2300, 0);
            close_window(512, 16'h2300, 0);
            service_ready = 1'b0;
            wait (service_valid);
            @(negedge clk_core);
            event_source = 0; event_block = 0; event_row_offset = 0;
            event_negate = 1'b0; window_base_row = 576;
            window_context = 16'h2301;
            event_valid = 1'b1; window_close_valid = 1'b1;
            service_ready = 1'b1;
            #0.1;
            if (!dut.request_collision || !protocol_error || service_valid
                    || service_accept || event_ready || window_close_ready)
                $fatal(1, "M106 event/close collision did not quarantine");
            prove_fault_sticky_then_reset("event close collision");

            if (protocol_attacks != 4 || reset_recoveries != 6)
                $fatal(1, "M106 fault/reset coverage mismatch attacks=%0d resets=%0d",
                       protocol_attacks, reset_recoveries);

            $display("PASS M106 independent positive windows=4 empty=1 full_keys=128 full_rows=64 ingress_events=8198 active_keys=133 load_tokens=399 event_tokens=8198 service_tokens=8597 stalls=7 overlaps=%0d event_grace_cycles=2 close_grace_cycles=2 protocol_attacks=4 reset_recoveries=6 presence_bits=16384 direction_bits=16384 bitmap_payload_bits=32768 bank_metadata_bits_min=314 accumulator_implemented=false actual_record_replay=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                     overlap_accepts);
            $finish;
        end
    endtask

    task automatic run_close_exact_hold_witness;
        begin
            scoreboard_enable = 1'b0;
            reset_dut();
            @(negedge clk_core);
            window_base_row = 0;
            window_context = 16'h3000;
            window_close_valid = 1'b1;
            @(posedge clk_core); #0.1;
            if (close_accept_observed != 1
                    || !dut.accepted_close_grace_q)
                $fatal(1, "M106 witness setup failed for exact close hold");
            // Keep the exact payload and valid asserted.  The frozen policy
            // requires ready/accept low; current RTL exposes the new fill bank.
            @(posedge clk_core); #0.1;
            if (close_accept_observed == 1 && !window_close_ready)
                $display("PASS M106 exact-close hold correctly blocked");
            else
                $fatal(1, "P0 M106 exact held close reaccepted across bank switch accepts=%0d ready=%0d grace=%0d fill_bank=%0d",
                       close_accept_observed, window_close_ready,
                       dut.accepted_close_grace_q, fill_bank);
            $finish;
        end
    endtask

    task automatic run_event_mutation_witness;
        begin
            scoreboard_enable = 1'b0;
            reset_dut();
            @(negedge clk_core);
            event_source = 1; event_block = 1; event_row_offset = 1;
            event_negate = 1'b0; window_base_row = 64;
            window_context = 16'h3100; event_valid = 1'b1;
            @(posedge clk_core); #0.1;
            if (event_accept_observed != 1
                    || !dut.accepted_event_grace_q)
                $fatal(1, "M106 witness setup failed for event mutation");
            @(negedge clk_core);
            event_row_offset = 2;
            #0.1;
            if (protocol_error && !event_ready)
                $display("PASS M106 held-event mutation correctly quarantined");
            else begin
                @(posedge clk_core); #0.1;
                $fatal(1, "P0 M106 held-event identity mutation accepted without valid-low accepts=%0d ready=%0d fault=%0d grace=%0d",
                       event_accept_observed, event_ready, protocol_error,
                       dut.accepted_event_grace_q);
            end
            $finish;
        end
    endtask

    task automatic run_close_mutation_witness;
        begin
            scoreboard_enable = 1'b0;
            reset_dut();
            @(negedge clk_core);
            window_base_row = 128;
            window_context = 16'h3200;
            window_close_valid = 1'b1;
            @(posedge clk_core); #0.1;
            if (close_accept_observed != 1
                    || !dut.accepted_close_grace_q)
                $fatal(1, "M106 witness setup failed for close mutation");
            @(negedge clk_core);
            window_base_row = 192;
            window_context = 16'h3201;
            #0.1;
            if (protocol_error && !window_close_ready)
                $display("PASS M106 held-close mutation correctly quarantined");
            else begin
                @(posedge clk_core); #0.1;
                $fatal(1, "P0 M106 held-close identity mutation accepted across bank switch accepts=%0d ready=%0d fault=%0d grace=%0d",
                       close_accept_observed, window_close_ready,
                       protocol_error, dut.accepted_close_grace_q);
            end
            $finish;
        end
    endtask

    always @(posedge clk_core) begin : independent_scoreboard
        if (!rst_core) begin
            if (window_close_accept)
                close_accept_observed++;
            if (event_accept)
                event_accept_observed++;
            if (service_valid && !service_ready)
                stall_cycles++;
            if (service_valid && event_accept
                    && dut.fill_bank_q != dut.drain_bank_q)
                overlap_accepts++;
            if (scoreboard_enable && service_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M106 independent unexpected service token");
                if (service_context == 16'h1000)
                    $fatal(1, "M106 empty window emitted a service token");
                if (service_is_event !== expected_is_event[expected_read]
                        || service_source !== expected_source[expected_read]
                        || service_block !== expected_block[expected_read]
                        || service_load_beat !== expected_beat[expected_read]
                        || service_row_offset !== expected_row[expected_read]
                        || service_destination_row
                           !== expected_destination[expected_read]
                        || service_negate !== expected_negate[expected_read]
                        || service_last_for_key !== expected_last[expected_read]
                        || service_context !== expected_context[expected_read])
                    $fatal(1, "M106 independent token mismatch idx=%0d event=%0d/%0d key=%0d:%0d/%0d:%0d beat=%0d/%0d row=%0d/%0d dest=%0d/%0d neg=%0d/%0d last=%0d/%0d context=%0h/%0h",
                           expected_read, service_is_event,
                           expected_is_event[expected_read], service_source,
                           service_block, expected_source[expected_read],
                           expected_block[expected_read], service_load_beat,
                           expected_beat[expected_read], service_row_offset,
                           expected_row[expected_read],
                           service_destination_row,
                           expected_destination[expected_read], service_negate,
                           expected_negate[expected_read],
                           service_last_for_key,
                           expected_last[expected_read], service_context,
                           expected_context[expected_read]);
                expected_read++;
                accepted_services++;
                if (service_is_event)
                    event_tokens++;
                else
                    load_tokens++;
            end
        end
    end

    initial begin : dispatch
        clk_core = 1'b0;
        rst_core = 1'b1;
        event_valid = 1'b0;
        event_source = '0;
        event_block = '0;
        event_row_offset = '0;
        event_negate = 1'b0;
        window_base_row = '0;
        window_context = '0;
        window_close_valid = 1'b0;
        service_ready = 1'b1;
        expected_write = 0;
        expected_read = 0;
        accepted_events = 0;
        accepted_closes = 0;
        accepted_services = 0;
        load_tokens = 0;
        event_tokens = 0;
        stall_cycles = 0;
        overlap_accepts = 0;
        event_grace_cycles = 0;
        close_grace_cycles = 0;
        protocol_attacks = 0;
        reset_recoveries = 0;
        empty_windows = 0;
        close_accept_observed = 0;
        event_accept_observed = 0;
        scoreboard_enable = 1'b0;
        test_case = "POSITIVE";
        void'($value$plusargs("CASE=%s", test_case));

        if (test_case == "POSITIVE")
            run_positive();
        else if (test_case == "CLOSE_EXACT_HOLD")
            run_close_exact_hold_witness();
        else if (test_case == "EVENT_MUTATION")
            run_event_mutation_witness();
        else if (test_case == "CLOSE_MUTATION")
            run_close_mutation_witness();
        else
            $fatal(1, "M106 unknown independent case=%s", test_case);
    end
endmodule

`default_nettype wire
