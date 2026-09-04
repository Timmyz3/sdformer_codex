`timescale 1ns/1ps
`default_nettype none

module tb_m110_w384_independent;
    localparam int WIN_ROWS = 384;
    localparam int ROW_W = 9;
    localparam int BASE_W = 12;
    localparam int CONTEXT_W = 16;
    localparam int KEYS = 128;
    localparam int WINDOWS = 2;
    localparam int EVENTS_PER_WINDOW = KEYS * WIN_ROWS;
    localparam int LOADS_PER_WINDOW = KEYS * 3;

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

    integer cycle_count;
    integer ingress_events;
    integer service_events;
    integer service_loads;
    integer service_tokens;
    integer stall_cycles;
    integer overlap_cycles;
    integer event_ii1_pairs;
    integer exact_event_grace_cycles;
    integer exact_close_grace_cycles;
    integer cross_bank_close_grace_cycles;
    integer protocol_attacks;
    integer reset_recoveries;
    integer expected_window;
    integer expected_key;
    integer expected_row;
    integer expected_beat;
    logic expected_event_phase;
    logic previous_event_accept;
    logic scoreboard_enable;
    logic stall_pattern_enable;
    logic manual_service_ready;

    m110_w384_bounded_bitmap_transpose_scheduler dut (.*);

    m110_w384_independent_assertions hammer_sva (
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
        .request_collision(dut.request_collision),
        .illegal_request(dut.illegal_request),
        .event_semantically_valid(dut.event_semantically_valid),
        .close_semantically_valid(dut.close_semantically_valid),
        .accepted_event_grace(dut.accepted_event_grace_q),
        .accepted_close_grace(dut.accepted_close_grace_q),
        .accepted_event_grace_match(dut.accepted_event_grace_match),
        .accepted_close_grace_match(dut.accepted_close_grace_match),
        .row_in_range(dut.row_in_range),
        .fill_available(dut.fill_available_q),
        .drain_active(dut.drain_active_q),
        .fill_bank(fill_bank), .drain_bank(drain_bank)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #3000000;
        $fatal(1, "M110 independent watchdog window=%0d key=%0d row=%0d tokens=%0d fault=%0d",
               expected_window, expected_key, expected_row, service_tokens,
               protocol_error);
    end

    initial begin : geometry_audit
        if ($bits(dut.row_valid_q) != 98304
                || $bits(dut.row_negate_q) != 98304)
            $fatal(1, "M110 bitmap geometry mismatch presence=%0d direction=%0d",
                   $bits(dut.row_valid_q), $bits(dut.row_negate_q));
        if ($bits(dut.active_key_q) != 256
                || $bits(dut.bank_base_q) != 24
                || $bits(dut.bank_context_q) != 32
                || $bits(dut.identity_valid_q) != 2)
            $fatal(1, "M110 metadata geometry mismatch active=%0d base=%0d context=%0d identity=%0d",
                   $bits(dut.active_key_q), $bits(dut.bank_base_q),
                   $bits(dut.bank_context_q),
                   $bits(dut.identity_valid_q));
    end

    function automatic logic expected_direction(
        input integer window_index,
        input integer key,
        input integer row
    );
        expected_direction = ((window_index * 3) ^ key ^ (row * 5)) & 1;
    endfunction

    function automatic integer expected_base(input integer window_index);
        expected_base = window_index == 0 ? 137 : 701;
    endfunction

    function automatic integer expected_context(input integer window_index);
        expected_context = window_index == 0 ? 16'h6a10 : 16'h6a11;
    endfunction

    always @(negedge clk_core) begin
        if (!rst_core) begin
            if (stall_pattern_enable)
                service_ready = ((cycle_count % 11) != 3)
                             && ((cycle_count % 23) != 7)
                             && ((cycle_count % 41) != 19);
            else
                service_ready = manual_service_ready;
        end
    end

    always @(posedge clk_core) begin : monitor
        if (rst_core) begin
            previous_event_accept = 1'b0;
        end else begin
            cycle_count = cycle_count + 1;
            if (scoreboard_enable) begin
                if (event_accept && previous_event_accept)
                    event_ii1_pairs = event_ii1_pairs + 1;
                previous_event_accept = event_accept;
                if (event_valid && service_valid)
                    overlap_cycles = overlap_cycles + 1;
                if (service_valid && !service_ready)
                    stall_cycles = stall_cycles + 1;
                if (protocol_error)
                    $fatal(1, "M110 unexpected fault in full-capacity phase");
                if (service_accept) begin
                    service_tokens = service_tokens + 1;
                    if (expected_window >= WINDOWS)
                        $fatal(1, "M110 extra service token");
                    if (service_source !== expected_key[6:3]
                            || service_block !== expected_key[2:0]
                            || service_context
                               !== expected_context(expected_window)[15:0])
                        $fatal(1, "M110 service identity mismatch win=%0d key=%0d",
                               expected_window, expected_key);
                    if (!expected_event_phase) begin
                        if (service_is_event
                                || service_load_beat !== expected_beat[1:0]
                                || service_row_offset !== 0
                                || service_destination_row !== 0
                                || service_negate || service_last_for_key)
                            $fatal(1, "M110 load mismatch win=%0d key=%0d beat=%0d",
                                   expected_window, expected_key,
                                   expected_beat);
                        service_loads = service_loads + 1;
                        if (expected_beat == 2) begin
                            expected_beat = 0;
                            expected_event_phase = 1'b1;
                            expected_row = 0;
                        end else begin
                            expected_beat = expected_beat + 1;
                        end
                    end else begin
                        if (!service_is_event
                                || service_row_offset !== expected_row[8:0]
                                || service_destination_row
                                   !== (expected_base(expected_window)
                                        + expected_row)
                                || service_negate !== expected_direction(
                                    expected_window, expected_key,
                                    expected_row)
                                || service_last_for_key
                                   !== (expected_row == WIN_ROWS - 1))
                            $fatal(1, "M110 event mismatch win=%0d key=%0d row=%0d",
                                   expected_window, expected_key,
                                   expected_row);
                        service_events = service_events + 1;
                        if (expected_row == WIN_ROWS - 1) begin
                            expected_event_phase = 1'b0;
                            expected_beat = 0;
                            expected_row = 0;
                            if (expected_key == KEYS - 1) begin
                                expected_key = 0;
                                expected_window = expected_window + 1;
                            end else begin
                                expected_key = expected_key + 1;
                            end
                        end else begin
                            expected_row = expected_row + 1;
                        end
                    end
                end
            end else begin
                previous_event_accept = 1'b0;
            end
        end
    end

    task automatic reset_dut;
        begin
            scoreboard_enable = 1'b0;
            stall_pattern_enable = 1'b0;
            manual_service_ready = 1'b1;
            @(negedge clk_core);
            rst_core = 1'b1;
            event_valid = 1'b0;
            event_source = 0;
            event_block = 0;
            event_row_offset = 0;
            event_negate = 0;
            window_base_row = 0;
            window_context = 0;
            window_close_valid = 1'b0;
            service_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            @(posedge clk_core);
            if (protocol_error || service_valid || !event_ready)
                $fatal(1, "M110 reset recovery failed");
            reset_recoveries = reset_recoveries + 1;
        end
    endtask

    task automatic accept_event_then_sample_low(
        input integer source, input integer block, input integer row,
        input logic negate, input integer base, input integer context_value
    );
        begin
            @(negedge clk_core);
            event_source = source[3:0];
            event_block = block[2:0];
            event_row_offset = row[8:0];
            event_negate = negate;
            window_base_row = base[11:0];
            window_context = context_value[15:0];
            event_valid = 1'b1;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M110 setup event was not accepted");
            @(negedge clk_core);
            event_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (dut.accepted_event_grace_q)
                $fatal(1, "M110 event grace did not clear");
        end
    endtask

    task automatic accept_close_then_sample_low(
        input integer base, input integer context_value
    );
        begin
            @(negedge clk_core);
            window_base_row = base[11:0];
            window_context = context_value[15:0];
            window_close_valid = 1'b1;
            @(posedge clk_core);
            if (!window_close_accept)
                $fatal(1, "M110 setup close was not accepted");
            @(negedge clk_core);
            window_close_valid = 1'b0;
            @(posedge clk_core);
        end
    endtask

    task automatic changed_close_stream_probe;
        begin
            reset_dut();
            @(negedge clk_core);
            window_base_row = 12'd32;
            window_context = 16'h6100;
            window_close_valid = 1'b1;
            @(posedge clk_core);
            if (!window_close_accept)
                $fatal(1, "M110 first empty close was not accepted");
            @(negedge clk_core);
            window_base_row = 12'd416;
            window_context = 16'h6101;
            @(posedge clk_core);
            if (!window_close_accept || protocol_error)
                $fatal(1, "M110 changed legal close did not stream at II1");
            @(posedge clk_core);
            if (window_close_accept || window_close_ready || protocol_error)
                $fatal(1, "M110 exact second close was not grace-held");
            @(negedge clk_core);
            window_close_valid = 1'b0;
        end
    endtask

    task automatic fill_full_reverse_window(input integer window_index);
        begin
            for (int key = KEYS - 1; key >= 0; key--) begin
                for (int row = WIN_ROWS - 1; row >= 0; row--) begin
                    @(negedge clk_core);
                    event_valid = 1'b1;
                    event_source = key[6:3];
                    event_block = key[2:0];
                    event_row_offset = row[8:0];
                    event_negate = expected_direction(
                        window_index, key, row);
                    window_base_row = expected_base(window_index)[11:0];
                    window_context = expected_context(window_index)[15:0];
                    @(posedge clk_core);
                    if (!event_accept)
                        $fatal(1, "M110 reverse stream bubble win=%0d key=%0d row=%0d",
                               window_index, key, row);
                    ingress_events = ingress_events + 1;
                end
            end
            // Hold the exact final accepted event for one extra edge.
            @(posedge clk_core);
            if (event_ready || event_accept || protocol_error
                    || !dut.accepted_event_grace_match)
                $fatal(1, "M110 exact event grace failed win=%0d",
                       window_index);
            exact_event_grace_cycles = exact_event_grace_cycles + 1;
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic close_full_window_with_exact_grace(
        input integer window_index
    );
        begin
            @(negedge clk_core);
            window_base_row = expected_base(window_index)[11:0];
            window_context = expected_context(window_index)[15:0];
            window_close_valid = 1'b1;
            @(posedge clk_core);
            if (!window_close_accept)
                $fatal(1, "M110 full close was not accepted win=%0d",
                       window_index);
            @(posedge clk_core);
            if (window_close_ready || window_close_accept || protocol_error
                    || !dut.accepted_close_grace_match)
                $fatal(1, "M110 full exact close grace failed win=%0d",
                       window_index);
            exact_close_grace_cycles = exact_close_grace_cycles + 1;
            if (dut.fill_available_q)
                cross_bank_close_grace_cycles =
                    cross_bank_close_grace_cycles + 1;
            @(negedge clk_core);
            window_close_valid = 1'b0;
        end
    endtask

    task automatic full_capacity_reverse_campaign;
        begin
            reset_dut();
            ingress_events = 0;
            service_events = 0;
            service_loads = 0;
            service_tokens = 0;
            stall_cycles = 0;
            overlap_cycles = 0;
            event_ii1_pairs = 0;
            exact_event_grace_cycles = 0;
            exact_close_grace_cycles = 0;
            cross_bank_close_grace_cycles = 0;
            expected_window = 0;
            expected_key = 0;
            expected_row = 0;
            expected_beat = 0;
            expected_event_phase = 1'b0;
            previous_event_accept = 1'b0;
            scoreboard_enable = 1'b1;
            stall_pattern_enable = 1'b1;

            fill_full_reverse_window(0);
            close_full_window_with_exact_grace(0);
            fill_full_reverse_window(1);
            close_full_window_with_exact_grace(1);

            do begin
                @(posedge clk_core); #0.1;
            end while (expected_window != WINDOWS || busy);
            repeat (3) @(posedge clk_core);

            if (ingress_events != WINDOWS * EVENTS_PER_WINDOW
                    || service_events != WINDOWS * EVENTS_PER_WINDOW
                    || service_loads != WINDOWS * LOADS_PER_WINDOW
                    || service_tokens
                       != WINDOWS * (EVENTS_PER_WINDOW + LOADS_PER_WINDOW)
                    || event_ii1_pairs != WINDOWS * (EVENTS_PER_WINDOW - 1)
                    || exact_event_grace_cycles != WINDOWS
                    || exact_close_grace_cycles != WINDOWS
                    || cross_bank_close_grace_cycles < 1
                    || overlap_cycles == 0 || stall_cycles == 0)
                $fatal(1, "M110 full campaign accounting ingress=%0d events=%0d loads=%0d tokens=%0d ii1=%0d egrace=%0d cgrace=%0d cross=%0d overlap=%0d stalls=%0d",
                       ingress_events, service_events, service_loads,
                       service_tokens, event_ii1_pairs,
                       exact_event_grace_cycles,
                       exact_close_grace_cycles,
                       cross_bank_close_grace_cycles, overlap_cycles,
                       stall_cycles);
            scoreboard_enable = 1'b0;
            stall_pattern_enable = 1'b0;
        end
    endtask

    task automatic prove_fault_then_reset(input string attack_name);
        begin
            protocol_attacks = protocol_attacks + 1;
            @(posedge clk_core);
            @(negedge clk_core);
            event_valid = 1'b0;
            window_close_valid = 1'b0;
            repeat (3) begin
                @(posedge clk_core); #0.1;
                if (!protocol_error || event_ready || window_close_ready
                        || service_valid || service_accept)
                    $fatal(1, "M110 fault was not sticky/quarantined: %s",
                           attack_name);
            end
            reset_dut();
        end
    endtask

    task automatic illegal_campaign;
        begin
            reset_dut();
            // Boundary attack: 384 is the first illegal 9-bit row code.
            @(negedge clk_core);
            event_source = 0; event_block = 0;
            event_row_offset = 9'd384; event_negate = 0;
            window_base_row = 12'd64; window_context = 16'h7100;
            event_valid = 1'b1;
            #0.1;
            if (dut.row_in_range || !dut.event_violation
                    || !protocol_error || event_ready || event_accept)
                $fatal(1, "M110 boundary range attack did not fail closed");
            prove_fault_then_reset("row384 range");

            accept_event_then_sample_low(1, 0, 17, 1'b0,
                                         128, 16'h7101);
            @(negedge clk_core);
            event_source = 1; event_block = 0;
            event_row_offset = 17; event_negate = 0;
            window_base_row = 128; window_context = 16'h7101;
            event_valid = 1'b1;
            #0.1;
            if (!dut.duplicate_event || !protocol_error || event_ready)
                $fatal(1, "M110 duplicate did not fail closed");
            prove_fault_then_reset("duplicate");

            // Same key/row but changed direction/context is not exact grace;
            // it is an illegal new transaction and must fault immediately.
            @(negedge clk_core);
            event_source = 2; event_block = 0;
            event_row_offset = 21; event_negate = 0;
            window_base_row = 192; window_context = 16'h7102;
            event_valid = 1'b1;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M110 changed-illegal setup not accepted");
            @(negedge clk_core);
            event_negate = 1'b1;
            window_context = 16'h7103;
            #0.1;
            if (dut.accepted_event_grace_match || !dut.event_violation
                    || !protocol_error || event_ready)
                $fatal(1, "M110 changed illegal payload did not fail closed");
            prove_fault_then_reset("changed illegal payload");

            @(negedge clk_core);
            event_source = 3; event_block = 0;
            event_row_offset = 3; event_negate = 0;
            window_base_row = 256; window_context = 16'h7104;
            event_valid = 1'b1;
            window_close_valid = 1'b1;
            #0.1;
            if (!dut.request_collision || !protocol_error
                    || event_ready || window_close_ready)
                $fatal(1, "M110 collision did not fail closed");
            prove_fault_then_reset("event-close collision");

            // Fill/close both banks while service is stalled, then present a
            // third event.  The frozen fail-fast interface calls this illegal.
            manual_service_ready = 1'b0;
            service_ready = 1'b0;
            accept_event_then_sample_low(4, 0, 4, 1'b0,
                                         320, 16'h7105);
            accept_close_then_sample_low(320, 16'h7105);
            accept_event_then_sample_low(5, 0, 5, 1'b1,
                                         704, 16'h7106);
            accept_close_then_sample_low(704, 16'h7106);
            if (dut.fill_available_q)
                $fatal(1, "M110 unavailable setup retained fill bank");
            @(negedge clk_core);
            event_source = 6; event_block = 0;
            event_row_offset = 6; event_negate = 0;
            window_base_row = 1088; window_context = 16'h7107;
            event_valid = 1'b1;
            #0.1;
            if (!dut.event_violation || !protocol_error || event_ready
                    || service_valid || service_accept)
                $fatal(1, "M110 unavailable ingress did not quarantine");
            prove_fault_then_reset("unavailable bank");

            if (protocol_attacks != 5)
                $fatal(1, "M110 attack count mismatch %0d",
                       protocol_attacks);
        end
    endtask

    initial begin : test
        clk_core = 1'b0;
        rst_core = 1'b1;
        event_valid = 1'b0;
        event_source = 0;
        event_block = 0;
        event_row_offset = 0;
        event_negate = 0;
        window_base_row = 0;
        window_context = 0;
        window_close_valid = 1'b0;
        service_ready = 1'b1;
        cycle_count = 0;
        ingress_events = 0;
        service_events = 0;
        service_loads = 0;
        service_tokens = 0;
        stall_cycles = 0;
        overlap_cycles = 0;
        event_ii1_pairs = 0;
        exact_event_grace_cycles = 0;
        exact_close_grace_cycles = 0;
        cross_bank_close_grace_cycles = 0;
        protocol_attacks = 0;
        reset_recoveries = 0;
        expected_window = 0;
        expected_key = 0;
        expected_row = 0;
        expected_beat = 0;
        expected_event_phase = 1'b0;
        previous_event_accept = 1'b0;
        scoreboard_enable = 1'b0;
        stall_pattern_enable = 1'b0;
        manual_service_ready = 1'b1;

        changed_close_stream_probe();
        full_capacity_reverse_campaign();
        illegal_campaign();

        $display("PASS M110 independent W384 full-capacity reverse windows=2 ingress_events=98304 active_keys=256 rows_per_key=384 load_tokens=768 event_tokens=98304 service_tokens=99072 event_ii1_pairs=98302 stalls=%0d overlap_cycles=%0d exact_event_grace=2 exact_close_grace=2 cross_bank_close_grace=1 changed_legal_close_run=2 protocol_attacks=5 reset_recoveries=%0d presence_bits=98304 direction_bits=98304 raw_bitmap_bits=196608 metadata_bits_min=314 accumulator_implemented=false m109_r2_ratio_2p535_is_projection=true actual_record_replay=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 stall_cycles, overlap_cycles, reset_recoveries);
        $finish;
    end
endmodule

`default_nettype wire
