`timescale 1ns/1ps
`default_nettype none

module tb_m106_r2_independent_streaming;
    localparam int MAX_EXPECTED = 9000;

    logic clk_core, rst_core;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic [5:0] event_row_offset;
    logic event_negate;
    logic [11:0] window_base_row;
    logic [15:0] window_context;
    logic event_accept;
    logic window_close_valid, window_close_ready, window_close_accept;
    logic service_valid, service_ready, service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [5:0] service_row_offset;
    logic [11:0] service_destination_row;
    logic service_negate, service_last_for_key;
    logic [15:0] service_context;
    logic service_accept;
    logic fill_bank, drain_bank;
    logic [1:0] bank_ready;
    logic protocol_error, busy;

    logic expected_is_event [0:MAX_EXPECTED-1];
    logic [3:0] expected_source [0:MAX_EXPECTED-1];
    logic [2:0] expected_block [0:MAX_EXPECTED-1];
    logic [1:0] expected_beat [0:MAX_EXPECTED-1];
    logic [5:0] expected_row [0:MAX_EXPECTED-1];
    logic [11:0] expected_destination [0:MAX_EXPECTED-1];
    logic expected_negate [0:MAX_EXPECTED-1];
    logic expected_last [0:MAX_EXPECTED-1];
    logic [15:0] expected_context [0:MAX_EXPECTED-1];

    integer expected_write, expected_read;
    integer service_tokens, load_tokens, event_tokens, stall_cycles;
    integer close_accepts, event_accepts, protocol_attacks;
    integer event_accept_run, event_accept_run_max;
    integer close_accept_run, close_accept_run_max;
    integer exact_event_grace_cycles, exact_close_grace_cycles;
    integer reset_recoveries;
    logic scoreboard_enable;

    m106_bounded_bitmap_transpose_scheduler dut (.*);

    m106_r2_independent_streaming_assertions hammer_sva (
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
        .fill_available(dut.fill_available_q),
        .drain_active(dut.drain_active_q),
        .fill_bank(fill_bank), .drain_bank(drain_bank)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #2000000;
        $fatal(1, "M106 r2 independent watchdog expected=%0d/%0d fault=%0d",
               expected_read, expected_write, protocol_error);
    end

    initial begin : geometry_audit
        if ($bits(dut.row_valid_q) != 16384
                || $bits(dut.row_negate_q) != 16384)
            $fatal(1, "M106 r2 bitmap geometry mismatch presence=%0d direction=%0d",
                   $bits(dut.row_valid_q), $bits(dut.row_negate_q));
        if ($bits(dut.active_key_q) != 256
                || $bits(dut.bank_base_q) != 24
                || $bits(dut.bank_context_q) != 32
                || $bits(dut.identity_valid_q) != 2)
            $fatal(1, "M106 r2 metadata geometry mismatch active=%0d base=%0d context=%0d identity=%0d",
                   $bits(dut.active_key_q), $bits(dut.bank_base_q),
                   $bits(dut.bank_context_q),
                   $bits(dut.identity_valid_q));
    end

    task automatic push_load(input integer key, input integer beat,
                             input integer context_value);
        begin
            expected_is_event[expected_write] = 1'b0;
            expected_source[expected_write] = key >> 3;
            expected_block[expected_write] = key & 7;
            expected_beat[expected_write] = beat;
            expected_row[expected_write] = '0;
            expected_destination[expected_write] = '0;
            expected_negate[expected_write] = 1'b0;
            expected_last[expected_write] = 1'b0;
            expected_context[expected_write] = context_value;
            expected_write++;
        end
    endtask

    task automatic push_event(input integer key, input integer row,
                              input integer base,
                              input integer context_value);
        begin
            expected_is_event[expected_write] = 1'b1;
            expected_source[expected_write] = key >> 3;
            expected_block[expected_write] = key & 7;
            expected_beat[expected_write] = '0;
            expected_row[expected_write] = row;
            expected_destination[expected_write] = base + row;
            expected_negate[expected_write] = (key ^ row) & 1;
            expected_last[expected_write] = row == 63;
            expected_context[expected_write] = context_value;
            expected_write++;
        end
    endtask

    task automatic build_full_expected(input integer base,
                                       input integer context_value);
        begin
            for (int key = 0; key < 128; key++) begin
                push_load(key, 0, context_value);
                push_load(key, 1, context_value);
                push_load(key, 2, context_value);
                for (int row = 0; row < 64; row++)
                    push_event(key, row, base, context_value);
            end
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
            if (protocol_error || service_valid || !event_ready)
                $fatal(1, "M106 r2 reset recovery failed");
            reset_recoveries++;
        end
    endtask

    task automatic accept_event_then_sample_low(
        input integer source, input integer block, input integer row,
        input logic negate, input integer base, input integer context_value
    );
        begin
            @(negedge clk_core);
            event_source = source;
            event_block = block;
            event_row_offset = row;
            event_negate = negate;
            window_base_row = base;
            window_context = context_value;
            event_valid = 1'b1;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M106 r2 helper event was not accepted");
            @(negedge clk_core);
            event_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (dut.accepted_event_grace_q)
                $fatal(1, "M106 r2 event grace did not clear");
        end
    endtask

    task automatic accept_close_then_sample_low(
        input integer base, input integer context_value
    );
        begin
            @(negedge clk_core);
            window_base_row = base;
            window_context = context_value;
            window_close_valid = 1'b1;
            @(posedge clk_core);
            if (!window_close_accept)
                $fatal(1, "M106 r2 helper close was not accepted");
            @(negedge clk_core);
            window_close_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (dut.accepted_close_grace_q)
                $fatal(1, "M106 r2 close grace did not clear");
        end
    endtask

    task automatic wait_idle_no_service;
        begin
            while (busy) begin
                @(posedge clk_core); #0.1;
                if (service_valid || service_accept)
                    $fatal(1, "M106 r2 empty window emitted service");
            end
        end
    endtask

    task automatic changed_close_stream_probe;
        integer before_closes;
        begin
            reset_dut();
            before_closes = close_accepts;
            @(negedge clk_core);
            window_base_row = 0;
            window_context = 16'h4000;
            window_close_valid = 1'b1;
            @(posedge clk_core);
            if (!window_close_accept)
                $fatal(1, "M106 r2 first streaming close not accepted");
            @(negedge clk_core);
            window_base_row = 64;
            window_context = 16'h4001;
            @(posedge clk_core);
            if (!window_close_accept || protocol_error)
                $fatal(1, "M106 r2 changed legal close did not stream");
            @(negedge clk_core);
            window_close_valid = 1'b0;
            wait_idle_no_service();
            if (close_accepts - before_closes != 2
                    || close_accept_run_max < 2)
                $fatal(1, "M106 r2 changed close accounting mismatch delta=%0d maxrun=%0d",
                       close_accepts - before_closes,
                       close_accept_run_max);
        end
    endtask

    task automatic full_stream_and_exact_grace;
        integer closed_bank;
        integer before_closes;
        begin
            reset_dut();
            scoreboard_enable = 1'b1;
            build_full_expected(128, 16'h4100);

            // 8,192 distinct events remain valid continuously.  Every changed
            // legal payload must accept on the next edge (II=1).
            for (int key = 127; key >= 0; key--) begin
                for (int row = 63; row >= 0; row--) begin
                    @(negedge clk_core);
                    event_source = key >> 3;
                    event_block = key & 7;
                    event_row_offset = row;
                    event_negate = (key ^ row) & 1;
                    window_base_row = 128;
                    window_context = 16'h4100;
                    event_valid = 1'b1;
                    @(posedge clk_core);
                    if (!event_accept || protocol_error)
                        $fatal(1, "M106 r2 legal event stream broke key=%0d row=%0d ready=%0d fault=%0d",
                               key, row, event_ready, protocol_error);
                end
            end

            // Exact final identity remains high for one more edge: grace only.
            @(posedge clk_core);
            if (event_accept || event_ready || protocol_error
                    || !dut.accepted_event_grace_match)
                $fatal(1, "M106 r2 exact held event reaccepted");
            exact_event_grace_cycles++;
            @(negedge clk_core);
            event_valid = 1'b0;
            @(posedge clk_core); #0.1;

            if (event_accept_run_max != 8192)
                $fatal(1, "M106 r2 event II1 run mismatch max=%0d",
                       event_accept_run_max);

            closed_bank = fill_bank;
            before_closes = close_accepts;
            @(negedge clk_core);
            window_base_row = 128;
            window_context = 16'h4100;
            window_close_valid = 1'b1;
            @(posedge clk_core);
            if (!window_close_accept)
                $fatal(1, "M106 r2 full close not accepted");
            #0.1;
            if (dut.active_key_q[closed_bank] !== {128{1'b1}})
                $fatal(1, "M106 r2 full key bitmap mismatch");
            for (int key = 0; key < 128; key++) begin
                if (dut.row_valid_q[closed_bank][key] !== {64{1'b1}})
                    $fatal(1, "M106 r2 full row bitmap mismatch key=%0d", key);
            end

            // Alternate bank is EMPTY and becomes FILL.  Exact held close must
            // remain grace-only, closing the r1 cross-bank P0.
            @(posedge clk_core);
            if (window_close_accept || window_close_ready || protocol_error
                    || !dut.accepted_close_grace_match)
                $fatal(1, "M106 r2 exact held close crossed bank boundary");
            exact_close_grace_cycles++;
            @(negedge clk_core);
            window_close_valid = 1'b0;

            wait (service_valid);
            @(negedge clk_core);
            service_ready = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            service_ready = 1'b1;

            wait (expected_read == expected_write && !service_valid
                  && bank_ready == 0 && !busy);
            repeat (2) @(posedge clk_core);
            if (close_accepts - before_closes != 1)
                $fatal(1, "M106 r2 phantom close observed delta=%0d",
                       close_accepts - before_closes);
            if (expected_write != 8576 || service_tokens != 8576
                    || load_tokens != 384 || event_tokens != 8192
                    || stall_cycles != 3 || protocol_error)
                $fatal(1, "M106 r2 full service totals mismatch expected=%0d service=%0d load=%0d event=%0d stalls=%0d fault=%0d",
                       expected_write, service_tokens, load_tokens,
                       event_tokens, stall_cycles, protocol_error);
            scoreboard_enable = 1'b0;
        end
    endtask

    task automatic prove_fault_then_reset(input [8*48-1:0] label);
        begin
            repeat (4) begin
                @(posedge clk_core); #0.1;
                if (!protocol_error || event_ready || window_close_ready
                        || service_valid || service_accept)
                    $fatal(1, "M106 r2 sticky quarantine failed: %s", label);
            end
            protocol_attacks++;
            reset_dut();
        end
    endtask

    task automatic illegal_campaign;
        begin
            reset_dut();

            // Duplicate after valid-low.
            accept_event_then_sample_low(2, 3, 4, 1'b0, 256, 16'h4200);
            @(negedge clk_core);
            event_source = 2; event_block = 3; event_row_offset = 4;
            event_negate = 1'b1; window_base_row = 256;
            window_context = 16'h4200; event_valid = 1'b1;
            #0.1;
            if (!dut.duplicate_event || !protocol_error || event_ready)
                $fatal(1, "M106 r2 duplicate did not fail closed");
            prove_fault_then_reset("duplicate");

            // Changed payload is not automatically legal: context/base drift
            // inside one open bank must fail in the same cycle.
            @(negedge clk_core);
            event_source = 4; event_block = 2; event_row_offset = 6;
            event_negate = 1'b0; window_base_row = 320;
            window_context = 16'h4300; event_valid = 1'b1;
            @(posedge clk_core);
            if (!event_accept)
                $fatal(1, "M106 r2 context attack setup failed");
            @(negedge clk_core);
            event_row_offset = 7;
            window_base_row = 384;
            window_context = 16'h4301;
            #0.1;
            if (!dut.event_violation || !protocol_error || event_ready)
                $fatal(1, "M106 r2 changed illegal context/base did not fail closed");
            prove_fault_then_reset("changed context/base");

            // Collision is illegal independent of otherwise legal payloads.
            @(negedge clk_core);
            event_source = 0; event_block = 0; event_row_offset = 0;
            event_negate = 1'b0; window_base_row = 448;
            window_context = 16'h4400;
            event_valid = 1'b1; window_close_valid = 1'b1;
            #0.1;
            if (!dut.request_collision || !protocol_error
                    || event_ready || window_close_ready)
                $fatal(1, "M106 r2 collision did not fail closed");
            prove_fault_then_reset("event close collision");

            // Occupy both banks and stall the first drain.  A third ingress is
            // explicitly illegal under the frozen unavailable-bank policy.
            service_ready = 1'b0;
            accept_event_then_sample_low(1, 0, 1, 1'b0, 512, 16'h4500);
            accept_close_then_sample_low(512, 16'h4500);
            accept_event_then_sample_low(2, 0, 2, 1'b1, 576, 16'h4501);
            accept_close_then_sample_low(576, 16'h4501);
            if (dut.fill_available_q)
                $fatal(1, "M106 r2 unavailable attack setup still has fill bank");
            @(negedge clk_core);
            event_source = 3; event_block = 0; event_row_offset = 3;
            event_negate = 1'b0; window_base_row = 640;
            window_context = 16'h4502; event_valid = 1'b1;
            #0.1;
            if (!dut.event_violation || !protocol_error || event_ready
                    || service_valid || service_accept)
                $fatal(1, "M106 r2 unavailable ingress did not quarantine stalled service");
            prove_fault_then_reset("unavailable bank");

            if (protocol_attacks != 4)
                $fatal(1, "M106 r2 illegal coverage mismatch attacks=%0d",
                       protocol_attacks);
        end
    endtask

    always @(posedge clk_core) begin : scoreboard_and_runs
        if (rst_core) begin
            event_accept_run = 0;
            close_accept_run = 0;
        end else begin
            if (event_accept) begin
                event_accepts++;
                event_accept_run++;
                if (event_accept_run > event_accept_run_max)
                    event_accept_run_max = event_accept_run;
            end else begin
                event_accept_run = 0;
            end
            if (window_close_accept) begin
                close_accepts++;
                close_accept_run++;
                if (close_accept_run > close_accept_run_max)
                    close_accept_run_max = close_accept_run;
            end else begin
                close_accept_run = 0;
            end
            if (service_valid && !service_ready)
                stall_cycles++;
            if (scoreboard_enable && service_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M106 r2 unexpected service token");
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
                    $fatal(1, "M106 r2 service mismatch idx=%0d event=%0d/%0d key=%0d:%0d/%0d:%0d beat=%0d/%0d row=%0d/%0d dest=%0d/%0d neg=%0d/%0d last=%0d/%0d context=%0h/%0h",
                           expected_read, service_is_event,
                           expected_is_event[expected_read], service_source,
                           service_block, expected_source[expected_read],
                           expected_block[expected_read], service_load_beat,
                           expected_beat[expected_read], service_row_offset,
                           expected_row[expected_read],
                           service_destination_row,
                           expected_destination[expected_read], service_negate,
                           expected_negate[expected_read],
                           service_last_for_key, expected_last[expected_read],
                           service_context, expected_context[expected_read]);
                expected_read++;
                service_tokens++;
                if (service_is_event)
                    event_tokens++;
                else
                    load_tokens++;
            end
        end
    end

    initial begin : test
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
        service_tokens = 0;
        load_tokens = 0;
        event_tokens = 0;
        stall_cycles = 0;
        close_accepts = 0;
        event_accepts = 0;
        protocol_attacks = 0;
        event_accept_run = 0;
        event_accept_run_max = 0;
        close_accept_run = 0;
        close_accept_run_max = 0;
        exact_event_grace_cycles = 0;
        exact_close_grace_cycles = 0;
        reset_recoveries = 0;
        scoreboard_enable = 1'b0;

        changed_close_stream_probe();
        full_stream_and_exact_grace();
        illegal_campaign();

        if (exact_event_grace_cycles != 1
                || exact_close_grace_cycles != 1
                || event_accept_run_max != 8192
                || close_accept_run_max < 2
                || protocol_attacks != 4)
            $fatal(1, "M106 r2 final coverage mismatch egrace=%0d cgrace=%0d event_run=%0d close_run=%0d attacks=%0d",
                   exact_event_grace_cycles, exact_close_grace_cycles,
                   event_accept_run_max, close_accept_run_max,
                   protocol_attacks);

        $display("PASS M106 r2 independent standard-streaming full_keys=128 full_rows=64 ingress_stream_run=8192 service_tokens=8576 load_tokens=384 event_tokens=8192 exact_event_grace=1 exact_cross_bank_close_grace=1 changed_legal_close_run=2 illegal_main_attacks=4 stalls=3 presence_bits=16384 direction_bits=16384 bitmap_payload_bits=32768 metadata_bits_min=314 m107_r1_cycle_exact=false accumulator_implemented=false dc_admitted=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false");
        $finish;
    end
endmodule

// Frozen production geometry has ROW_W=6 and WIN_ROWS=64, so no binary row
// code is out of range.  This review-only parameter probe validates the generic
// comparator with WIN_ROWS=63; SYNTHESIS disables the production-geometry
// elaboration guard but does not alter the request-audit logic.
module tb_m106_r2_independent_range_probe;
    logic clk_core, rst_core;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic [5:0] event_row_offset;
    logic event_negate;
    logic [11:0] window_base_row;
    logic [15:0] window_context;
    logic event_accept;
    logic window_close_valid, window_close_ready, window_close_accept;
    logic service_valid, service_ready, service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [5:0] service_row_offset;
    logic [11:0] service_destination_row;
    logic service_negate, service_last_for_key;
    logic [15:0] service_context;
    logic service_accept;
    logic fill_bank, drain_bank;
    logic [1:0] bank_ready;
    logic protocol_error, busy;

    m106_bounded_bitmap_transpose_scheduler #(
        .WIN_ROWS(63), .ROW_W(6), .BASE_W(12), .CONTEXT_W(16)
    ) dut (.*);

    always #1.5 clk_core = ~clk_core;

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        event_valid = 1'b0;
        event_source = 0;
        event_block = 0;
        event_row_offset = 0;
        event_negate = 0;
        window_base_row = 0;
        window_context = 16'h5100;
        window_close_valid = 0;
        service_ready = 1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;
        @(negedge clk_core);
        event_row_offset = 6'd63;
        event_valid = 1'b1;
        #0.1;
        if (dut.row_in_range || !dut.event_violation
                || !protocol_error || event_ready || event_accept)
            $fatal(1, "M106 r2 generic range violation did not fail closed");
        @(posedge clk_core); #0.1;
        event_valid = 0;
        repeat (3) begin
            @(posedge clk_core); #0.1;
            if (!protocol_error || event_ready || service_valid)
                $fatal(1, "M106 r2 generic range fault not sticky");
        end
        @(negedge clk_core);
        rst_core = 1;
        event_row_offset = 0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;
        @(posedge clk_core); #0.1;
        if (protocol_error || !event_ready)
            $fatal(1, "M106 r2 generic range reset recovery failed");
        $display("PASS M106 r2 independent generic-range-probe win_rows=63 row_w=6 attacked_row=63 fail_closed=true sticky=true reset_only=true production_range_code_unrepresentable=true");
        $finish;
    end
endmodule

`default_nettype wire
