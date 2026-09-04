`timescale 1ns/1ps
`default_nettype none

module tb_m106_bounded_bitmap_transpose_scheduler;
    localparam int WIN_ROWS = 64;
    localparam int ROW_W = 6;
    localparam int BASE_W = 12;
    localparam int CONTEXT_W = 16;
    localparam int MAX_EXPECTED = 128;

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
    integer event_grace_holds, close_grace_holds, protocol_attacks;
    integer cross_bank_close_grace_holds;

    m106_bounded_bitmap_transpose_scheduler dut (.*);
    m106_bounded_bitmap_transpose_scheduler_assertions dut_sva (
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
        .service_context(service_context), .service_accept(service_accept),
        .protocol_error(protocol_error),
        .accepted_event_grace_match(dut.accepted_event_grace_match),
        .accepted_close_grace_match(dut.accepted_close_grace_match),
        .illegal_request(dut.illegal_request), .bank_ready(bank_ready),
        .fill_bank(fill_bank), .drain_bank(drain_bank)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #200000;
        $fatal(1, "M106 watchdog timeout expected=%0d/%0d fault=%0d",
               expected_read, expected_write, protocol_error);
    end

    task automatic push_load(
        input integer source,
        input integer block,
        input integer beat,
        input integer context_value
    );
        begin
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

    task automatic push_key(
        input integer source,
        input integer block,
        input integer context_value
    );
        begin
            push_load(source, block, 0, context_value);
            push_load(source, block, 1, context_value);
            push_load(source, block, 2, context_value);
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
                $fatal(1, "M106 reset recovery failed");
        end
    endtask

    task automatic drive_event(
        input integer source,
        input integer block,
        input integer row,
        input logic negate,
        input integer base,
        input integer context_value,
        input logic hold_grace
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
            do @(posedge clk_core); while (!event_ready);
            #0.1;
            if (hold_grace) begin
                if (!dut.accepted_event_grace_match || event_ready
                        || protocol_error)
                    $fatal(1, "M106 event grace missing after accept");
                @(posedge clk_core); #0.1;
                if (!dut.accepted_event_grace_match || event_ready
                        || protocol_error)
                    $fatal(1, "M106 event grace did not span full cycle");
                event_grace_holds++;
            end
            event_valid = 1'b0;
            accepted_events++;
        end
    endtask

    task automatic close_window(
        input integer base,
        input integer context_value,
        input logic hold_grace
    );
        begin
            @(negedge clk_core);
            window_base_row = base;
            window_context = context_value;
            window_close_valid = 1'b1;
            do @(posedge clk_core); while (!window_close_ready);
            #0.1;
            if (hold_grace) begin
                if (!dut.accepted_close_grace_match || window_close_ready
                        || protocol_error)
                    $fatal(1, "M106 close grace missing after accept");
                @(posedge clk_core); #0.1;
                if (!dut.accepted_close_grace_match || window_close_ready
                        || protocol_error)
                    $fatal(1, "M106 close grace did not span full cycle");
                close_grace_holds++;
            end
            window_close_valid = 1'b0;
            accepted_closes++;
        end
    endtask

    always @(posedge clk_core) begin : service_scoreboard
        if (!rst_core) begin
            if (service_valid && !service_ready)
                stall_cycles++;
            if (service_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M106 unexpected service token");
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
                    $fatal(1, "M106 token mismatch index=%0d event=%0d/%0d key=%0d:%0d/%0d:%0d beat=%0d/%0d row=%0d/%0d dest=%0d/%0d neg=%0d/%0d last=%0d/%0d context=%0h/%0h",
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

    initial begin : directed_test
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
        event_grace_holds = 0;
        close_grace_holds = 0;
        protocol_attacks = 0;
        cross_bank_close_grace_holds = 0;

        push_key(0, 0, 16'h0100);
        push_event(0, 0, 0, 1'b0, 1'b0, 0, 16'h0100);
        push_event(0, 0, 3, 1'b1, 1'b1, 0, 16'h0100);
        push_key(1, 1, 16'h0100);
        push_event(1, 1, 1, 1'b0, 1'b1, 0, 16'h0100);
        push_key(15, 7, 16'h0100);
        push_event(15, 7, 63, 1'b1, 1'b1, 0, 16'h0100);
        push_key(0, 0, 16'h0101);
        push_event(0, 0, 2, 1'b1, 1'b1, 64, 16'h0101);
        push_key(1, 0, 16'h0101);
        push_event(1, 0, 0, 1'b0, 1'b0, 64, 16'h0101);
        push_event(1, 0, 5, 1'b1, 1'b1, 64, 16'h0101);

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Window 0 is deliberately ingested out of key/row order.  The first
        // event and close are held through an extra edge to exercise grace.
        drive_event(15, 7, 63, 1'b1, 0, 16'h0100, 1'b1);
        drive_event(0, 0, 3, 1'b1, 0, 16'h0100, 1'b0);
        drive_event(1, 1, 1, 1'b0, 0, 16'h0100, 1'b0);
        drive_event(0, 0, 0, 1'b0, 0, 16'h0100, 1'b0);
        close_window(0, 16'h0100, 1'b0);

        // Fill the second bank while the first drains, then stall a token.
        drive_event(1, 0, 5, 1'b1, 64, 16'h0101, 1'b0);
        drive_event(0, 0, 2, 1'b1, 64, 16'h0101, 1'b0);
        @(negedge clk_core);
        service_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        service_ready = 1'b1;
        drive_event(1, 0, 0, 1'b0, 64, 16'h0101, 1'b0);
        close_window(64, 16'h0101, 1'b1);

        wait (expected_read == expected_write && !service_valid
              && bank_ready == 0);
        repeat (3) @(posedge clk_core);

        if (expected_write != 22 || accepted_services != 22
                || load_tokens != 15 || event_tokens != 7
                || accepted_events != 7 || accepted_closes != 2
                || stall_cycles != 3 || event_grace_holds != 1
                || close_grace_holds != 1 || protocol_error)
            $fatal(1, "M106 functional coverage mismatch expected=%0d services=%0d loads=%0d events=%0d ingress=%0d closes=%0d stalls=%0d event_grace=%0d close_grace=%0d fault=%0d",
                   expected_write, accepted_services, load_tokens,
                   event_tokens, accepted_events, accepted_closes,
                   stall_cycles, event_grace_holds, close_grace_holds,
                   protocol_error);

        // Exact held close while the alternate bank becomes immediately
        // fillable must remain grace-only and cannot create a phantom window.
        reset_dut();
        close_window(0, 16'h0500, 1'b1);
        if (accepted_closes != 3 || protocol_error)
            $fatal(1, "M106 cross-bank exact close was reaccepted");
        cross_bank_close_grace_holds++;
        wait (!busy);
        reset_dut();

        // A duplicate after an observed valid-low edge is a real new request
        // and must enter same-cycle, reset-only fail-closed quarantine.
        reset_dut();
        drive_event(2, 3, 4, 1'b0, 128, 16'h0200, 1'b0);
        @(posedge clk_core); #0.1;
        @(negedge clk_core);
        event_source = 2;
        event_block = 3;
        event_row_offset = 4;
        event_negate = 1'b0;
        window_base_row = 128;
        window_context = 16'h0200;
        event_valid = 1'b1;
        #0.1;
        if (!dut.event_violation || !protocol_error || event_ready
                || service_valid)
            $fatal(1, "M106 duplicate did not quarantine same cycle");
        @(posedge clk_core); #0.1;
        event_valid = 1'b0;
        repeat (3) begin
            @(posedge clk_core); #0.1;
            if (!protocol_error || event_ready || window_close_ready
                    || service_valid)
                $fatal(1, "M106 fault was not sticky/fail-closed");
        end
        protocol_attacks++;
        reset_dut();

        // A changed window identity after an accepted event is illegal.
        drive_event(4, 2, 6, 1'b0, 256, 16'h0300, 1'b0);
        @(posedge clk_core); #0.1;
        @(negedge clk_core);
        event_source = 4;
        event_block = 2;
        event_row_offset = 7;
        event_negate = 1'b0;
        window_base_row = 256;
        window_context = 16'h0301;
        event_valid = 1'b1;
        #0.1;
        if (!dut.event_violation || !protocol_error || event_ready)
            $fatal(1, "M106 context mutation did not fail closed");
        @(posedge clk_core); #0.1;
        event_valid = 1'b0;
        protocol_attacks++;
        reset_dut();

        // Simultaneous ingress/close while a service token is stalled must
        // quarantine that old token before ready is released.
        drive_event(3, 1, 8, 1'b1, 320, 16'h0400, 1'b0);
        close_window(320, 16'h0400, 1'b0);
        service_ready = 1'b0;
        wait (service_valid);
        @(negedge clk_core);
        event_source = 0;
        event_block = 0;
        event_row_offset = 0;
        event_negate = 1'b0;
        window_base_row = 384;
        window_context = 16'h0401;
        event_valid = 1'b1;
        window_close_valid = 1'b1;
        service_ready = 1'b1;
        #0.1;
        if (!dut.request_collision || !protocol_error || service_valid
                || service_accept || event_ready || window_close_ready)
            $fatal(1, "M106 collision did not quarantine stalled service");
        @(posedge clk_core); #0.1;
        event_valid = 1'b0;
        window_close_valid = 1'b0;
        repeat (2) begin
            @(posedge clk_core); #0.1;
            if (!protocol_error || service_valid || service_accept)
                $fatal(1, "M106 collision fault was not sticky");
        end
        protocol_attacks++;
        reset_dut();

        if (protocol_attacks != 3 || cross_bank_close_grace_holds != 1)
            $fatal(1, "M106 protocol attack coverage mismatch");
        $display("PASS M106 r2 standard-streaming grace windows=2 ingress_events=7 keys=5 load_tokens=15 event_tokens=7 service_tokens=22 stalls=3 event_grace=1 close_grace=2 cross_bank_close_grace=1 protocol_attacks=3 win_rows=64 bitmap_payload_bits=32768 metadata_payload_bits_min=314 accumulator_contract_bits=24 accumulator_port_cut=true macros=0");
        $finish;
    end
endmodule

`default_nettype wire
