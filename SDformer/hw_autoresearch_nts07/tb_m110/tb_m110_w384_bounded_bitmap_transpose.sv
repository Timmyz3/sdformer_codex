`timescale 1ns/1ps
`default_nettype none

module tb_m110_w384_bounded_bitmap_transpose;
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

    int cycle_count;
    int ingress_events;
    int close_accepts;
    int service_events;
    int service_loads;
    int service_tokens;
    int stall_cycles;
    int ii1_pairs;
    int overlap_cycles;
    int close_grace_cycles;
    int expected_window;
    int expected_key;
    int expected_row;
    int expected_beat;
    bit expected_event_phase;
    bit previous_event_accept;
    bit positive_phase;

    m110_w384_bounded_bitmap_transpose_scheduler dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .event_valid(event_valid),
        .event_ready(event_ready),
        .event_source(event_source),
        .event_block(event_block),
        .event_row_offset(event_row_offset),
        .event_negate(event_negate),
        .window_base_row(window_base_row),
        .window_context(window_context),
        .event_accept(event_accept),
        .window_close_valid(window_close_valid),
        .window_close_ready(window_close_ready),
        .window_close_accept(window_close_accept),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_destination_row(service_destination_row),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_context(service_context),
        .service_accept(service_accept),
        .fill_bank(fill_bank),
        .drain_bank(drain_bank),
        .bank_ready(bank_ready),
        .protocol_error(protocol_error),
        .busy(busy)
    );

    m110_w384_bounded_bitmap_transpose_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .event_valid(event_valid),
        .event_ready(event_ready),
        .event_accept(event_accept),
        .window_close_valid(window_close_valid),
        .window_close_ready(window_close_ready),
        .window_close_accept(window_close_accept),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_accept(service_accept),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_destination_row(service_destination_row),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_context(service_context),
        .fill_bank(fill_bank),
        .drain_bank(drain_bank),
        .bank_ready(bank_ready),
        .protocol_error(protocol_error),
        .busy(busy)
    );

    always #1 clk_core = ~clk_core;

    function automatic logic expected_direction(
        input int window_index,
        input int key,
        input int row
    );
        expected_direction = (window_index ^ key ^ row) & 1;
    endfunction

    function automatic int expected_base(input int window_index);
        expected_base = window_index == 0 ? 100 : 600;
    endfunction

    function automatic int expected_context(input int window_index);
        expected_context = window_index == 0 ? 16'h1111 : 16'h2222;
    endfunction

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            service_ready <= 1'b0;
            previous_event_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            service_ready <= ((cycle_count % 17) != 5)
                          && ((cycle_count % 29) != 11);
            if (service_valid && !service_ready)
                stall_cycles <= stall_cycles + 1;
            if (event_accept && previous_event_accept)
                ii1_pairs <= ii1_pairs + 1;
            previous_event_accept <= event_accept;
            if (event_valid && service_valid)
                overlap_cycles <= overlap_cycles + 1;

            if (positive_phase && protocol_error)
                $fatal(1, "M110 unexpected protocol_error cycle=%0d", cycle_count);

            if (service_accept) begin
                service_tokens <= service_tokens + 1;
                if (expected_window >= WINDOWS)
                    $fatal(1, "M110 extra service token after two windows");
                if (service_source !== expected_key[6:3]
                        || service_block !== expected_key[2:0])
                    $fatal(1, "M110 key order mismatch win=%0d expected=%0d got=%0d/%0d",
                           expected_window, expected_key,
                           service_source, service_block);
                if (service_context !== expected_context(expected_window)[CONTEXT_W-1:0])
                    $fatal(1, "M110 context mismatch win=%0d", expected_window);
                if (!expected_event_phase) begin
                    if (service_is_event || service_load_beat !== expected_beat[1:0]
                            || service_row_offset !== '0
                            || service_destination_row !== '0
                            || service_negate || service_last_for_key)
                        $fatal(1, "M110 load token mismatch win=%0d key=%0d beat=%0d",
                               expected_window, expected_key, expected_beat);
                    service_loads <= service_loads + 1;
                    if (expected_beat == 2) begin
                        expected_beat <= 0;
                        expected_event_phase <= 1'b1;
                        expected_row <= 0;
                    end else begin
                        expected_beat <= expected_beat + 1;
                    end
                end else begin
                    if (!service_is_event
                            || service_row_offset !== expected_row[ROW_W-1:0]
                            || service_destination_row !==
                               (expected_base(expected_window) + expected_row)
                            || service_negate !== expected_direction(
                                expected_window, expected_key, expected_row)
                            || service_last_for_key !== (expected_row == WIN_ROWS-1))
                        $fatal(1, "M110 event mismatch win=%0d key=%0d row=%0d",
                               expected_window, expected_key, expected_row);
                    service_events <= service_events + 1;
                    if (expected_row == WIN_ROWS-1) begin
                        expected_event_phase <= 1'b0;
                        expected_beat <= 0;
                        expected_row <= 0;
                        if (expected_key == KEYS-1) begin
                            expected_key <= 0;
                            expected_window <= expected_window + 1;
                        end else begin
                            expected_key <= expected_key + 1;
                        end
                    end else begin
                        expected_row <= expected_row + 1;
                    end
                end
            end
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            event_valid = 1'b0;
            window_close_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic fill_full_window(input int window_index);
        int key;
        int row;
        begin
            for (key = 0; key < KEYS; key++) begin
                for (row = 0; row < WIN_ROWS; row++) begin
                    @(negedge clk_core);
                    event_valid = 1'b1;
                    event_source = key[6:3];
                    event_block = key[2:0];
                    event_row_offset = row[ROW_W-1:0];
                    event_negate = expected_direction(window_index, key, row);
                    window_base_row = expected_base(window_index)[BASE_W-1:0];
                    window_context = expected_context(window_index)[CONTEXT_W-1:0];
                    do @(posedge clk_core); while (!event_accept);
                    ingress_events = ingress_events + 1;
                end
            end
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic close_window_with_exact_grace(input int window_index);
        begin
            @(negedge clk_core);
            window_base_row = expected_base(window_index)[BASE_W-1:0];
            window_context = expected_context(window_index)[CONTEXT_W-1:0];
            window_close_valid = 1'b1;
            do @(posedge clk_core); while (!window_close_accept);
            close_accepts = close_accepts + 1;
            @(posedge clk_core);
            if (window_close_accept || window_close_ready || protocol_error)
                $fatal(1, "M110 exact close grace failed win=%0d", window_index);
            close_grace_cycles = close_grace_cycles + 1;
            @(negedge clk_core);
            window_close_valid = 1'b0;
        end
    endtask

    initial begin
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
        service_ready = 1'b0;
        cycle_count = 0;
        ingress_events = 0;
        close_accepts = 0;
        service_events = 0;
        service_loads = 0;
        service_tokens = 0;
        stall_cycles = 0;
        ii1_pairs = 0;
        overlap_cycles = 0;
        close_grace_cycles = 0;
        expected_window = 0;
        expected_key = 0;
        expected_row = 0;
        expected_beat = 0;
        expected_event_phase = 1'b0;
        previous_event_accept = 1'b0;
        positive_phase = 1'b1;

        reset_dut();
        fill_full_window(0);
        close_window_with_exact_grace(0);
        fill_full_window(1);
        close_window_with_exact_grace(1);

        while (expected_window != WINDOWS || busy) begin
            @(posedge clk_core);
            if (cycle_count > 250000)
                $fatal(1, "M110 positive-phase watchdog timeout");
        end
        repeat (3) @(posedge clk_core);
        if (ingress_events != WINDOWS * EVENTS_PER_WINDOW
                || service_events != WINDOWS * EVENTS_PER_WINDOW
                || service_loads != WINDOWS * LOADS_PER_WINDOW
                || service_tokens != WINDOWS * (EVENTS_PER_WINDOW + LOADS_PER_WINDOW)
                || close_accepts != WINDOWS || close_grace_cycles != WINDOWS)
            $fatal(1, "M110 conservation mismatch in=%0d out=%0d loads=%0d tokens=%0d closes=%0d grace=%0d",
                   ingress_events, service_events, service_loads, service_tokens,
                   close_accepts, close_grace_cycles);
        if (ii1_pairs < WINDOWS * EVENTS_PER_WINDOW - 4)
            $fatal(1, "M110 II1 coverage too small: %0d", ii1_pairs);
        if (overlap_cycles == 0 || stall_cycles == 0)
            $fatal(1, "M110 overlap/stall coverage missing overlap=%0d stalls=%0d",
                   overlap_cycles, stall_cycles);

        positive_phase = 1'b0;
        reset_dut();
        @(negedge clk_core);
        event_valid = 1'b1;
        event_source = 4'h0;
        event_block = 3'h0;
        event_row_offset = 9'd511;
        event_negate = 1'b0;
        window_base_row = 12'd0;
        window_context = 16'hbad0;
        @(posedge clk_core);
        if (event_ready || event_accept || !protocol_error)
            $fatal(1, "M110 out-of-range row did not fail closed");
        @(negedge clk_core);
        event_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        if (!protocol_error || event_ready || window_close_ready || service_valid)
            $fatal(1, "M110 protocol fault did not quarantine sticky");

        $display("PASS M110 W384 full-capacity VCS windows=%0d ingress_events=%0d active_keys=%0d rows_per_key=%0d load_tokens=%0d event_tokens=%0d service_tokens=%0d ii1_pairs=%0d stalls=%0d overlap_cycles=%0d close_grace=%0d protocol_attacks=1 win_rows=%0d bitmap_payload_bits=%0d accumulator_contract_bits=24 accumulator_implemented=false macros=0 scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 WINDOWS, ingress_events, WINDOWS * KEYS, WIN_ROWS,
                 service_loads, service_events, service_tokens, ii1_pairs,
                 stall_cycles, overlap_cycles, close_grace_cycles, WIN_ROWS,
                 2 * KEYS * WIN_ROWS * 2);
        $finish;
    end

endmodule

`default_nettype wire
