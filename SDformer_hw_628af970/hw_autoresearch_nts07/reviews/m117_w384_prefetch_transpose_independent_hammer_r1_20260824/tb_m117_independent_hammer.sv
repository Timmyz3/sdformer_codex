`timescale 1ns/1ps
`default_nettype none

module tb_m117_independent_hammer;
    localparam int WIN_ROWS = 384;
    localparam int ROW_W = 9;
    localparam int BASE_W = 12;
    localparam int CONTEXT_W = 16;
    localparam int KEYS = 128;

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
    logic weight_prefetch_valid, weight_prefetch_ready;
    logic [3:0] weight_prefetch_source;
    logic [2:0] weight_prefetch_block;
    logic [CONTEXT_W-1:0] weight_prefetch_context;
    logic weight_prefetch_accept;
    logic descriptor_done, descriptor_done_empty;
    logic [BASE_W-1:0] descriptor_done_base_row;
    logic [CONTEXT_W-1:0] descriptor_done_context;
    logic fill_bank, drain_bank;
    logic [1:0] bank_ready;
    logic protocol_error, busy;

    bit exp_valid [0:1][0:KEYS-1][0:WIN_ROWS-1];
    bit exp_neg [0:1][0:KEYS-1][0:WIN_ROWS-1];
    int exp_total [0:1];
    int expected_done_total, observed_done;
    bit expected_done_empty [0:7];
    int expected_done_base [0:7];
    int expected_done_context [0:7];

    bit positive_phase, auto_ready, main_scoreboard;
    bit zero_bubble_due;
    int zero_bubble_key, zero_bubble_context;
    int cycle_count, watchdog;
    int ingress_accepts, close_accepts, service_accepts;
    int event_outputs, load_outputs, prefetch_accepts;
    int main_prefetch_window, main_prefetch_key;
    int main_window, main_key, main_row, main_beat;
    bit main_event_phase;
    int zero_bubble_transitions;
    int service_stall_cycles, service_stall_run, max_service_stall_run;
    int repeated_stall_releases, overlap_cycles;
    int exact_event_grace_cycles, exact_close_grace_cycles;
    int empty_done_count, nonempty_done_count, consecutive_done_cycles;
    bit previous_descriptor_done;
    int manual_prefetch_accepts, manual_initial_stall_cycles;
    int manual_final_stall_cycles, manual_post_prefetch_stall_cycles;
    int manual_duplicate_prefetches;
    int first_fill_bank, second_fill_bank;

    m117_w384_prefetch_transpose_scheduler dut (.*);

    m117_w384_prefetch_transpose_assertions production_checks (.*);

    m117_independent_assertions independent_checks (.*);

    always #1 clk_core = ~clk_core;

    function automatic int expected_base(input int window_index);
        expected_base = window_index == 0 ? 12'd73 : 12'd907;
    endfunction

    function automatic int expected_context(input int window_index);
        expected_context = window_index == 0 ? 16'h31a7 : 16'hc05e;
    endfunction

    function automatic int first_pending_key(input int window_index,
                                               input int start_key);
        first_pending_key = -1;
        for (int key = start_key; key < KEYS; key++) begin
            for (int row = 0; row < WIN_ROWS; row++) begin
                if (first_pending_key < 0 && exp_valid[window_index][key][row])
                    first_pending_key = key;
            end
        end
    endfunction

    function automatic int first_pending_row(input int window_index,
                                               input int key_index,
                                               input int start_row);
        first_pending_row = -1;
        for (int row = start_row; row < WIN_ROWS; row++) begin
            if (first_pending_row < 0 && exp_valid[window_index][key_index][row])
                first_pending_row = row;
        end
    endfunction

    task automatic clear_expected;
        begin
            for (int window_index = 0; window_index < 2; window_index++) begin
                exp_total[window_index] = 0;
                for (int key = 0; key < KEYS; key++) begin
                    for (int row = 0; row < WIN_ROWS; row++) begin
                        exp_valid[window_index][key][row] = 1'b0;
                        exp_neg[window_index][key][row] = 1'b0;
                    end
                end
            end
        end
    endtask

    task automatic prepare_sparse_window(input int window_index,
                                          input int unsigned seed_initial);
        int unsigned seed;
        int count, row;
        begin
            seed = seed_initial;
            for (int key = 0; key < KEYS; key++) begin
                count = 1 + ((key * 11 + window_index * 7) % 5);
                for (int slot = 0; slot < count; slot++) begin
                    seed = seed * 32'd1664525 + 32'd1013904223;
                    row = (seed >> 8) % WIN_ROWS;
                    if (key == 0 && slot == 0)
                        row = 0;
                    if (key == KEYS-1 && slot == 0)
                        row = WIN_ROWS-1;
                    while (exp_valid[window_index][key][row])
                        row = (row + 37) % WIN_ROWS;
                    exp_valid[window_index][key][row] = 1'b1;
                    exp_neg[window_index][key][row]
                        = ((seed >> 31) ^ key ^ row ^ window_index) & 1;
                    exp_total[window_index]++;
                end
            end
        end
    endtask

    task automatic reset_dut;
        begin
            auto_ready = 1'b0;
            @(negedge clk_core);
            rst_core = 1'b1;
            event_valid = 1'b0;
            window_close_valid = 1'b0;
            service_ready = 1'b0;
            weight_prefetch_ready = 1'b0;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic drive_event(input int window_index,
                               input int key,
                               input int row);
        begin
            @(negedge clk_core);
            event_valid = 1'b1;
            event_source = key[6:3];
            event_block = key[2:0];
            event_row_offset = row[ROW_W-1:0];
            event_negate = exp_neg[window_index][key][row];
            window_base_row = expected_base(window_index)[BASE_W-1:0];
            window_context = expected_context(window_index)[CONTEXT_W-1:0];
            do @(posedge clk_core); while (!event_accept);
            ingress_accepts++;
        end
    endtask

    task automatic close_after_exact_last_event_grace(input int window_index);
        begin
            // The just-accepted final ingress payload remains unchanged for
            // one complete cycle.  It must be grace-held, not reaccepted.
            @(posedge clk_core);
            if (event_ready || event_accept || protocol_error)
                $fatal(1, "M117 independent exact event grace failed");
            exact_event_grace_cycles++;
            @(negedge clk_core);
            event_valid = 1'b0;
            window_close_valid = 1'b1;
            window_base_row = expected_base(window_index)[BASE_W-1:0];
            window_context = expected_context(window_index)[CONTEXT_W-1:0];
            do @(posedge clk_core); while (!window_close_accept);
            close_accepts++;
            @(posedge clk_core);
            if (window_close_ready || window_close_accept || protocol_error)
                $fatal(1, "M117 independent exact close grace failed");
            exact_close_grace_cycles++;
            @(negedge clk_core);
            window_close_valid = 1'b0;
        end
    endtask

    task automatic drive_sparse_window_reverse(input int window_index);
        int remaining;
        begin
            remaining = exp_total[window_index];
            for (int key = KEYS-1; key >= 0; key--) begin
                for (int row = WIN_ROWS-1; row >= 0; row--) begin
                    if (exp_valid[window_index][key][row]) begin
                        drive_event(window_index, key, row);
                        remaining--;
                        if (remaining == 0)
                            close_after_exact_last_event_grace(window_index);
                    end
                end
            end
            if (remaining != 0)
                $fatal(1, "M117 independent ingress permutation incomplete");
        end
    endtask

    task automatic configure_done(input int count);
        begin
            expected_done_total = count;
            observed_done = 0;
            for (int index = 0; index < 8; index++) begin
                expected_done_empty[index] = 1'b0;
                expected_done_base[index] = 0;
                expected_done_context[index] = 0;
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (!rst_core && auto_ready) begin
            // Deterministic pseudo-random repeated service stalls.  Weight
            // prefetch is deliberately ideal only in the 254/254 experiment.
            service_ready = !(((cycle_count % 37) >= 9
                            && (cycle_count % 37) <= 13)
                           || ((cycle_count % 113) >= 70
                            && (cycle_count % 113) <= 78));
            weight_prefetch_ready = 1'b1;
        end
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            zero_bubble_due = 1'b0;
            service_stall_run = 0;
            previous_descriptor_done = 1'b0;
        end else begin
            cycle_count++;
            if (positive_phase && protocol_error)
                $fatal(1, "M117 independent unexpected protocol_error cycle=%0d",
                       cycle_count);

            if (service_valid && !service_ready) begin
                service_stall_cycles++;
                service_stall_run++;
                if (service_stall_run > max_service_stall_run)
                    max_service_stall_run = service_stall_run;
            end else begin
                if (service_stall_run >= 3)
                    repeated_stall_releases++;
                service_stall_run = 0;
            end
            if (event_valid && service_valid && fill_bank != drain_bank)
                overlap_cycles++;

            if (descriptor_done) begin
                if (observed_done >= expected_done_total)
                    $fatal(1, "M117 independent extra descriptor_done");
                if (descriptor_done_empty
                        !== expected_done_empty[observed_done]
                        || descriptor_done_base_row
                           !== expected_done_base[observed_done][BASE_W-1:0]
                        || descriptor_done_context
                           !== expected_done_context[observed_done][CONTEXT_W-1:0])
                    $fatal(1, "M117 independent descriptor_done mismatch index=%0d",
                           observed_done);
                if (descriptor_done_empty)
                    empty_done_count++;
                else
                    nonempty_done_count++;
                if (previous_descriptor_done)
                    consecutive_done_cycles++;
                observed_done++;
            end
            previous_descriptor_done = descriptor_done;

            if (main_scoreboard) begin
                if (zero_bubble_due) begin
                    if (!service_valid || service_is_event
                            || service_load_beat != 0
                            || {service_source, service_block}
                               != zero_bubble_key[6:0]
                            || service_context
                               != zero_bubble_context[CONTEXT_W-1:0])
                        $fatal(1, "M117 independent missing/wrong zero-bubble load0 key=%0d",
                               zero_bubble_key);
                    zero_bubble_transitions++;
                    zero_bubble_due = 1'b0;
                end

                if (weight_prefetch_accept) begin
                    if (main_prefetch_window >= 2
                            || {weight_prefetch_source, weight_prefetch_block}
                               != main_prefetch_key[6:0]
                            || weight_prefetch_context
                               != expected_context(main_prefetch_window)[CONTEXT_W-1:0])
                        $fatal(1, "M117 independent duplicate/skip prefetch w=%0d k=%0d got=%0d",
                               main_prefetch_window, main_prefetch_key,
                               {weight_prefetch_source, weight_prefetch_block});
                    prefetch_accepts++;
                    if (main_prefetch_key == KEYS-1) begin
                        main_prefetch_key = 0;
                        main_prefetch_window++;
                    end else begin
                        main_prefetch_key++;
                    end
                end

                if (service_accept) begin
                    int found_row, found_key;
                    service_accepts++;
                    if (main_window >= 2)
                        $fatal(1, "M117 independent extra service token");
                    if ({service_source, service_block} != main_key[6:0]
                            || service_context
                               != expected_context(main_window)[CONTEXT_W-1:0])
                        $fatal(1, "M117 independent service identity mismatch w=%0d k=%0d",
                               main_window, main_key);
                    if (!main_event_phase) begin
                        if (service_is_event
                                || service_load_beat != main_beat[1:0]
                                || service_row_offset != 0
                                || service_destination_row != 0
                                || service_negate || service_last_for_key)
                            $fatal(1, "M117 independent load mismatch w=%0d k=%0d beat=%0d",
                                   main_window, main_key, main_beat);
                        load_outputs++;
                        if (main_beat == 2) begin
                            main_beat = 0;
                            main_event_phase = 1'b1;
                            main_row = first_pending_row(main_window,
                                                         main_key, 0);
                        end else begin
                            main_beat++;
                        end
                    end else begin
                        found_row = first_pending_row(main_window, main_key,
                                                      main_row + 1);
                        if (!service_is_event
                                || service_row_offset != main_row[ROW_W-1:0]
                                || service_destination_row
                                   != (expected_base(main_window) + main_row)
                                || service_negate
                                   != exp_neg[main_window][main_key][main_row]
                                || !exp_valid[main_window][main_key][main_row]
                                || service_last_for_key != (found_row < 0))
                            $fatal(1, "M117 independent event mismatch w=%0d k=%0d row=%0d",
                                   main_window, main_key, main_row);
                        exp_valid[main_window][main_key][main_row] = 1'b0;
                        event_outputs++;
                        if (found_row >= 0) begin
                            main_row = found_row;
                        end else begin
                            found_key = first_pending_key(main_window,
                                                          main_key + 1);
                            main_event_phase = 1'b0;
                            main_beat = 0;
                            if (found_key >= 0) begin
                                zero_bubble_due = 1'b1;
                                zero_bubble_key = found_key;
                                zero_bubble_context
                                    = expected_context(main_window);
                                main_key = found_key;
                            end else begin
                                main_window++;
                                if (main_window < 2)
                                    main_key = first_pending_key(main_window, 0);
                            end
                        end
                    end
                end
            end
        end
    end

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
        weight_prefetch_ready = 1'b0;
        positive_phase = 1'b1;
        auto_ready = 1'b0;
        main_scoreboard = 1'b0;
        zero_bubble_due = 1'b0;
        cycle_count = 0;
        ingress_accepts = 0;
        close_accepts = 0;
        service_accepts = 0;
        event_outputs = 0;
        load_outputs = 0;
        prefetch_accepts = 0;
        zero_bubble_transitions = 0;
        service_stall_cycles = 0;
        service_stall_run = 0;
        max_service_stall_run = 0;
        repeated_stall_releases = 0;
        overlap_cycles = 0;
        exact_event_grace_cycles = 0;
        exact_close_grace_cycles = 0;
        empty_done_count = 0;
        nonempty_done_count = 0;
        consecutive_done_cycles = 0;
        previous_descriptor_done = 1'b0;
        manual_prefetch_accepts = 0;
        manual_initial_stall_cycles = 0;
        manual_final_stall_cycles = 0;
        manual_post_prefetch_stall_cycles = 0;
        manual_duplicate_prefetches = 0;

        // Scenario A: two full-key sparse descriptors.  Ingress arrives in
        // reverse key/row order while drain is ascending.  The service port
        // sees deterministic random multi-cycle stalls, but the idealized
        // identity-only prefetch port remains ready for the 254/254 claim.
        clear_expected();
        prepare_sparse_window(0, 32'h1170a55a);
        prepare_sparse_window(1, 32'h11715aa5);
        configure_done(2);
        expected_done_empty[0] = 1'b0;
        expected_done_base[0] = expected_base(0);
        expected_done_context[0] = expected_context(0);
        expected_done_empty[1] = 1'b0;
        expected_done_base[1] = expected_base(1);
        expected_done_context[1] = expected_context(1);
        main_prefetch_window = 0;
        main_prefetch_key = 0;
        main_window = 0;
        main_key = 0;
        main_row = 0;
        main_beat = 0;
        main_event_phase = 1'b0;
        reset_dut();
        main_scoreboard = 1'b1;
        auto_ready = 1'b1;
        first_fill_bank = fill_bank;
        drive_sparse_window_reverse(0);
        second_fill_bank = fill_bank;
        drive_sparse_window_reverse(1);
        watchdog = 0;
        while (main_window != 2 || observed_done != 2 || busy) begin
            @(posedge clk_core);
            watchdog++;
            if (watchdog > 50000)
                $fatal(1, "M117 independent main watchdog");
        end
        repeat (3) @(posedge clk_core);
        if (main_prefetch_window != 2 || main_prefetch_key != 0
                || prefetch_accepts != 256
                || zero_bubble_transitions != 254
                || event_outputs != exp_total[0] + exp_total[1]
                || load_outputs != 768
                || first_fill_bank == second_fill_bank
                || overlap_cycles == 0 || repeated_stall_releases == 0
                || max_service_stall_run < 3)
            $fatal(1, "M117 independent main conservation/coverage failed pref=%0d zero=%0d events=%0d/%0d loads=%0d banks=%0d/%0d overlap=%0d releases=%0d maxstall=%0d",
                   prefetch_accepts, zero_bubble_transitions, event_outputs,
                   exp_total[0] + exp_total[1], load_outputs,
                   first_fill_bank, second_fill_bank, overlap_cycles,
                   repeated_stall_releases, max_service_stall_run);
        main_scoreboard = 1'b0;
        auto_ready = 1'b0;

        // Scenario B: two empty descriptors are streamed on successive
        // legal ready/valid payloads, exercising ordered done and ping-pong.
        configure_done(2);
        expected_done_empty[0] = 1'b1;
        expected_done_base[0] = 12'd1200;
        expected_done_context[0] = 16'he001;
        expected_done_empty[1] = 1'b1;
        expected_done_base[1] = 12'd1300;
        expected_done_context[1] = 16'he002;
        reset_dut();
        service_ready = 1'b1;
        weight_prefetch_ready = 1'b1;
        @(negedge clk_core);
        first_fill_bank = fill_bank;
        window_base_row = 12'd1200;
        window_context = 16'he001;
        window_close_valid = 1'b1;
        do @(posedge clk_core); while (!window_close_accept);
        close_accepts++;
        @(negedge clk_core);
        second_fill_bank = fill_bank;
        window_base_row = 12'd1300;
        window_context = 16'he002;
        // Keeping valid high with a changed legal identity is standard
        // ready/valid streaming, not a duplicate of the accepted close.
        do @(posedge clk_core); while (!window_close_accept);
        close_accepts++;
        @(negedge clk_core);
        window_close_valid = 1'b0;
        watchdog = 0;
        while (observed_done != 2 || busy) begin
            @(posedge clk_core);
            watchdog++;
            if (watchdog > 100)
                $fatal(1, "M117 independent empty watchdog");
        end
        if (first_fill_bank == second_fill_bank
                || empty_done_count < 2)
            $fatal(1, "M117 independent empty/pingpong failed");

        // Scenario C: explicit initial-key and last-event lookahead stalls.
        // The next-key identity must be accepted exactly once before the
        // stalled final event retires, then produce load0 without a bubble.
        configure_done(1);
        expected_done_empty[0] = 1'b0;
        expected_done_base[0] = 12'd1440;
        expected_done_context[0] = 16'hf117;
        reset_dut();
        service_ready = 1'b0;
        weight_prefetch_ready = 1'b0;
        @(negedge clk_core);
        event_valid = 1'b1;
        event_source = 4'd12;
        event_block = 3'd4; // key 100, deliberately arrives first
        event_row_offset = 9'd383;
        event_negate = 1'b1;
        window_base_row = 12'd1440;
        window_context = 16'hf117;
        do @(posedge clk_core); while (!event_accept);
        ingress_accepts++;
        @(negedge clk_core);
        event_source = 4'd0;
        event_block = 3'd3; // key 3 is serviced first
        event_row_offset = 9'd0;
        event_negate = 1'b0;
        do @(posedge clk_core); while (!event_accept);
        ingress_accepts++;
        @(negedge clk_core);
        event_valid = 1'b0;
        window_close_valid = 1'b1;
        do @(posedge clk_core); while (!window_close_accept);
        close_accepts++;
        @(negedge clk_core);
        window_close_valid = 1'b0;
        do @(posedge clk_core); while (!weight_prefetch_valid);
        repeat (7) begin
            if (!weight_prefetch_valid || weight_prefetch_ready
                    || {weight_prefetch_source, weight_prefetch_block} != 7'd3
                    || weight_prefetch_context != 16'hf117)
                $fatal(1, "M117 independent initial prefetch identity drift");
            manual_initial_stall_cycles++;
            @(posedge clk_core);
        end
        @(negedge clk_core);
        weight_prefetch_ready = 1'b1;
        do @(posedge clk_core); while (!weight_prefetch_accept);
        manual_prefetch_accepts++;
        @(negedge clk_core);
        if (!service_valid || service_is_event || service_load_beat != 0
                || {service_source, service_block} != 7'd3)
            $fatal(1, "M117 independent initial prefetch did not launch load0");
        service_ready = 1'b1;
        for (int beat = 0; beat < 3; beat++) begin
            do @(posedge clk_core); while (!service_accept);
            if (service_is_event || service_load_beat != beat[1:0]
                    || {service_source, service_block} != 7'd3)
                $fatal(1, "M117 independent manual key3 load mismatch");
        end
        @(negedge clk_core);
        service_ready = 1'b0;
        weight_prefetch_ready = 1'b0;
        repeat (6) begin
            @(posedge clk_core);
            if (!service_valid || !service_is_event || !service_last_for_key
                    || {service_source, service_block} != 7'd3
                    || !weight_prefetch_valid
                    || {weight_prefetch_source, weight_prefetch_block} != 7'd100
                    || weight_prefetch_context != 16'hf117)
                $fatal(1, "M117 independent final-event prefetch identity drift");
            manual_final_stall_cycles++;
        end
        @(negedge clk_core);
        weight_prefetch_ready = 1'b1;
        do @(posedge clk_core); while (!weight_prefetch_accept);
        manual_prefetch_accepts++;
        @(negedge clk_core);
        weight_prefetch_ready = 1'b0;
        repeat (4) begin
            @(posedge clk_core);
            if (!service_valid || !service_is_event || !service_last_for_key
                    || {service_source, service_block} != 7'd3
                    || weight_prefetch_valid || weight_prefetch_accept)
                $fatal(1, "M117 independent duplicate early prefetch or event drift");
            manual_post_prefetch_stall_cycles++;
        end
        @(negedge clk_core);
        service_ready = 1'b1;
        do @(posedge clk_core); while (!service_accept);
        @(negedge clk_core);
        if (!service_valid || service_is_event || service_load_beat != 0
                || {service_source, service_block} != 7'd100)
            $fatal(1, "M117 independent early prefetch inserted/wrong load0");
        for (int beat = 0; beat < 3; beat++) begin
            do @(posedge clk_core); while (!service_accept);
            if (service_is_event || service_load_beat != beat[1:0]
                    || {service_source, service_block} != 7'd100)
                $fatal(1, "M117 independent manual key100 load mismatch");
        end
        do @(posedge clk_core); while (!service_accept);
        if (!service_is_event || !service_last_for_key
                || service_row_offset != 9'd383
                || service_destination_row != (12'd1440 + 12'd383)
                || !service_negate)
            $fatal(1, "M117 independent manual key100 event mismatch");
        watchdog = 0;
        while (observed_done != 1 || busy) begin
            @(posedge clk_core);
            watchdog++;
            if (watchdog > 100)
                $fatal(1, "M117 independent directed prefetch watchdog");
        end
        if (manual_prefetch_accepts != 2
                || manual_initial_stall_cycles != 7
                || manual_final_stall_cycles != 6
                || manual_post_prefetch_stall_cycles != 4
                || manual_duplicate_prefetches != 0
                || empty_done_count != 2 || nonempty_done_count != 3
                || consecutive_done_cycles != 1)
            $fatal(1, "M117 independent directed prefetch accounting failed");

        // Allow assertion coverage bookkeeping to sample the final registered
        // descriptor_done before ending the commercial simulation.
        repeat (3) @(posedge clk_core);

        $display("PASS M117 INDEPENDENT HAMMER commercial_vcs=true sparse_seeded_windows=2 full_key_windows=2 ingress_events=%0d weight_prefetches=256 zero_bubble_scoreboard=254 zero_bubble_expected=254 service_events=%0d load_tokens=768 service_stall_cycles=%0d max_repeated_stall=%0d stall_releases=%0d pingpong_overlap=%0d empty_descriptors=%0d nonempty_descriptors=%0d consecutive_empty_done=%0d exact_event_grace=%0d exact_close_grace=%0d manual_prefetch_accepts=2 initial_prefetch_stalls=7 final_event_prefetch_stalls=6 post_prefetch_event_stalls=4 duplicate_prefetches=0 first_key_no_skip=true next_key_no_skip=true stall_identity_stable=true weight_payload_memory=false lane_sram_768b=false shared_arbiter=false numeric_mapper=false m109_2p535_is_projection=true one_bubble_per_group_ratio=2.4886483878017676 scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 ingress_accepts, event_outputs, service_stall_cycles,
                 max_service_stall_run, repeated_stall_releases,
                 overlap_cycles, empty_done_count, nonempty_done_count,
                 consecutive_done_cycles,
                 exact_event_grace_cycles, exact_close_grace_cycles);
        $finish;
    end
endmodule

`default_nettype wire
