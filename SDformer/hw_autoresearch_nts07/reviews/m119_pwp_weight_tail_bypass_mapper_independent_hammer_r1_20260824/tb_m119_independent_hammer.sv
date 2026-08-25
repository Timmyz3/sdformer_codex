`timescale 1ns/1ps
`default_nettype none

module tb_m119_independent_hammer;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int DELTA_BITS = LANES * ACC_BITS;

    logic clk_core, rst_core;
    logic service_valid, service_ready, service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [8:0] service_row_offset;
    logic service_negate, service_last_for_key, service_accept;
    logic weight_rd_en;
    logic [6:0] weight_rd_key;
    logic [1:0] weight_rd_beat;
    logic [255:0] weight_rd_data;
    logic update_valid, update_ready;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [DELTA_BITS-1:0] update_delta;
    logic update_accept;
    logic payload_active, tail_bypass_available, protocol_error, busy;

    typedef struct packed {
        logic [6:0] key;
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic negate;
        logic main_item;
    } expected_update_t;
    expected_update_t expected_q[$];
    expected_update_t expected_item;

    bit main_count_enable, capture_updates, positive_phase;
    bit previous_load2_accept, previous_event_accept;
    bit previous_weight_read;
    logic [6:0] previous_weight_key;
    logic [1:0] previous_weight_beat;
    int cycle_count;
    int main_loads, main_reads, main_response_checks;
    int main_events, main_updates, main_lane_checks;
    int main_tail_hits, main_negate_events, main_event_ii1_pairs;
    int main_update_stalls, main_negate_min_checks, main_positive_max_checks;
    int event_backpressure_cycles, simultaneous_retire_accepts;
    int output_backpressure_cycles;
    int protocol_attack_classes, fault_reads, fault_service_accepts;
    int duplicate_retry_event_accepts, duplicate_retry_updates;
    int older_update_fault_drains;
    int total_update_accepts;
    int boundary_minus128_checks, boundary_plus127_checks;
    int boundary_negated_minus128_to_plus128_checks;

    m119_pwp_weight_tail_bypass_mapper dut (.*);
    m119_pwp_weight_tail_bypass_mapper_assertions production_checks (.*);
    m119_independent_assertions independent_checks (.*);

    always #1 clk_core = ~clk_core;

    function automatic integer signed independent_weight_value(
        input int key,
        input int lane
    );
        integer raw;
        begin
            if (lane == 0)
                independent_weight_value = -128;
            else if (lane == 1)
                independent_weight_value = 127;
            else begin
                raw = (key * 53 + lane * 97 + 8'h5d) & 8'hff;
                independent_weight_value = raw >= 128 ? raw - 256 : raw;
            end
        end
    endfunction

    function automatic int group_key(input int group_index);
        if (group_index < 64)
            group_key = 127 - group_index;
        else if (group_index < 128)
            group_key = ((group_index - 64) * 37 + 11) & 63;
        else
            group_key = 73;
    endfunction

    function automatic logic [255:0] expected_beat_payload(
        input int key,
        input int beat
    );
        logic [255:0] result;
        integer signed value;
        begin
            result = '0;
            for (int byte_index = 0; byte_index < 32; byte_index++) begin
                value = independent_weight_value(
                    key, beat * 32 + byte_index);
                result[byte_index * 8 +: 8] = value[7:0];
            end
            expected_beat_payload = result;
        end
    endfunction

    // Independent fixed one-cycle synchronous 256-bit weight memory.
    always @(posedge clk_core) begin : independent_weight_memory
        if (weight_rd_en)
            weight_rd_data <= expected_beat_payload(weight_rd_key,
                                                     weight_rd_beat);
    end

    always @(posedge clk_core) begin : independent_scoreboard
        integer signed expected_value;
        if (rst_core) begin
            previous_load2_accept <= 1'b0;
            previous_event_accept <= 1'b0;
            previous_weight_read <= 1'b0;
            previous_weight_key <= '0;
            previous_weight_beat <= '0;
        end else begin
            cycle_count++;
            if (positive_phase && protocol_error)
                $fatal(1, "M119 independent unexpected protocol_error cycle=%0d",
                       cycle_count);

            if (previous_weight_read) begin
                if (weight_rd_data
                        !== expected_beat_payload(previous_weight_key,
                                                  previous_weight_beat))
                    $fatal(1, "M119 independent fixed-latency response mismatch key=%0d beat=%0d",
                           previous_weight_key, previous_weight_beat);
                if (main_count_enable)
                    main_response_checks++;
            end
            previous_weight_read <= weight_rd_en;
            previous_weight_key <= weight_rd_key;
            previous_weight_beat <= weight_rd_beat;

            if (weight_rd_en && main_count_enable)
                main_reads++;
            if (service_accept && !service_is_event && main_count_enable)
                main_loads++;

            if (service_accept && service_is_event && capture_updates) begin
                expected_item.key = {service_source, service_block};
                expected_item.block_id = service_block;
                expected_item.row_id = service_row_offset;
                expected_item.negate = service_negate;
                expected_item.main_item = main_count_enable;
                expected_q.push_back(expected_item);
                if (main_count_enable) begin
                    main_events++;
                    if (service_negate)
                        main_negate_events++;
                    if (previous_event_accept)
                        main_event_ii1_pairs++;
                    if (previous_load2_accept) begin
                        if (!tail_bypass_available)
                            $fatal(1, "M119 independent first event missed beat2 tail bypass");
                        main_tail_hits++;
                    end
                end
            end
            previous_event_accept <= service_accept && service_is_event
                                   && main_count_enable;
            previous_load2_accept <= service_accept && !service_is_event
                                  && service_load_beat == 2
                                  && main_count_enable;

            if (main_count_enable && update_valid && !update_ready)
                main_update_stalls++;
            if (update_valid && !update_ready)
                output_backpressure_cycles++;

            if (update_accept && capture_updates) begin
                total_update_accepts++;
                if (expected_q.size() == 0)
                    $fatal(1, "M119 independent update without accepted event");
                expected_item = expected_q.pop_front();
                if (update_block !== expected_item.block_id
                        || update_row !== expected_item.row_id)
                    $fatal(1, "M119 independent update identity mismatch");
                for (int lane = 0; lane < LANES; lane++) begin
                    expected_value = independent_weight_value(
                        expected_item.key, lane);
                    if (expected_item.negate)
                        expected_value = -expected_value;
                    if ($signed(update_delta[lane * ACC_BITS +: ACC_BITS])
                            !== expected_value)
                        $fatal(1, "M119 independent signed map mismatch key=%0d row=%0d lane=%0d got=%0d exp=%0d",
                               expected_item.key, expected_item.row_id, lane,
                               $signed(update_delta[
                                   lane * ACC_BITS +: ACC_BITS]),
                               expected_value);
                    if (expected_item.main_item) begin
                        main_lane_checks++;
                        if (lane == 0) begin
                            boundary_minus128_checks++;
                            if (expected_item.negate) begin
                                if (expected_value != 128)
                                    $fatal(1, "M119 -(-128) oracle failure");
                                boundary_negated_minus128_to_plus128_checks++;
                                main_negate_min_checks++;
                            end
                        end
                        if (lane == 1) begin
                            boundary_plus127_checks++;
                            if (!expected_item.negate)
                                main_positive_max_checks++;
                        end
                    end
                end
                if (expected_item.main_item)
                    main_updates++;
            end
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            service_valid = 1'b0;
            update_ready = 1'b0;
            expected_q.delete();
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic drive_load(input int key, input int beat);
        begin
            @(negedge clk_core);
            service_valid = 1'b1;
            service_is_event = 1'b0;
            service_source = key[6:3];
            service_block = key[2:0];
            service_load_beat = beat[1:0];
            service_row_offset = '0;
            service_negate = 1'b0;
            service_last_for_key = 1'b0;
            do @(posedge clk_core); while (!service_accept);
        end
    endtask

    task automatic drive_event(
        input int key,
        input int row,
        input bit negate,
        input bit last_for_key
    );
        begin
            @(negedge clk_core);
            service_valid = 1'b1;
            service_is_event = 1'b1;
            service_source = key[6:3];
            service_block = key[2:0];
            service_load_beat = '0;
            service_row_offset = row[8:0];
            service_negate = negate;
            service_last_for_key = last_for_key;
            do @(posedge clk_core); while (!service_accept);
        end
    endtask

    task automatic stop_service;
        begin
            @(negedge clk_core);
            service_valid = 1'b0;
        end
    endtask

    task automatic wait_updates_drained;
        int wait_cycles;
        begin
            wait_cycles = 0;
            while (expected_q.size() != 0 || update_valid) begin
                @(posedge clk_core);
                wait_cycles++;
                if (wait_cycles > 1000)
                    $fatal(1, "M119 independent update drain watchdog");
            end
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic expect_sticky_fault;
        begin
            @(posedge clk_core);
            if (!protocol_error || service_ready || service_accept
                    || weight_rd_en)
                $fatal(1, "M119 independent malformed token not quarantined");
            protocol_attack_classes++;
            @(negedge clk_core);
            service_valid = 1'b0;
            repeat (2) @(posedge clk_core);
            if (!protocol_error)
                $fatal(1, "M119 independent fault not sticky");
        end
    endtask

    initial begin
        int key, events_in_group, row, global_event, dup_updates_before;
        logic [DELTA_BITS-1:0] held_delta;
        logic [2:0] held_block;
        logic [8:0] held_row;

        clk_core = 1'b0;
        rst_core = 1'b1;
        service_valid = 1'b0;
        service_is_event = 1'b0;
        service_source = '0;
        service_block = '0;
        service_load_beat = '0;
        service_row_offset = '0;
        service_negate = 1'b0;
        service_last_for_key = 1'b0;
        weight_rd_data = 'x;
        update_ready = 1'b1;
        main_count_enable = 1'b0;
        capture_updates = 1'b1;
        positive_phase = 1'b1;
        cycle_count = 0;
        main_loads = 0;
        main_reads = 0;
        main_response_checks = 0;
        main_events = 0;
        main_updates = 0;
        main_lane_checks = 0;
        main_tail_hits = 0;
        main_negate_events = 0;
        main_event_ii1_pairs = 0;
        main_update_stalls = 0;
        main_negate_min_checks = 0;
        main_positive_max_checks = 0;
        event_backpressure_cycles = 0;
        simultaneous_retire_accepts = 0;
        output_backpressure_cycles = 0;
        protocol_attack_classes = 0;
        fault_reads = 0;
        fault_service_accepts = 0;
        duplicate_retry_event_accepts = 0;
        duplicate_retry_updates = 0;
        older_update_fault_drains = 0;
        total_update_accepts = 0;
        boundary_minus128_checks = 0;
        boundary_plus127_checks = 0;
        boundary_negated_minus128_to_plus128_checks = 0;

        // Main independent conservation run.  Keys 127..64 arrive in reverse,
        // keys 0..63 use a seeded permutation, then key73 is repeated as the
        // 129th group.  Rows are deterministic pseudo-random and polarity is
        // chosen to produce exactly 257 negated events.
        reset_dut();
        update_ready = 1'b1;
        main_count_enable = 1'b1;
        global_event = 0;
        for (int group = 0; group < 129; group++) begin
            key = group_key(group);
            drive_load(key, 0);
            drive_load(key, 1);
            drive_load(key, 2);
            events_in_group = group < 128 ? 4 : 1;
            for (int event_index = 0;
                    event_index < events_in_group; event_index++) begin
                row = (key * 71 + event_index * 113 + group * 19) % 384;
                if (group == 0 && event_index == 0)
                    row = 383;
                if (group == 127 && event_index == 3)
                    row = 0;
                if (group == 128)
                    update_ready = 1'b0;
                drive_event(key, row, (global_event & 1) == 0,
                            event_index == events_in_group - 1);
                global_event++;
            end
        end
        stop_service();
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        update_ready = 1'b1;
        wait_updates_drained();
        main_count_enable = 1'b0;

        if (main_loads != 387 || main_reads != 387
                || main_response_checks != 387
                || main_events != 513 || main_updates != 513
                || main_lane_checks != 49248 || main_tail_hits != 129
                || main_negate_events != 257 || main_update_stalls != 3
                || boundary_minus128_checks != 513
                || boundary_plus127_checks != 513
                || boundary_negated_minus128_to_plus128_checks != 257
                || main_negate_min_checks != 257
                || main_positive_max_checks != 256)
            $fatal(1, "M119 independent main mismatch loads=%0d reads=%0d resp=%0d events=%0d updates=%0d lanes=%0d tail=%0d neg=%0d stalls=%0d -128=%0d +127=%0d negmin=%0d",
                   main_loads, main_reads, main_response_checks, main_events,
                   main_updates, main_lane_checks, main_tail_hits,
                   main_negate_events, main_update_stalls,
                   boundary_minus128_checks, boundary_plus127_checks,
                   boundary_negated_minus128_to_plus128_checks);

        // Independent elastic attack: hold a second event stable while the
        // older update owns the one-entry output.  The release cycle must
        // retire the old update and accept the new event simultaneously.
        reset_dut();
        update_ready = 1'b1;
        drive_load(5, 0);
        drive_load(5, 1);
        drive_load(5, 2);
        update_ready = 1'b0;
        drive_event(5, 10, 1'b0, 1'b0);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b1;
        service_source = 0;
        service_block = 5;
        service_load_beat = 0;
        service_row_offset = 9'd11;
        service_negate = 1'b1;
        service_last_for_key = 1'b1;
        repeat (5) begin
            @(posedge clk_core);
            if (service_ready || service_accept || !update_valid)
                $fatal(1, "M119 independent event backpressure failed");
            event_backpressure_cycles++;
        end
        @(negedge clk_core);
        update_ready = 1'b1;
        @(posedge clk_core);
        if (!service_accept || !update_accept)
            $fatal(1, "M119 independent elastic retire/accept failed");
        simultaneous_retire_accepts++;
        stop_service();
        wait_updates_drained();

        // Malformed beat/type/key attacks are reset-isolated and must not
        // issue a weight read or service acceptance.
        positive_phase = 1'b0;

        // Wrong first beat.
        reset_dut();
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 0;
        service_block = 0;
        service_load_beat = 1;
        expect_sticky_fault();

        // Event before any three-beat payload.
        reset_dut();
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b1;
        service_source = 0;
        service_block = 0;
        service_row_offset = 1;
        expect_sticky_fault();

        // Accepted beat0 retried/duplicated instead of beat1.
        reset_dut();
        drive_load(9, 0);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 1;
        service_block = 1;
        service_load_beat = 0;
        expect_sticky_fault();

        // Beat1 is skipped.
        reset_dut();
        drive_load(10, 0);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 1;
        service_block = 2;
        service_load_beat = 2;
        expect_sticky_fault();

        // Beat1 carries a different key.
        reset_dut();
        drive_load(10, 0);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 1;
        service_block = 3;
        service_load_beat = 1;
        expect_sticky_fault();

        // Event carries a different key after a valid three-beat load.
        reset_dut();
        drive_load(10, 0);
        drive_load(10, 1);
        drive_load(10, 2);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b1;
        service_source = 1;
        service_block = 3;
        service_row_offset = 4;
        expect_sticky_fault();

        // Exact retry/duplicate event is NOT detected.  After the first event
        // is accepted into a stalled output, holding the same valid token
        // causes it to be accepted again when the older update retires.
        reset_dut();
        positive_phase = 1'b1;
        dup_updates_before = total_update_accepts;
        update_ready = 1'b1;
        drive_load(23, 0);
        drive_load(23, 1);
        drive_load(23, 2);
        update_ready = 1'b0;
        drive_event(23, 123, 1'b1, 1'b0);
        duplicate_retry_event_accepts++;
        repeat (4) begin
            @(posedge clk_core);
            if (service_ready || service_accept || !update_valid
                    || protocol_error)
                $fatal(1, "M119 duplicate retry stall setup failed");
        end
        @(negedge clk_core);
        update_ready = 1'b1;
        @(posedge clk_core);
        if (!service_accept || !update_accept || protocol_error)
            $fatal(1, "M119 exact duplicate retry was not accepted as expected");
        duplicate_retry_event_accepts++;
        stop_service();
        wait_updates_drained();
        duplicate_retry_updates = total_update_accepts - dup_updates_before;
        if (duplicate_retry_event_accepts != 2
                || duplicate_retry_updates != 2)
            $fatal(1, "M119 duplicate retry evidence missing");

        // A malformed younger token must not corrupt or suppress an older
        // already-accepted update.  The old identity and delta remain stable
        // through fault quarantine and drain when update_ready rises.
        positive_phase = 1'b0;
        reset_dut();
        update_ready = 1'b1;
        drive_load(30, 0);
        drive_load(30, 1);
        drive_load(30, 2);
        update_ready = 1'b0;
        drive_event(30, 321, 1'b1, 1'b1);
        stop_service();
        @(posedge clk_core);
        if (!update_valid)
            $fatal(1, "M119 older update missing before fault");
        held_delta = update_delta;
        held_block = update_block;
        held_row = update_row;
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 0;
        service_block = 0;
        service_load_beat = 2;
        repeat (2) begin
            @(posedge clk_core);
            if (!protocol_error || service_accept || weight_rd_en
                    || !update_valid || update_delta !== held_delta
                    || update_block !== held_block || update_row !== held_row)
                $fatal(1, "M119 older accepted update changed under fault");
        end
        @(negedge clk_core);
        service_valid = 1'b0;
        update_ready = 1'b1;
        @(posedge clk_core);
        if (!protocol_error || !update_accept
                || update_delta !== held_delta
                || update_block !== held_block || update_row !== held_row)
            $fatal(1, "M119 older accepted update did not drain under fault");
        older_update_fault_drains++;
        protocol_attack_classes++;
        wait_updates_drained();

        if (protocol_attack_classes != 7
                || event_backpressure_cycles != 5
                || simultaneous_retire_accepts != 1
                || older_update_fault_drains != 1)
            $fatal(1, "M119 independent attack accounting mismatch classes=%0d eventstall=%0d simultaneous=%0d olddrain=%0d",
                   protocol_attack_classes, event_backpressure_cycles,
                   simultaneous_retire_accepts, older_update_fault_drains);

        repeat (3) @(posedge clk_core);
        $display("PASS M119 INDEPENDENT HAMMER commercial_vcs=true groups=129 reverse_keys=64 permuted_keys=64 repeated_key_groups=1 weight_loads=387 weight_reads=387 fixed_1cycle_responses=387 events=513 updates=513 lane_checks=49248 tail_bypass_first_events=129 negate_events=257 int8_minus128_checks=513 int8_plus127_checks=513 negate_minus128_to_plus128=257 output_stall_cycles=3 event_backpressure_cycles=5 simultaneous_retire_accept=1 malformed_attack_classes=7 beat_retry_fail_closed=true beat_skip_fail_closed=true wrong_key_fail_closed=true wrong_type_fail_closed=true exact_event_retry_detected=false duplicate_event_accepts=2 duplicate_updates=2 older_accepted_update_fault_drains=1 behavioral_sync256=true foundry_sram=false m117_payload_p0_standalone_closed=true m117_integrated_p0_closed=false m118_exact_once_p0_closed=false m109_2p535_is_projection=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false");
        $finish;
    end
endmodule

`default_nettype wire
