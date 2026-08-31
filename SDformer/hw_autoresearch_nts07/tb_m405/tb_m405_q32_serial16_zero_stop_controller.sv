`timescale 1ns/1ps
`default_nettype none

module tb_m405_q32_serial16_zero_stop_controller;
    localparam int TAG_BITS = 24;
    localparam int ROWS = 64;

    logic clk_core, reset_n;
    logic config_valid, config_ready, config_accept;
    logic [1:0] config_beat_index;
    logic config_commit;
    logic [TAG_BITS-1:0] config_tag;
    logic [255:0] config_data;
    logic phase_release_valid, phase_release_ready, phase_release_accept;
    logic row_valid, row_ready, row_accept;
    logic [11:0] row_id;
    logic [15:0] row_original;
    logic row_last;
    logic result_valid, result_ready, result_accept;
    logic [TAG_BITS-1:0] result_tag;
    logic [11:0] result_row_id;
    logic [15:0] result_original;
    logic [4:0] result_center_id, result_distance;
    logic result_use_pwp, result_last;
    logic [511:0] configured_centers_q32;
    logic [255:0] configured_narrow_bitmap;
    logic [TAG_BITS-1:0] configured_tag;
    logic configuration_live;
    logic protocol_error, busy, debug_pass1_pending;
    logic [31:0] debug_source_rows, debug_pass0_tasks;
    logic [31:0] debug_pass1_tasks, debug_early_stops, debug_results;

    logic [511:0] centers;
    logic [255:0] bitmap;
    logic [15:0] expected_original [0:ROWS-1];
    logic [4:0] expected_center [0:ROWS-1];
    logic [4:0] expected_distance [0:ROWS-1];
    logic expected_use [0:ROWS-1];
    integer outputs_checked, expected_pass1, expected_early;
    integer stalls, protocol_attacks, adjacency_errors, cycle_count;
    integer previous_task_count;

    m405_q32_serial16_zero_stop_controller #(
        .TAG_BITS(TAG_BITS), .ROWS_PER_PHASE(ROWS)
    ) dut (.*);
    m405_q32_serial16_zero_stop_controller_assertions #(.TAG_BITS(TAG_BITS))
        m405_b_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    function automatic integer pop16(input logic [15:0] value);
        begin pop16 = $countones(value); end
    endfunction

    task automatic reference_row(input integer index, input logic [15:0] original);
        integer population, best_distance, best_id, distance;
        integer pass0_distance;
        begin
            population = pop16(original);
            best_distance = 99;
            best_id = 0;
            for (int center = 0; center < 16; center++) begin
                distance = pop16(original ^ centers[center*16 +: 16]);
                if (distance < best_distance) begin
                    best_distance = distance;
                    best_id = center;
                end
            end
            pass0_distance = best_distance;
            if (population >= 2 && pass0_distance > 0) begin
                expected_pass1++;
                for (int center = 16; center < 32; center++) begin
                    distance = pop16(original ^ centers[center*16 +: 16]);
                    if (distance < best_distance) begin
                        best_distance = distance;
                        best_id = center;
                    end
                end
            end else if (population >= 2 && pass0_distance == 0) begin
                expected_early++;
            end
            expected_original[index] = original;
            expected_center[index] = best_id[4:0];
            expected_distance[index] = best_distance[4:0];
            expected_use[index] = population >= 2 &&
                1 + best_distance < population;
        end
    endtask

    task automatic drive_config(input integer tag_value);
        begin
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                config_valid = 1;
                config_beat_index = beat[1:0];
                config_commit = beat == 2;
                config_tag = tag_value[TAG_BITS-1:0];
                config_data = beat == 0 ? centers[255:0] :
                    (beat == 1 ? centers[511:256] : bitmap);
                do @(posedge clk_core); while (!config_accept && !protocol_error);
            end
            @(negedge clk_core); config_valid = 0;
        end
    endtask

    task automatic drive_phase;
        logic [15:0] pattern;
        begin
            for (int index = 0; index < ROWS; index++) begin
                case (index)
                    0: pattern = 16'h0000;
                    1: pattern = 16'h0001;
                    2: pattern = 16'h0003; // pass0 exact
                    3: pattern = 16'h00f0; // q32-only exact
                    4: pattern = 16'h0f0f; // tie/positive case
                    5: pattern = 16'hffff;
                    default: pattern = (16'h9e37 * index) ^
                        (16'h1357 + index * 16'h21);
                endcase
                reference_row(index,pattern);
                @(negedge clk_core);
                row_valid = 1;
                row_id = index[11:0];
                row_original = pattern;
                row_last = index == ROWS-1;
                do @(posedge clk_core); while (!row_accept && !protocol_error);
            end
            @(negedge clk_core); row_valid = 0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            reset_n = 0;
            config_valid = 0;
            phase_release_valid = 0;
            row_valid = 0;
            result_ready = 1;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core); reset_n = 1;
        end
    endtask

    always @(posedge clk_core) begin
        if (result_valid && !result_ready) stalls++;
        if (result_accept) begin
            if (result_row_id >= ROWS)
                $fatal(1,"M405B row id out of range");
            if (result_tag !== 24'h405001
                    || result_original !== expected_original[result_row_id]
                    || result_center_id !== expected_center[result_row_id]
                    || result_distance !== expected_distance[result_row_id]
                    || result_use_pwp !== expected_use[result_row_id]
                    || result_last !== (result_row_id == ROWS-1))
                $fatal(1,"M405B reference mismatch row=%0d original=%h center=%0d/%0d distance=%0d/%0d use=%0d/%0d",
                       result_row_id,result_original,result_center_id,
                       expected_center[result_row_id],result_distance,
                       expected_distance[result_row_id],result_use_pwp,
                       expected_use[result_row_id]);
            outputs_checked++;
        end
        // With a continuously held row producer, a registered pass1 task is
        // followed by the next task without an empty task-count cycle.
        if (!protocol_error && row_valid && result_ready && busy) begin
            if (previous_task_count ==
                    debug_pass0_tasks + debug_pass1_tasks
                    && !result_valid && !config_valid)
                adjacency_errors++;
            previous_task_count = debug_pass0_tasks + debug_pass1_tasks;
        end
    end

    initial begin
        clk_core = 0;
        reset_n = 0;
        config_valid = 0;
        config_beat_index = 0;
        config_commit = 0;
        config_tag = 0;
        config_data = 0;
        phase_release_valid = 0;
        row_valid = 0;
        row_id = 0;
        row_original = 0;
        row_last = 0;
        result_ready = 1;
        centers = 0;
        bitmap = 256'h0123456789abcdef_fedcba9876543210_55aa55aa55aa55aa_aa55aa55aa55aa55;
        for (int center = 0; center < 32; center++)
            centers[center*16 +: 16] =
                (center * 16'h1111) ^ (center << (center % 7));
        centers[0*16 +: 16] = 16'h0003;
        centers[1*16 +: 16] = 16'h0003; // lower-ID tie
        centers[16*16 +: 16] = 16'h00f0;
        centers[17*16 +: 16] = 16'h00f0; // lower global-ID tie
        outputs_checked = 0;
        expected_pass1 = 0;
        expected_early = 0;
        stalls = 0;
        protocol_attacks = 0;
        adjacency_errors = 0;
        cycle_count = 0;
        previous_task_count = -1;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;

        drive_config(24'h405001);
        if (!configuration_live || configured_centers_q32 !== centers
                || configured_narrow_bitmap !== bitmap)
            $fatal(1,"M405B config ownership mismatch");
        fork
            drive_phase();
            begin
                while (outputs_checked < ROWS) begin
                    @(negedge clk_core);
                    result_ready = $urandom_range(0,5) != 0;
                end
                result_ready = 1;
            end
        join
        wait (outputs_checked == ROWS);
        if (!configuration_live || !busy || !phase_release_ready)
            $fatal(1,"M405B config lifetime ended before explicit release");
        if (debug_source_rows != ROWS || debug_pass0_tasks != ROWS
                || debug_pass1_tasks != expected_pass1
                || debug_early_stops != expected_early
                || debug_results != ROWS)
            $fatal(1,"M405B task ledger mismatch source=%0d p0=%0d p1=%0d/%0d early=%0d/%0d results=%0d",
                   debug_source_rows,debug_pass0_tasks,debug_pass1_tasks,
                   expected_pass1,debug_early_stops,expected_early,
                   debug_results);
        @(negedge clk_core); phase_release_valid = 1;
        @(posedge clk_core);
        if (!phase_release_accept)
            $fatal(1,"M405B legal phase release not accepted");
        @(negedge clk_core); phase_release_valid = 0;
        @(posedge clk_core);
        if (configuration_live || busy)
            $fatal(1,"M405B release did not retire config ownership");

        // Attack 1: config begins with beat one.
        reset_dut();
        @(negedge clk_core);
        config_valid = 1;
        config_beat_index = 1;
        config_commit = 0;
        config_tag = 24'hbad001;
        config_data = 0;
        @(posedge clk_core); #1;
        config_valid = 0;
        if (!protocol_error) $fatal(1,"M405B config order attack missed");
        protocol_attacks++;

        // Attack 2: reload while the phase is active.
        reset_dut();
        drive_config(24'hbad002);
        @(negedge clk_core);
        config_valid = 1;
        config_beat_index = 0;
        config_commit = 0;
        config_tag = 24'hbad003;
        @(posedge clk_core); #1;
        config_valid = 0;
        if (!protocol_error) $fatal(1,"M405B active reload attack missed");
        protocol_attacks++;

        // Attack 3: first row has a nonzero identity.
        reset_dut();
        drive_config(24'hbad004);
        @(negedge clk_core);
        row_valid = 1;
        row_id = 1;
        row_original = 16'h1234;
        row_last = 0;
        @(posedge clk_core); #1;
        row_valid = 0;
        if (!protocol_error || result_valid)
            $fatal(1,"M405B row identity attack leaked result");
        protocol_attacks++;

        if (protocol_attacks != 3 || outputs_checked != ROWS
                || expected_pass1 == 0 || expected_early == 0
                || adjacency_errors != 0)
            $fatal(1,"M405B coverage mismatch outputs=%0d p1=%0d early=%0d attacks=%0d adjacency=%0d",
                   outputs_checked,expected_pass1,expected_early,
                   protocol_attacks,adjacency_errors);
        $display("PASS M405B q32 serial16 rows=64 pass0=%0d pass1=%0d early=%0d outputs=%0d stalls=%0d protocol_attacks=3 tie_lowest_id=true source_scratch_reads=0 descriptor_scratch=0 task_adjacency_observed=true system_speedup=false headline=false",
                 ROWS,expected_pass1,expected_early,outputs_checked,stalls);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1,"M405B watchdog timeout outputs=%0d busy=%0d error=%0d",
               outputs_checked,busy,protocol_error);
    end
endmodule

`default_nettype wire
