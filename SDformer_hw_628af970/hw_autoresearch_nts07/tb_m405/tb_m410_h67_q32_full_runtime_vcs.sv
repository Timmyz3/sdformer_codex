`timescale 1ns/1ps
`default_nettype none

module tb_m410_h67_q32_full_runtime_vcs;
    localparam int TAG_BITS = 24;
    localparam int PARTITIONS = 432;
    localparam int PHASES = 17280;
    localparam int ROWS_PER_PHASE = 3000;
    localparam int TOTAL_ROWS = 51840000;
    localparam int EXPECTED_PASS1 = 16037540;
    localparam int EXPECTED_EARLY = 3751608;
    localparam int EXPECTED_PWP = 16971357;

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

    logic [767:0] phase_configs [0:PHASES-1];
    logic [31:0] runtime_rows [0:TOTAL_ROWS-1];
    string config_path, row_path;
    integer expected_result_index;
    integer checked_results, checked_pwp, checked_last;
    integer metadata_mismatches, arithmetic_mismatches;
    integer config_mismatches, task_flag_mismatches;
    integer cycle_count;

    m405_q32_serial16_zero_stop_controller #(
        .TAG_BITS(TAG_BITS), .ROWS_PER_PHASE(ROWS_PER_PHASE)
    ) dut (.*);
    m405_q32_serial16_zero_stop_controller_assertions #(
        .TAG_BITS(TAG_BITS)
    ) m410_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    task automatic drive_config(input integer phase_index);
        begin
            for (int beat = 0; beat < 3; beat++) begin
                if (clk_core !== 1'b0) @(negedge clk_core);
                config_valid = 1;
                config_beat_index = beat[1:0];
                config_commit = beat == 2;
                config_tag = phase_index[TAG_BITS-1:0];
                config_data = phase_configs[phase_index][beat*256 +: 256];
                do @(posedge clk_core); while (!config_accept && !protocol_error);
            end
            @(negedge clk_core); config_valid = 0;
            #1;
            if (!configuration_live
                    || configured_tag !== phase_index[TAG_BITS-1:0]
                    || configured_centers_q32 !==
                       phase_configs[phase_index][511:0]
                    || configured_narrow_bitmap !==
                       phase_configs[phase_index][767:512]) begin
                config_mismatches++;
                $fatal(1,"M410 configuration mismatch phase=%0d",phase_index);
            end
        end
    endtask

    task automatic drive_phase_rows(input integer phase_index);
        integer global_index;
        logic [31:0] record;
        begin
            for (int local_row = 0; local_row < ROWS_PER_PHASE;
                 local_row++) begin
                global_index = phase_index * ROWS_PER_PHASE + local_row;
                record = runtime_rows[global_index];
                if (record[31:29] !== 0
                        || (record[27] && record[28])) begin
                    task_flag_mismatches++;
                    $fatal(1,"M410 illegal expected row flags index=%0d record=%h",
                           global_index,record);
                end
                if (clk_core !== 1'b0) @(negedge clk_core);
                row_valid = 1;
                row_id = local_row[11:0];
                row_original = record[15:0];
                row_last = local_row == ROWS_PER_PHASE-1;
                do @(posedge clk_core); while (!row_accept && !protocol_error);
            end
            @(negedge clk_core); row_valid = 0;
        end
    endtask

    task automatic release_phase(input integer phase_index);
        begin
            wait (checked_results == (phase_index + 1) * ROWS_PER_PHASE);
            @(negedge clk_core);
            if (!configuration_live || !phase_release_ready)
                $fatal(1,"M410 release not ready phase=%0d checked=%0d",
                       phase_index,checked_results);
            phase_release_valid = 1;
            @(posedge clk_core);
            if (!phase_release_accept)
                $fatal(1,"M410 legal release rejected phase=%0d",phase_index);
            @(negedge clk_core); phase_release_valid = 0;
            @(posedge clk_core); #1;
            if (configuration_live)
                $fatal(1,"M410 release did not clear config phase=%0d",
                       phase_index);
        end
    endtask

    always @(posedge clk_core) begin
        integer expected_phase, expected_row;
        logic [31:0] record;
        if (result_accept) begin
            if (expected_result_index >= TOTAL_ROWS)
                $fatal(1,"M410 result overrun");
            record = runtime_rows[expected_result_index];
            expected_phase = expected_result_index / ROWS_PER_PHASE;
            expected_row = expected_result_index % ROWS_PER_PHASE;
            if (result_tag !== expected_phase[TAG_BITS-1:0]
                    || result_row_id !== expected_row[11:0]
                    || result_original !== record[15:0]
                    || result_last !==
                       (expected_row == ROWS_PER_PHASE-1)) begin
                metadata_mismatches++;
                $fatal(1,"M410 metadata/order mismatch index=%0d tag=%0d/%0d row=%0d/%0d original=%h/%h last=%0d",
                       expected_result_index,result_tag,expected_phase,
                       result_row_id,expected_row,result_original,
                       record[15:0],result_last);
            end
            if (result_center_id !== record[20:16]
                    || result_distance !== record[25:21]
                    || result_use_pwp !== record[26]) begin
                arithmetic_mismatches++;
                $fatal(1,"M410 exact matcher mismatch index=%0d center=%0d/%0d distance=%0d/%0d use=%0d/%0d pass1=%0d early=%0d",
                       expected_result_index,result_center_id,record[20:16],
                       result_distance,record[25:21],result_use_pwp,
                       record[26],record[27],record[28]);
            end
            checked_results++;
            checked_pwp += record[26];
            checked_last += expected_row == ROWS_PER_PHASE-1;
            expected_result_index++;
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
        expected_result_index = 0;
        checked_results = 0;
        checked_pwp = 0;
        checked_last = 0;
        metadata_mismatches = 0;
        arithmetic_mismatches = 0;
        config_mismatches = 0;
        task_flag_mismatches = 0;
        cycle_count = 0;
        if (!$value$plusargs("M410_CONFIG=%s",config_path)
                || !$value$plusargs("M410_ROWS=%s",row_path))
            $fatal(1,"M410 missing stimulus plusargs");
        $readmemh(config_path,phase_configs);
        $readmemh(row_path,runtime_rows);
        if (^phase_configs[0] === 1'bx
                || ^phase_configs[PHASES-1] === 1'bx
                || ^runtime_rows[0] === 1'bx
                || ^runtime_rows[TOTAL_ROWS-1] === 1'bx)
            $fatal(1,"M410 stimulus extent/read failure");

        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;
        for (int phase = 0; phase < PHASES; phase++) begin
            drive_config(phase);
            drive_phase_rows(phase);
            release_phase(phase);
            if ((phase + 1) % PARTITIONS == 0)
                $display("M410_PROGRESS phases=%0d/%0d rows=%0d",
                         phase+1,PHASES,checked_results);
        end
        @(negedge clk_core);
        if (protocol_error || busy || configuration_live
                || checked_results != TOTAL_ROWS
                || expected_result_index != TOTAL_ROWS
                || checked_pwp != EXPECTED_PWP
                || checked_last != PHASES
                || debug_source_rows != TOTAL_ROWS
                || debug_pass0_tasks != TOTAL_ROWS
                || debug_pass1_tasks != EXPECTED_PASS1
                || debug_early_stops != EXPECTED_EARLY
                || debug_results != TOTAL_ROWS
                || metadata_mismatches != 0
                || arithmetic_mismatches != 0
                || config_mismatches != 0
                || task_flag_mismatches != 0)
            $fatal(1,"M410 final ledger failure rows=%0d/%0d pwp=%0d/%0d pass1=%0d/%0d early=%0d/%0d results=%0d protocol=%0d busy=%0d metadata=%0d arithmetic=%0d config=%0d flags=%0d",
                   checked_results,TOTAL_ROWS,checked_pwp,EXPECTED_PWP,
                   debug_pass1_tasks,EXPECTED_PASS1,debug_early_stops,
                   EXPECTED_EARLY,debug_results,protocol_error,busy,
                   metadata_mismatches,arithmetic_mismatches,
                   config_mismatches,task_flag_mismatches);
        $display("PASS M410 full ordered q32 runtime phases=17280 configs=17280 rows=51840000 pass0=51840000 pass1=16037540 early=3751608 pwp=16971357 results=51840000 metadata_mismatches=0 arithmetic_mismatches=0 config_mismatches=0 task_flag_mismatches=0 tie_lowest_id=true exact_runtime_order=true system_speedup=false headline=false cycles=%0d",
                 cycle_count);
        $finish;
    end

    initial begin
        #400000000;
        $fatal(1,"M410 watchdog phases_completed=%0d rows=%0d cycle=%0d busy=%0d error=%0d",
               checked_last,checked_results,cycle_count,busy,protocol_error);
    end
endmodule

`default_nettype wire
