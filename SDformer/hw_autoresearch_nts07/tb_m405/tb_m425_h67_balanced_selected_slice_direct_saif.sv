`timescale 1ns/1ps
`default_nettype none

module tb_m425_h67_balanced_selected_slice_direct_saif;
    localparam int TAG_BITS = 24;
    localparam int PHASES = 64;
    localparam int ROWS_PER_PHASE = 3000;
    localparam int TOTAL_ROWS = 192000;
    localparam int STATIC_BLOCKS = 16384;
    localparam int EXPECTED_PASS1 = 61285;
    localparam int EXPECTED_EARLY = 11923;
    localparam int EXPECTED_ZERO = 93037;
    localparam int EXPECTED_POP1 = 25755;
    localparam int EXPECTED_PWP_ROWS = 63067;
    localparam int EXPECTED_LOW = 504536;
    localparam int EXPECTED_HIGH = 416630;
    localparam int EXPECTED_NARROW = 87906;
    localparam int EXPECTED_WIDE = 416630;
    localparam int EXPECTED_CONTRIBUTIONS = 921166;

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
    logic pwp_low_valid, pwp_low_ready, pwp_low_accept;
    logic [TAG_BITS-1:0] pwp_low_tag;
    logic pwp_low_tile;
    logic [4:0] pwp_low_center_id;
    logic [2:0] pwp_low_output_block;
    logic [767:0] pwp_low_data;
    logic pwp_high_valid, pwp_high_ready, pwp_high_accept;
    logic [TAG_BITS-1:0] pwp_high_tag;
    logic pwp_high_tile;
    logic [4:0] pwp_high_center_id;
    logic [2:0] pwp_high_output_block;
    logic [511:0] pwp_high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow, contribution_part_high, contribution_last;
    logic [1151:0] contribution_data;
    logic protocol_error, busy;

    logic [767:0] phase_configs [0:PHASES-1];
    logic [31:0] runtime_rows [0:TOTAL_ROWS-1];
    logic [1280:0] static_pwp [0:STATIC_BLOCKS-1];
    string config_path, row_path, pwp_path;

    integer expected_result_index;
    integer checked_results, checked_pwp_rows;
    integer checked_low, checked_high, checked_narrow, checked_wide;
    integer checked_contributions, checked_reconstructed_lanes;
    integer checked_phases, zero_rows, pop1_rows;
    integer metadata_mismatches, matcher_arithmetic_mismatches;
    integer codec_arithmetic_mismatches, reconstruction_mismatches;
    integer bitmap_mismatches, unknown_transaction_count;
    integer measurement_cycles;
    logic m425_measurement_window;

    integer expected_pwp_phase, expected_pwp_row, expected_pwp_block;
    logic [4:0] expected_pwp_center;
    logic expected_pwp_narrow;
    integer expected_contribution_part;
    logic expected_block_active;
    logic [1151:0] captured_low_part;

    m405_q32_elastic_selected_slice #(
        .TAG_BITS(TAG_BITS), .ROWS_PER_PHASE(ROWS_PER_PHASE)
    ) dut (.*);
    m405_q32_elastic_selected_slice_assertions shell_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (m425_measurement_window)
            measurement_cycles <= measurement_cycles + 1;
    end

    function automatic integer popcount16(input logic [15:0] value);
        integer count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 16; bit_index++)
                count += value[bit_index];
            popcount16 = count;
        end
    endfunction

    function automatic integer pwp_index(
        input integer phase_index,
        input integer center_index,
        input integer block_index
    );
        pwp_index = (phase_index * 32 + center_index) * 8 + block_index;
    endfunction

    function automatic logic [1151:0] expected_vector(
        input integer physical_index,
        input integer part_index
    );
        logic [1151:0] value;
        logic [7:0] low_lane;
        logic [3:0] high_lane;
        begin
            value = '0;
            for (int lane = 0; lane < 96; lane++) begin
                low_lane = static_pwp[physical_index][lane*8 +: 8];
                high_lane = static_pwp[physical_index][768+lane*4 +: 4];
                if (static_pwp[physical_index][1280])
                    value[lane*12 +: 12] = {{4{low_lane[7]}},low_lane};
                else if (part_index == 0)
                    value[lane*12 +: 12] = {4'b0000,low_lane};
                else
                    value[lane*12 +: 12] = {high_lane,8'b0};
            end
            expected_vector = value;
        end
    endfunction

    task automatic drive_config(input integer phase_index);
        begin
            for (int beat = 0; beat < 3; beat++) begin
                if (clk_core !== 1'b0) @(negedge clk_core);
                config_valid = 1;
                config_beat_index = beat[1:0];
                config_commit = beat == 2;
                config_tag = phase_index[TAG_BITS-1:0];
                config_data = phase_configs[phase_index][beat*256 +: 256];
                do @(posedge clk_core); while (!config_accept
                                               && !protocol_error);
            end
            @(negedge clk_core);
            config_valid = 0;
            #1;
            if (!dut.configuration_live_w
                    || dut.configured_tag_w !== phase_index[TAG_BITS-1:0]
                    || dut.centers_w !== phase_configs[phase_index][511:0]
                    || dut.narrow_bitmap_w !==
                       phase_configs[phase_index][767:512]) begin
                metadata_mismatches++;
                $fatal(1,"M425 configuration mismatch phase=%0d",phase_index);
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
                if (record[31:29] !== 0 || (record[27] && record[28]))
                    $fatal(1,"M425 illegal row flags index=%0d",global_index);
                if (clk_core !== 1'b0) @(negedge clk_core);
                row_valid = 1;
                row_id = local_row[11:0];
                row_original = record[15:0];
                row_last = local_row == ROWS_PER_PHASE-1;
                do @(posedge clk_core); while (!row_accept
                                               && !protocol_error);
            end
            @(negedge clk_core);
            row_valid = 0;
            wait (checked_results == (phase_index+1)*ROWS_PER_PHASE);
        end
    endtask

    task automatic drive_low_block(
        input integer phase_index,
        input integer row_index,
        input integer center_index,
        input integer block_index,
        input integer physical_index
    );
        begin
            if (clk_core !== 1'b0) @(negedge clk_core);
            expected_pwp_phase = phase_index;
            expected_pwp_row = row_index;
            expected_pwp_block = block_index;
            expected_pwp_center = center_index[4:0];
            expected_pwp_narrow = static_pwp[physical_index][1280];
            expected_contribution_part = 0;
            expected_block_active = 1;
            pwp_low_valid = 1;
            pwp_low_tag = phase_index[TAG_BITS-1:0];
            pwp_low_tile = block_index >= 4;
            pwp_low_center_id = center_index[4:0];
            pwp_low_output_block = block_index[2:0];
            pwp_low_data = static_pwp[physical_index][767:0];
            do @(posedge clk_core); while (!pwp_low_accept
                                           && !protocol_error);
            @(negedge clk_core);
            pwp_low_valid = 0;
        end
    endtask

    task automatic drive_high_block(
        input integer phase_index,
        input integer center_index,
        input integer block_index,
        input integer physical_index
    );
        begin
            if (clk_core !== 1'b0) @(negedge clk_core);
            pwp_high_valid = 1;
            pwp_high_tag = phase_index[TAG_BITS-1:0];
            pwp_high_tile = block_index >= 4;
            pwp_high_center_id = center_index[4:0];
            pwp_high_output_block = block_index[2:0];
            pwp_high_data = static_pwp[physical_index][1279:768];
            do @(posedge clk_core); while (!pwp_high_accept
                                           && !protocol_error);
            @(negedge clk_core);
            pwp_high_valid = 0;
        end
    endtask

    task automatic replay_phase_pwp(input integer phase_index);
        integer global_index, center_index, physical_index;
        logic [31:0] record;
        begin
            for (int local_row = 0; local_row < ROWS_PER_PHASE;
                 local_row++) begin
                global_index = phase_index * ROWS_PER_PHASE + local_row;
                record = runtime_rows[global_index];
                if (!record[26])
                    continue;
                center_index = record[20:16];
                checked_pwp_rows++;
                for (int block_index = 0; block_index < 8;
                     block_index++) begin
                    physical_index = pwp_index(
                        phase_index,center_index,block_index);
                    if (static_pwp[physical_index][1280] !==
                            phase_configs[phase_index]
                                [512+center_index*8+block_index]) begin
                        bitmap_mismatches++;
                        $fatal(1,"M425 bitmap/static mismatch phase=%0d center=%0d block=%0d",
                               phase_index,center_index,block_index);
                    end
                    drive_low_block(phase_index,local_row,center_index,
                                    block_index,physical_index);
                    if (!static_pwp[physical_index][1280])
                        drive_high_block(phase_index,center_index,
                                         block_index,physical_index);
                    wait (!expected_block_active || protocol_error);
                    if (protocol_error)
                        $fatal(1,"M425 protocol error during PWP replay phase=%0d row=%0d block=%0d",
                               phase_index,local_row,block_index);
                end
            end
        end
    endtask

    task automatic release_phase(input integer phase_index);
        begin
            if (clk_core !== 1'b0) @(negedge clk_core);
            if (!dut.configuration_live_w || !phase_release_ready)
                $fatal(1,"M425 release not ready phase=%0d",phase_index);
            phase_release_valid = 1;
            @(posedge clk_core);
            if (!phase_release_accept)
                $fatal(1,"M425 release rejected phase=%0d",phase_index);
            // Drop valid in the same NBA update as the accepted release.
            // Holding it through the following half-cycle would create a
            // false combinational shell_violation after ready legitimately
            // falls, even though no subsequent clock can latch the fault.
            phase_release_valid <= 0;
            @(negedge clk_core);
            @(posedge clk_core);
            #1;
            if (dut.configuration_live_w || busy)
                $fatal(1,"M425 release did not quiesce phase=%0d",phase_index);
            checked_phases++;
        end
    endtask

    always @(posedge clk_core) begin : matcher_checker
        integer expected_phase, expected_row, population;
        logic [31:0] record;
        if (result_accept) begin
            if (expected_result_index >= TOTAL_ROWS)
                $fatal(1,"M425 result overrun");
            record = runtime_rows[expected_result_index];
            expected_phase = expected_result_index / ROWS_PER_PHASE;
            expected_row = expected_result_index % ROWS_PER_PHASE;
            population = popcount16(record[15:0]);
            if (result_tag !== expected_phase[TAG_BITS-1:0]
                    || result_row_id !== expected_row[11:0]
                    || result_original !== record[15:0]
                    || result_last !==
                       (expected_row == ROWS_PER_PHASE-1)) begin
                metadata_mismatches++;
                $fatal(1,"M425 result metadata mismatch index=%0d",
                       expected_result_index);
            end
            if (result_center_id !== record[20:16]
                    || result_distance !== record[25:21]
                    || result_use_pwp !== record[26]
                    || result_use_pwp !==
                       (({1'b0,result_distance}+6'd1) < population)) begin
                matcher_arithmetic_mismatches++;
                $fatal(1,"M425 matcher mismatch index=%0d",expected_result_index);
            end
            zero_rows += population == 0;
            pop1_rows += population == 1;
            checked_results++;
            expected_result_index++;
        end
    end

    always @(posedge clk_core) begin : contribution_checker
        integer physical_index;
        logic [1151:0] vector;
        logic expected_high, expected_last;
        logic [11:0] reconstructed, raw12;
        if (contribution_accept) begin
            if (!expected_block_active)
                $fatal(1,"M425 unexpected contribution");
            physical_index = pwp_index(expected_pwp_phase,
                                       expected_pwp_center,
                                       expected_pwp_block);
            expected_high = !expected_pwp_narrow
                && expected_contribution_part == 1;
            expected_last = expected_pwp_narrow || expected_high;
            vector = expected_vector(physical_index,
                                     expected_contribution_part);
            if (contribution_tag !==
                    expected_pwp_phase[TAG_BITS-1:0]
                    || contribution_tile !== (expected_pwp_block >= 4)
                    || contribution_center_id !== expected_pwp_center
                    || contribution_output_block !==
                       expected_pwp_block[2:0]
                    || contribution_narrow !== expected_pwp_narrow
                    || contribution_part_high !== expected_high
                    || contribution_last !== expected_last) begin
                metadata_mismatches++;
                $fatal(1,"M425 contribution metadata mismatch phase=%0d row=%0d block=%0d part=%0d",
                       expected_pwp_phase,expected_pwp_row,
                       expected_pwp_block,expected_contribution_part);
            end
            if (contribution_data !== vector) begin
                codec_arithmetic_mismatches++;
                $fatal(1,"M425 contribution arithmetic mismatch phase=%0d row=%0d block=%0d part=%0d",
                       expected_pwp_phase,expected_pwp_row,
                       expected_pwp_block,expected_contribution_part);
            end
            checked_contributions++;
            if (expected_pwp_narrow) begin
                checked_low++;
                checked_narrow++;
                checked_reconstructed_lanes += 96;
                expected_block_active = 0;
            end else if (!expected_high) begin
                checked_low++;
                captured_low_part = contribution_data;
                expected_contribution_part = 1;
            end else begin
                checked_high++;
                checked_wide++;
                for (int lane = 0; lane < 96; lane++) begin
                    reconstructed = captured_low_part[lane*12 +: 12]
                        + contribution_data[lane*12 +: 12];
                    raw12 = {static_pwp[physical_index]
                                [768+lane*4 +: 4],
                             static_pwp[physical_index]
                                [lane*8 +: 8]};
                    if (reconstructed !== raw12) begin
                        reconstruction_mismatches++;
                        $fatal(1,"M425 12-bit reconstruction mismatch phase=%0d row=%0d block=%0d lane=%0d",
                               expected_pwp_phase,expected_pwp_row,
                               expected_pwp_block,lane);
                    end
                end
                checked_reconstructed_lanes += 96;
                expected_block_active = 0;
            end
        end
    end

    always @(posedge clk_core) begin : unknown_audit
        if (reset_n) begin
            if ($isunknown({config_valid,config_ready,config_accept,
                    phase_release_valid,phase_release_ready,
                    phase_release_accept,row_valid,row_ready,row_accept,
                    result_valid,result_ready,result_accept,
                    pwp_low_valid,pwp_low_ready,pwp_low_accept,
                    pwp_high_valid,pwp_high_ready,pwp_high_accept,
                    contribution_valid,contribution_ready,
                    contribution_accept,protocol_error,busy})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown transaction control");
            end
            if (config_accept && $isunknown({config_beat_index,
                    config_commit,config_tag,config_data})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown accepted config");
            end
            if (row_accept && $isunknown({row_id,row_original,row_last})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown accepted row");
            end
            if (result_accept && $isunknown({result_tag,result_row_id,
                    result_original,result_center_id,result_distance,
                    result_use_pwp,result_last})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown accepted result");
            end
            if (pwp_low_accept && $isunknown({pwp_low_tag,pwp_low_tile,
                    pwp_low_center_id,pwp_low_output_block,pwp_low_data})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown accepted PWP low");
            end
            if (pwp_high_accept && $isunknown({pwp_high_tag,pwp_high_tile,
                    pwp_high_center_id,pwp_high_output_block,
                    pwp_high_data})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown accepted PWP high");
            end
            if (contribution_accept && $isunknown({contribution_tag,
                    contribution_tile,contribution_center_id,
                    contribution_output_block,contribution_narrow,
                    contribution_part_high,contribution_last,
                    contribution_data})) begin
                unknown_transaction_count++;
                $fatal(1,"M425 unknown accepted contribution");
            end
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
        pwp_low_valid = 0;
        pwp_low_tag = 0;
        pwp_low_tile = 0;
        pwp_low_center_id = 0;
        pwp_low_output_block = 0;
        pwp_low_data = 0;
        pwp_high_valid = 0;
        pwp_high_tag = 0;
        pwp_high_tile = 0;
        pwp_high_center_id = 0;
        pwp_high_output_block = 0;
        pwp_high_data = 0;
        contribution_ready = 1;
        expected_result_index = 0;
        checked_results = 0;
        checked_pwp_rows = 0;
        checked_low = 0;
        checked_high = 0;
        checked_narrow = 0;
        checked_wide = 0;
        checked_contributions = 0;
        checked_reconstructed_lanes = 0;
        checked_phases = 0;
        zero_rows = 0;
        pop1_rows = 0;
        metadata_mismatches = 0;
        matcher_arithmetic_mismatches = 0;
        codec_arithmetic_mismatches = 0;
        reconstruction_mismatches = 0;
        bitmap_mismatches = 0;
        unknown_transaction_count = 0;
        measurement_cycles = 0;
        m425_measurement_window = 0;
        expected_block_active = 0;
        captured_low_part = 0;

        if (!$value$plusargs("M425_CONFIG=%s",config_path)
                || !$value$plusargs("M425_ROWS=%s",row_path)
                || !$value$plusargs("M425_PWP=%s",pwp_path))
            $fatal(1,"M425 missing stimulus plusargs");
        $readmemh(config_path,phase_configs);
        $readmemh(row_path,runtime_rows);
        $readmemh(pwp_path,static_pwp);
        if (^phase_configs[0] === 1'bx
                || ^phase_configs[PHASES-1] === 1'bx
                || ^runtime_rows[0] === 1'bx
                || ^runtime_rows[TOTAL_ROWS-1] === 1'bx
                || ^static_pwp[0] === 1'bx
                || ^static_pwp[STATIC_BLOCKS-1] === 1'bx)
            $fatal(1,"M425 stimulus extent/read failure");

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        reset_n = 1;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        m425_measurement_window = 1;
        $display("M425_SAIF_WINDOW_START time_ns=%0t",$realtime);
        #1;
        for (int phase = 0; phase < PHASES; phase++) begin
            drive_config(phase);
            drive_phase_rows(phase);
            replay_phase_pwp(phase);
            release_phase(phase);
            if ((phase+1) % 8 == 0)
                $display("M425_PROGRESS phases=%0d/64 rows=%0d pwp_rows=%0d contributions=%0d",
                         phase+1,checked_results,checked_pwp_rows,
                         checked_contributions);
        end
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        m425_measurement_window = 0;
        $display("M425_SAIF_WINDOW_STOP time_ns=%0t measurement_cycles=%0d",
                 $realtime,measurement_cycles);
        if (protocol_error || busy || dut.configuration_live_w
                || checked_phases != PHASES
                || checked_results != TOTAL_ROWS
                || checked_pwp_rows != EXPECTED_PWP_ROWS
                || checked_low != EXPECTED_LOW
                || checked_high != EXPECTED_HIGH
                || checked_narrow != EXPECTED_NARROW
                || checked_wide != EXPECTED_WIDE
                || checked_contributions != EXPECTED_CONTRIBUTIONS
                || checked_reconstructed_lanes != EXPECTED_LOW*96
                || zero_rows != EXPECTED_ZERO || pop1_rows != EXPECTED_POP1
                || dut.matcher_debug_source != TOTAL_ROWS
                || dut.matcher_debug_pass0 != TOTAL_ROWS
                || dut.matcher_debug_pass1 != EXPECTED_PASS1
                || dut.matcher_debug_early != EXPECTED_EARLY
                || dut.matcher_debug_results != TOTAL_ROWS
                || dut.adapter_debug_low != EXPECTED_LOW
                || dut.adapter_debug_high != EXPECTED_HIGH
                || dut.adapter_debug_narrow != EXPECTED_NARROW
                || dut.adapter_debug_wide != EXPECTED_WIDE
                || dut.adapter_debug_contributions !=
                   EXPECTED_CONTRIBUTIONS
                || metadata_mismatches != 0
                || matcher_arithmetic_mismatches != 0
                || codec_arithmetic_mismatches != 0
                || reconstruction_mismatches != 0
                || bitmap_mismatches != 0
                || unknown_transaction_count != 0)
            $fatal(1,"M425 final ledger failure phases=%0d rows=%0d pwp_rows=%0d low=%0d high=%0d narrow=%0d wide=%0d contributions=%0d reconstructed_lanes=%0d zero=%0d pop1=%0d pass1=%0d early=%0d protocol=%0d busy=%0d metadata=%0d matcher=%0d codec=%0d reconstruction=%0d bitmap=%0d unknown=%0d",
                   checked_phases,checked_results,checked_pwp_rows,
                   checked_low,checked_high,checked_narrow,checked_wide,
                   checked_contributions,checked_reconstructed_lanes,
                   zero_rows,pop1_rows,dut.matcher_debug_pass1,
                   dut.matcher_debug_early,protocol_error,busy,
                   metadata_mismatches,matcher_arithmetic_mismatches,
                   codec_arithmetic_mismatches,reconstruction_mismatches,
                   bitmap_mismatches,unknown_transaction_count);
        $display("PASS M425 H67 balanced selected-slice direct-SAIF activity phases=64 rows=192000 pass0=192000 pass1=61285 early=11923 zero=93037 pop1=25755 pwp_rows=63067 low=504536 high=416630 narrow=87906 wide=416630 contributions=921166 reconstructed_lanes=48435456 metadata_mismatches=0 matcher_arithmetic_mismatches=0 codec_arithmetic_mismatches=0 reconstruction_mismatches=0 bitmap_mismatches=0 unknown_transactions=0 protocol_error=0 balanced_m414=true exploratory_pre_macro_power_activity=true paper_power_eligible=false power=false energy=false system_speedup=false headline=false measurement_cycles=%0d",
                 measurement_cycles);
        if ($test$plusargs("M425_UCLI_SAIF_STOP"))
            $stop;
        else
            $finish;
    end

    initial begin
        #20000000;
        $fatal(1,"M425 watchdog phases=%0d rows=%0d pwp_rows=%0d contributions=%0d busy=%0d error=%0d",
               checked_phases,checked_results,checked_pwp_rows,
               checked_contributions,busy,protocol_error);
    end
endmodule

`default_nettype wire
