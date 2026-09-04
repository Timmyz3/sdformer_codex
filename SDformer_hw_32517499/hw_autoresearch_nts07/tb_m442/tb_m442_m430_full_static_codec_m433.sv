`timescale 1ns/1ps
`default_nettype none

module tb_m442_m430_full_static_codec_m433;
    localparam int TAG_BITS = 24;
    localparam int EXPECTED_BLOCKS = 442368;
    localparam int EXPECTED_NARROW = 70503;

    logic clk_core, reset_n, config_reload;
    logic request_valid, request_ready, request_accept;
    logic [TAG_BITS-1:0] low_tag, high_tag;
    logic low_tile, high_tile;
    logic [4:0] low_center_id, high_center_id;
    logic [2:0] low_output_block, high_output_block;
    logic request_narrow;
    logic [767:0] low_data;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow;
    logic [1151:0] contribution_data;
    logic protocol_error, busy, debug_output_full;
    logic [31:0] debug_request_accepts, debug_narrow_accepts;
    logic [31:0] debug_wide_accepts, debug_contributions;
    logic [31:0] debug_protocol_faults;

    logic [TAG_BITS-1:0] expected_tag [0:3];
    logic expected_tile [0:3];
    logic [4:0] expected_center [0:3];
    logic [2:0] expected_block [0:3];
    logic expected_narrow [0:3];
    logic [1151:0] expected_data [0:3];
    integer expected_head, expected_tail, expected_count;
    integer accepted, retired, narrow_accepted, wide_accepted;
    integer metadata_mismatches, arithmetic_mismatches, unknown_outputs;
    integer simultaneous_pop_push, stall_cycles, maximum_queue_depth;
    integer source_lines, scan_rc, stimulus_fd, cycle_counter;
    string stimulus_path;
    logic [TAG_BITS-1:0] scan_tag;
    integer scan_tile, scan_center, scan_block, scan_narrow;
    logic [767:0] scan_low;
    logic [511:0] scan_high;
    logic [1151:0] scan_expected;

    m433_exact_dualbank_coread_pwp_adapter #(.TAG_BITS(TAG_BITS)) dut (.*);
    m433_exact_dualbank_coread_pwp_adapter_assertions #(
        .TAG_BITS(TAG_BITS)) m442_a_sva (.*);

    always #1.5 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (!reset_n) begin
            expected_head = 0;
            expected_tail = 0;
            expected_count = 0;
        end else begin
            if (contribution_accept) begin
                if (expected_count <= 0)
                    $fatal(1,"M442 unexpected contribution");
                if ($isunknown({contribution_tag,contribution_tile,
                                contribution_center_id,
                                contribution_output_block,
                                contribution_narrow,contribution_data})) begin
                    unknown_outputs++;
                    $fatal(1,"M442 unknown output tag=%0d",contribution_tag);
                end
                if (contribution_tag !== expected_tag[expected_head] ||
                        contribution_tile !== expected_tile[expected_head] ||
                        contribution_center_id !== expected_center[expected_head] ||
                        contribution_output_block !== expected_block[expected_head] ||
                        contribution_narrow !== expected_narrow[expected_head]) begin
                    metadata_mismatches++;
                    $fatal(1,"M442 metadata mismatch retired=%0d",retired);
                end
                if (contribution_data !== expected_data[expected_head]) begin
                    arithmetic_mismatches++;
                    $fatal(1,"M442 arithmetic mismatch retired=%0d xor=%h",
                           retired,contribution_data ^ expected_data[expected_head]);
                end
                expected_head = (expected_head + 1) & 3;
                expected_count--;
                retired++;
            end
            if (request_accept) begin
                if (expected_count >= 4)
                    $fatal(1,"M442 scoreboard overflow");
                expected_tag[expected_tail] = low_tag;
                expected_tile[expected_tail] = low_tile;
                expected_center[expected_tail] = low_center_id;
                expected_block[expected_tail] = low_output_block;
                expected_narrow[expected_tail] = request_narrow;
                expected_data[expected_tail] = scan_expected;
                expected_tail = (expected_tail + 1) & 3;
                expected_count++;
                accepted++;
                if (request_narrow) narrow_accepted++;
                else wide_accepted++;
            end
            if (request_accept && contribution_accept)
                simultaneous_pop_push++;
            if (contribution_valid && !contribution_ready)
                stall_cycles++;
            if (expected_count > maximum_queue_depth)
                maximum_queue_depth = expected_count;
            if (protocol_error)
                $fatal(1,"M442 unexpected fail-closed protocol error");
        end
    end

    initial begin : ready_driver
        contribution_ready = 1'b1;
        cycle_counter = 0;
        forever begin
            @(negedge clk_core);
            cycle_counter++;
            // Sparse deterministic stalls exercise payload retention without
            // turning this population replay into a random performance test.
            contribution_ready = ((cycle_counter % 4096) != 0);
        end
    end

    task automatic drive_scanned_record;
        begin
            low_tag = scan_tag;
            low_tile = scan_tile[0];
            low_center_id = scan_center[4:0];
            low_output_block = scan_block[2:0];
            request_narrow = scan_narrow[0];
            low_data = scan_low;
            if (scan_narrow != 0) begin
                high_tag = '0;
                high_tile = 1'b0;
                high_center_id = '0;
                high_output_block = '0;
                high_data = '0;
            end else begin
                high_tag = scan_tag;
                high_tile = scan_tile[0];
                high_center_id = scan_center[4:0];
                high_output_block = scan_block[2:0];
                high_data = scan_high;
            end
            request_valid = 1'b1;
            do @(posedge clk_core); while (!request_accept);
        end
    endtask

    initial begin : main
        clk_core = 1'b0;
        reset_n = 1'b0;
        config_reload = 1'b0;
        request_valid = 1'b0;
        low_tag = '0;
        low_tile = 1'b0;
        low_center_id = '0;
        low_output_block = '0;
        request_narrow = 1'b0;
        low_data = '0;
        high_tag = '0;
        high_tile = 1'b0;
        high_center_id = '0;
        high_output_block = '0;
        high_data = '0;
        accepted = 0;
        retired = 0;
        narrow_accepted = 0;
        wide_accepted = 0;
        metadata_mismatches = 0;
        arithmetic_mismatches = 0;
        unknown_outputs = 0;
        simultaneous_pop_push = 0;
        stall_cycles = 0;
        maximum_queue_depth = 0;
        source_lines = 0;
        if (!$value$plusargs("M442_STIMULUS=%s",stimulus_path))
            $fatal(1,"M442 missing +M442_STIMULUS=<path>");
        stimulus_fd = $fopen(stimulus_path,"r");
        if (stimulus_fd == 0)
            $fatal(1,"M442 cannot open stimulus %0s",stimulus_path);
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1'b1;

        while (!$feof(stimulus_fd)) begin
            scan_rc = $fscanf(stimulus_fd,"%h %d %h %h %d %h %h %h\n",
                              scan_tag,scan_tile,scan_center,scan_block,
                              scan_narrow,scan_low,scan_high,scan_expected);
            if (scan_rc == -1)
                break;
            if (scan_rc != 8)
                $fatal(1,"M442 malformed stimulus line=%0d fields=%0d",
                       source_lines,scan_rc);
            if (scan_narrow != 0 && scan_high !== 512'b0)
                $fatal(1,"M442 narrow high-side generator drift line=%0d",
                       source_lines);
            drive_scanned_record();
            source_lines++;
            @(negedge clk_core);
            request_valid = 1'b0;
        end
        $fclose(stimulus_fd);
        wait (retired == accepted && expected_count == 0 && !busy);
        repeat (3) @(posedge clk_core);
        if (source_lines != EXPECTED_BLOCKS || accepted != EXPECTED_BLOCKS ||
                retired != EXPECTED_BLOCKS ||
                narrow_accepted != EXPECTED_NARROW ||
                wide_accepted != EXPECTED_BLOCKS-EXPECTED_NARROW ||
                metadata_mismatches != 0 || arithmetic_mismatches != 0 ||
                unknown_outputs != 0 || debug_protocol_faults != 0 ||
                debug_request_accepts != EXPECTED_BLOCKS ||
                debug_contributions != EXPECTED_BLOCKS ||
                simultaneous_pop_push < EXPECTED_BLOCKS-1000 ||
                stall_cycles < 64 || maximum_queue_depth < 1)
            $fatal(1,"M442 final gate failed src=%0d acc=%0d ret=%0d n=%0d w=%0d meta=%0d arith=%0d unknown=%0d poppush=%0d stall=%0d q=%0d",
                   source_lines,accepted,retired,narrow_accepted,wide_accepted,
                   metadata_mismatches,arithmetic_mismatches,unknown_outputs,
                   simultaneous_pop_push,stall_cycles,maximum_queue_depth);
        $display("PASS M442 M430 full static codec through M433 blocks=%0d lanes=%0d narrow=%0d wide=%0d metadata_mismatches=0 arithmetic_mismatches=0 unknown_outputs=0 protocol_faults=0 pop_push=%0d stall_cycles=%0d max_queue=%0d runtime_issue_population=false cycles=false system_speedup=false power=false ppa=false headline=false",
                 retired,retired*96,narrow_accepted,wide_accepted,
                 simultaneous_pop_push,stall_cycles,maximum_queue_depth);
        $finish;
    end

    initial begin : watchdog
        repeat (900000) @(posedge clk_core);
        $fatal(1,"M442 watchdog accepted=%0d retired=%0d source=%0d",
               accepted,retired,source_lines);
    end
endmodule

`default_nettype wire
