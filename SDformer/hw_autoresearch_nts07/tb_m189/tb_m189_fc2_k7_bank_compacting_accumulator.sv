`timescale 1ns/1ps
`default_nettype none

module tb_m189_fc2_k7_bank_compacting_accumulator;
    localparam int LANES = 96;
    localparam int MAX_RESULTS = 620;

    logic clk_core;
    logic rst_core;
    logic issue_valid;
    logic issue_ready;
    logic [23:0] issue_tag;
    logic issue_last;
    logic [7:0] issue_bank_valid;
    logic signed [7:0] issue_weight_bank [0:7][0:LANES-1];
    logic signed [23:0] issue_accumulator [0:LANES-1];
    logic issue_accept;
    logic result_valid;
    logic result_ready;
    logic [23:0] result_tag;
    logic result_last;
    logic [3:0] result_source_count;
    logic [7:0] result_bank_mask;
    logic signed [23:0] result_accumulator [0:LANES-1];
    logic result_accept;
    logic [8*LANES-1:0] accepted_weight_bank_active_mask;
    logic [7*LANES-1:0] accepted_compacted_lane_active_mask;
    logic protocol_error;
    logic numeric_overflow;
    logic busy;

    logic random_stall_mode;
    logic force_stall_mode;
    logic throughput_mode;
    integer expected_write;
    integer expected_read;
    integer issue_count;
    integer result_count;
    integer source_histogram [1:7];
    integer accepted_weight_terms;
    integer output_stall_cycles;
    integer same_cycle_replace_count;
    integer consecutive_issue_hits;
    integer previous_issue_cycle;
    integer cycle_count;

    logic [23:0] expected_tag [0:MAX_RESULTS-1];
    logic expected_last [0:MAX_RESULTS-1];
    logic [3:0] expected_count [0:MAX_RESULTS-1];
    logic [7:0] expected_bank_mask [0:MAX_RESULTS-1];
    logic signed [23:0] expected_accumulator
        [0:MAX_RESULTS-1][0:LANES-1];

    m189_fc2_k7_bank_compacting_accumulator dut (.*);

    bind m189_fc2_k7_bank_compacting_accumulator
        m189_fc2_k7_bank_compacting_accumulator_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic integer popcount8(input logic [7:0] mask);
        begin
            popcount8 = 0;
            for (int bank = 0; bank < 8; bank++)
                popcount8 = popcount8 + mask[bank];
        end
    endfunction

    task automatic load_legal_issue(
        input integer tag_value,
        input logic [7:0] bank_mask
    );
        integer slot_index;
        integer source_count;
        integer total;
        begin
            slot_index = expected_write;
            if (slot_index >= MAX_RESULTS)
                $fatal(1, "M189 expected scoreboard overflow");
            source_count = popcount8(bank_mask);
            if (source_count < 1 || source_count > 7)
                $fatal(1, "M189 TB attempted illegal legal issue mask=%02x",
                    bank_mask);
            issue_tag = tag_value;
            issue_last = ((tag_value % 11) == 0);
            issue_bank_valid = bank_mask;
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < LANES; lane++)
                    issue_weight_bank[bank][lane]
                        = (($urandom % 255) - 127);
            end
            expected_tag[slot_index] = issue_tag;
            expected_last[slot_index] = issue_last;
            expected_count[slot_index] = source_count;
            expected_bank_mask[slot_index] = issue_bank_valid;
            for (int lane = 0; lane < LANES; lane++) begin
                issue_accumulator[lane] = (($urandom % 100001) - 50000);
                total = $signed(issue_accumulator[lane]);
                for (int bank = 0; bank < 8; bank++) begin
                    if (issue_bank_valid[bank])
                        total = total + $signed(issue_weight_bank[bank][lane]);
                end
                expected_accumulator[slot_index][lane] = total;
            end
            expected_write = expected_write + 1;
        end
    endtask

    task automatic load_overflow_issue;
        integer slot_index;
        integer total;
        begin
            slot_index = expected_write;
            issue_tag = 24'hfff891;
            issue_last = 1'b1;
            issue_bank_valid = 8'hfe;
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < LANES; lane++)
                    issue_weight_bank[bank][lane] = 8'sd127;
            end
            expected_tag[slot_index] = issue_tag;
            expected_last[slot_index] = issue_last;
            expected_count[slot_index] = 4'd7;
            expected_bank_mask[slot_index] = 8'hfe;
            for (int lane = 0; lane < LANES; lane++) begin
                issue_accumulator[lane] = 24'sh7ffff0;
                total = $signed(issue_accumulator[lane]) + 7*127;
                expected_accumulator[slot_index][lane] = total;
            end
            expected_write = expected_write + 1;
        end
    endtask

    task automatic drive_all_legal_masks(input integer tag_base);
        integer sent;
        begin
            sent = 1;
            @(negedge clk_core);
            load_legal_issue(tag_base, sent[7:0]);
            issue_valid = 1'b1;
            while (sent <= 254) begin
                @(posedge clk_core);
                if (issue_accept) begin
                    sent = sent + 1;
                    if (sent <= 254) begin
                        @(negedge clk_core);
                        load_legal_issue(tag_base + sent - 1, sent[7:0]);
                    end else begin
                        @(negedge clk_core);
                        issue_valid = 1'b0;
                    end
                end
            end
        end
    endtask

    task automatic drive_pattern_issues(
        input integer count,
        input integer tag_base
    );
        integer sent;
        integer mask_value;
        begin
            sent = 0;
            mask_value = 1;
            @(negedge clk_core);
            load_legal_issue(tag_base, mask_value[7:0]);
            issue_valid = 1'b1;
            while (sent < count) begin
                @(posedge clk_core);
                if (issue_accept) begin
                    sent = sent + 1;
                    if (sent < count) begin
                        mask_value = (sent % 254) + 1;
                        @(negedge clk_core);
                        load_legal_issue(tag_base + sent,
                            mask_value[7:0]);
                    end else begin
                        @(negedge clk_core);
                        issue_valid = 1'b0;
                    end
                end
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core || force_stall_mode)
            result_ready <= 1'b0;
        else if (random_stall_mode)
            result_ready <= ($urandom_range(0, 3) != 0);
        else
            result_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        integer active_terms;
        if (rst_core) begin
            cycle_count <= 0;
            issue_count <= 0;
            result_count <= 0;
            accepted_weight_terms <= 0;
            output_stall_cycles <= 0;
            same_cycle_replace_count <= 0;
            consecutive_issue_hits <= 0;
            previous_issue_cycle <= -1;
            for (int count = 1; count <= 7; count++)
                source_histogram[count] <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (issue_accept) begin
                active_terms = popcount8(issue_bank_valid);
                issue_count <= issue_count + 1;
                source_histogram[active_terms]
                    <= source_histogram[active_terms] + 1;
                accepted_weight_terms <= accepted_weight_terms + active_terms;
                for (int bank = 0; bank < 8; bank++) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (accepted_weight_bank_active_mask[(bank*LANES)+lane]
                                !== issue_bank_valid[bank])
                            $fatal(1, "M189 structural activity mismatch");
                    end
                end
                for (int slot = 0; slot < 7; slot++) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (accepted_compacted_lane_active_mask[(slot*LANES)+lane]
                                !== (slot < active_terms))
                            $fatal(1, "M189 compacted activity mismatch");
                    end
                end
                if (throughput_mode) begin
                    if (previous_issue_cycle >= 0) begin
                        if (cycle_count - previous_issue_cycle != 1)
                            $fatal(1, "M189 issue II drift=%0d",
                                cycle_count - previous_issue_cycle);
                        consecutive_issue_hits <= consecutive_issue_hits + 1;
                    end
                    previous_issue_cycle <= cycle_count;
                end
            end
            if (result_valid && !result_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (result_accept && issue_accept)
                same_cycle_replace_count <= same_cycle_replace_count + 1;
            if (result_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M189 unexpected result");
                if (result_tag !== expected_tag[expected_read]
                        || result_last !== expected_last[expected_read]
                        || result_source_count !== expected_count[expected_read]
                        || result_bank_mask !== expected_bank_mask[expected_read])
                    $fatal(1, "M189 header mismatch index=%0d", expected_read);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (result_accumulator[lane]
                            !== expected_accumulator[expected_read][lane])
                        $fatal(1, "M189 accumulator mismatch index=%0d lane=%0d got=%0d expected=%0d",
                            expected_read, lane, result_accumulator[lane],
                            expected_accumulator[expected_read][lane]);
                end
                expected_read <= expected_read + 1;
                result_count <= result_count + 1;
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        issue_valid = 1'b0;
        issue_tag = '0;
        issue_last = 1'b0;
        issue_bank_valid = '0;
        result_ready = 1'b0;
        random_stall_mode = 1'b0;
        force_stall_mode = 1'b0;
        throughput_mode = 1'b0;
        expected_write = 0;
        expected_read = 0;
        issue_count = 0;
        result_count = 0;
        accepted_weight_terms = 0;
        output_stall_cycles = 0;
        same_cycle_replace_count = 0;
        consecutive_issue_hits = 0;
        previous_issue_cycle = -1;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        throughput_mode = 1'b1;
        drive_all_legal_masks(24'h610000);
        wait (expected_read == expected_write);
        @(negedge clk_core);
        throughput_mode = 1'b0;
        previous_issue_cycle = -1;

        random_stall_mode = 1'b1;
        drive_pattern_issues(320, 24'h620000);
        wait (expected_read == expected_write);
        wait (!busy);
        @(negedge clk_core);
        random_stall_mode = 1'b0;

        // A legal seven-source request overflows Acc24 and remains drainable.
        force_stall_mode = 1'b1;
        @(negedge clk_core);
        load_overflow_issue();
        issue_valid = 1'b1;
        do @(posedge clk_core); while (!issue_accept);
        @(negedge clk_core);
        issue_tag = 24'hfff892;
        issue_bank_valid = 8'hff;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_tag = 24'hfff893;
        issue_bank_valid = 8'h00;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 1'b0;
        force_stall_mode = 1'b0;
        wait (expected_read == expected_write);
        repeat (2) @(posedge clk_core);

        if (!numeric_overflow || !protocol_error || issue_ready)
            $fatal(1, "M189 numeric/protocol fail-close missing");
        if (issue_count != 575 || result_count != 575
                || source_histogram[1] != 23
                || source_histogram[2] != 73
                || source_histogram[3] != 132
                || source_histogram[4] != 155
                || source_histogram[5] != 118
                || source_histogram[6] != 57
                || source_histogram[7] != 17
                || accepted_weight_terms != 2236)
            $fatal(1, "M189 population drift issues=%0d results=%0d hist=%0d/%0d/%0d/%0d/%0d/%0d/%0d terms=%0d",
                issue_count, result_count, source_histogram[1],
                source_histogram[2], source_histogram[3],
                source_histogram[4], source_histogram[5],
                source_histogram[6], source_histogram[7],
                accepted_weight_terms);
        if (consecutive_issue_hits < 253 || output_stall_cycles == 0
                || same_cycle_replace_count == 0)
            $fatal(1, "M189 recurrence/cover counters missing");

        $display("PASS M189 FC2 K7 bank-compacting accumulator VCS issues=575 results=575 one_source=23 two_source=73 three_source=132 four_source=155 five_source=118 six_source=57 seven_source=17 accepted_weight_terms=2236 legal_masks_exhausted=254 output_lanes=96 accumulator_bits=24 weight_bits=8 structural_weight_banks=8 compacted_weight_lanes=7 max_sources_per_issue=7 consecutive_issue_ii1_hits=%0d same_cycle_result_replace=%0d output_stall_cycles=%0d overflow_attacks=1 empty_mask_attacks=1 full_mask_attacks=1 arbitrary_nonprefix_masks=true increasing_bank_order_compaction=true multipliers=0 structural_input_bits=6144 compacted_internal_bits=5376 nominal_lane_reduction_percent=12.5 m187_k7_over_k8_cycle_penalty_percent=0.088857646 sn2_threshold_frozen_one_required=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false",
            consecutive_issue_hits, same_cycle_replace_count,
            output_stall_cycles);
        $finish;
    end

    initial begin
        #400000;
        $fatal(1, "M189 watchdog timeout");
    end
endmodule

`default_nettype wire
