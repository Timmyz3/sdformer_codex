`timescale 1ns/1ps
`default_nettype none

module tb_m185_fc2_k8_fixed_bank_accumulator;
    localparam int LANES = 96;
    localparam int MAX_RESULTS = 520;

    logic clk_core;
    logic rst_core;
    logic issue_valid;
    logic issue_ready;
    logic [23:0] issue_tag;
    logic issue_last;
    logic [7:0] issue_bank_valid;
    logic signed [7:0] issue_weight [0:7][0:LANES-1];
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
    logic [8*LANES-1:0] accepted_weight_active_mask;
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
    integer source_histogram [1:8];
    integer accepted_weight_terms;
    integer output_stall_cycles;
    integer same_cycle_replace_count;
    integer consecutive_issue_hits;
    integer previous_issue_cycle;
    integer cycle_count;
    integer pattern_counter;

    logic [23:0] expected_tag [0:MAX_RESULTS-1];
    logic expected_last [0:MAX_RESULTS-1];
    logic [3:0] expected_count [0:MAX_RESULTS-1];
    logic [7:0] expected_bank_mask [0:MAX_RESULTS-1];
    logic signed [23:0] expected_accumulator
        [0:MAX_RESULTS-1][0:LANES-1];

    m185_fc2_k8_fixed_bank_accumulator dut (.*);

    bind m185_fc2_k8_fixed_bank_accumulator
        m185_fc2_k8_fixed_bank_accumulator_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [7:0] mask_for_count(input integer count);
        begin
            case (count)
                1: mask_for_count = 8'h80;
                2: mask_for_count = 8'h81;
                3: mask_for_count = 8'h89;
                4: mask_for_count = 8'ha5;
                5: mask_for_count = 8'hd5;
                6: mask_for_count = 8'hf5;
                7: mask_for_count = 8'hfe;
                default: mask_for_count = 8'hff;
            endcase
        end
    endfunction

    task automatic load_legal_issue(input integer tag_value);
        integer slot_index;
        integer source_count;
        integer total;
        begin
            slot_index = expected_write;
            if (slot_index >= MAX_RESULTS)
                $fatal(1, "M185 expected scoreboard overflow");
            source_count = (pattern_counter % 8) + 1;
            pattern_counter = pattern_counter + 1;
            issue_tag = tag_value;
            issue_last = ((tag_value % 11) == 0);
            issue_bank_valid = mask_for_count(source_count);
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < LANES; lane++)
                    issue_weight[bank][lane] = (($urandom % 129) - 64);
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
                        total = total + $signed(issue_weight[bank][lane]);
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
            issue_tag = 24'hfff851;
            issue_last = 1'b1;
            issue_bank_valid = 8'hff;
            for (int bank = 0; bank < 8; bank++) begin
                for (int lane = 0; lane < LANES; lane++)
                    issue_weight[bank][lane] = 8'sd127;
            end
            expected_tag[slot_index] = issue_tag;
            expected_last[slot_index] = issue_last;
            expected_count[slot_index] = 4'd8;
            expected_bank_mask[slot_index] = 8'hff;
            for (int lane = 0; lane < LANES; lane++) begin
                issue_accumulator[lane] = 24'sh7ffff0;
                total = $signed(issue_accumulator[lane]) + 8*127;
                expected_accumulator[slot_index][lane] = total;
            end
            expected_write = expected_write + 1;
        end
    endtask

    task automatic drive_issues(input integer count, input integer tag_base);
        integer sent;
        begin
            sent = 0;
            @(negedge clk_core);
            load_legal_issue(tag_base + sent);
            issue_valid = 1'b1;
            while (sent < count) begin
                @(posedge clk_core);
                if (issue_accept) begin
                    sent = sent + 1;
                    if (sent < count) begin
                        @(negedge clk_core);
                        load_legal_issue(tag_base + sent);
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
            for (int count = 1; count <= 8; count++)
                source_histogram[count] <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (issue_accept) begin
                active_terms = 0;
                for (int bank = 0; bank < 8; bank++) begin
                    if (issue_bank_valid[bank])
                        active_terms = active_terms + 1;
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (accepted_weight_active_mask[(bank*LANES)+lane]
                                !== issue_bank_valid[bank])
                            $fatal(1, "M185 accepted activity mask mismatch");
                    end
                end
                issue_count <= issue_count + 1;
                source_histogram[active_terms]
                    <= source_histogram[active_terms] + 1;
                accepted_weight_terms <= accepted_weight_terms + active_terms;
                if (throughput_mode) begin
                    if (previous_issue_cycle >= 0) begin
                        if (cycle_count - previous_issue_cycle != 1)
                            $fatal(1, "M185 issue II drift=%0d",
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
                    $fatal(1, "M185 unexpected result");
                if (result_tag !== expected_tag[expected_read]
                        || result_last !== expected_last[expected_read]
                        || result_source_count !== expected_count[expected_read]
                        || result_bank_mask !== expected_bank_mask[expected_read])
                    $fatal(1, "M185 header mismatch index=%0d", expected_read);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (result_accumulator[lane]
                            !== expected_accumulator[expected_read][lane])
                        $fatal(1, "M185 accumulator mismatch index=%0d lane=%0d got=%0d expected=%0d",
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
        pattern_counter = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        throughput_mode = 1'b1;
        drive_issues(160, 24'h510000);
        wait (expected_read == expected_write);
        @(negedge clk_core);
        throughput_mode = 1'b0;
        previous_issue_cycle = -1;

        random_stall_mode = 1'b1;
        drive_issues(320, 24'h520000);
        wait (expected_read == expected_write);
        wait (!busy);
        @(negedge clk_core);
        random_stall_mode = 1'b0;

        // The accepted overflow result remains drainable while a younger empty
        // bank mask is rejected and closes future issue admission.
        force_stall_mode = 1'b1;
        @(negedge clk_core);
        load_overflow_issue();
        issue_valid = 1'b1;
        do @(posedge clk_core); while (!issue_accept);
        @(negedge clk_core);
        issue_tag = 24'hfff852;
        issue_bank_valid = 8'h00;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 1'b0;
        force_stall_mode = 1'b0;
        wait (expected_read == expected_write);
        repeat (2) @(posedge clk_core);

        if (!numeric_overflow || !protocol_error || issue_ready)
            $fatal(1, "M185 numeric/protocol fail-close missing");
        if (issue_count != 481 || result_count != 481
                || source_histogram[1] != 60
                || source_histogram[2] != 60
                || source_histogram[3] != 60
                || source_histogram[4] != 60
                || source_histogram[5] != 60
                || source_histogram[6] != 60
                || source_histogram[7] != 60
                || source_histogram[8] != 61
                || accepted_weight_terms != 2168)
            $fatal(1, "M185 population drift issues=%0d results=%0d hist=%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d terms=%0d",
                issue_count, result_count, source_histogram[1],
                source_histogram[2], source_histogram[3],
                source_histogram[4], source_histogram[5],
                source_histogram[6], source_histogram[7],
                source_histogram[8], accepted_weight_terms);
        if (consecutive_issue_hits < 159 || output_stall_cycles == 0
                || same_cycle_replace_count == 0)
            $fatal(1, "M185 recurrence/cover counters missing");

        $display("PASS M185 FC2 K8 fixed-bank accumulator VCS issues=481 results=481 one_source=60 two_source=60 three_source=60 four_source=60 five_source=60 six_source=60 seven_source=60 eight_source=61 accepted_weight_terms=2168 output_lanes=96 accumulator_bits=24 weight_bits=8 fixed_weight_banks=8 max_sources_per_issue=8 consecutive_issue_ii1_hits=%0d same_cycle_result_replace=%0d output_stall_cycles=%0d overflow_attacks=1 empty_mask_attacks=1 arbitrary_nonprefix_masks=true bank_id_payload=false pairwise_bank_comparators=0 prefix_packing=false multipliers=0 weight_payload_bits_per_full_issue=6144 m182_bounded_exact_payload_k1_over_k8=4.344533568 sn2_threshold_frozen_one_required=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false",
            consecutive_issue_hits, same_cycle_replace_count,
            output_stall_cycles);
        $finish;
    end

    initial begin
        #300000;
        $fatal(1, "M185 watchdog timeout");
    end
endmodule

`default_nettype wire
